"""
SurgicalMamba — shared-input dual-path SSM for surgical phase recognition.

Architecture (within each SurgicalMamba call):

1. Shared input, separate projections:
   - Fast: in_proj(h) → x, z     (d_model → 2*d_inner)
   - Slow: in_proj_slow(h) → x_slow  (d_model → d_inner)
   Same input h, different learned projections extract fast/slow aspects.

2. Dual SSM scan:
   - Fast SSM: CUDA parallel scan on x. Zero-init per clip.
   - Slow SSM: PyTorch sequential scan on x_slow. State carries across clips.

3. Outputs:
   - Main: (scale_fast*y_fast + scale_slow*y_slow) * silu(z) → norm → out_proj

Cross-clip state carry: forward() accepts/returns (slow_ssm_state, slow_conv_state).
Both slow SSM state and slow conv state carry across clips.
Fast state is always zero-init per clip (handled by selective_scan_fn).
Step-wise interface (step()) packs fast||slow states for online inference.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None


class _PerHeadMLP(nn.Module):
    """Per-head 2-layer MLP: (B, H, in_dim) → (B, H, out_dim).

    Implemented via batched einsum with head-specific weights (no Python loop).
    """

    def __init__(self, n_heads: int, in_dim: int, hidden_dim: int, out_dim: int,
                 w1_gain: float = 1.0, w2_gain: float = 1.0,
                 device=None, dtype=None):
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.W1 = nn.Parameter(torch.empty(n_heads, hidden_dim, in_dim, **factory))
        self.b1 = nn.Parameter(torch.zeros(n_heads, hidden_dim, **factory))
        self.W2 = nn.Parameter(torch.empty(n_heads, out_dim, hidden_dim, **factory))
        self.b2 = nn.Parameter(torch.zeros(n_heads, out_dim, **factory))
        with torch.no_grad():
            for h in range(n_heads):
                nn.init.xavier_normal_(self.W1[h], gain=w1_gain)
                nn.init.xavier_normal_(self.W2[h], gain=w2_gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.einsum("bhp,hop->bho", x, self.W1.float()) + self.b1.float().unsqueeze(0)
        h = F.silu(h)
        return torch.einsum("bho,hqo->bhq", h, self.W2.float()) + self.b2.float().unsqueeze(0)


def _graph_safe_matrix_exp(Q, scale_power: int = 5, taylor_order: int = 6):
    """CUDA graph-safe matrix exponential via fixed scaling-and-squaring.

    Avoids torch.matrix_exp's dynamic branching (norm-dependent Padé order selection).
    exp(Q) = (exp(Q / 2^s))^(2^s), with exp(Q/2^s) via fixed-order Taylor.

    For ||Q||_op up to ~10, scale_power=5 + taylor_order=6 gives ~1e-6 accuracy.
    Cost: (taylor_order + scale_power) matmuls — negligible for small n.

    Args:
        Q: (..., n, n) matrix
        scale_power: s, scales Q → Q/2^s before Taylor
        taylor_order: k, Taylor truncation order
    Returns:
        exp(Q): (..., n, n)
    """
    Q_scaled = Q / (2 ** scale_power)
    eye = torch.eye(Q.shape[-1], device=Q.device, dtype=Q.dtype)
    T = eye.expand_as(Q).clone()
    term = eye.expand_as(Q).clone()
    for k in range(1, taylor_order + 1):
        term = torch.matmul(term, Q_scaled) / k
        T = T + term
    for _ in range(scale_power):
        T = torch.matmul(T, T)
    return T


class SurgicalMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_state_slow=None,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        slow_decay_scale=0.5,
        slow_dt_shift=-2.0,
        conv_bias=True,
        bias=False,
        # Ablation flags — defaults preserve full-model behavior.
        t_rank=16,
        chunk_size=32,           # symmetric default (used if *_fast / *_slow not set)
        chunk_size_fast=None,    # override fast-path SSD/T_state chunk granularity
        chunk_size_slow=None,    # override slow-path SSD/T_state chunk granularity
        use_t_state=True,        # per-chunk Cayley orthogonal rotation
        use_intensity=True,      # λ time-warp modulation on slow dt
        use_fast_path=True,      # fast SSM path (per-clip fresh, gated, conditioned)
        use_slow_path=True,      # slow SSM path (cross-clip carry, λ-modulated)
        device=None,
        dtype=None,
    ):
        # Sanity: at least one path must be active.
        assert use_fast_path or use_slow_path, "At least one of fast/slow path must be enabled."
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.use_t_state    = use_t_state
        self.use_intensity  = use_intensity
        self.use_fast_path  = use_fast_path
        self.use_slow_path  = use_slow_path

        self.d_model = d_model
        self.d_state = d_state
        self.d_state_slow = d_state_slow if d_state_slow is not None else d_state
        self.d_conv  = d_conv
        self.expand  = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.activation = "silu"
        self.act = nn.SiLU()

        # ── Fast path input projection (d_model → 2 * d_inner: x + z) ──
        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )

        # ── Slow path input projection (d_model → d_inner, no z gate) ──
        # Slow keeps long-context flow uninterrupted (no output gating by design).
        self.in_proj_slow = nn.Linear(
            self.d_model, self.d_inner, bias=bias, **factory_kwargs
        )

        # ── Fast path internals (Mamba2 SSD multi-head) ──────────────
        #   Mamba2-standard split: d_head=64, n_heads=24 (= d_inner/64).
        #   ngroups=4 → B/C split into 4 groups (6 heads/group share B/C) → state diversity.
        self.d_head_fast = 64
        assert self.d_inner % self.d_head_fast == 0, \
            f"d_inner ({self.d_inner}) must be divisible by d_head_fast ({self.d_head_fast})"
        self.n_heads_fast = self.d_inner // self.d_head_fast
        # ngroups = n_heads → per-head independent B, C (max diversity).
        self.ngroups_fast = self.n_heads_fast
        self.chunk_size_fast = chunk_size_fast if chunk_size_fast is not None else chunk_size

        self.conv1d_fast = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1,
            bias=conv_bias, **factory_kwargs,
        )
        # x_proj_fast takes [x_conv_fast, y_slow] concatenated → dt_rank + 2*ngroups*d_state
        self.x_proj_fast = nn.Linear(
            self.d_inner * 2,
            self.dt_rank + 2 * self.ngroups_fast * self.d_state,
            bias=False, **factory_kwargs,
        )

        # ── Intensity head: scalar output per frame ──────────
        #   Simplified: d_inner → 1 scalar λ per frame (broadcast to all heads).
        #   Reduces per-frame fluctuation (single scalar less noisy than 24-dim).
        intensity_bottleneck = max(self.d_inner // 16, 8)
        self.intensity_net = nn.Sequential(
            nn.Linear(self.d_inner, intensity_bottleneck, bias=True, **factory_kwargs),
            nn.SiLU(),
            nn.Linear(intensity_bottleneck, 1, bias=True, **factory_kwargs),
        )
        # dt_proj: dt_rank → n_heads (per-head dt, not per-channel)
        self.dt_proj_fast = nn.Linear(
            self.dt_rank, self.n_heads_fast, bias=True, **factory_kwargs,
        )
        # A per-head scalar, init uniform in (1, d_state) then log.
        A_fast = torch.empty(self.n_heads_fast, dtype=torch.float32, device=device).uniform_(1.0, float(self.d_state))
        self.A_log_fast = nn.Parameter(torch.log(A_fast))
        self.A_log_fast._no_weight_decay = True
        # D per-head scalar
        self.D_fast = nn.Parameter(torch.ones(self.n_heads_fast, device=device))
        self.D_fast._no_weight_decay = True

        # ── Slow path internals (Mamba2 SSD multi-head) ────────────────
        #   d_head=64, ngroups=n_heads (per-head B,C), chunked scan with carry.
        #   T_state (per-head, low-rank) applied at each chunk boundary to mix h.
        self.d_head_slow = 64
        assert self.d_inner % self.d_head_slow == 0, \
            f"d_inner ({self.d_inner}) must be divisible by d_head_slow ({self.d_head_slow})"
        self.n_heads_slow = self.d_inner // self.d_head_slow
        # ngroups = n_heads → per-head independent B, C projection (max diversity).
        self.ngroups_slow = self.n_heads_slow
        self.chunk_size_slow = chunk_size_slow if chunk_size_slow is not None else chunk_size

        self.conv1d_slow = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1,
            bias=conv_bias, **factory_kwargs,
        )
        # x_proj output: dt_rank + 2*ngroups*d_state (B and C)
        self.x_proj_slow = nn.Linear(
            self.d_inner,
            self.dt_rank + 2 * self.ngroups_slow * self.d_state_slow,
            bias=False, **factory_kwargs,
        )
        # dt_proj: dt_rank → n_heads (one dt per head, not per channel)
        self.dt_proj_slow = nn.Linear(
            self.dt_rank, self.n_heads_slow, bias=True, **factory_kwargs,
        )
        # A per-head scalar — uniform(1, d_state_slow), same init range as fast.
        A_slow = torch.empty(self.n_heads_slow, dtype=torch.float32, device=device).uniform_(1.0, float(self.d_state_slow))
        self.A_log_slow = nn.Parameter(torch.log(A_slow))
        self.A_log_slow._no_weight_decay = True
        # D per-head scalar
        self.D_slow = nn.Parameter(torch.ones(self.n_heads_slow, device=device))
        self.D_slow._no_weight_decay = True

        # ── Chunk-level state-space SVD filter (per-head MLP, batched einsum) ──
        # Each head has its OWN MLP weights (W1, b1, W2, b2) — strong differentiation.
        # Computed via batched einsum (no Python loop) for GPU efficiency.
        n_slow = self.d_state_slow
        self.T_rank = t_rank
        stab_hidden = 128

        # ── Slow per-head MLP — UV shared trunk (output split), σ separate trunk ──
        H_s = self.n_heads_slow
        r = self.T_rank
        self.UV_mlp_slow = _PerHeadMLP(H_s, self.d_head_slow, stab_hidden,
                                        2 * n_slow * r, w1_gain=1.0, w2_gain=2.0,
                                        **factory_kwargs)
        self.S_mlp_slow  = _PerHeadMLP(H_s, self.d_head_slow, stab_hidden,
                                        r, w1_gain=1.0, w2_gain=2.0,
                                        **factory_kwargs)
        # Init UV bias — per-head Gram-Schmidt: U_init's orthogonal across heads,
        # V_init's orthogonal across heads (in flattened n*r space). Promotes
        # head diversity at init (lowers cold-start headMsim).
        # σ via softplus (unbounded); σ_logit bias = 0.5 → σ ≈ 0.97.
        _slow_total = n_slow * r
        _slow_scale = 0.5 * math.sqrt(_slow_total)        # match (randn*0.5).norm
        _Q_U_slow, _ = torch.linalg.qr(torch.randn(_slow_total, H_s))   # (n*r, H_s) orthonormal cols
        _Q_V_slow, _ = torch.linalg.qr(torch.randn(_slow_total, H_s))
        with torch.no_grad():
            self.S_mlp_slow.b2.fill_(0.3)
            for h in range(H_s):
                U_init = _Q_U_slow[:, h] * _slow_scale     # (n*r,)
                V_init = _Q_V_slow[:, h] * _slow_scale
                self.UV_mlp_slow.b2[h].copy_(torch.cat([U_init, V_init]).to(
                    self.UV_mlp_slow.b2.device, dtype=self.UV_mlp_slow.b2.dtype))

        # ── Fast per-head MLP — UV shared trunk (output split), σ separate trunk ──
        H_f = self.n_heads_fast
        n_f = self.d_state
        self.UV_mlp_fast = _PerHeadMLP(H_f, self.d_head_fast, stab_hidden,
                                        2 * n_f * r, w1_gain=1.0, w2_gain=2.0,
                                        **factory_kwargs)
        self.S_mlp_fast  = _PerHeadMLP(H_f, self.d_head_fast, stab_hidden,
                                        r, w1_gain=1.0, w2_gain=2.0,
                                        **factory_kwargs)
        # Same Gram-Schmidt orthogonalization for fast path
        _fast_total = n_f * r
        _fast_scale = 0.5 * math.sqrt(_fast_total)
        _Q_U_fast, _ = torch.linalg.qr(torch.randn(_fast_total, H_f))
        _Q_V_fast, _ = torch.linalg.qr(torch.randn(_fast_total, H_f))
        with torch.no_grad():
            self.S_mlp_fast.b2.fill_(0.3)
            for h in range(H_f):
                U_init = _Q_U_fast[:, h] * _fast_scale
                V_init = _Q_V_fast[:, h] * _fast_scale
                self.UV_mlp_fast.b2[h].copy_(torch.cat([U_init, V_init]).to(
                    self.UV_mlp_fast.b2.device, dtype=self.UV_mlp_fast.b2.dtype))

        # UV / σ MLPs: receive weight decay (regularization to prevent overfit).
        # Only A_log, D scalars are excluded from WD (above).

        self.register_buffer("_T_identity", torch.eye(n_slow))

        # ── dt initialization ────────────────────────────────────────
        # Fast path (Mamba2): per-head dt bias (n_heads_fast,)
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj_fast.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj_fast.weight, -dt_init_std, dt_init_std)
        dt_fast = torch.exp(
            torch.rand(self.n_heads_fast, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt_fast = dt_fast + torch.log(-torch.expm1(-dt_fast))
        with torch.no_grad():
            self.dt_proj_fast.bias.copy_(inv_dt_fast)
        self.dt_proj_fast.bias._no_reinit = True

        # Slow path (Mamba2): per-head dt bias (n_heads_slow,)
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj_slow.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj_slow.weight, -dt_init_std, dt_init_std)
        dt_slow = torch.exp(
            torch.rand(self.n_heads_slow, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt_slow = dt_slow + torch.log(-torch.expm1(-dt_slow))
        with torch.no_grad():
            self.dt_proj_slow.bias.copy_(inv_dt_slow)
            self.dt_proj_slow.bias.data += slow_dt_shift
        self.dt_proj_slow.bias._no_reinit = True

        # ── Per-channel learnable scale for fast/slow balancing (sum fusion) ──
        self.scale_fast = nn.Parameter(torch.ones(self.d_inner, **factory_kwargs))
        self.scale_slow = nn.Parameter(torch.ones(self.d_inner, **factory_kwargs))

        # ── Pre-projection normalization (Mamba2 standard) ──
        self.norm_pre_out = nn.RMSNorm(self.d_inner, eps=1e-5, **factory_kwargs)

        # ── Output projection ────────────────────────────────────────
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )

    # ── Properties for external state allocation ─────────────────────

    @property
    def state_d_inner(self):
        """d_inner for state allocation — doubled because fast||slow packed."""
        return self.d_inner * 2

    # ── Forward (batch, parallel scan) ───────────────────────────────

    def forward(self, hidden_states, slow_state=None):
        """
        Shared-input forward: same input feeds both fast and slow paths.

        Args:
            hidden_states: (B, L, D) — single stream input
            slow_state:    (ssm, conv) or None — carried across clips
        Returns:
            out:            (B, L, D)
            new_slow_state: (ssm, conv)
            lam_seq:        (B, L) intensity logits (for BCE)
        """
        batch, seqlen, dim = hidden_states.shape

        # ── Fast path input (shared → x, z) ──────────────────────────────────
        # Keep in_proj alive whenever the module exists so that even no_fast
        # retains the z gate (only the x half / fast scan is removed).
        if getattr(self, "in_proj", None) is not None:
            xz = rearrange(
                self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
                "d (b l) -> b d l", l=seqlen,
            )
            if self.in_proj.bias is not None:
                xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
            x, z = xz.chunk(2, dim=1)
            if not self.use_fast_path:
                x = None       # discard x; fast SSM scan is skipped
        else:
            x, z = None, None

        # ── Slow path input — gated on use_slow_path. ──────────────────────────
        if self.use_slow_path:
            x_slow = rearrange(
                self.in_proj_slow.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
                "d (b l) -> b d l", l=seqlen,
            )
            if self.in_proj_slow.bias is not None:
                x_slow = x_slow + rearrange(self.in_proj_slow.bias.to(dtype=x_slow.dtype), "d -> d 1")
        else:
            x_slow = None

        if slow_state is not None:
            slow_ssm_state, slow_conv_state = slow_state[0], slow_state[1]
        else:
            slow_ssm_state, slow_conv_state = None, None

        # ── 1. Slow path (intensity-modulated dt + chunk T_state) ──
        ref_dtype = (x if x is not None else x_slow).dtype
        if self.use_slow_path:
            y_slow, new_slow_ssm, new_slow_conv, lam_seq = self._scan_slow(
                x_slow, seqlen, slow_ssm_state, slow_conv_state,
            )
        else:
            y_slow = torch.zeros(batch, self.d_inner, seqlen,
                                  device=hidden_states.device, dtype=ref_dtype)
            new_slow_ssm = slow_ssm_state
            new_slow_conv = slow_conv_state
            lam_seq = None

        # ── 2. Fast path (chunked T_state, conditioned on y_slow, no λ) ──
        if self.use_fast_path:
            y_fast = self._scan_fast(x, seqlen, condition=(y_slow if self.use_slow_path else None))
        else:
            y_fast = torch.zeros_like(y_slow)

        # ── Sum fusion + gating ──────────────────────────────────────────────
        # Ablation variants may omit scale_fast/scale_slow (when a path is removed).
        sf = self.scale_fast.view(1, -1, 1) if getattr(self, "scale_fast", None) is not None else 1.0
        ss = self.scale_slow.view(1, -1, 1) if getattr(self, "scale_slow", None) is not None else 1.0
        y = sf * y_fast + ss * y_slow
        if z is not None:                       # gate only when fast path provides z
            y = y * self.act(z)
        y = rearrange(y, "b d l -> b l d")
        y = self.norm_pre_out(y)
        out = self.out_proj(y)

        # When slow path is disabled there is no state to carry — return None
        # for the entire tuple so train.py's detach loop short-circuits cleanly.
        new_slow_state = (new_slow_ssm, new_slow_conv) if self.use_slow_path else None
        return out, new_slow_state, lam_seq

    def _scan_fast(self, x, seqlen, condition=None):
        """Fast path: chunked SSD scan with per-chunk T_state (per-clip fresh state).
        condition: y_slow (B, d_inner, L) — slow path context for conditioning.
        No intensity modulation (λ moved to slow path).
        """
        # ── Conv (no carry-over: each clip starts fresh) ─────────
        if causal_conv1d_fn is None:
            x_conv = self.act(self.conv1d_fast(x)[..., :seqlen])
        else:
            x_conv = causal_conv1d_fn(
                x, rearrange(self.conv1d_fast.weight, "d 1 w -> d w"),
                self.conv1d_fast.bias, activation=self.activation,
            )

        H = self.n_heads_fast
        P = self.d_head_fast
        G = self.ngroups_fast
        n_f = self.d_state

        x_conv_flat = rearrange(x_conv, "b d l -> (b l) d")                   # (B*L, d_inner)

        # ── Conditioned x_proj: [x_conv_fast, y_slow] → dt, B, C ──
        # LayerNorm each stream so y_slow's larger magnitude doesn't drown out
        # x_conv in the fusion linear (which would make fast dt track y_slow only).
        if condition is not None:
            cond_flat = rearrange(condition, "b d l -> (b l) d")
            fusion_input = torch.cat([x_conv_flat, cond_flat], dim=-1)
        else:
            fusion_input = torch.cat([x_conv_flat, torch.zeros_like(x_conv_flat)], dim=-1)

        x_dbl = self.x_proj_fast(fusion_input)
        dt_raw, B_all, C_all = torch.split(
            x_dbl, [self.dt_rank, G * n_f, G * n_f], dim=-1,
        )
        # dt per-head: dt_rank → n_heads, softplus + bias (no intensity modulation)
        dt_all = self.dt_proj_fast.weight @ dt_raw.t()                        # (H, B*L)
        dt_all = rearrange(dt_all, "h (b l) -> b l h", l=seqlen).contiguous()

        B_all = rearrange(B_all, "(b l) (g n) -> b l g n", l=seqlen, g=G).contiguous()
        C_all = rearrange(C_all, "(b l) (g n) -> b l g n", l=seqlen, g=G).contiguous()
        x_rs  = rearrange(x_conv, "b (h p) l -> b l h p", p=P).contiguous()

        A_fast  = -torch.exp(self.A_log_fast.float())                         # (H,)
        D_fast  = self.D_fast.float()                                         # (H,)
        dt_bias_f = self.dt_proj_fast.bias.float()                            # (H,)

        # ── Chunked SSD loop with per-chunk T_state ──────────────
        inp_dtype = x.dtype
        chunk_size = self.chunk_size_fast
        n_chunks = (seqlen + chunk_size - 1) // chunk_size
        outputs = []
        h = None  # fresh per clip

        r_f = self.T_rank
        for c in range(n_chunks):
            t_s = c * chunk_size
            t_e = min(t_s + chunk_size, seqlen)
            cs = t_e - t_s
            y_ch, h = mamba_chunk_scan_combined(
                x_rs[:, t_s:t_e],
                dt_all[:, t_s:t_e],
                A_fast,
                B_all[:, t_s:t_e],
                C_all[:, t_s:t_e],
                chunk_size=cs,
                D=D_fast,
                dt_bias=dt_bias_f,
                dt_softplus=True,
                initial_states=h,
                return_final_states=True,
            )
            y_ch = rearrange(y_ch, "b l h p -> b (h p) l").contiguous()       # (B, d_inner, cs)

            if self.use_t_state:
                # Per-chunk SVD filter (fast): UV shared MLP (output split) + σ separate.
                B_ = y_ch.shape[0]
                chunk_feat_ph = y_ch.view(B_, H, P, cs).mean(dim=-1).float()      # (B, H, P)
                chunk_feat_ph = F.layer_norm(chunk_feat_ph, [P])
                UV_flat = self.UV_mlp_fast(chunk_feat_ph)                           # (B, H, 2nr)
                U_flat, V_flat = UV_flat.chunk(2, dim=-1)
                U = F.normalize(U_flat.view(B_, H, n_f, r_f), dim=-2)
                V = F.normalize(V_flat.view(B_, H, n_f, r_f), dim=-2)
                sigma_logit = self.S_mlp_fast(chunk_feat_ph)                        # (B, H, r)
                with torch.amp.autocast("cuda", enabled=False):
                    sigma = F.softplus(sigma_logit.float())
                    UV = torch.einsum("bhnr,bhr,bhmr->bhnm", U.float(), sigma, V.float())
                    S = UV - UV.transpose(-2, -1)
                    I_S = torch.eye(n_f, device=S.device, dtype=S.dtype).expand_as(S)
                    S_half = S * 0.5
                    M = torch.linalg.solve(I_S - S_half, I_S + S_half)              # Cayley
                    h = torch.einsum("bhpn,bhnm->bhpm", h.float(), M)

            outputs.append(y_ch.to(inp_dtype))

        y = torch.cat(outputs, dim=2)                                         # (B, d_inner, L)
        return y

    def _scan_slow(self, x, seqlen, initial_state=None, conv_init=None):
        """Slow path: chunked SSD scan with λ-modulated dt + per-chunk T_state.
        conv_init: (B, d_inner, d_conv-1) — carried conv state from previous clip.
        Returns:
            y, h, new_conv_state, lam_logit_seq
        """
        # Conv with state carry: prepend saved state instead of zero padding
        if conv_init is not None:
            x_padded = torch.cat([conv_init, x], dim=2)
            x_conv = F.conv1d(
                x_padded, self.conv1d_slow.weight, self.conv1d_slow.bias,
                groups=self.d_inner,
            )
            x_conv = self.act(x_conv)  # output length = seqlen (exact)
        else:
            x_conv = self.act(self.conv1d_slow(x)[..., :seqlen])

        # Save last d_conv-1 frames of x for next clip
        new_conv_state = x[:, :, -(self.d_conv - 1):].clone()

        # ── Batched projections (Mamba2 multi-head) ──────────────────────
        H  = self.n_heads_slow
        P  = self.d_head_slow
        G  = self.ngroups_slow
        n  = self.d_state_slow

        x_conv_flat = rearrange(x_conv, "b d l -> (b l) d")
        x_dbl = self.x_proj_slow(x_conv_flat)
        dt_raw, B_all, C_all = torch.split(
            x_dbl, [self.dt_rank, G * n, G * n], dim=-1,
        )
        # ── Scalar intensity λ(t): applied as dt modulation on slow path ──
        # Gated so no_intensity ablation can omit the intensity_net module.
        # Return None for lam_logit_seq so train.py's intensity-loss path is
        # genuinely skipped (otherwise a zeros tensor still triggers BCE).
        if getattr(self, "intensity_net", None) is not None:
            lam_logit = self.intensity_net(x_conv_flat)
            lam_logit = rearrange(lam_logit, "(b l) 1 -> b l 1", l=seqlen)
            lam = torch.sigmoid(lam_logit)
            lam_logit_seq = lam_logit.squeeze(-1)
        else:
            lam = torch.zeros(x_conv.shape[0], seqlen, 1, device=x_conv.device)
            lam_logit_seq = None
        # dt: dt_rank → n_heads (per-head dt, not per-channel), then softplus + λ modulation (broadcast)
        dt_all = self.dt_proj_slow.weight @ dt_raw.t()            # (n_heads, B*L)
        dt_all = rearrange(dt_all, "h (b l) -> b l h", l=seqlen).contiguous()
        dt_bias_pre = self.dt_proj_slow.bias.float().view(1, 1, -1)
        dt_all = F.softplus(dt_all.float() + dt_bias_pre)                      # (B, L, H)
        # Ablation: w/o intensity → skip λ-modulation of dt.
        if self.use_intensity:
            dt_all = dt_all * (1.0 + lam)                                      # (B, L, H), broadcast H
        B_all  = rearrange(B_all, "(b l) (g n) -> b l g n", l=seqlen, g=G).contiguous()
        C_all  = rearrange(C_all, "(b l) (g n) -> b l g n", l=seqlen, g=G).contiguous()
        x_rs   = rearrange(x_conv, "b (h p) l -> b l h p", p=P).contiguous()

        A_slow  = -torch.exp(self.A_log_slow.float())              # (n_heads,)
        dt_bias = self.dt_proj_slow.bias.float()                   # (n_heads,)
        D_slow  = self.D_slow.float()                              # (n_heads,)

        bsz = x.shape[0]
        inp_dtype = x.dtype

        # Initial SSM state: (B, n_heads, d_head, d_state) or None (kernel zero-inits)
        h = initial_state.to(torch.float32) if initial_state is not None else None

        # ── Chunked SSD loop: per-chunk SSM via kernel + per-chunk T_state ──
        chunk_size = self.chunk_size_slow
        n_chunks = (seqlen + chunk_size - 1) // chunk_size

        outputs = []

        for c in range(n_chunks):
            t_s = c * chunk_size
            t_e = min(t_s + chunk_size, seqlen)
            cs  = t_e - t_s

            # SSD kernel: (B, cs, H, P) output + (B, H, P, n) final state
            # dt already softplus'd + λ-modulated above → dt_softplus=False, dt_bias=None.
            y_ch, h = mamba_chunk_scan_combined(
                x_rs[:, t_s:t_e],
                dt_all[:, t_s:t_e],
                A_slow,
                B_all[:, t_s:t_e],
                C_all[:, t_s:t_e],
                chunk_size=cs,
                D=D_slow,
                dt_bias=None,
                dt_softplus=False,
                initial_states=h,
                return_final_states=True,
            )
            y_ch = rearrange(y_ch, "b l h p -> b (h p) l").contiguous()   # (B, d_inner, cs)

            if self.use_t_state:
                # ── Per-chunk SVD filter: separate U, V, σ MLPs ──
                n_s = self.d_state_slow
                H_s = self.n_heads_slow
                r   = self.T_rank
                B_  = y_ch.shape[0]
                P_s = self.d_head_slow
                chunk_feat_ph = y_ch.view(B_, H_s, P_s, cs).mean(dim=-1).float()  # (B, H, P)
                chunk_feat_ph = F.layer_norm(chunk_feat_ph, [P_s])
                UV_flat = self.UV_mlp_slow(chunk_feat_ph)                          # (B, H, 2nr)
                U_flat, V_flat = UV_flat.chunk(2, dim=-1)
                U = F.normalize(U_flat.view(B_, H_s, n_s, r), dim=-2)
                V = F.normalize(V_flat.view(B_, H_s, n_s, r), dim=-2)
                sigma_logit = self.S_mlp_slow(chunk_feat_ph)                       # (B, H, r)
                # Force fp32 for Cayley (autocast wraps everything in bf16 otherwise).
                # Cayley: M = (I + S/2) · (I − S/2)⁻¹  — orthogonal, numerically stable for any ‖S‖.
                with torch.amp.autocast("cuda", enabled=False):
                    sigma = F.softplus(sigma_logit.float())
                    UV = torch.einsum("bhnr,bhr,bhmr->bhnm", U.float(), sigma, V.float())
                    S = UV - UV.transpose(-2, -1)                                   # skew-symmetric
                    I_S = torch.eye(n_s, device=S.device, dtype=S.dtype).expand_as(S)
                    S_half = S * 0.5
                    M = torch.linalg.solve(I_S - S_half, I_S + S_half)              # Cayley, orthogonal
                    h = torch.einsum("bhpn,bhnm->bhpm", h.float(), M)

            outputs.append(y_ch.to(inp_dtype))

        y = torch.cat(outputs, dim=2)                                       # (B, d_inner, L)
        return y, h, new_conv_state, lam_logit_seq

    # ── Step (single frame, online inference) ────────────────────────

    def step(self, hidden_states, conv_state, ssm_state, lower_mem=None):
        """Single-frame autoregressive step.

        conv_state: (B, 2*d_inner, d_conv) — fast||slow packed.
        ssm_state:  5-tuple (ssm_fast, ssm_slow, y_sum_slow, y_sum_fast, frame_in_chunk).
        lower_mem:  unused (kept for API back-compat).
        """
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1
        h = hidden_states.squeeze(1)
        xz = self.in_proj(h)                           # (B, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)                     # (B, d_inner) each

        x_slow = self.in_proj_slow(h)                  # (B, d_inner) — no z gate

        d = self.d_inner
        conv_fast, conv_slow = conv_state[:, :d], conv_state[:, d:]
        # ssm_state: 5-tuple = (ssm_fast, ssm_slow, y_sum_slow, y_sum_fast, frame_in_chunk)
        ssm_fast, ssm_slow, y_sum_slow, y_sum_fast, frame_in_chunk = ssm_state

        # ── 1. Slow step (intensity-modulated dt + chunk T_state) ──
        y_slow, y_sum_slow, frame_in_chunk, lam_t = self._step_slow(
            x_slow, conv_slow, ssm_slow,
            y_sum=y_sum_slow, frame_in_chunk=frame_in_chunk,
        )

        # ── 2. Fast step (chunk T_state via running mean, no intensity) ─────
        y_fast, y_sum_fast = self._step_fast(
            x, conv_fast, ssm_fast, y_slow,
            y_sum=y_sum_fast, frame_in_chunk=frame_in_chunk,
        )

        # ── Sum fusion (mirrors scan): scale·fast + scale·slow → gate(z) → norm → out_proj ──
        y = self.scale_fast * y_fast + self.scale_slow * y_slow
        y = y * self.act(z)
        y = self.norm_pre_out(y)
        out = self.out_proj(y)

        return out.unsqueeze(1), conv_state, (ssm_fast, ssm_slow, y_sum_slow, y_sum_fast, frame_in_chunk), lam_t

    def _step_fast(self, x, conv_state, ssm_state, y_slow_cond,
                   y_sum=None, frame_in_chunk=None):
        """Fast path single step. No intensity. Chunk-end T_state mixing from chunk-mean y.

        y_sum: (B, d_inner) running sum for chunk-mean computation (mirrors scan's y_ch.mean).
        """
        dtype = x.dtype

        # Conv update
        if causal_conv1d_update is None:
            conv_state[:, :, :-1] = conv_state[:, :, 1:].clone()
            conv_state[:, :, -1] = x
            x_conv = torch.sum(
                conv_state * rearrange(self.conv1d_fast.weight, "d 1 w -> d w"), dim=-1
            )
            if self.conv1d_fast.bias is not None:
                x_conv = x_conv + self.conv1d_fast.bias
            x_conv = self.act(x_conv).to(dtype=dtype)
        else:
            x_conv = causal_conv1d_update(
                x, conv_state,
                rearrange(self.conv1d_fast.weight, "d 1 w -> d w"),
                self.conv1d_fast.bias, self.activation,
            )

        H = self.n_heads_fast
        P = self.d_head_fast
        G = self.ngroups_fast
        n = self.d_state

        # ── Conditioned x_proj: [x_conv, y_slow] → dt_raw, B, C ──
        fusion_input = torch.cat([x_conv, y_slow_cond], dim=-1)
        x_db = self.x_proj_fast(fusion_input)
        dt_raw, B, C = torch.split(x_db, [self.dt_rank, G * n, G * n], dim=-1)

        # dt: dt_rank → n_heads, softplus + bias (no intensity)
        dt = F.linear(dt_raw, self.dt_proj_fast.weight)                           # (B, H)
        dt = F.softplus(dt.float() + self.dt_proj_fast.bias.float())              # (B, H)

        # Per-head B, C
        B = rearrange(B, "b (g n) -> b g n", g=G)                                 # (B, G, n)
        C = rearrange(C, "b (g n) -> b g n", g=G)
        heads_per_group = H // G
        Bh = B.repeat_interleave(heads_per_group, dim=1)                          # (B, H, n)
        Ch = C.repeat_interleave(heads_per_group, dim=1)

        A_fast = -torch.exp(self.A_log_fast.float())                              # (H,)
        D_head = self.D_fast.to(dtype)                                            # (H,)

        x_ssd = rearrange(x_conv, "b (h p) -> b h p", p=P)                        # (B, H, P)

        # Mamba2 multi-head SSM step
        dA  = torch.exp(A_fast.unsqueeze(0) * dt)                                 # (B, H)
        dBx = torch.einsum("bh,bhn,bhp->bhpn", dt, Bh.float(), x_ssd.float())     # (B, H, P, n)
        ssm_state.mul_(rearrange(dA, "b h -> b h 1 1")).add_(dBx)
        y = torch.einsum("bhpn,bhn->bhp", ssm_state.to(dtype), Ch)                # (B, H, P)
        y = y + D_head.view(1, H, 1) * x_ssd
        y = rearrange(y, "b h p -> b (h p)").contiguous()                         # (B, d_inner)

        # ── Per-chunk T_state for fast (mirrors scan: chunk-mean y input) ──
        # NOTE: frame_in_chunk is advanced by slow step; post-advance pos==0 marks chunk-end.
        # Accumulate y into y_sum each frame; at chunk-end, use mean = y_sum/chunk_size.
        if y_sum is not None:
            y_sum.add_(y.to(y_sum.dtype))
        if frame_in_chunk is not None:
            pos_advanced = int(frame_in_chunk[0].item())
            if pos_advanced == 0:
                chunk_size = self.chunk_size_fast
                r_f = self.T_rank
                n_f = self.d_state
                B_ = y.shape[0]
                chunk_mean = (y_sum / chunk_size).float() if y_sum is not None else y.float()
                chunk_feat_ph = chunk_mean.view(B_, H, P)                          # (B, H, P)
                chunk_feat_ph = F.layer_norm(chunk_feat_ph, [P])
                UV_flat = self.UV_mlp_fast(chunk_feat_ph)                          # (B, H, 2nr)
                U_flat, V_flat = UV_flat.chunk(2, dim=-1)
                U = F.normalize(U_flat.view(B_, H, n_f, r_f), dim=-2)
                V = F.normalize(V_flat.view(B_, H, n_f, r_f), dim=-2)
                sigma_logit = self.S_mlp_fast(chunk_feat_ph)                       # (B, H, r)
                with torch.amp.autocast("cuda", enabled=False):
                    sigma = F.softplus(sigma_logit.float())
                    UV = torch.einsum("bhnr,bhr,bhmr->bhnm", U.float(), sigma, V.float())
                    S = UV - UV.transpose(-2, -1)
                    I_S = torch.eye(n_f, device=S.device, dtype=S.dtype).expand_as(S)
                    S_half = S * 0.5
                    M = torch.linalg.solve(I_S - S_half, I_S + S_half)          # Cayley
                    ssm_state.copy_(
                        torch.einsum("bhpn,bhnm->bhpm", ssm_state.float(), M).to(ssm_state.dtype)
                    )
                if y_sum is not None:
                    y_sum.zero_()

        return y, y_sum

    def _step_slow(self, x, conv_state, ssm_state,
                   y_sum=None, frame_in_chunk=None):
        """Single-frame Mamba2 SSD step for slow path with λ modulation + T_state at chunk-end."""
        dtype = x.dtype

        if causal_conv1d_update is None:
            conv_state[:, :, :-1] = conv_state[:, :, 1:].clone()
            conv_state[:, :, -1] = x
            x_conv = torch.sum(
                conv_state * rearrange(self.conv1d_slow.weight, "d 1 w -> d w"), dim=-1
            )
            if self.conv1d_slow.bias is not None:
                x_conv = x_conv + self.conv1d_slow.bias
            x_conv = self.act(x_conv).to(dtype=dtype)
        else:
            x_conv = causal_conv1d_update(
                x, conv_state,
                rearrange(self.conv1d_slow.weight, "d 1 w -> d w"),
                self.conv1d_slow.bias, self.activation,
            )

        H = self.n_heads_slow
        P = self.d_head_slow
        G = self.ngroups_slow
        n = self.d_state_slow

        x_db = self.x_proj_slow(x_conv)
        dt_raw, B, C = torch.split(x_db, [self.dt_rank, G * n, G * n], dim=-1)

        # Scalar intensity λ (moved from fast to slow — safe due to memory backup).
        # LayerNorm input → bound MLP magnitude.

        lam_logit = self.intensity_net(x_conv)                           # (B, 1)
        lam = torch.sigmoid(lam_logit)                                        # (B, 1)
        # dt: dt_rank → n_heads, softplus + bias, then λ modulation (broadcast H)
        dt = F.linear(dt_raw, self.dt_proj_slow.weight)
        dt = F.softplus(dt.float() + self.dt_proj_slow.bias.float())         # (B, H)
        dt = dt * (1.0 + lam)                                                 # (B, H), broadcast

        # Per-head B, C: expand groups to heads (heads_per_group = H/G)
        B = rearrange(B, "b (g n) -> b g n", g=G)                             # (B, G, n)
        C = rearrange(C, "b (g n) -> b g n", g=G)
        heads_per_group = H // G
        Bh = B.repeat_interleave(heads_per_group, dim=1)                      # (B, H, n)
        Ch = C.repeat_interleave(heads_per_group, dim=1)

        A_slow = -torch.exp(self.A_log_slow.float())                          # (H,)
        D_head = self.D_slow.to(dtype)                                        # (H,)

        x_ssd = rearrange(x_conv, "b (h p) -> b h p", p=P)                    # (B, H, P)

        # Mamba2 multi-head SSM step
        dA  = torch.exp(A_slow.unsqueeze(0) * dt.float())                     # (B, H)
        dBx = torch.einsum(
            "bh,bhn,bhp->bhpn",
            dt.float(), Bh.float(), x_ssd.float(),
        )                                                                     # (B, H, P, n)
        ssm_state.mul_(rearrange(dA, "b h -> b h 1 1")).add_(dBx)
        y = torch.einsum("bhpn,bhn->bhp", ssm_state.to(dtype), Ch)            # (B, H, P)
        y = y + D_head.view(1, H, 1) * x_ssd
        y = rearrange(y, "b h p -> b (h p)").contiguous()                     # (B, d_inner)

        # ── Accumulate y + at chunk-end apply T_state ──
        chunk_size = self.chunk_size_slow
        pos = int(frame_in_chunk[0].item()) if frame_in_chunk is not None else 0
        if y_sum is not None:
            y_sum.add_(y.to(y_sum.dtype))
        new_pos = (pos + 1) % chunk_size

        if new_pos == 0:  # chunk-end
            n_s = self.d_state_slow
            H_s = self.n_heads_slow
            r   = self.T_rank
            P_s = self.d_head_slow
            B_  = y.shape[0]
            chunk_mean = (y_sum / chunk_size).float() if y_sum is not None else y.float()
            chunk_feat_ph = chunk_mean.view(B_, H_s, P_s)                   # (B, H, P)
            chunk_feat_ph = F.layer_norm(chunk_feat_ph, [P_s])
            UV_flat = self.UV_mlp_slow(chunk_feat_ph)                       # (B, H, 2nr)
            U_flat, V_flat = UV_flat.chunk(2, dim=-1)
            U = F.normalize(U_flat.view(B_, H_s, n_s, r), dim=-2)
            V = F.normalize(V_flat.view(B_, H_s, n_s, r), dim=-2)
            sigma_logit = self.S_mlp_slow(chunk_feat_ph)                    # (B, H, r)
            with torch.amp.autocast("cuda", enabled=False):
                sigma = F.softplus(sigma_logit.float())
                UV = torch.einsum("bhnr,bhr,bhmr->bhnm", U.float(), sigma, V.float())
                S = UV - UV.transpose(-2, -1)
                I_S = torch.eye(n_s, device=S.device, dtype=S.dtype).expand_as(S)
                S_half = S * 0.5
                M = torch.linalg.solve(I_S - S_half, I_S + S_half)              # Cayley
                ssm_state.copy_(
                    torch.einsum("bhpn,bhnm->bhpm", ssm_state.float(), M).to(ssm_state.dtype)
                )
            if y_sum is not None:
                y_sum.zero_()
        if frame_in_chunk is not None:
            frame_in_chunk.fill_(new_pos)

        return y, y_sum, frame_in_chunk, lam_logit
