"""SurgicalMamba: shared-input dual-path SSM for surgical phase recognition.

Per call:
- Fast path: in_proj → SSD scan (zero-init per clip), gated by z = silu(z).
- Slow path: in_proj_slow → SSD scan with state carried across clips and
  λ-modulated dt; per-chunk Cayley orthogonal rotation (T_state) on the SSM
  state at each chunk boundary.
- Fusion: scale_fast * y_fast + scale_slow * y_slow → silu(z) gate → norm → out_proj.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None


class _PerHeadMLP(nn.Module):
    """Per-head 2-layer MLP: (B, H, in_dim) → (B, H, out_dim).

    Each head owns its own (W1, b1, W2, b2); evaluated via batched einsum.
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
        t_rank=16,
        chunk_size=32,
        chunk_size_fast=None,
        chunk_size_slow=None,
        use_t_state=True,
        use_intensity=True,
        use_fast_path=True,
        use_slow_path=True,
        device=None,
        dtype=None,
    ):
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

        # ── Fast path projections ────────────────────────────────────────
        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )
        self.in_proj_slow = nn.Linear(
            self.d_model, self.d_inner, bias=bias, **factory_kwargs
        )

        # ── Fast path: Mamba2 SSD with d_head=64, ngroups=n_heads ────────
        self.d_head_fast = 64
        assert self.d_inner % self.d_head_fast == 0, \
            f"d_inner ({self.d_inner}) must be divisible by d_head_fast ({self.d_head_fast})"
        self.n_heads_fast = self.d_inner // self.d_head_fast
        self.ngroups_fast = self.n_heads_fast
        self.chunk_size_fast = chunk_size_fast if chunk_size_fast is not None else chunk_size

        self.conv1d_fast = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1,
            bias=conv_bias, **factory_kwargs,
        )
        # x_proj_fast input is [x_conv_fast, y_slow] (concatenated)
        self.x_proj_fast = nn.Linear(
            self.d_inner * 2,
            self.dt_rank + 2 * self.ngroups_fast * self.d_state,
            bias=False, **factory_kwargs,
        )

        # Intensity head: per-frame scalar λ (broadcast across heads)
        intensity_bottleneck = max(self.d_inner // 16, 8)
        self.intensity_net = nn.Sequential(
            nn.Linear(self.d_inner, intensity_bottleneck, bias=True, **factory_kwargs),
            nn.SiLU(),
            nn.Linear(intensity_bottleneck, 1, bias=True, **factory_kwargs),
        )
        self.dt_proj_fast = nn.Linear(
            self.dt_rank, self.n_heads_fast, bias=True, **factory_kwargs,
        )
        A_fast = torch.empty(self.n_heads_fast, dtype=torch.float32, device=device).uniform_(1.0, float(self.d_state))
        self.A_log_fast = nn.Parameter(torch.log(A_fast))
        self.A_log_fast._no_weight_decay = True
        self.D_fast = nn.Parameter(torch.ones(self.n_heads_fast, device=device))
        self.D_fast._no_weight_decay = True

        # ── Slow path: Mamba2 SSD with cross-clip carry ──────────────────
        self.d_head_slow = 64
        assert self.d_inner % self.d_head_slow == 0, \
            f"d_inner ({self.d_inner}) must be divisible by d_head_slow ({self.d_head_slow})"
        self.n_heads_slow = self.d_inner // self.d_head_slow
        self.ngroups_slow = self.n_heads_slow
        self.chunk_size_slow = chunk_size_slow if chunk_size_slow is not None else chunk_size

        self.conv1d_slow = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1,
            bias=conv_bias, **factory_kwargs,
        )
        self.x_proj_slow = nn.Linear(
            self.d_inner,
            self.dt_rank + 2 * self.ngroups_slow * self.d_state_slow,
            bias=False, **factory_kwargs,
        )
        self.dt_proj_slow = nn.Linear(
            self.dt_rank, self.n_heads_slow, bias=True, **factory_kwargs,
        )
        A_slow = torch.empty(self.n_heads_slow, dtype=torch.float32, device=device).uniform_(1.0, float(self.d_state_slow))
        self.A_log_slow = nn.Parameter(torch.log(A_slow))
        self.A_log_slow._no_weight_decay = True
        self.D_slow = nn.Parameter(torch.ones(self.n_heads_slow, device=device))
        self.D_slow._no_weight_decay = True

        # ── Per-chunk T_state: per-head SVD filter (UV shared MLP + σ MLP) ──
        n_slow = self.d_state_slow
        self.T_rank = t_rank
        stab_hidden = 128

        H_s = self.n_heads_slow
        r = self.T_rank
        self.UV_mlp_slow = _PerHeadMLP(H_s, self.d_head_slow, stab_hidden,
                                        2 * n_slow * r, w1_gain=1.0, w2_gain=2.0,
                                        **factory_kwargs)
        self.S_mlp_slow  = _PerHeadMLP(H_s, self.d_head_slow, stab_hidden,
                                        r, w1_gain=1.0, w2_gain=2.0,
                                        **factory_kwargs)
        # Per-head Gram-Schmidt: orthogonalise U_init / V_init across heads in
        # flattened n*r space → diverse rotations at init.
        _slow_total = n_slow * r
        _slow_scale = 0.5 * math.sqrt(_slow_total)
        _Q_U_slow, _ = torch.linalg.qr(torch.randn(_slow_total, H_s))
        _Q_V_slow, _ = torch.linalg.qr(torch.randn(_slow_total, H_s))
        with torch.no_grad():
            self.S_mlp_slow.b2.fill_(0.3)
            for h in range(H_s):
                U_init = _Q_U_slow[:, h] * _slow_scale
                V_init = _Q_V_slow[:, h] * _slow_scale
                self.UV_mlp_slow.b2[h].copy_(torch.cat([U_init, V_init]).to(
                    self.UV_mlp_slow.b2.device, dtype=self.UV_mlp_slow.b2.dtype))

        H_f = self.n_heads_fast
        n_f = self.d_state
        self.UV_mlp_fast = _PerHeadMLP(H_f, self.d_head_fast, stab_hidden,
                                        2 * n_f * r, w1_gain=1.0, w2_gain=2.0,
                                        **factory_kwargs)
        self.S_mlp_fast  = _PerHeadMLP(H_f, self.d_head_fast, stab_hidden,
                                        r, w1_gain=1.0, w2_gain=2.0,
                                        **factory_kwargs)
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

        self.register_buffer("_T_identity", torch.eye(n_slow))

        # ── dt initialization ────────────────────────────────────────
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

        # Per-channel learnable scale for fast/slow balancing (sum fusion).
        self.scale_fast = nn.Parameter(torch.ones(self.d_inner, **factory_kwargs))
        self.scale_slow = nn.Parameter(torch.ones(self.d_inner, **factory_kwargs))

        self.norm_pre_out = nn.RMSNorm(self.d_inner, eps=1e-5, **factory_kwargs)
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )

    @property
    def state_d_inner(self):
        """d_inner for state allocation — doubled because fast||slow packed."""
        return self.d_inner * 2

    def forward(self, hidden_states, slow_state=None):
        """Shared-input forward: same input feeds both fast and slow paths.

        Args:
            hidden_states: (B, L, D) input stream.
            slow_state:    (ssm, conv) or None — carried across clips.

        Returns:
            out:            (B, L, D)
            new_slow_state: (ssm, conv) or None when ``use_slow_path=False``.
            lam_seq:        (B, L) intensity logits for BCE, or None when
                            ``use_slow_path=False``.
        """
        batch, seqlen, dim = hidden_states.shape

        if getattr(self, "in_proj", None) is not None:
            xz = rearrange(
                self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
                "d (b l) -> b d l", l=seqlen,
            )
            if self.in_proj.bias is not None:
                xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
            x, z = xz.chunk(2, dim=1)
            if not self.use_fast_path:
                x = None
        else:
            x, z = None, None

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

        if self.use_fast_path:
            y_fast = self._scan_fast(x, seqlen, condition=(y_slow if self.use_slow_path else None))
        else:
            y_fast = torch.zeros_like(y_slow)

        sf = self.scale_fast.view(1, -1, 1) if getattr(self, "scale_fast", None) is not None else 1.0
        ss = self.scale_slow.view(1, -1, 1) if getattr(self, "scale_slow", None) is not None else 1.0
        y = sf * y_fast + ss * y_slow
        if z is not None:
            y = y * self.act(z)
        y = rearrange(y, "b d l -> b l d")
        y = self.norm_pre_out(y)
        out = self.out_proj(y)

        new_slow_state = (new_slow_ssm, new_slow_conv) if self.use_slow_path else None
        return out, new_slow_state, lam_seq

    def _scan_fast(self, x, seqlen, condition=None):
        """Fast path: chunked SSD scan with per-chunk T_state, state fresh per clip.

        ``condition`` is y_slow (B, d_inner, L) — slow path context for
        conditioning. No intensity modulation here (λ lives on slow path).
        """
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

        x_conv_flat = rearrange(x_conv, "b d l -> (b l) d")

        if condition is not None:
            cond_flat = rearrange(condition, "b d l -> (b l) d")
            fusion_input = torch.cat([x_conv_flat, cond_flat], dim=-1)
        else:
            fusion_input = torch.cat([x_conv_flat, torch.zeros_like(x_conv_flat)], dim=-1)

        x_dbl = self.x_proj_fast(fusion_input)
        dt_raw, B_all, C_all = torch.split(
            x_dbl, [self.dt_rank, G * n_f, G * n_f], dim=-1,
        )
        dt_all = self.dt_proj_fast.weight @ dt_raw.t()
        dt_all = rearrange(dt_all, "h (b l) -> b l h", l=seqlen).contiguous()

        B_all = rearrange(B_all, "(b l) (g n) -> b l g n", l=seqlen, g=G).contiguous()
        C_all = rearrange(C_all, "(b l) (g n) -> b l g n", l=seqlen, g=G).contiguous()
        x_rs  = rearrange(x_conv, "b (h p) l -> b l h p", p=P).contiguous()

        A_fast  = -torch.exp(self.A_log_fast.float())
        D_fast  = self.D_fast.float()
        dt_bias_f = self.dt_proj_fast.bias.float()

        inp_dtype = x.dtype
        chunk_size = self.chunk_size_fast
        n_chunks = (seqlen + chunk_size - 1) // chunk_size
        outputs = []
        h = None

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
            y_ch = rearrange(y_ch, "b l h p -> b (h p) l").contiguous()

            if self.use_t_state:
                B_ = y_ch.shape[0]
                chunk_feat_ph = y_ch.view(B_, H, P, cs).mean(dim=-1).float()
                chunk_feat_ph = F.layer_norm(chunk_feat_ph, [P])
                UV_flat = self.UV_mlp_fast(chunk_feat_ph)
                U_flat, V_flat = UV_flat.chunk(2, dim=-1)
                U = F.normalize(U_flat.view(B_, H, n_f, r_f), dim=-2)
                V = F.normalize(V_flat.view(B_, H, n_f, r_f), dim=-2)
                sigma_logit = self.S_mlp_fast(chunk_feat_ph)
                # Cayley in fp32 (autocast wraps everything else in bf16).
                with torch.amp.autocast("cuda", enabled=False):
                    sigma = F.softplus(sigma_logit.float())
                    UV = torch.einsum("bhnr,bhr,bhmr->bhnm", U.float(), sigma, V.float())
                    S = UV - UV.transpose(-2, -1)
                    I_S = torch.eye(n_f, device=S.device, dtype=S.dtype).expand_as(S)
                    S_half = S * 0.5
                    M = torch.linalg.solve(I_S - S_half, I_S + S_half)
                    h = torch.einsum("bhpn,bhnm->bhpm", h.float(), M)

            outputs.append(y_ch.to(inp_dtype))

        y = torch.cat(outputs, dim=2)
        return y

    def _scan_slow(self, x, seqlen, initial_state=None, conv_init=None):
        """Slow path: chunked SSD scan with λ-modulated dt + per-chunk T_state.

        ``conv_init`` is the (B, d_inner, d_conv-1) conv state carried from the
        previous clip; ``initial_state`` is the SSM state. Returns
        ``(y, new_ssm_state, new_conv_state, lam_logit_seq)``.
        """
        # Carry conv state by prepending it; output length stays exactly seqlen.
        if conv_init is not None:
            x_padded = torch.cat([conv_init, x], dim=2)
            x_conv = F.conv1d(
                x_padded, self.conv1d_slow.weight, self.conv1d_slow.bias,
                groups=self.d_inner,
            )
            x_conv = self.act(x_conv)
        else:
            x_conv = self.act(self.conv1d_slow(x)[..., :seqlen])

        new_conv_state = x[:, :, -(self.d_conv - 1):].clone()

        H  = self.n_heads_slow
        P  = self.d_head_slow
        G  = self.ngroups_slow
        n  = self.d_state_slow

        x_conv_flat = rearrange(x_conv, "b d l -> (b l) d")
        x_dbl = self.x_proj_slow(x_conv_flat)
        dt_raw, B_all, C_all = torch.split(
            x_dbl, [self.dt_rank, G * n, G * n], dim=-1,
        )
        # Scalar intensity λ(t) per frame, applied as dt modulation.
        if getattr(self, "intensity_net", None) is not None:
            lam_logit = self.intensity_net(x_conv_flat)
            lam_logit = rearrange(lam_logit, "(b l) 1 -> b l 1", l=seqlen)
            lam = torch.sigmoid(lam_logit)
            lam_logit_seq = lam_logit.squeeze(-1)
        else:
            lam = torch.zeros(x_conv.shape[0], seqlen, 1, device=x_conv.device)
            lam_logit_seq = None

        dt_all = self.dt_proj_slow.weight @ dt_raw.t()
        dt_all = rearrange(dt_all, "h (b l) -> b l h", l=seqlen).contiguous()
        dt_bias_pre = self.dt_proj_slow.bias.float().view(1, 1, -1)
        dt_all = F.softplus(dt_all.float() + dt_bias_pre)
        if self.use_intensity:
            dt_all = dt_all * (1.0 + lam)
        B_all  = rearrange(B_all, "(b l) (g n) -> b l g n", l=seqlen, g=G).contiguous()
        C_all  = rearrange(C_all, "(b l) (g n) -> b l g n", l=seqlen, g=G).contiguous()
        x_rs   = rearrange(x_conv, "b (h p) l -> b l h p", p=P).contiguous()

        A_slow  = -torch.exp(self.A_log_slow.float())
        D_slow  = self.D_slow.float()

        inp_dtype = x.dtype
        h = initial_state.to(torch.float32) if initial_state is not None else None

        chunk_size = self.chunk_size_slow
        n_chunks = (seqlen + chunk_size - 1) // chunk_size

        outputs = []

        for c in range(n_chunks):
            t_s = c * chunk_size
            t_e = min(t_s + chunk_size, seqlen)
            cs  = t_e - t_s

            # dt already softplus'd + λ-modulated → dt_softplus=False, dt_bias=None.
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
            y_ch = rearrange(y_ch, "b l h p -> b (h p) l").contiguous()

            if self.use_t_state:
                n_s = self.d_state_slow
                H_s = self.n_heads_slow
                r   = self.T_rank
                B_  = y_ch.shape[0]
                P_s = self.d_head_slow
                chunk_feat_ph = y_ch.view(B_, H_s, P_s, cs).mean(dim=-1).float()
                chunk_feat_ph = F.layer_norm(chunk_feat_ph, [P_s])
                UV_flat = self.UV_mlp_slow(chunk_feat_ph)
                U_flat, V_flat = UV_flat.chunk(2, dim=-1)
                U = F.normalize(U_flat.view(B_, H_s, n_s, r), dim=-2)
                V = F.normalize(V_flat.view(B_, H_s, n_s, r), dim=-2)
                sigma_logit = self.S_mlp_slow(chunk_feat_ph)
                # Cayley: M = (I + S/2)·(I − S/2)⁻¹, orthogonal. fp32 for stability.
                with torch.amp.autocast("cuda", enabled=False):
                    sigma = F.softplus(sigma_logit.float())
                    UV = torch.einsum("bhnr,bhr,bhmr->bhnm", U.float(), sigma, V.float())
                    S = UV - UV.transpose(-2, -1)
                    I_S = torch.eye(n_s, device=S.device, dtype=S.dtype).expand_as(S)
                    S_half = S * 0.5
                    M = torch.linalg.solve(I_S - S_half, I_S + S_half)
                    h = torch.einsum("bhpn,bhnm->bhpm", h.float(), M)

            outputs.append(y_ch.to(inp_dtype))

        y = torch.cat(outputs, dim=2)
        return y, h, new_conv_state, lam_logit_seq

    def step(self, hidden_states, conv_state, ssm_state, lower_mem=None):
        """Single-frame autoregressive step.

        Args:
            conv_state: (B, 2*d_inner, d_conv) — fast||slow packed.
            ssm_state:  5-tuple (ssm_fast, ssm_slow, y_sum_slow, y_sum_fast,
                        frame_in_chunk). ``frame_in_chunk`` is a Python int
                        (host-side counter — keeps the per-frame step path
                        free of GPU→CPU syncs).
            lower_mem:  unused (kept for API back-compat).
        """
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1
        h = hidden_states.squeeze(1)
        xz = self.in_proj(h)
        x, z = xz.chunk(2, dim=-1)
        x_slow = self.in_proj_slow(h)

        d = self.d_inner
        conv_fast, conv_slow = conv_state[:, :d], conv_state[:, d:]
        ssm_fast, ssm_slow, y_sum_slow, y_sum_fast, frame_in_chunk = ssm_state

        y_slow, y_sum_slow, frame_in_chunk, lam_t = self._step_slow(
            x_slow, conv_slow, ssm_slow,
            y_sum=y_sum_slow, frame_in_chunk=frame_in_chunk,
        )
        y_fast, y_sum_fast = self._step_fast(
            x, conv_fast, ssm_fast, y_slow,
            y_sum=y_sum_fast, frame_in_chunk=frame_in_chunk,
        )

        y = self.scale_fast * y_fast + self.scale_slow * y_slow
        y = y * self.act(z)
        y = self.norm_pre_out(y)
        out = self.out_proj(y)

        return out.unsqueeze(1), conv_state, (ssm_fast, ssm_slow, y_sum_slow, y_sum_fast, frame_in_chunk), lam_t

    def _step_fast(self, x, conv_state, ssm_state, y_slow_cond,
                   y_sum=None, frame_in_chunk=None):
        """Fast path single step.

        Mirrors the scan path: T_state mixing at each chunk boundary uses the
        chunk-mean of y, accumulated via ``y_sum``.
        """
        dtype = x.dtype

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

        fusion_input = torch.cat([x_conv, y_slow_cond], dim=-1)
        x_db = self.x_proj_fast(fusion_input)
        dt_raw, B, C = torch.split(x_db, [self.dt_rank, G * n, G * n], dim=-1)

        dt = F.linear(dt_raw, self.dt_proj_fast.weight)
        dt = F.softplus(dt.float() + self.dt_proj_fast.bias.float())

        B = rearrange(B, "b (g n) -> b g n", g=G)
        C = rearrange(C, "b (g n) -> b g n", g=G)
        heads_per_group = H // G
        if heads_per_group == 1:
            Bh, Ch = B, C
        else:
            Bh = B.repeat_interleave(heads_per_group, dim=1)
            Ch = C.repeat_interleave(heads_per_group, dim=1)

        A_fast = -torch.exp(self.A_log_fast.float())
        D_head = self.D_fast.to(dtype)
        x_ssd = rearrange(x_conv, "b (h p) -> b h p", p=P)

        dA  = torch.exp(A_fast.unsqueeze(0) * dt)
        dBx = torch.einsum("bh,bhn,bhp->bhpn", dt, Bh.float(), x_ssd.float())
        ssm_state.mul_(rearrange(dA, "b h -> b h 1 1")).add_(dBx)
        y = torch.einsum("bhpn,bhn->bhp", ssm_state.to(dtype), Ch)
        y = y + D_head.view(1, H, 1) * x_ssd
        y = rearrange(y, "b h p -> b (h p)").contiguous()

        # frame_in_chunk (Python int) is advanced by the slow step; a post-advance
        # value of 0 marks a chunk boundary. Accumulate y each frame, mix via
        # T_state at chunk-end.
        if y_sum is not None:
            y_sum.add_(y.to(y_sum.dtype))
        if frame_in_chunk is not None and frame_in_chunk == 0:
            chunk_size = self.chunk_size_fast
            r_f = self.T_rank
            n_f = self.d_state
            B_ = y.shape[0]
            chunk_mean = (y_sum / chunk_size).float() if y_sum is not None else y.float()
            chunk_feat_ph = chunk_mean.view(B_, H, P)
            chunk_feat_ph = F.layer_norm(chunk_feat_ph, [P])
            UV_flat = self.UV_mlp_fast(chunk_feat_ph)
            U_flat, V_flat = UV_flat.chunk(2, dim=-1)
            U = F.normalize(U_flat.view(B_, H, n_f, r_f), dim=-2)
            V = F.normalize(V_flat.view(B_, H, n_f, r_f), dim=-2)
            sigma_logit = self.S_mlp_fast(chunk_feat_ph)
            with torch.amp.autocast("cuda", enabled=False):
                sigma = F.softplus(sigma_logit.float())
                UV = torch.einsum("bhnr,bhr,bhmr->bhnm", U.float(), sigma, V.float())
                S = UV - UV.transpose(-2, -1)
                I_S = torch.eye(n_f, device=S.device, dtype=S.dtype).expand_as(S)
                S_half = S * 0.5
                M = torch.linalg.solve(I_S - S_half, I_S + S_half)
                ssm_state.copy_(
                    torch.einsum("bhpn,bhnm->bhpm", ssm_state.float(), M).to(ssm_state.dtype)
                )
            if y_sum is not None:
                y_sum.zero_()

        return y, y_sum

    def _step_slow(self, x, conv_state, ssm_state,
                   y_sum=None, frame_in_chunk=None):
        """Slow path single step: λ-modulated dt + T_state mixing at chunk-end."""
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

        lam_logit = self.intensity_net(x_conv)
        lam = torch.sigmoid(lam_logit)
        dt = F.linear(dt_raw, self.dt_proj_slow.weight)
        dt = F.softplus(dt.float() + self.dt_proj_slow.bias.float())
        dt = dt * (1.0 + lam)

        B = rearrange(B, "b (g n) -> b g n", g=G)
        C = rearrange(C, "b (g n) -> b g n", g=G)
        heads_per_group = H // G
        if heads_per_group == 1:
            Bh, Ch = B, C
        else:
            Bh = B.repeat_interleave(heads_per_group, dim=1)
            Ch = C.repeat_interleave(heads_per_group, dim=1)

        A_slow = -torch.exp(self.A_log_slow.float())
        D_head = self.D_slow.to(dtype)
        x_ssd = rearrange(x_conv, "b (h p) -> b h p", p=P)

        dA  = torch.exp(A_slow.unsqueeze(0) * dt.float())
        dBx = torch.einsum(
            "bh,bhn,bhp->bhpn",
            dt.float(), Bh.float(), x_ssd.float(),
        )
        ssm_state.mul_(rearrange(dA, "b h -> b h 1 1")).add_(dBx)
        y = torch.einsum("bhpn,bhn->bhp", ssm_state.to(dtype), Ch)
        y = y + D_head.view(1, H, 1) * x_ssd
        y = rearrange(y, "b h p -> b (h p)").contiguous()

        chunk_size = self.chunk_size_slow
        pos = frame_in_chunk if frame_in_chunk is not None else 0
        if y_sum is not None:
            y_sum.add_(y.to(y_sum.dtype))
        new_pos = (pos + 1) % chunk_size

        if new_pos == 0:
            n_s = self.d_state_slow
            H_s = self.n_heads_slow
            r   = self.T_rank
            P_s = self.d_head_slow
            B_  = y.shape[0]
            chunk_mean = (y_sum / chunk_size).float() if y_sum is not None else y.float()
            chunk_feat_ph = chunk_mean.view(B_, H_s, P_s)
            chunk_feat_ph = F.layer_norm(chunk_feat_ph, [P_s])
            UV_flat = self.UV_mlp_slow(chunk_feat_ph)
            U_flat, V_flat = UV_flat.chunk(2, dim=-1)
            U = F.normalize(U_flat.view(B_, H_s, n_s, r), dim=-2)
            V = F.normalize(V_flat.view(B_, H_s, n_s, r), dim=-2)
            sigma_logit = self.S_mlp_slow(chunk_feat_ph)
            with torch.amp.autocast("cuda", enabled=False):
                sigma = F.softplus(sigma_logit.float())
                UV = torch.einsum("bhnr,bhr,bhmr->bhnm", U.float(), sigma, V.float())
                S = UV - UV.transpose(-2, -1)
                I_S = torch.eye(n_s, device=S.device, dtype=S.dtype).expand_as(S)
                S_half = S * 0.5
                M = torch.linalg.solve(I_S - S_half, I_S + S_half)
                ssm_state.copy_(
                    torch.einsum("bhpn,bhnm->bhpm", ssm_state.float(), M).to(ssm_state.dtype)
                )
            if y_sum is not None:
                y_sum.zero_()

        return y, y_sum, new_pos, lam_logit
