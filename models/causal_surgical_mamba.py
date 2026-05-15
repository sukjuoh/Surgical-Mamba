
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from omegaconf import DictConfig

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

from .extractors import ConvNeXtTinyExtractor
from .surgical_mamba import SurgicalMamba


# ── FFN ───────────────────────────────────────────────────────────────────────

class FFN(nn.Module):
    """2-layer FFN with GELU. Hidden dim = ``hidden_ratio · d_model``."""

    def __init__(self, d_model: int, hidden_ratio: float = 4.0, bias: bool = True):
        super().__init__()
        d_hidden = int(round(hidden_ratio * d_model))
        self.fc1 = nn.Linear(d_model, d_hidden, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_hidden, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ── Building blocks ────────────────────────────────────────────────────────────

class _CausalMamba2Block(nn.Module):
    """Mamba2 block with per-chunk T_state + FFN. Used inside MambaHead."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2,
                 d_head=64, chunk_size=32, T_rank=16, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_head = d_head
        assert self.d_inner % d_head == 0, \
            f"d_inner ({self.d_inner}) must be divisible by d_head ({d_head})"
        self.n_heads = self.d_inner // d_head
        self.ngroups = self.n_heads
        self.d_state = d_state
        self.d_conv = d_conv
        self.chunk_size = chunk_size
        self.T_rank = T_rank
        self.dt_rank = max(math.ceil(d_model / 16), 16)
        self.activation = "silu"
        self.act = nn.SiLU()

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1, bias=True,
        )

        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + 2 * self.ngroups * d_state,
            bias=False,
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.n_heads, bias=True)
        # dt bias init: softplus inverse of uniform(0.001, 0.1).
        dt = torch.exp(
            torch.rand(self.n_heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = torch.empty(self.n_heads).uniform_(1.0, float(d_state))
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.n_heads))
        self.D._no_weight_decay = True

        self.norm_pre_out = nn.RMSNorm(self.d_inner, eps=1e-5)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        from .surgical_mamba import _PerHeadMLP
        stab_hidden = 128
        self.UV_mlp = _PerHeadMLP(self.n_heads, d_head, stab_hidden, 2 * d_state * T_rank,
                                  w1_gain=1.0, w2_gain=2.0)
        self.S_mlp  = _PerHeadMLP(self.n_heads, d_head, stab_hidden, T_rank,
                                  w1_gain=1.0, w2_gain=2.0)
        # Per-head Gram-Schmidt: orthogonalise U_init / V_init across heads in
        # flattened n*r space → diverse rotations at init.
        _head_total = d_state * T_rank
        _head_scale = 0.5 * math.sqrt(_head_total)
        _Q_U_head, _ = torch.linalg.qr(torch.randn(_head_total, self.n_heads))
        _Q_V_head, _ = torch.linalg.qr(torch.randn(_head_total, self.n_heads))
        with torch.no_grad():
            self.S_mlp.b2.fill_(0.3)
            for h in range(self.n_heads):
                U_init = _Q_U_head[:, h] * _head_scale
                V_init = _Q_V_head[:, h] * _head_scale
                self.UV_mlp.b2[h].copy_(torch.cat([U_init, V_init]).to(
                    self.UV_mlp.b2.device, dtype=self.UV_mlp.b2.dtype))

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model)
        self.drop = nn.Dropout(dropout)

    def _apply_T_state(self, h, chunk_feat_per_head):
        """Apply per-chunk Cayley orthogonal rotation to SSM state ``h``.

        Args:
            h: (B, H, P, n) SSM state.
            chunk_feat_per_head: (B, H, P) chunk-level feature per head.
        """
        B, H, P, n = h.shape
        r = self.T_rank
        chunk_feat_per_head = F.layer_norm(chunk_feat_per_head, [P])
        UV_flat = self.UV_mlp(chunk_feat_per_head)
        U_flat, V_flat = UV_flat.chunk(2, dim=-1)
        U = F.normalize(U_flat.view(B, H, n, r), dim=-2)
        V = F.normalize(V_flat.view(B, H, n, r), dim=-2)
        sigma_logit = self.S_mlp(chunk_feat_per_head)
        # Cayley: M = (I + S/2)·(I − S/2)⁻¹, orthogonal. fp32 for stability.
        with torch.amp.autocast("cuda", enabled=False):
            sigma = F.softplus(sigma_logit.float())
            UV = torch.einsum("bhnr,bhr,bhmr->bhnm", U.float(), sigma, V.float())
            S = UV - UV.transpose(-2, -1)
            I_S = torch.eye(n, device=S.device, dtype=S.dtype).expand_as(S)
            S_half = S * 0.5
            M = torch.linalg.solve(I_S - S_half, I_S + S_half)
            return torch.einsum("bhpn,bhnm->bhpm", h.float(), M)

    def forward(self, x):
        """x: (B, L, d_model). Returns (B, L, d_model)."""
        residual = x
        x_norm = self.norm(x)
        B, L, D = x_norm.shape

        xz = self.in_proj(x_norm)
        xz = rearrange(xz, "b l d -> b d l")
        x_in, z = xz.chunk(2, dim=1)

        if causal_conv1d_fn is None:
            x_conv = self.act(self.conv1d(x_in)[..., :L])
        else:
            x_conv = causal_conv1d_fn(
                x_in, rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias, activation=self.activation,
            )

        x_conv_flat = rearrange(x_conv, "b d l -> (b l) d")
        x_db = self.x_proj(x_conv_flat)
        dt_raw, B_all, C_all = torch.split(
            x_db, [self.dt_rank, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1,
        )
        dt = self.dt_proj.weight @ dt_raw.t()
        dt = rearrange(dt, "h (b l) -> b l h", l=L).contiguous()

        B_all = rearrange(B_all, "(b l) (g n) -> b l g n", l=L, g=self.ngroups).contiguous()
        C_all = rearrange(C_all, "(b l) (g n) -> b l g n", l=L, g=self.ngroups).contiguous()
        x_rs = rearrange(x_conv, "b (h p) l -> b l h p", p=self.d_head).contiguous()

        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        dt_bias = self.dt_proj.bias.float()

        outputs = []
        h = None
        inp_dtype = x.dtype
        for c in range(0, L, self.chunk_size):
            cs = min(self.chunk_size, L - c)
            y_ch, h = mamba_chunk_scan_combined(
                x_rs[:, c:c+cs], dt[:, c:c+cs], A,
                B_all[:, c:c+cs], C_all[:, c:c+cs],
                chunk_size=cs, D=D, dt_bias=dt_bias, dt_softplus=True,
                initial_states=h, return_final_states=True,
            )
            y_ch = rearrange(y_ch, "b l h p -> b (h p) l").contiguous()

            chunk_feat_ph = y_ch.view(B, self.n_heads, self.d_head, cs).mean(dim=-1).float()
            h = self._apply_T_state(h, chunk_feat_ph)

            outputs.append(y_ch)

        y = torch.cat(outputs, dim=2)
        y = y * self.act(rearrange(z, "b d l -> b d l"))
        y = rearrange(y, "b d l -> b l d")
        y = self.norm_pre_out(y)
        y = self.out_proj(y)

        out = residual + self.drop(y)
        out = out + self.drop(self.ffn(self.ffn_norm(out)))
        return out

    def step(self, x, conv_state, ssm_state, y_sum, frame_in_chunk):
        """Single-frame step. ``x``: (B, d_model). Tensor states are updated in
        place; ``frame_in_chunk`` is a Python int returned updated."""
        dtype = x.dtype
        residual = x
        x_norm = self.norm(x)
        xz = self.in_proj(x_norm)
        x_in, z = xz.chunk(2, dim=-1)

        if causal_conv1d_update is None:
            conv_state[:, :, :-1] = conv_state[:, :, 1:].clone()
            conv_state[:, :, -1] = x_in
            x_conv = torch.sum(
                conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
            )
            if self.conv1d.bias is not None:
                x_conv = x_conv + self.conv1d.bias
            x_conv = self.act(x_conv).to(dtype=dtype)
        else:
            x_conv = causal_conv1d_update(
                x_in, conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias, self.activation,
            )

        x_db = self.x_proj(x_conv)
        dt_raw, B, C = torch.split(
            x_db, [self.dt_rank, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1,
        )
        dt = F.linear(dt_raw, self.dt_proj.weight)
        dt = F.softplus(dt.float() + self.dt_proj.bias.float())

        B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
        C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
        # ngroups == n_heads — already per-head, no repeat_interleave needed.
        Bh, Ch = B, C

        A = -torch.exp(self.A_log.float())
        D_head = self.D.to(dtype)
        x_ssd = rearrange(x_conv, "b (h p) -> b h p", p=self.d_head)

        dA = torch.exp(A.unsqueeze(0) * dt)
        dBx = torch.einsum("bh,bhn,bhp->bhpn", dt, Bh.float(), x_ssd.float())
        ssm_state.mul_(rearrange(dA, "b h -> b h 1 1")).add_(dBx)
        y = torch.einsum("bhpn,bhn->bhp", ssm_state.to(dtype), Ch)
        y = y + D_head.view(1, self.n_heads, 1) * x_ssd
        y = rearrange(y, "b h p -> b (h p)").contiguous()

        y_sum.add_(y.to(y_sum.dtype))
        new_pos = (frame_in_chunk + 1) % self.chunk_size
        if new_pos == 0:
            chunk_mean = (y_sum / self.chunk_size).float()
            chunk_feat_ph = chunk_mean.view(y.shape[0], self.n_heads, self.d_head)
            ssm_state.copy_(
                self._apply_T_state(ssm_state.float(), chunk_feat_ph).to(ssm_state.dtype)
            )
            y_sum.zero_()

        out = y * self.act(z)
        out = self.norm_pre_out(out)
        out = self.out_proj(out)
        out = residual + self.drop(out)
        out = out + self.drop(self.ffn(self.ffn_norm(out)))
        return out, conv_state, ssm_state, y_sum, new_pos

    def allocate_inference_cache(self, batch_size, device=None, dtype=None):
        device = device or next(self.parameters()).device
        dt = dtype or torch.float32
        conv = torch.zeros(batch_size, self.d_inner, self.d_conv,
                           device=device, dtype=dt)
        ssm  = torch.zeros(batch_size, self.n_heads, self.d_head, self.d_state,
                           device=device, dtype=dt)
        y_sum = torch.zeros(batch_size, self.d_inner, device=device, dtype=dt)
        return conv, ssm, y_sum, 0


class _HybridBlock(nn.Module):
    """SurgicalMamba + FFN."""

    def __init__(self, d_model, d_state=16, d_state_slow=None, d_conv=4, expand=2,
                 dropout=0.0,
                 t_rank=16, chunk_size=32,
                 chunk_size_fast=None, chunk_size_slow=None,
                 use_t_state=True, use_intensity=True,
                 use_fast_path=True, use_slow_path=True):
        super().__init__()
        self.norm_mamba = nn.LayerNorm(d_model)
        self.mamba = SurgicalMamba(
            d_model=d_model, d_state=d_state, d_state_slow=d_state_slow,
            d_conv=d_conv, expand=expand,
            t_rank=t_rank, chunk_size=chunk_size,
            chunk_size_fast=chunk_size_fast, chunk_size_slow=chunk_size_slow,
            use_t_state=use_t_state, use_intensity=use_intensity,
            use_fast_path=use_fast_path, use_slow_path=use_slow_path,
        )

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, slow_state=None):
        """Returns (h, new_slow_state, lam).
        slow_state: (ssm_state, conv_state) tuple or None.
        """
        mamba_out, new_slow, lam = self.mamba(self.norm_mamba(h), slow_state)
        h = h + self.drop(mamba_out)
        h = h + self.drop(self.ffn(self.ffn_norm(h)))
        return h, new_slow, lam

    def step(self, h, conv, ssm):
        out, conv, ssm, _lam = self.mamba.step(
            self.norm_mamba(h).unsqueeze(1), conv, ssm,
        )
        h = h + self.drop(out.squeeze(1))
        h = h + self.drop(self.ffn(self.ffn_norm(h)))
        return h, conv, ssm


class MambaHead(nn.Module):
    """Stack of Mamba2 blocks with per-chunk T_state + Linear head."""

    def __init__(self, d_model, num_classes, n_layers=1,
                 d_state=16, d_conv=4, expand=2, dropout=0.1,
                 d_head=64, chunk_size=32, T_rank=16):
        super().__init__()
        self.blocks = nn.ModuleList([
            _CausalMamba2Block(d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                               d_head=d_head, chunk_size=chunk_size, T_rank=T_rank,
                               dropout=dropout)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x)

    def step(self, x, conv_states, ssm_states, y_sums, frame_in_chunks):
        for i, block in enumerate(self.blocks):
            x, conv_states[i], ssm_states[i], y_sums[i], frame_in_chunks[i] = block.step(
                x, conv_states[i], ssm_states[i], y_sums[i], frame_in_chunks[i],
            )
        return self.head(x), conv_states, ssm_states, y_sums, frame_in_chunks

    def allocate_inference_cache(self, batch_size, device=None, dtype=None):
        conv_states, ssm_states, y_sums, frame_in_chunks = [], [], [], []
        for block in self.blocks:
            c, s, y, f = block.allocate_inference_cache(batch_size, device=device, dtype=dtype)
            conv_states.append(c)
            ssm_states.append(s)
            y_sums.append(y)
            frame_in_chunks.append(f)
        return conv_states, ssm_states, y_sums, frame_in_chunks


# ── Main model ────────────────────────────────────────────────────────────────

class CausalSurgicalMamba(nn.Module):
    """ConvNeXt backbone → visual projector → SurgicalMamba blocks → MambaHead."""

    def __init__(
        self,
        num_phases:                int   = 7,
        backbone:                  str   = "convnext_tiny",
        freeze_backbone:           bool  = True,
        backbone_trainable_stages: int   = 1,
        grad_checkpointing:        bool  = False,
        d_model:                   int   = 768,
        n_layers:                  int   = 4,
        d_state:                   int   = 16,
        d_state_slow:              int   = None,
        d_conv:                    int   = 4,
        expand:                    int   = 2,
        output_dropout:            float = 0.1,
        mamba_dropout:             float = 0.1,
        head_layers:               int   = 1,
        head_chunk_size:           int   = 32,
        t_rank_block:              int   = 16,
        chunk_size_block:          int   = 32,
        chunk_size_fast_block:     int   = None,
        chunk_size_slow_block:     int   = None,
        use_t_state:               bool  = True,
        use_intensity:             bool  = True,
        use_fast_path:             bool  = True,
        use_slow_path:             bool  = True,
    ):
        super().__init__()

        assert d_model % 64 == 0, f"d_model={d_model} must be a multiple of 64"

        self.d_model    = d_model
        self.num_phases = num_phases

        # ── 1. Visual backbone ────────────────────────────────────────────────
        backbone = backbone.lower()
        if backbone == "convnext_v2_tiny":
            backbone = "convnextv2_tiny"
        if backbone not in ("convnext_tiny", "convnextv2_tiny"):
            raise ValueError(f"Unknown backbone: {backbone!r}")
        self.extractor = ConvNeXtTinyExtractor(
            pretrained=True,
            model_name=backbone,
            grad_checkpointing=grad_checkpointing,
        )

        if freeze_backbone:
            for p in self.extractor.backbone.parameters():
                p.requires_grad_(False)
            if backbone_trainable_stages > 0:
                stages = list(self.extractor.backbone.stages)
                for stage in stages[-backbone_trainable_stages:]:
                    for p in stage.parameters():
                        p.requires_grad_(True)
                for p in self.extractor.backbone.norm_pre.parameters():
                    p.requires_grad_(True)

        self.d_visual = self.extractor.num_features

        # ── 2. Visual projector ───────────────────────────────────────────────
        self.visual_proj = nn.Sequential(
            nn.Linear(self.d_visual, d_model),
            nn.LayerNorm(d_model),
        )

        # ── 3. Hybrid blocks ─────────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            _HybridBlock(
                d_model=d_model, d_state=d_state, d_state_slow=d_state_slow,
                d_conv=d_conv, expand=expand, dropout=mamba_dropout,
                t_rank=t_rank_block, chunk_size=chunk_size_block,
                chunk_size_fast=chunk_size_fast_block,
                chunk_size_slow=chunk_size_slow_block,
                use_t_state=use_t_state, use_intensity=use_intensity,
                use_fast_path=use_fast_path, use_slow_path=use_slow_path,
            )
            for _ in range(n_layers)
        ])

        self.frame_norm = nn.LayerNorm(d_model)

        # ── 4. Output head ────────────────────────────────────────────────────
        self.output_dropout = nn.Dropout(output_dropout)
        self.output_head = MambaHead(
            d_model=d_model, num_classes=num_phases,
            n_layers=head_layers, d_state=d_state, d_conv=d_conv,
            expand=expand, dropout=output_dropout,
            chunk_size=head_chunk_size,
        )


    def forward_clip(self, frames, slow_states=None):
        """Process one T-frame clip.

        Args:
            frames:      (B, T, 3, H, W) input clip.
            slow_states: per-block list of (ssm_state, conv_state), or None.

        Returns:
            logits:          (B, T, num_phases) per-frame phase logits.
            new_slow_states: per-block list carried to the next clip.
            frame_hidden:    (B, T, d_model) detached penultimate features.
            lambdas:         list of (B, T) intensity logits per layer.
        """
        B, T, C, H, W = frames.shape

        f = self.extractor(frames.view(B * T, C, H, W)).view(B, T, self.d_visual)
        h = self.visual_proj(f)

        if slow_states is None:
            slow_states = [None] * len(self.blocks)
        new_slow_states = [None] * len(self.blocks)
        lambdas = []

        for i, block in enumerate(self.blocks):
            h, new_slow_states[i], lam = block(h, slow_state=slow_states[i])
            if lam is not None:
                lambdas.append(lam)

        h = self.frame_norm(h)
        logits = self.output_head(self.output_dropout(h))

        return logits, new_slow_states, h.detach(), lambdas

    def forward(self, frames):
        logits, _, _, _ = self.forward_clip(frames)
        return logits

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "CausalSurgicalMamba":
        m = cfg.model
        return cls(
            num_phases                = cfg.data.num_phases,
            backbone                  = m.get("backbone", "convnext_tiny"),
            freeze_backbone           = m.get("freeze_backbone", True),
            backbone_trainable_stages = m.get("backbone_trainable_stages", 1),
            grad_checkpointing        = m.get("grad_checkpointing", False),
            d_model                   = m.get("d_model", 768),
            n_layers                  = m.get("n_layers", 4),
            d_state                   = m.get("d_state", 16),
            d_state_slow              = m.get("d_state_slow", None),
            d_conv                    = m.get("d_conv", 4),
            expand                    = m.get("expand", 2),
            output_dropout            = m.get("output_dropout", 0.1),
            mamba_dropout             = m.get("mamba_dropout", 0.1),
            head_layers               = m.get("head_layers", 1),
            head_chunk_size           = m.get("head_chunk_size", 32),
            t_rank_block              = m.get("t_rank_block", 16),
            chunk_size_block          = m.get("chunk_size_block", 32),
            chunk_size_fast_block     = m.get("chunk_size_fast_block", None),
            chunk_size_slow_block     = m.get("chunk_size_slow_block", None),
            use_t_state               = m.get("use_t_state", True),
            use_intensity             = m.get("use_intensity", True),
            use_fast_path             = m.get("use_fast_path", True),
            use_slow_path             = m.get("use_slow_path", True),
        )


# ── Online frame-by-frame inference ───────────────────────────────────────────

class OnlineSession:
    """Frame-by-frame online inference for CausalSurgicalMamba.

    Slow SSM state and fast conv state carry across clips; fast SSM and head
    states reset at the end of every clip (matches training).

    With ``use_cuda_graph=True`` the normal per-frame path is captured into a
    CUDA graph and replayed; frames that land on a chunk boundary (block or
    head) run eager, since the per-chunk Cayley T_state is a different kernel
    sequence. Most frames hit the fast graph replay.
    """

    def __init__(self, model, clip_len=128, device=None, dtype=None,
                 use_cuda_graph=False):
        self.model    = model.eval()
        self.clip_len = clip_len
        self.device   = torch.device(device) if device is not None \
            else next(model.parameters()).device
        self.dtype    = dtype
        self.use_cuda_graph = use_cuda_graph
        if use_cuda_graph and self.device.type != "cuda":
            raise ValueError("use_cuda_graph=True requires a CUDA device.")
        self._graph = None
        self._graph_input = None
        self._graph_output = None
        self.reset()

    def reset(self):
        self._frame_idx    = 0
        self._global_pos   = 0   # frames since session start (block chunk counter)
        self._block_states = self._zero_states()
        (self._head_conv_states, self._head_ssm_states,
         self._head_y_sums, self._head_frame_in_chunks) = self._zero_head_states()
        self._graph = None       # state tensors reallocated → old graph invalid

    @torch.no_grad()
    def step(self, frame):
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)
        frame = frame.to(self.device)
        with torch.autocast("cuda", dtype=self.dtype, enabled=self.dtype is not None):
            if self.use_cuda_graph:
                logit = self._step_cuda_graph(frame)
            else:
                logit = self._step_impl(frame)

        self._frame_idx  += 1
        self._global_pos += 1
        if self._frame_idx >= self.clip_len:
            self._frame_idx = 0
            self._end_of_clip()

        return logit

    def _step_cuda_graph(self, frame):
        cs_block = self.model.blocks[0].mamba.chunk_size_slow
        cs_head  = self.model.output_head.blocks[0].chunk_size

        if self._graph is None:
            # Warmup triggers Triton / cuDNN autotuning (which syncs — illegal
            # during capture). State is then zeroed so capture starts clean.
            self._graph_input = torch.empty_like(frame)
            self._graph_input.copy_(frame)
            _ = self._step_impl(self._graph_input)
            self._reset_all_states_inplace()

            # Capture the normal path: counters are 0, so no chunk-end branch.
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                self._graph_output = self._step_impl(self._graph_input)

            self._reset_all_states_inplace()
            self._frame_idx  = 0
            self._global_pos = 0

        # A chunk boundary for the block (carries across clips) or the head
        # (resets each clip) needs the eager Cayley T_state path.
        block_end = (self._global_pos % cs_block) == (cs_block - 1)
        head_end  = (self._frame_idx  % cs_head)  == (cs_head - 1)
        if block_end or head_end:
            self._sync_counters()
            return self._step_impl(frame)

        self._graph_input.copy_(frame)
        self._graph.replay()
        return self._graph_output.clone()

    def _sync_counters(self):
        """Resync host-side chunk counters before an eager step — graph
        replays advance neither the block nor the head counter."""
        cs_block = self.model.blocks[0].mamba.chunk_size_slow
        cs_head  = self.model.output_head.blocks[0].chunk_size
        block_pos = self._global_pos % cs_block
        head_pos  = self._frame_idx  % cs_head
        for i, (conv, ssm) in enumerate(self._block_states):
            self._block_states[i] = (conv, (*ssm[:4], block_pos))
        self._head_frame_in_chunks = [head_pos] * len(self._head_frame_in_chunks)

    def _reset_all_states_inplace(self):
        """Zero every state tensor in place and reset host-side counters."""
        for i, (conv, ssm) in enumerate(self._block_states):
            conv.zero_()
            for s in ssm[:4]:
                s.zero_()
            self._block_states[i] = (conv, (*ssm[:4], 0))
        for lst in (self._head_conv_states, self._head_ssm_states, self._head_y_sums):
            for t in lst:
                t.zero_()
        self._head_frame_in_chunks = [0] * len(self._head_frame_in_chunks)

    def _step_impl(self, frame):
        m = self.model

        f = m.extractor(frame)
        h = m.visual_proj(f)

        for i, block in enumerate(m.blocks):
            conv, ssm = self._block_states[i]
            h, conv, ssm = block.step(h, conv, ssm)
            self._block_states[i] = (conv, ssm)

        h = m.frame_norm(h)

        logit, self._head_conv_states, self._head_ssm_states, self._head_y_sums, self._head_frame_in_chunks = \
            m.output_head.step(
                m.output_dropout(h), self._head_conv_states, self._head_ssm_states,
                self._head_y_sums, self._head_frame_in_chunks,
            )

        return logit

    def _zero_states(self):
        # ssm tuple = (ssm_fast, ssm_slow, y_sum_slow, y_sum_fast, frame_in_chunk).
        states = []
        for block in self.model.blocks:
            mb = block.mamba
            conv = torch.zeros(1, mb.d_inner * 2, mb.d_conv, device=self.device)
            ssm = (
                torch.zeros(1, mb.n_heads_fast, mb.d_head_fast, mb.d_state, device=self.device),
                torch.zeros(1, mb.n_heads_slow, mb.d_head_slow, mb.d_state_slow, device=self.device),
                torch.zeros(1, mb.d_inner, device=self.device),
                torch.zeros(1, mb.d_inner, device=self.device),
                0,  # frame_in_chunk — host-side counter
            )
            states.append((conv, ssm))
        return states

    def _zero_head_states(self, in_place=False):
        if in_place and hasattr(self, '_head_conv_states'):
            for lst in (self._head_conv_states, self._head_ssm_states,
                        self._head_y_sums):
                for t in lst:
                    t.zero_()
            self._head_frame_in_chunks = [0] * len(self._head_frame_in_chunks)
            return (self._head_conv_states, self._head_ssm_states,
                    self._head_y_sums, self._head_frame_in_chunks)
        return self.model.output_head.allocate_inference_cache(
            1, device=self.device,
        )

    def _reset_fast_states(self):
        # Fast conv + fast SSM reset per clip (matches training: zero-padded
        # conv1d, fresh fast scan). Slow conv, slow SSM, y_sums and the
        # chunk-position counter carry across clips.
        for i, block in enumerate(self.model.blocks):
            mb = block.mamba
            conv, ssm = self._block_states[i]
            conv[:, :mb.d_inner, :].zero_()
            ssm[0].zero_()

    def _end_of_clip(self):
        self._reset_fast_states()
        self._zero_head_states(in_place=True)
