"""
Surgical Phase Recognition via LLM Reprogramming.

Architecture (per clip):

  frames (B, T, 3, H, W)
       │
       ▼ VMamba-Tiny extractor
  visual_feats (B, T, 768)
       │
       ▼ TemporalRefiner (Mamba2 × N layers)
  refined_feats (B, T, 768)   — temporally-aware frame features
       │
       ├──────────────────────────────────────┐
       │  VisualProjector                     │ ClipHintEncoder
       │  ┌──────────────────────────────┐    │  temporal-segment biased queries
       │  │ direct = Linear(768→d_llm)   │    │  + transition stream (frame diffs)
       │  │ reprogram = Reprogram(...)   │    │  + slot embeddings + self-attn + FFN
       │  │ fused = LN(direct+reprogram) │    │  → (B, N_hints, d_llm)
       │  │ visual_tokens = LN(fused     │    │
       │  │               + FFN(fused))  │    │
       │  └──────────┬───────────────────┘    │
       │             │ visual_tokens           │ hint_tokens (B, N_hints, d_llm)
       └─────────────┼────────────────────────┘
                     │
  Cross-clip context (from previous clip, both detached):
    prev_memory  (B, N_hints, d_llm)  — Mamba SSM global summary of all past clips
    prev_visual  (B, T, d_llm)        → LocalContextCompressor → (B, T//4, d_llm)
                     │
                     ▼
  LLM input: [enriched_memory(N_hints) | hint_tokens | tool_text | compressed_prev(T//4) | visual_tokens]
  with past_key_values = prompt_kv (fixed, computed once per video)
                     │
                     ▼ Frozen LLM
  hidden (B, *, d_llm)
                     │
   take last T positions, first d_ff dims
                     │
                     ▼ output_head
  logits (B, T, num_phases)

  + new_memory = CrossClipMemory(hints, prev_memory)  → passed to next clip (detached)
  + visual_tokens (detached)                          → passed to next clip as prev_visual
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from .vmamba_extractor import VMambaTinyExtractor
from .reprogramming import ReprogrammingLayer
from .clip_hint import ClipHintEncoder
from .temporal_mamba import TemporalRefiner, CrossClipMemory, LocalContextCompressor


CHOLEC80_PHASES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]


CHOLEC80_TOOLS = [
    "Grasper",
    "Bipolar",
    "Hook",
    "Scissors",
    "Clipper",
    "Irrigator",
    "SpecimenBag",
]

_DEFAULT_PROMPT = (
    "<|start_prompt|>"
    "Dataset description: Laparoscopic cholecystectomy surgical video recorded at 1 fps. "
    "Task description: Recognize the surgical phase label for each video frame token "
    "given the sequence of visual feature tokens. "
    "Phase 0 - Preparation: initial setup and trocar insertion before the procedure begins. "
    "Phase 1 - CalotTriangleDissection: dissection of peritoneum around the Calot triangle "
    "to expose the cystic duct and artery. "
    "Phase 2 - ClippingCutting: application of clips and cutting of the cystic duct and artery. "
    "Phase 3 - GallbladderDissection: dissection of the gallbladder from the liver bed. "
    "Phase 4 - GallbladderPackaging: placement of the gallbladder into a retrieval bag. "
    "Phase 5 - CleaningCoagulation: irrigation, cleaning, and coagulation of the surgical site. "
    "Phase 6 - GallbladderRetraction: retraction and extraction of the gallbladder. "
    "The 7 surgical tools used in this procedure are: "
    "Grasper: grasps and retracts tissue to expose the surgical field. "
    "Bipolar: applies bipolar electrical current for hemostasis and coagulation. "
    "Hook: dissects and coagulates tissue using a hook-shaped electrosurgical tip. "
    "Scissors: cuts tissue and structures during dissection. "
    "Clipper: applies metal clips to seal the cystic duct and cystic artery before cutting. "
    "Irrigator: irrigates the surgical field with saline and aspirates fluid. "
    "SpecimenBag: a retrieval bag used to contain the gallbladder for safe extraction. "
    "Typical tool usage per phase: "
    "Phase 0 - Preparation: no tools or Grasper only. "
    "Phase 1 - CalotTriangleDissection: Grasper, Hook, Bipolar. "
    "Phase 2 - ClippingCutting: Grasper, Clipper, Scissors. "
    "Phase 3 - GallbladderDissection: Grasper, Hook, Bipolar. "
    "Phase 4 - GallbladderPackaging: Grasper, SpecimenBag. "
    "Phase 5 - CleaningCoagulation: Grasper, Bipolar, Irrigator. "
    "Phase 6 - GallbladderRetraction: Grasper, SpecimenBag. "
    "Input sequence structure per clip: "
    "[Global memory tokens] A fixed number of tokens summarising all previously seen clips "
    "via a Mamba SSM updated with cross-attention to the current clip, providing long-range "
    "surgical context across the entire video. "
    "[Local hint tokens] Clip-level visual summary tokens follow the global memory. "
    "Each local hint token is a learned compression of the current clip visual features, "
    "capturing dominant scene content, temporal dynamics, and phase transition signals. "
    "[Tool context] A text description of which surgical tools are active in each temporal "
    "segment of the current clip follows the local hint tokens. "
    "[Previous clip context] A temporally compressed representation of the immediately "
    "preceding clip's visual tokens follows the tool context, providing dense local "
    "continuity between consecutive clips. "
    "[Visual tokens] Per-frame visual feature tokens follow the previous clip context. "
    "Each visual token encodes one second of video as a 768-dimensional feature "
    "vector extracted by a VMamba-Tiny backbone and refined by a Mamba temporal encoder. "
    "<|end_prompt|>"
)

# Number of temporal segments for per-segment tool text
_N_TOOL_SEGMENTS = 4


class SurgicalPhaseLLM(nn.Module):
    """
    Surgical phase recognition model with:
      - VMamba-Tiny visual extractor
      - TemporalRefiner: Mamba2 blocks for temporal feature refinement
      - VisualProjector: Linear + ReprogrammingLayer + FFN (transformer-style)
      - ClipHintEncoder: temporal-segment-biased + transition-aware hints
      - Dynamic per-segment tool text injection
      - Frozen LLM with cached prompt KV
      - Detached context prefix for cross-clip temporal continuity
      - Per-frame output head

    Args:
        llm_model_name:     HuggingFace model ID
        num_phases:         surgical phase count
        vmamba_pretrained:  VMamba checkpoint path (optional)
        freeze_llm:         freeze LLM weights
        freeze_vmamba:      freeze VMamba extractor
        mamba_layers:       number of Mamba2 temporal refinement layers
        n_heads:            heads for reprogramming cross-attention
        d_ff:               output bottleneck dimension
        num_tokens:         vocabulary reduction for reprogramming K/V
        n_hints:            number of ClipHint tokens per clip
        prompt:             task description text prefix
        attention_dropout:  dropout in ReprogrammingLayer
        hint_dropout:       dropout in ClipHintEncoder
        output_dropout:     dropout before output head
    """

    def __init__(
        self,
        llm_model_name: str = "Qwen/Qwen2.5-0.5B",
        num_phases: int = 7,
        vmamba_pretrained: str = None,
        freeze_llm: bool = True,
        freeze_vmamba: bool = False,
        mamba_layers: int = 2,
        n_heads: int = 8,
        d_ff: int = 256,
        num_tokens: int = 1000,
        n_hints: int = 8,
        local_context_ratio: int = 4,
        prompt: str = _DEFAULT_PROMPT,
        attention_dropout: float = 0.1,
        hint_dropout: float = 0.1,
        output_dropout: float = 0.1,
    ):
        super().__init__()
        self.d_ff = d_ff
        self.n_hints = n_hints

        # ── 1. VMamba visual extractor ───────────────────────────────────────
        self.extractor = VMambaTinyExtractor(pretrained=vmamba_pretrained)
        self.d_visual = self.extractor.num_features  # 768

        if freeze_vmamba:
            for p in self.extractor.parameters():
                p.requires_grad_(False)

        # ── 2. Mamba temporal refiner ────────────────────────────────────────
        # Refines per-frame features across the temporal dimension before
        # LLM reprogramming. Captures temporal dependencies that VMamba's
        # per-frame spatial processing cannot see.
        self.temporal_refiner = TemporalRefiner(
            d_model=self.d_visual,
            num_layers=mamba_layers,
        )

        # ── 3. LLM (frozen by default) ───────────────────────────────────────
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.d_llm = self.llm.config.hidden_size

        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad_(False)

        # ── 4. Visual projector: Linear + Reprogram + FFN ───────────────────
        # Direct path
        self.linear_proj = nn.Linear(self.d_visual, self.d_llm)

        # Semantic reprogramming path
        word_emb_weight = self.llm.get_input_embeddings().weight  # (V, d_llm)
        self.vocab_size = word_emb_weight.shape[0]
        self.mapping_layer = nn.Linear(self.vocab_size, num_tokens)
        self.reprogramming = ReprogrammingLayer(
            d_model=self.d_visual,
            n_heads=n_heads,
            d_llm=self.d_llm,
            attention_dropout=attention_dropout,
        )
        self.register_buffer("word_embeddings", word_emb_weight.detach().float())

        # Fusion + FFN (transformer-style projector block)
        self.visual_norm1 = nn.LayerNorm(self.d_llm)
        self.visual_ffn = nn.Sequential(
            nn.Linear(self.d_llm, self.d_llm * 2),
            nn.GELU(),
            nn.Linear(self.d_llm * 2, self.d_llm),
        )
        self.visual_norm2 = nn.LayerNorm(self.d_llm)

        # ── 5. Clip hint encoder ─────────────────────────────────────────────
        self.hint_encoder = ClipHintEncoder(
            n_hints=n_hints,
            d_visual=self.d_visual,
            d_llm=self.d_llm,
            n_heads=n_heads,
            dropout=hint_dropout,
            max_seq=512,
        )

        # ── 6. Prompt KV cache setup ─────────────────────────────────────────
        self.prompt_text = prompt
        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).input_ids

        with torch.no_grad():
            prompt_emb = self.llm.get_input_embeddings()(prompt_ids)  # (1, P, d_llm)

        self.register_buffer("prompt_emb", prompt_emb.detach())
        self.prompt_len = prompt_emb.shape[1]
        self._prompt_kv: tuple = None

        # ── 7. Cross-clip memory (SSM-based) + local context compressor ──────
        # CrossClipMemory has two separate paths:
        #   Read:  prev_memory(Q) × visual_tokens(K/V) → enriched_memory
        #     Memory slots refresh themselves by attending to current clip.
        #     enriched_memory is the LLM's global prefix; visual_tokens unchanged.
        #   Write: Mamba([prev_memory | raw hints])[-N:] → new_memory
        #     Stored for next clip; independent of read path (no contamination).
        # LocalContextCompressor reduces the previous clip's visual tokens
        #   to T//4 tokens via average pooling, prepended after global memory.
        self.cross_clip_memory = CrossClipMemory(d_model=self.d_llm)
        self.local_compressor = LocalContextCompressor(ratio=local_context_ratio)

        # ── 8. Output head ───────────────────────────────────────────────────
        self.output_dropout = nn.Dropout(output_dropout)
        self.output_head = nn.Linear(d_ff, num_phases)

    # ── Prompt KV cache ──────────────────────────────────────────────────────

    @torch.no_grad()
    def build_prompt_kv(self) -> tuple:
        """
        Compute and cache the prompt's KV pairs from the frozen LLM.
        Stored as legacy tuple format so each forward_clip builds a fresh
        DynamicCache — prevents in-place mutation across clips.
        """
        device = self.prompt_emb.device
        out = self.llm(
            inputs_embeds=self.prompt_emb.to(device),
            use_cache=True,
            output_hidden_states=False,
        )
        kv = out.past_key_values
        if hasattr(kv, "to_legacy_cache"):
            kv = kv.to_legacy_cache()
        self._prompt_kv = kv
        return self._prompt_kv

    # ── Per-segment tool text embedding ──────────────────────────────────────

    @torch.no_grad()
    def _make_tool_emb(self, tool_annots: torch.Tensor) -> torch.Tensor:
        """
        Build per-segment tool text embeddings for the current clip.

        Instead of a single clip-level summary, the clip is divided into
        _N_TOOL_SEGMENTS equal segments. For each segment, we describe
        which tools were active. This gives the LLM fine-grained temporal
        information about tool usage (e.g. "frames 0-31: Grasper; frames
        32-63: Hook, Clipper"), which correlates strongly with phase.

        Args:
            tool_annots: (B, T, 7) binary/float tool presence per frame

        Returns:
            tool_emb: (B, L_tool, d_llm)
        """
        device = tool_annots.device
        B, T, _ = tool_annots.shape
        embed_fn = self.llm.get_input_embeddings()
        n_seg = _N_TOOL_SEGMENTS

        embs = []
        for b in range(B):
            seg_size = max(T // n_seg, 1)
            parts = []
            for s in range(n_seg):
                start = s * seg_size
                end   = start + seg_size if s < n_seg - 1 else T
                seg_tools = tool_annots[b, start:end, :]          # (seg_len, 7)
                present = seg_tools.sum(dim=0) > 0                # (7,)
                active = [name for name, flag in zip(CHOLEC80_TOOLS, present.tolist()) if flag]
                tool_str = f"frames {start}-{end-1}: " + (", ".join(active) if active else "none")
                parts.append(tool_str)

            full_str = "Tool context: " + "; ".join(parts) + "."
            ids = self.tokenizer(
                full_str, return_tensors="pt", truncation=True, max_length=128
            ).input_ids.to(device)
            embs.append(embed_fn(ids))                            # (1, L, d_llm)

        max_len = max(e.shape[1] for e in embs)
        padded = torch.zeros(B, max_len, embs[0].shape[-1], device=device, dtype=embs[0].dtype)
        for b, e in enumerate(embs):
            padded[b, :e.shape[1]] = e[0]
        return padded  # (B, L_tool, d_llm)

    # ── Per-clip forward ─────────────────────────────────────────────────────

    def forward_clip(
        self,
        frames: torch.Tensor,
        tool_annots: torch.Tensor = None,
        memory: torch.Tensor = None,
        prev_visual: torch.Tensor = None,
        prompt_kv: tuple = None,
    ):
        """
        Process one T-frame clip.

        Args:
            frames:      (B, T, 3, H, W)
            tool_annots: (B, T, 7) binary/float tool presence, optional
            memory:      (B, N_hints, d_llm) detached CrossClipMemory output from
                         previous clip; prepended directly as global hint tokens
            prev_visual: (B, T, d_llm) detached visual_tokens from previous clip;
                         compressed to T//4 and prepended as local context
            prompt_kv:   cached KV from build_prompt_kv(); uses self._prompt_kv if None

        Returns:
            logits:          (B, T, num_phases)
            new_memory:      (B, N_hints, d_llm) detached – pass as memory for next clip
            new_prev_visual: (B, T, d_llm) detached  – pass as prev_visual for next clip
            hints:           (B, N_hints, d_llm) – for hint diversity loss
            attn_focus_loss: scalar – attention focus loss from hint encoder
        """
        if prompt_kv is None:
            prompt_kv = self._prompt_kv

        B, T, C, H, W = frames.shape

        # 1. VMamba: per-frame spatial features ───────────────────────────────
        feats = self.extractor(frames.view(B * T, C, H, W)).view(B, T, self.d_visual)

        # 2. Mamba temporal refinement ────────────────────────────────────────
        # Captures temporal dependencies across the clip before LLM projection
        feats = self.temporal_refiner(feats)                           # (B, T, d_visual)

        # 3. Visual projector (transformer block style) ───────────────────────
        direct   = self.linear_proj(feats)                             # (B, T, d_llm)
        src_emb  = self.mapping_layer(
            self.word_embeddings.permute(1, 0)).permute(1, 0)          # (num_tokens, d_llm)
        reprogram = self.reprogramming(feats, src_emb)                 # (B, T, d_llm)

        # Residual 1: fusion
        fused = self.visual_norm1(direct + reprogram)                  # (B, T, d_llm)
        # Residual 2: FFN
        visual_tokens = self.visual_norm2(fused + self.visual_ffn(fused))  # (B, T, d_llm)

        # 4. Clip hint tokens ─────────────────────────────────────────────────
        hints, attn_focus_loss = self.hint_encoder(feats)               # (B, N_hints, d_llm)

        # 4b. Cross-clip memory update ────────────────────────────────────────
        # Read:  prev_memory(Q) × visual_tokens(K/V) → enriched_memory (B, N, d)
        #   Each memory slot attends to the current clip and refreshes itself.
        #   enriched_memory replaces raw memory in the LLM prefix, giving the
        #   LLM a global context tuned to the current clip.
        #   visual_tokens are untouched — clean separation of roles.
        # Write: Mamba([prev_memory | raw hints])[-N:] → new_memory
        #   Stored (detached) for the NEXT clip; independent of read path.
        new_memory, enriched_memory = self.cross_clip_memory(
            visual_tokens, hints, memory
        )                                                               # enriched: (B, N, d_llm)

        # 5. Build LLM input sequence ─────────────────────────────────────────
        # Order matches the prompt description exactly:
        #   [Global memory] | [Local hints] | [Tool context] |
        #   [Previous clip context] | [Visual tokens]
        parts = []
        if enriched_memory is not None:
            parts.append(enriched_memory)                               # [Global memory tokens]
        elif memory is not None:
            parts.append(memory)                                        # first-clip fallback
        parts.append(hints)                                             # [Local hint tokens]
        if tool_annots is not None:
            parts.append(self._make_tool_emb(tool_annots))             # [Tool context]
        if prev_visual is not None:
            parts.append(self.local_compressor(prev_visual))           # [Previous clip context]
        parts.append(visual_tokens)                                     # [Visual tokens]

        llm_dtype = next(self.llm.parameters()).dtype
        llm_input = torch.cat([p.to(llm_dtype) for p in parts], dim=1)  # (B, *, d_llm)

        # 6. Frozen LLM forward ───────────────────────────────────────────────
        # Fresh DynamicCache each clip so the stored prompt tensors are not
        # mutated in-place (which would break the second backward call).
        if isinstance(prompt_kv, tuple):
            kv_cache = DynamicCache.from_legacy_cache(prompt_kv)
        else:
            kv_cache = prompt_kv

        lm_out = self.llm(
            inputs_embeds=llm_input,
            past_key_values=kv_cache,
            use_cache=False,
            output_hidden_states=True,
        )
        hidden = lm_out.hidden_states[-1]                             # (B, *, d_llm)

        # 7. Extract frame positions and predict ──────────────────────────────
        frame_hidden = hidden[:, -T:, :self.d_ff].float()             # (B, T, d_ff)
        logits = self.output_head(self.output_dropout(frame_hidden))  # (B, T, num_phases)

        return (
            logits, new_memory.detach(), visual_tokens.detach(),
            hints, attn_focus_loss,
        )

    # ── Convenience ──────────────────────────────────────────────────────────

    def forward(self, frames: torch.Tensor, tool_annots: torch.Tensor = None):
        """Single-clip forward without temporal context (for quick testing)."""
        if self._prompt_kv is None:
            self.build_prompt_kv()
        logits, _, _, _, _ = self.forward_clip(frames, tool_annots=tool_annots)
        return logits

    # ── Config factory ───────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "SurgicalPhaseLLM":
        """Build model from OmegaConf config (cfg.model section)."""
        m = cfg.model
        return cls(
            llm_model_name   = m.llm_model_name,
            num_phases       = m.num_phases,
            vmamba_pretrained= m.vmamba_pretrained,
            freeze_llm       = m.freeze_llm,
            freeze_vmamba    = m.freeze_vmamba,
            mamba_layers     = m.mamba_layers,
            n_heads          = m.n_heads,
            d_ff             = m.d_ff,
            num_tokens       = m.num_tokens,
            n_hints          = m.n_hints,
            local_context_ratio = m.local_context_ratio,
            prompt           = m.prompt,
            attention_dropout= m.attention_dropout,
            hint_dropout     = m.hint_dropout,
            output_dropout   = m.output_dropout,
        )
