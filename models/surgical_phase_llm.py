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

from .vmamba_extractor import VMambaTinyExtractor, CLIPExtractor
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

# ── Prompt prefix (static, KV-cached) ────────────────────────────────────────
# Describes dataset context, phase semantics, and tool semantics.
# Does NOT describe input structure — section headers below do that inline.
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
    "<|end_prompt|>"
)

# ── Section headers (static, pre-embedded as buffers) ────────────────────────
# Each header is placed immediately before its corresponding token group,
# so the LLM sees "what these tokens are" right at the point they appear.
_SEG_MEMORY = "The following tokens summarise the global surgical context from all previous clips:"
_SEG_HINTS  = "The following tokens are visual summary tokens for the current clip:"
_SEG_PREV   = "The following tokens are a compressed representation of the previous clip:"
_SEG_VISUAL = "The following tokens are per-frame visual feature tokens for the current clip:"

# Number of temporal segments for per-segment tool text
_N_TOOL_SEGMENTS = 4

# ── Surgical/medical domain vocabulary for reprogramming token selection ──────
# Used to select semantically relevant LLM tokens as K/V bank for the
# ReprogrammingLayer, replacing the naive "first num_tokens tokens" approach.
_SURGICAL_SEED_WORDS = [
    # Phase names (full + decomposed)
    "Preparation", "preparation", "prepare", "prepared", "preparing",
    "Dissection", "dissection", "dissect", "dissecting",
    "Clipping", "clipping", "clip", "clips",
    "Cutting", "cutting", "cut", "cutter",
    "Gallbladder", "gallbladder", "gall", "bladder",
    "Packaging", "packaging", "package",
    "Retraction", "retraction", "retract", "retracting",
    "Coagulation", "coagulation", "coagulate", "coagulating",
    "Cleaning", "cleaning", "clean",
    "Irrigation", "irrigation", "irrigate", "irrigating",
    "Calot", "calot", "triangle",
    # Anatomy
    "peritoneum", "cystic", "duct", "artery",
    "liver", "bile", "abdomen", "abdominal", "hepatic",
    "hepatocystic", "cholecyst", "cholecystectomy",
    "trocar", "laparoscopic", "laparoscopy", "endoscopic",
    # Tools
    "Grasper", "grasper", "grasp", "grasping",
    "Bipolar", "bipolar",
    "Hook", "hook",
    "Scissors", "scissors", "scissor",
    "Clipper", "clipper",
    "Irrigator", "irrigator",
    "SpecimenBag", "specimen", "bag",
    # Surgical actions
    "insert", "remove", "place", "apply",
    "cut", "suture", "staple", "stitch",
    "irrigate", "aspirate", "coagulate",
    "dissect", "expose", "identify", "separate", "mobilize",
    "grasp", "retract", "lift", "pull", "push",
    "visualize", "inspect", "examine",
    # Medical/surgical context
    "surgical", "surgery", "operation", "procedure", "operative", "intraoperative",
    "tissue", "vessel", "hemorrhage", "bleeding", "hemostasis",
    "incision", "visualization", "exposure", "identification",
    "instrument", "device", "tool",
    "phase", "stage", "step", "transition",
    "visible", "identified", "completed", "ongoing", "initiated",
    "patient", "anatomy", "blood", "fat", "adipose",
    "camera", "endoscope", "laparoscope", "scope", "port",
    "frame", "video", "recognition", "detection",
]


def _select_domain_token_ids(tokenizer, vocab_size: int, num_tokens: int) -> torch.Tensor:
    """
    Build a set of token IDs biased toward surgical/medical domain concepts.

    Strategy:
      1. Tokenize each word in _SURGICAL_SEED_WORDS (with/without leading space).
      2. Collect all unique token IDs that appear in those tokenizations.
      3. If fewer than num_tokens, pad with sequential IDs not yet included.

    Returns:
        LongTensor of shape (num_tokens,)
    """
    seen: set = set()
    ordered: list = []

    for word in _SURGICAL_SEED_WORDS:
        for variant in [word, " " + word, word.lower(), " " + word.lower()]:
            ids = tokenizer(variant, add_special_tokens=False).input_ids
            for tid in ids:
                if 0 <= tid < vocab_size and tid not in seen:
                    seen.add(tid)
                    ordered.append(tid)
                    if len(ordered) >= num_tokens:
                        break
            if len(ordered) >= num_tokens:
                break
        if len(ordered) >= num_tokens:
            break

    # Pad with sequential tokens if surgical vocab didn't fill num_tokens
    if len(ordered) < num_tokens:
        for tid in range(vocab_size):
            if tid not in seen:
                ordered.append(tid)
                seen.add(tid)
            if len(ordered) >= num_tokens:
                break

    return torch.tensor(ordered[:num_tokens], dtype=torch.long)


def _build_hint_init_embeddings(
    embed_fn, tokenizer, n_hints: int, d_llm: int
) -> torch.Tensor:
    """
    Initialize hint queries from LLM embeddings of phase/tool concept names.

    Each phase and tool name is tokenized and its token embeddings are mean-pooled
    to a single d_llm vector. Vectors are tiled to fill n_hints slots.

    Returns:
        FloatTensor of shape (n_hints, d_llm)
    """
    concepts = CHOLEC80_PHASES + CHOLEC80_TOOLS  # 14 concepts

    emb_list = []
    for concept in concepts:
        ids = tokenizer(concept, add_special_tokens=False, return_tensors="pt").input_ids
        with torch.no_grad():
            tok_embs = embed_fn(ids)            # (1, L, d_llm)
        emb_list.append(tok_embs.mean(dim=1).squeeze(0))  # (d_llm,)

    embs = torch.stack(emb_list, dim=0).float()            # (14, d_llm)

    # Tile to fill n_hints
    if embs.shape[0] < n_hints:
        repeats = (n_hints + embs.shape[0] - 1) // embs.shape[0]
        embs = embs.repeat(repeats, 1)[:n_hints]
    else:
        embs = embs[:n_hints]

    return embs  # (n_hints, d_llm)


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
        llm_trainable_layers: int = 0,
        freeze_vmamba: bool = False,
        use_mapping_layer: bool = True,
        vmamba_trainable_stages: int = 0,
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
        visual_backbone: str = "vmamba",       # "vmamba" | "clip"
        clip_trainable_layers: int = 0,        # clip: unfreeze last N ViT layers (0=fully frozen)
        # ── QLoRA fine-tuning ────────────────────────────────────────────────
        use_qlora: bool = False,               # enable 4-bit QLoRA for LLM fine-tuning
        qlora_r: int = 8,                      # LoRA rank
        qlora_alpha: int = 16,                 # LoRA alpha (scaling = alpha/r)
        qlora_dropout: float = 0.05,
        qlora_target_modules: list = None,     # None → auto-detect q/k/v/o projections
    ):
        super().__init__()
        self.d_ff = d_ff
        self.n_hints = n_hints
        self._use_qlora = use_qlora

        # ── 1. Visual extractor (VMamba or CLIP) ─────────────────────────────
        if visual_backbone == "clip":
            # CLIPExtractor freezes vision_model internally;
            # pool_query / pool_attn / proj are always trainable.
            self.extractor = CLIPExtractor(
                freeze=freeze_vmamba,
                trainable_layers=clip_trainable_layers,
            )
        else:
            self.extractor = VMambaTinyExtractor(pretrained=vmamba_pretrained)
            if freeze_vmamba:
                # Freeze entire VMamba backbone first
                for p in self.extractor.parameters():
                    p.requires_grad_(False)
                # Then selectively unfreeze the last N stages + classifier norm
                if vmamba_trainable_stages > 0:
                    backbone = self.extractor.backbone
                    for layer in backbone.layers[-vmamba_trainable_stages:]:
                        for p in layer.parameters():
                            p.requires_grad_(True)
                    for p in backbone.classifier.parameters():
                        p.requires_grad_(True)

        self.d_visual = self.extractor.num_features  # 768 for both

        # ── 2. Mamba temporal refiner ────────────────────────────────────────
        # Refines per-frame features across the temporal dimension before
        # LLM reprogramming. Captures temporal dependencies that VMamba's
        # per-frame spatial processing cannot see.
        self.temporal_refiner = TemporalRefiner(
            d_model=self.d_visual,
            num_layers=mamba_layers,
        )

        # ── 3. LLM ───────────────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if use_qlora:
            # ── QLoRA path: 4-bit quantised base + LoRA adapters ─────────────
            # KV caching is disabled during QLoRA fine-tuning because the LoRA
            # adapters modify the K/V projections; cached KVs from previous steps
            # would be stale after every parameter update.
            from transformers import BitsAndBytesConfig
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_model_name,
                quantization_config=bnb_config,
                device_map="auto",
            )
            self.llm = prepare_model_for_kbit_training(self.llm)

            if qlora_target_modules is None:
                # Q+V only: original LoRA paper recommendation.
                # Q controls what visual patterns to attend to;
                # V controls what information gets extracted — both critical for
                # learning to interpret visual tokens in the LLM's embedding space.
                # Add "k_proj" or "o_proj" in configs if stronger adaptation is needed.
                qlora_target_modules = ["q_proj", "v_proj"]

            lora_cfg = LoraConfig(
                r=qlora_r,
                lora_alpha=qlora_alpha,
                lora_dropout=qlora_dropout,
                target_modules=qlora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm = get_peft_model(self.llm, lora_cfg)
            self.llm.print_trainable_parameters()
            # Prompt KV cache is disabled — forward_clip prepends prompt_emb directly
            self._prompt_kv = None
        else:
            # ── Normal path: frozen LLM (default) ────────────────────────────
            self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)

            if freeze_llm:
                for p in self.llm.parameters():
                    p.requires_grad_(False)
                # Selectively unfreeze the last N transformer layers
                if llm_trainable_layers > 0:
                    for layer in self.llm.model.layers[-llm_trainable_layers:]:
                        for p in layer.parameters():
                            p.requires_grad_(True)
                    # Also unfreeze the final norm (feeds directly into classifier)
                    for p in self.llm.model.norm.parameters():
                        p.requires_grad_(True)

        self.d_llm = self.llm.config.hidden_size

        # ── 4. Visual projector: Linear + Reprogram + FFN ───────────────────
        # Direct path
        self.linear_proj = nn.Linear(self.d_visual, self.d_llm)

        # Semantic reprogramming path
        word_emb_weight = self.llm.get_input_embeddings().weight  # (V, d_llm)
        self.vocab_size = word_emb_weight.shape[0]
        self.use_mapping_layer = use_mapping_layer

        # Build domain-biased token index (surgical/medical vocab → token IDs)
        domain_ids = _select_domain_token_ids(self.tokenizer, self.vocab_size, num_tokens)
        # domain_ids: (num_tokens,) LongTensor of surgical-domain token IDs

        if use_mapping_layer:
            # Learnable: full vocab → num_tokens weighted combination.
            # Initialise the mapping so domain tokens receive high initial weight.
            self.mapping_layer = nn.Linear(self.vocab_size, num_tokens)
            with torch.no_grad():
                self.mapping_layer.weight.zero_()
                self.mapping_layer.weight[
                    torch.arange(num_tokens), domain_ids
                ] = 1.0  # start as a hard domain-token selection, soften during training
        else:
            # Frozen: use domain-relevant token embeddings directly as K/V
            self.register_buffer(
                "src_word_embeddings",
                word_emb_weight[domain_ids].detach().float()
            )
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
        # Initialize hint queries from LLM embeddings of phase/tool names so
        # each query starts "looking for" a specific surgical concept rather
        # than a random direction in embedding space.
        embed_fn = self.llm.get_input_embeddings()
        hint_init = _build_hint_init_embeddings(embed_fn, self.tokenizer, n_hints, self.d_llm)
        self.hint_encoder = ClipHintEncoder(
            n_hints=n_hints,
            d_visual=self.d_visual,
            d_llm=self.d_llm,
            n_heads=n_heads,
            dropout=hint_dropout,
            max_seq=512,
            init_embeddings=hint_init,
        )

        # ── 6. Prompt KV cache setup + section header embeddings ─────────────
        self.prompt_text = prompt
        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).input_ids

        embed_fn = self.llm.get_input_embeddings()
        with torch.no_grad():
            prompt_emb = embed_fn(prompt_ids)                          # (1, P, d_llm)

        self.register_buffer("prompt_emb", prompt_emb.detach())
        self.prompt_len = prompt_emb.shape[1]
        self._prompt_kv: tuple = None

        # Pre-embed section headers — placed inline just before each token group
        # so the LLM immediately knows what the following tokens represent.
        for attr, text in [
            ("seg_memory_emb", _SEG_MEMORY),
            ("seg_hints_emb",  _SEG_HINTS),
            ("seg_prev_emb",   _SEG_PREV),
            ("seg_visual_emb", _SEG_VISUAL),
        ]:
            ids = self.tokenizer(text, return_tensors="pt").input_ids
            with torch.no_grad():
                emb = embed_fn(ids)                                    # (1, L, d_llm)
            self.register_buffer(attr, emb.detach())

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

    def to(self, *args, **kwargs):
        """Override to avoid moving quantised LLM when QLoRA is active."""
        if self._use_qlora:
            # 4-bit quantised layers are pinned to their CUDA device by bitsandbytes.
            # Move every other module normally; leave self.llm untouched.
            for name, child in self.named_children():
                if name != "llm":
                    child.to(*args, **kwargs)
            # Keep buffers (prompt_emb, etc.) on the same device as non-LLM modules
            for buf_name, buf in self.named_buffers():
                if not buf_name.startswith("llm."):
                    setattr(self, buf_name.split(".")[-1],
                            buf.to(*args, **kwargs))
            return self
        return super().to(*args, **kwargs)

    # ── Prompt KV cache ──────────────────────────────────────────────────────

    @torch.no_grad()
    def build_prompt_kv(self) -> tuple:
        """
        Compute and cache the prompt's KV pairs from the frozen LLM.
        Stored as legacy tuple format so each forward_clip builds a fresh
        DynamicCache — prevents in-place mutation across clips.

        Returns None when QLoRA is active: prompt is prepended directly
        each forward pass to avoid stale cached KVs after LoRA updates.
        """
        if self._use_qlora:
            # QLoRA: LoRA adapters change K/V projections during training →
            # a cached KV from before an optimizer step would be wrong.
            # forward_clip prepends prompt_emb directly instead.
            self._prompt_kv = None
            return None

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
        ablate_visual: bool = False,
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
        if self.use_mapping_layer:
            src_emb = self.mapping_layer(
                self.word_embeddings.permute(1, 0)).permute(1, 0)      # (num_tokens, d_llm)
        else:
            src_emb = self.src_word_embeddings                         # (num_tokens, d_llm)
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

        # 5. Build LLM input sequence (interleaved text headers + token groups)
        # Each token group is immediately preceded by its section header so the
        # LLM sees "what these tokens are" right at the point they appear —
        # the same pattern used in time-series LLM reprogramming papers.
        llm_dtype = next(self.llm.parameters()).dtype

        def _seg(buf: torch.Tensor) -> torch.Tensor:
            """Expand pre-embedded header (1, L, d) → (B, L, d) and cast dtype."""
            return buf.expand(B, -1, -1).to(llm_dtype)

        parts = []
        mem_tokens = enriched_memory if enriched_memory is not None else memory
        if mem_tokens is not None:
            parts.append(_seg(self.seg_memory_emb))                     # "Global context:"
            parts.append(mem_tokens)                                    # [memory tokens]

        parts.append(_seg(self.seg_hints_emb))                         # "Visual summary:"
        parts.append(hints)                                             # [hint tokens]

        if tool_annots is not None:
            parts.append(self._make_tool_emb(tool_annots))             # "Tool context: ..."

        if prev_visual is not None:
            parts.append(_seg(self.seg_prev_emb))                      # "Previous clip:"
            parts.append(self.local_compressor(prev_visual))           # [prev tokens]

        parts.append(_seg(self.seg_visual_emb))                        # "Visual tokens:"
        vt = torch.zeros_like(visual_tokens) if ablate_visual else visual_tokens
        parts.append(vt)                                                # [visual tokens]

        llm_input = torch.cat([p.to(llm_dtype) for p in parts], dim=1)  # (B, *, d_llm)

        # 6. LLM forward ──────────────────────────────────────────────────────
        if prompt_kv is not None:
            # Frozen-LLM mode: reuse cached prompt KV (fast, no re-encoding prompt).
            # Fresh DynamicCache each clip so stored tensors are not mutated in-place.
            kv_cache = (
                DynamicCache.from_legacy_cache(prompt_kv)
                if isinstance(prompt_kv, tuple)
                else prompt_kv
            )
            lm_out = self.llm(
                inputs_embeds=llm_input,
                past_key_values=kv_cache,
                use_cache=False,
                output_hidden_states=True,
            )
        else:
            # QLoRA / no-cache mode: prepend prompt embeddings directly each clip.
            # LoRA adapters update K/V projections every step, so cached KVs would
            # be stale. Concatenating prompt_emb keeps the full context visible.
            prefix = self.prompt_emb.expand(B, -1, -1).to(llm_dtype)
            full_input = torch.cat([prefix, llm_input], dim=1)  # (B, P+*, d_llm)
            lm_out = self.llm(
                inputs_embeds=full_input,
                use_cache=False,
                output_hidden_states=True,
            )
        # 7. Extract frame positions and predict ──────────────────────────────
        # Fuse middle layer + second-to-last layer hidden states.
        # hidden_states[0] = embedding output, [-1] = final layer.
        # hidden_states[0]=embedding, hidden_states[-1]=last transformer layer (before LM head)
        final_hidden = lm_out.hidden_states[-1][:, -T:, :self.d_ff].float()

        logits = self.output_head(self.output_dropout(final_hidden))  # (B, T, num_phases)

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
            freeze_llm              = m.freeze_llm,
            llm_trainable_layers    = m.get("llm_trainable_layers", 0),
            freeze_vmamba           = m.freeze_vmamba,
            use_mapping_layer       = m.get("use_mapping_layer", True),
            vmamba_trainable_stages = m.get("vmamba_trainable_stages", 0),
            mamba_layers            = m.mamba_layers,
            n_heads          = m.n_heads,
            d_ff             = m.d_ff,
            num_tokens       = m.num_tokens,
            n_hints          = m.n_hints,
            local_context_ratio = m.local_context_ratio,
            prompt           = m.prompt,
            attention_dropout= m.attention_dropout,
            hint_dropout     = m.hint_dropout,
            output_dropout   = m.output_dropout,
            visual_backbone       = m.get("visual_backbone", "vmamba"),
            clip_trainable_layers = m.get("clip_trainable_layers", 0),
            use_qlora             = m.get("use_qlora", False),
            qlora_r               = m.get("qlora_r", 8),
            qlora_alpha           = m.get("qlora_alpha", 16),
            qlora_dropout         = m.get("qlora_dropout", 0.05),
            qlora_target_modules  = m.get("qlora_target_modules", None),
        )
