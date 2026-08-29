"""
model/sar_vlm.py
----------------
SARVLM: full SAR Visual Language Model wrapper.

Data flow:
    sar_input  [B, 1, 512, 512]
        ↓  SAREncoder (frozen)
    sar_features  [B, N_visual=256, D_sar=1024]      (f3 of SwinV2-Base, flattened)
        ↓  SARProjector (trainable MLP)
    sar_tokens  [B, N_visual, llm_hidden_size]
        ↓  concatenate with text embeddings
    inputs_embeds  [B, N_visual + N_text, llm_hidden_size]
        ↓  HybridLlamaForCausalLM (Vicuna base frozen, LoRA trainable)
    logits / loss

Trainable components:
    - SARProjector (MLP weights)
    - Vicuna LoRA matrices (q/k/v/o_proj in every attention layer)

Frozen components:
    - SAREncoder (MaRS SwinV2-Base checkpoint)
    - Vicuna base weights

Label convention (VQA):
    labels = [-100] * N_visual + [-100] * N_question + [answer_token_ids...]
    Loss is computed only on answer tokens.
"""
from typing import Optional

import torch
from torch import nn

from .hybrid_llama import HybridLlamaForCausalLM
from .sar_projector import SARProjector


# ---------------------------------------------------------------------------
# SAR Encoder — placeholder and real-encoder factory
# ---------------------------------------------------------------------------

class SAREncoderPlaceholder(nn.Module):
    """
    Offline placeholder that mimics the output shape of the real MaRS encoder.

    The real encoder is:
        timm.create_model(
            'swinv2_base_window8_256',
            pretrained=False,
            features_only=True,
            in_chans=1,
            img_size=512,
            checkpoint_path='mars_large_sar_encoder_only.pth'
        )

    For 512×512 input the last feature stage (f3) is [B, 1024, 16, 16].
    Spatially flattening → [B, 256, 1024].

    This placeholder returns zeros with that shape so tests run offline
    without the checkpoint.  Replace it with the real encoder (via
    build_sar_encoder()) before any real training.
    """

    def __init__(self, d_sar: int = 1024, n_visual: int = 256):
        super().__init__()
        self.d_sar = d_sar
        self.n_visual = n_visual

    def forward(self, sar_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sar_input: [B, 1, 512, 512] — single-channel SAR image
        Returns:
            [B, n_visual, d_sar] — flat visual token features (zeros as stub)
        """
        B = sar_input.shape[0]
        return torch.zeros(
            B, self.n_visual, self.d_sar,
            dtype=sar_input.dtype,
            device=sar_input.device,
        )


def build_sar_encoder(
    checkpoint_path: str,
    freeze: bool = True,
    d_sar: int = 1024,
) -> nn.Module:
    """
    Load the real MaRS SwinV2-Base SAR encoder from a local checkpoint.

    The model is created via timm (must be installed in the environment).

    Architecture:
        swinv2_base_window8_256 — SwinV2-Base
        features_only=True      — returns multi-scale feature maps
        in_chans=1              — single-channel SAR input
        img_size=512            — training image size

    The encoder wrapper takes care of selecting f3 and flattening so that
    forward() always returns [B, N_visual, d_sar].

    Args:
        checkpoint_path: Path to mars_large_sar_encoder_only.pth.
        freeze: If True, all encoder parameters are frozen (requires_grad=False).
        d_sar: Expected output channel count (1024 for SwinV2-Base).

    Returns:
        nn.Module with forward(sar_input) → [B, N_visual, d_sar].
    """
    try:
        import timm
    except ImportError:
        raise ImportError("timm is required to load the real SAR encoder. pip install timm")

    backbone = timm.create_model(
        "swinv2_base_window8_256",
        pretrained=False,
        features_only=True,
        in_chans=1,
        img_size=512,
    )
    print("check point path is ",checkpoint_path)
    # Load checkpoint — handle various save formats
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    if isinstance(state, dict):
        # Common keys: 'model', 'state_dict', 'encoder'
        for key in ("model", "state_dict", "encoder"):
            if key in state:
                state = state[key]
                break
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    if missing:
        print(f"[build_sar_encoder] Missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"[build_sar_encoder] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

    if freeze:
        backbone.requires_grad_(False)
        backbone.eval()

    return _SAREncoderWrapper(backbone, d_sar=d_sar)


class _SAREncoderWrapper(nn.Module):
    """
    Wraps a timm `features_only` backbone so its forward() always returns
    [B, N_visual, d_sar] by selecting f3 and spatially flattening.
    """

    def __init__(self, backbone: nn.Module, d_sar: int = 1024):
        super().__init__()
        self.backbone = backbone
        self.d_sar = d_sar

    def forward(self, sar_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sar_input: [B, 1, 512, 512]
        Returns:
            [B, H*W, d_sar]  where H=W=16 for 512px SwinV2-Base → [B, 256, 1024]
        """
        feature_maps = self.backbone(sar_input)  # list [f0, f1, f2, f3]
        f3 = feature_maps[-1]                    # timm swinv2 returns [B, H, W, C]
        B, H, W, C = f3.shape
        # Flatten spatial dimensions: [B, H, W, C] → [B, H*W, C]
        return f3.reshape(B, H * W, C)


# ---------------------------------------------------------------------------
# SARVLM
# ---------------------------------------------------------------------------

class SARVLM(nn.Module):
    """
    Full SAR-VQA Vision-Language Model.

    Components:
        sar_encoder   — frozen MaRS SwinV2-Base (or placeholder for tests)
        projector     — trainable two-layer MLP
        hybrid_vicuna — Vicuna 1.5 7B with bidirectional SAR attention +
                        LoRA adapters on q/k/v/o_proj

    Usage (training):
        vlm = SARVLM.from_vicuna(
            vicuna_path="lmsys/vicuna-7b-v1.5",
            sar_encoder=build_sar_encoder("mars_large_sar_encoder_only.pth"),
        )
        loss = vlm(sar_input, input_ids, attention_mask, labels).loss
        loss.backward()

    Usage (generation):
        token_ids = vlm.generate(sar_input, input_ids, attention_mask,
                                  max_new_tokens=50)
    """

    # Default LoRA hyper-parameters (configurable at construction time)
    DEFAULT_LORA_R = 16
    DEFAULT_LORA_ALPHA = 32
    DEFAULT_LORA_DROPOUT = 0.05
    DEFAULT_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

    def __init__(
        self,
        sar_encoder: nn.Module,
        projector: SARProjector,
        hybrid_vicuna: HybridLlamaForCausalLM,
        lora_r: int = DEFAULT_LORA_R,
        lora_alpha: int = DEFAULT_LORA_ALPHA,
        lora_dropout: float = DEFAULT_LORA_DROPOUT,
        lora_target_modules: list = None,
        apply_lora: bool = True,
    ):
        super().__init__()

        self.sar_encoder = sar_encoder
        self.projector = projector

        # Keep a direct reference to the inner HybridLlamaModel BEFORE PEFT
        # wraps the attention projections.  PEFT modifies the linear layers
        # inside the model but does not replace the HybridLlamaModel object
        # itself, so this reference stays valid.
        self._llama_model_ref = hybrid_vicuna.model  # HybridLlamaModel

        if apply_lora:
            hybrid_vicuna = self._apply_lora(
                hybrid_vicuna,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules or self.DEFAULT_LORA_TARGETS,
            )

        self.hybrid_vicuna = hybrid_vicuna

        # Ensure encoder is frozen and projector is trainable.
        self._freeze_encoder()
        self.projector.requires_grad_(True)

        self._print_trainable_params()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_vicuna(
        cls,
        vicuna_path: str = "lmsys/vicuna-7b-v1.5",
        sar_encoder: Optional[nn.Module] = None,
        d_sar: int = 1024,
        n_visual: int = 256,
        projector_hidden_dim: Optional[int] = None,
        torch_dtype=torch.float16,
        **kwargs,
    ) -> "SARVLM":
        """
        Convenience constructor that loads Vicuna weights from HuggingFace
        (or a local path) and wires everything together.

        Args:
            vicuna_path: HuggingFace model ID or local directory.
                e.g. "lmsys/vicuna-7b-v1.5"
            sar_encoder: Pre-built SAR encoder module.  If None, uses
                SAREncoderPlaceholder (for testing only).
            d_sar: SAR encoder output dimension (1024 for SwinV2-Base).
            n_visual: Number of visual tokens (256 for 512px/f3).
            projector_hidden_dim: Inner MLP dim for SARProjector.
            torch_dtype: dtype for Vicuna weights.
            **kwargs: Extra arguments passed to SARVLM.__init__.
        """
        from transformers import AutoModelForCausalLM, AutoConfig

        print(f"[SARVLM] Loading Vicuna config from {vicuna_path} ...")
        config = AutoConfig.from_pretrained(vicuna_path)
        config._attn_implementation = "eager"

        print(f"[SARVLM] Loading Vicuna weights from {vicuna_path} ...")
        
        # Load weights on CPU first to avoid GPU OOM during loading
        from transformers import AutoModelForCausalLM
        vicuna_base = AutoModelForCausalLM.from_pretrained(
            vicuna_path,
            torch_dtype=torch_dtype,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        
        # Build HybridLlamaForCausalLM
        hybrid_vicuna = HybridLlamaForCausalLM(config)
        
        # Copy weights from CPU to hybrid_vicuna
        result = hybrid_vicuna.load_state_dict(vicuna_base.state_dict(), strict=True)
        if result.missing_keys or result.unexpected_keys:
            print(f"[SARVLM] WARNING — missing: {result.missing_keys}, "
                  f"unexpected: {result.unexpected_keys}")
        else:
            print("[SARVLM] Vicuna weights loaded successfully (0 missing, 0 unexpected).")

        # Immediately delete vicuna_base and force garbage collection
        del vicuna_base
        import gc
        gc.collect()

        # SAR encoder
        if sar_encoder is None:
            print("[SARVLM] No SAR encoder provided — using SAREncoderPlaceholder.")
            sar_encoder = SAREncoderPlaceholder(d_sar=d_sar, n_visual=n_visual)

        # Projector
        llm_hidden_size = config.hidden_size
        projector = SARProjector(
            d_sar=d_sar,
            llm_hidden_size=llm_hidden_size,
            hidden_dim=projector_hidden_dim,
        )
        # Cast projector to match LLM dtype
        projector = projector.to(dtype=torch_dtype)

        return cls(sar_encoder=sar_encoder, projector=projector,
                   hybrid_vicuna=hybrid_vicuna, **kwargs)

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_lora(
        model: HybridLlamaForCausalLM,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        target_modules: list,
    ) -> nn.Module:
        """Apply PEFT LoRA to the hybrid Vicuna model and print stats."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError("peft is required for LoRA. pip install peft")

        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        peft_model = get_peft_model(model, lora_config)
        return peft_model

    # ------------------------------------------------------------------
    # Freezing
    # ------------------------------------------------------------------

    def _freeze_encoder(self):
        """Freeze all SAR encoder parameters."""
        for param in self.sar_encoder.parameters():
            param.requires_grad_(False)

    def _print_trainable_params(self):
        """Print a summary of trainable vs frozen parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"\n[SARVLM] Parameter summary:\n"
            f"  Total parameters    : {total:,}\n"
            f"  Trainable           : {trainable:,}  ({100*trainable/max(total,1):.2f}%)\n"
            f"  Frozen              : {total - trainable:,}\n"
        )

    # ------------------------------------------------------------------
    # Forward pass (training / single inference)
    # ------------------------------------------------------------------

    def _encode_sar(self, sar_input: torch.Tensor) -> torch.Tensor:
        """
        Run the SAR encoder (frozen) and return [B, N_visual, d_sar].

        For the real MaRS encoder (wrapped by _SAREncoderWrapper) this just
        calls forward.  For the placeholder it returns zeros.

        We use torch.no_grad() inside here because the encoder is frozen —
        no gradient should flow into it, and skipping autograd for the encoder
        avoids storing unnecessary intermediate activations.
        """
        with torch.no_grad():
            features = self.sar_encoder(sar_input)  # [B, N_visual, d_sar]
        # Detach so the projector's gradient graph starts at sar_features.
        # Cast to match projector dtype to avoid dtype mismatch
        return features.detach().to(self.projector.proj[0].weight.dtype)

    def forward(
        self,
        sar_input: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        """
        Training / single-pass inference forward.

        Args:
            sar_input: [B, 1, 512, 512] — raw SAR image tensor
            input_ids: [B, N_text] — tokenised question [+ answer] ids
            attention_mask: [B, N_text] — 1 for real tokens, 0 for padding
            labels: [B, N_text] — -100 for question/padding, token ids for answers

        Returns:
            CausalLMOutputWithPast (contains .loss if labels provided)

        Label convention:
            The caller provides labels aligned with `input_ids` (N_text tokens).
            Internally this method prepends N_visual copies of -100 so that
            loss is never computed on SAR visual positions.

            Full label layout after prepending:
                [-100]*N_visual | [-100]*N_question | answer_token_ids
        """
        B = sar_input.shape[0]

        # 1. SAR encoder (frozen) → projector (trainable)
        sar_features = self._encode_sar(sar_input)         # [B, N_v, d_sar]
        sar_tokens = self.projector(sar_features)           # [B, N_v, H]
        N_v = sar_tokens.shape[1]

        # 2. Text embeddings from Vicuna's embedding table
        embed_fn = (
            self.hybrid_vicuna.get_input_embeddings()
            if not hasattr(self.hybrid_vicuna, "base_model")
            else self.hybrid_vicuna.base_model.model.get_input_embeddings()
        )
        text_embeds = embed_fn(input_ids)                  # [B, N_t, H]

        # 3. Concatenate: [SAR tokens | text tokens]
        inputs_embeds = torch.cat([sar_tokens, text_embeds], dim=1)  # [B, N_v+N_t, H]

        # 4. Extend attention mask to cover visual tokens
        if attention_mask is not None:
            visual_attn = torch.ones(
                B, N_v, dtype=attention_mask.dtype, device=attention_mask.device
            )
            full_attention_mask = torch.cat([visual_attn, attention_mask], dim=1)
        else:
            full_attention_mask = None

        # 5. Extend labels: prepend -100 for every visual position
        if labels is not None:
            visual_labels = torch.full(
                (B, N_v), fill_value=-100, dtype=labels.dtype, device=labels.device
            )
            full_labels = torch.cat([visual_labels, labels], dim=1)
        else:
            full_labels = None

        # 6. Forward through hybrid Vicuna
        self._llama_model_ref._num_visual_tokens = N_v
        return self.hybrid_vicuna(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
            num_visual_tokens=N_v,
        )

    # ------------------------------------------------------------------
    # Generation (KV-cache compatible)
    # ------------------------------------------------------------------

    def generate(
        self,
        sar_input: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **generate_kwargs,
    ):
        """
        Autoregressive generation with SAR visual context.

        The SAR tokens are prepended to the prompt embeddings for the
        initial prefill.  Subsequent decode steps use the KV cache —
        the visual tokens are already cached and the new text queries
        attend to them through normal causal attention (kv_idx <= q_idx).

        The SAR visual mask function stays active throughout generation
        but has no effect during decode steps because new text-query
        positions satisfy q_idx >= N_visual, making the visual_mask
        function return False for all kv positions.

        Args:
            sar_input: [B, 1, 512, 512]
            input_ids: [B, N_prompt] — prompt token ids (question)
            attention_mask: [B, N_prompt] — 1 for real, 0 for padding
            **generate_kwargs: forwarded to model.generate()
                               (e.g. max_new_tokens, do_sample, temperature)

        Returns:
            Generated token id tensor [B, N_prompt + N_generated].
        """
        B = sar_input.shape[0]

        # Compute visual tokens (frozen encoder + trainable projector)
        with torch.no_grad():
            sar_features = self._encode_sar(sar_input)
            sar_tokens = self.projector(sar_features)

        N_v = sar_tokens.shape[1]

        embed_fn = (
            self.hybrid_vicuna.get_input_embeddings()
            if not hasattr(self.hybrid_vicuna, "base_model")
            else self.hybrid_vicuna.base_model.model.get_input_embeddings()
        )
        with torch.no_grad():
            text_embeds = embed_fn(input_ids)

        inputs_embeds = torch.cat([sar_tokens, text_embeds], dim=1)

        if attention_mask is not None:
            visual_attn = torch.ones(
                B, N_v, dtype=attention_mask.dtype, device=attention_mask.device
            )
            full_mask = torch.cat([visual_attn, attention_mask], dim=1)
        else:
            full_mask = None

        # Tell the inner LlamaModel how many visual positions to treat
        # bidirectionally.  This persists across all generate() steps.
        self._llama_model_ref._num_visual_tokens = N_v

        try:
            output_ids = self.hybrid_vicuna.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=full_mask,
                **generate_kwargs,
            )
        finally:
            # Reset after generation so the model is clean for next call.
            self._llama_model_ref._num_visual_tokens = 0

        return output_ids
