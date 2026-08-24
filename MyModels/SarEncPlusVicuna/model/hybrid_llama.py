"""
model/hybrid_llama.py
---------------------
HybridLlamaModel and HybridLlamaForCausalLM for SAR-VQA.

Design principle:
    Modify the MINIMUM possible relative to stock LlamaModel/LlamaForCausalLM.
    The only behavioural change is injecting `or_mask_function` into
    `create_causal_mask()` so that the leading N_visual positions attend
    bidirectionally to each other while text tokens remain causal.

No site-packages are modified.
Weight keys are 100 % identical to lmsys/vicuna-7b-v1.5 — load_state_dict
with strict=True produces missing_keys=[], unexpected_keys=[].
"""
from typing import Optional, Union

import torch
from torch import nn

from transformers import LlamaConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
)


# ---------------------------------------------------------------------------
# SAR bidirectional mask factory
# ---------------------------------------------------------------------------

def make_sar_visual_mask(num_visual_tokens: int):
    """
    Returns a mask function compatible with create_causal_mask(or_mask_function=...).

    The function returns True (attention allowed) when both the query token
    and the key/value token are within the SAR visual prefix.

    OR-ing with the causal mask produces the desired pattern:

        SAR → SAR  : 1  (bidirectional, added by this function)
        SAR → text : 0  (q_idx < N_v but kv_idx >= N_v → function returns False,
                         causal mask also False since SAR precedes text → 0)
        text→ SAR  : 1  (q_idx >= N_v, kv_idx < N_v → function False, but
                         causal mask True because kv_idx <= q_idx → 1)
        text→ text : causal lower-triangular

    Args:
        num_visual_tokens: number of leading SAR visual positions (N_visual).

    Returns:
        Callable with signature (batch_idx, head_idx, q_idx, kv_idx) -> bool.
    """
    def visual_mask(
        batch_idx: int,
        head_idx: int,
        q_idx: int,
        kv_idx: int,
    ) -> bool:
        # Both positions must be inside the visual prefix.
        return (q_idx < num_visual_tokens) & (kv_idx < num_visual_tokens)

    return visual_mask


# ---------------------------------------------------------------------------
# HybridLlamaModel
# ---------------------------------------------------------------------------

class HybridLlamaModel(LlamaModel):
    """
    LlamaModel with SAR visual bidirectional attention.

    Identical to LlamaModel in every respect (no new parameters, same
    state_dict keys) except that `create_causal_mask` receives an
    `or_mask_function` that allows the first `_num_visual_tokens` positions
    to attend bidirectionally.

    `_num_visual_tokens` is an *instance variable* (not a model parameter).
    It is set by `HybridLlamaForCausalLM.forward()` (or `SARVLM`) before
    every forward call and defaults to 0 (standard causal mask).

    During KV-cache decode steps the token index of the new query is always
    >= N_visual, so the visual_mask function evaluates to False and has
    zero effect — no special handling is needed for generation.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        # Communication slot — not a learnable parameter.
        self._num_visual_tokens: int = 0

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        """
        Verbatim copy of LlamaModel.forward (transformers 4.57.6) with ONE
        addition: `or_mask_function=make_sar_visual_mask(self._num_visual_tokens)`
        is passed to create_causal_mask when _num_visual_tokens > 0.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # --- Only change from base LlamaModel.forward ---
        # Build the SAR visual mask when we have visual tokens in this forward.
        # When _num_visual_tokens == 0 (pure-text or decode step where visual
        # tokens are already in the cache), or_mask_fn is None and we get the
        # standard causal mask.
        or_mask_fn = (
            make_sar_visual_mask(self._num_visual_tokens)
            if self._num_visual_tokens > 0
            else None
        )

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
            or_mask_function=or_mask_fn,          # ← SAR visual mask injection
        )
        # -------------------------------------------------

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


# ---------------------------------------------------------------------------
# HybridLlamaForCausalLM
# ---------------------------------------------------------------------------

class HybridLlamaForCausalLM(LlamaForCausalLM):
    """
    Drop-in replacement for LlamaForCausalLM that uses HybridLlamaModel.

    Weight compatibility:
        All parameter names match lmsys/vicuna-7b-v1.5 exactly because
        HybridLlamaModel subclasses LlamaModel without adding new parameters.

        >>> result = hybrid.load_state_dict(vicuna.state_dict(), strict=True)
        >>> assert result.missing_keys == [] and result.unexpected_keys == []

    Extra forward argument:
        num_visual_tokens (int, default 0) — the number of leading positions
        that are SAR visual tokens.  Set to N_visual during prefill/training,
        leave 0 during pure-text decoding.

    Generation compatibility:
        For model.generate() call SARVLM.generate(), which sets
        self.model._num_visual_tokens = N_v before calling generate().
        The visual mask function is safe to keep active during decode steps
        because q_idx >= N_v for any new text token, making the function
        return False and having no effect.
    """

    def __init__(self, config: LlamaConfig):
        # Call the grandparent (LlamaPreTrainedModel = PreTrainedModel) directly.
        # This initialises all HF machinery without allocating a vanilla LlamaModel
        # that would immediately be replaced — avoids double memory allocation.
        LlamaPreTrainedModel.__init__(self, config)

        # Build our hybrid model and standard LM head.
        # Parameter names are identical to LlamaForCausalLM.
        self.model = HybridLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Single post_init call: weight initialisation + weight tying.
        self.post_init()

    # The four embedding/head accessors are inherited from LlamaForCausalLM
    # (they refer to self.model.embed_tokens and self.lm_head by name, which
    # exist on this class with identical paths → no override needed).

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        num_visual_tokens: int = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Same as LlamaForCausalLM.forward plus `num_visual_tokens`.

        `num_visual_tokens` is written to self.model._num_visual_tokens before
        the inner model forward so that HybridLlamaModel picks it up.
        """
        # Inject visual token count for this forward pass.
        self.model._num_visual_tokens = num_visual_tokens

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
