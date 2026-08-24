import torch

from transformers import LlamaConfig
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import (
    create_causal_mask,
)


NUM_VISUAL_TOKENS = 3

config = LlamaConfig(
    hidden_size=4096,
    intermediate_size=11008,
    num_hidden_layers=2,
    num_attention_heads=32,
    num_key_value_heads=32,
)

config._attn_implementation = "eager"


# ------------------------------------------------------------
# 1. Create a cache containing 6 previous tokens
#
# positions:
#
# 0  1  2 | 3  4  5
# V1 V2 V3 | T1 T2 T3
# ------------------------------------------------------------

cache = DynamicCache(config=config)

num_cached_tokens = 6

key = torch.randn(
    1,
    config.num_key_value_heads,
    num_cached_tokens,
    config.head_dim,
)

value = torch.randn(
    1,
    config.num_key_value_heads,
    num_cached_tokens,
    config.head_dim,
)

cache.update(
    key,
    value,
    layer_idx=0,
)


print("Cache sequence length:")
print(cache.get_seq_length())


# ------------------------------------------------------------
# 2. Current input = ONE new token
# ------------------------------------------------------------

inputs_embeds = torch.randn(
    1,
    1,
    config.hidden_size,
)


# New token is at absolute position 6
cache_position = torch.tensor(
    [6],
    dtype=torch.long,
)

position_ids = cache_position.unsqueeze(0)


# ------------------------------------------------------------
# 3. Your visual mask
# ------------------------------------------------------------

def make_sar_visual_mask(num_visual_tokens):

    def visual_mask(
        batch_idx,
        head_idx,
        q_idx,
        kv_idx,
    ):
        return (
            (q_idx < num_visual_tokens)
            &
            (kv_idx < num_visual_tokens)
        )

    return visual_mask


visual_mask = make_sar_visual_mask(
    NUM_VISUAL_TOKENS
)


# ------------------------------------------------------------
# 4. Create the ACTUAL hybrid mask
# ------------------------------------------------------------

mask = create_causal_mask(
    config=config,
    input_embeds=inputs_embeds,
    attention_mask=None,
    past_key_values=cache,
    position_ids=position_ids,
    cache_position=cache_position,
    or_mask_function=visual_mask,
)


print("\nMask shape:")
print(mask.shape)

print("\nMask:")
print(mask[0, 0])