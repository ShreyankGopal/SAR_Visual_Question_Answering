"""
hybrid_llama_model.py (project root)
-------------------------------------
Backward-compatible shim — the canonical implementation lives in model/hybrid_llama.py.

This file is kept so that existing scripts (e.g. test_mask.py at the project
root) that do `from hybrid_llama_model import make_sar_visual_mask` continue
to work unchanged.
"""
from model.hybrid_llama import (         # noqa: F401 — re-export
    make_sar_visual_mask,
    HybridLlamaModel,
    HybridLlamaForCausalLM,
)