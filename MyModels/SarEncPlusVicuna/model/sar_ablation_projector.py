import torch
import torch.nn as nn


class SARAblationProjector(nn.Module):
    """
    MLP projector for SAR encoder features to LLM embedding space.
    
    Maps SAR encoder output [B, N_patches, d_sar] to [B, N_patches, llm_hidden_size]
    where d_sar=1024 (SwinV2-Base) and llm_hidden_size=4096 (Vicuna-7B).
    
    This is a 2-layer MLP with GELU activation, trained while SAR encoder and LLM remain frozen.
    """
    def __init__(self, d_sar=1024, llm_hidden_size=4096, hidden_dim=4096):
        super().__init__()
        self.d_sar = d_sar
        self.llm_hidden_size = llm_hidden_size
        
        self.proj = nn.Sequential(
            nn.Linear(d_sar, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_hidden_size)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, N_patches, d_sar] - SAR encoder features
        
        Returns:
            [B, N_patches, llm_hidden_size] - Projected features for LLM
        """
        return self.proj(x)
    
    @property
    def config(self):
        return {"mm_projector_type": 'sar_mlp'}

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load weights from a checkpoint path (.pth file or directory containing projector.pth).
        Handles nested dicts ('projector_state_dict', 'projector', 'model', 'state_dict')
        and strips common prefixes ('projector.', 'mm_projector.', 'model.mm_projector.').
        """
        import os
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Projector checkpoint path does not exist: {checkpoint_path}")

        if os.path.isdir(checkpoint_path):
            candidates = ["projector.pth", "sar_projector.pth"]
            file_path = None
            for cand in candidates:
                cand_path = os.path.join(checkpoint_path, cand)
                if os.path.exists(cand_path):
                    file_path = cand_path
                    break
            if file_path is None:
                pths = [os.path.join(checkpoint_path, f) for f in os.listdir(checkpoint_path) if f.endswith(".pth")]
                if pths:
                    file_path = pths[0]
                else:
                    raise FileNotFoundError(f"No .pth file found in directory: {checkpoint_path}")
        else:
            file_path = checkpoint_path

        try:
            state = torch.load(file_path, map_location="cpu", weights_only=True)
        except Exception:
            state = torch.load(file_path, map_location="cpu", weights_only=False)

        # Unwrap if nested
        for key in ("projector_state_dict", "projector", "model", "state_dict"):
            if isinstance(state, dict) and key in state and isinstance(state[key], dict):
                state = state[key]
                break

        # Strip prefixes
        cleaned_state = {}
        for k, v in state.items():
            new_k = k
            for prefix in ("model.mm_projector.", "mm_projector.", "projector."):
                if new_k.startswith(prefix):
                    new_k = new_k[len(prefix):]
                    break
            cleaned_state[new_k] = v

        # Match dtype
        target_dtype = next(self.parameters()).dtype
        cleaned_state = {
            k: (v.to(dtype=target_dtype) if torch.is_tensor(v) and v.is_floating_point() else v)
            for k, v in cleaned_state.items()
        }

        missing, unexpected = self.load_state_dict(cleaned_state, strict=False)
        return file_path, missing, unexpected
