import gc
import math
from typing import Optional, List
import torch
from torch import nn
# ---------------------------------------------------------------------------
# Gated LoRA
# ---------------------------------------------------------------------------

class GatedLoRA(nn.Module):
    """
    Gated LoRA with GSD-aware gating mechanism.

    The gating function uses learnable thresholds T_k and a learnable scale beta:
        s = log10(gsd)
        h_k(s) = sigmoid(beta * (T_k - s))

    The final LoRA output is:
        output = Wx + B * diag(h(s)) * Ax

    where:
        - W: original linear layer weights (frozen)
        - A: LoRA down-projection [in_features, rank]
        - B: LoRA up-projection [rank, out_features]
        - h(s): gating vector of size [rank]
        - diag(h(s)): diagonal matrix formed from gating values

    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: LoRA rank (number of gating functions)
        alpha: LoRA scaling factor
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank)) # make A matrix samples from normal distribution
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features)) # let be matrix be zero as it is

        # Gating parameters
        # T0 to T_{rank-1}: learnable thresholds for each rank
        self.gate_thresholds = nn.Parameter(torch.zeros(rank))
        # beta: learnable scale for sigmoid
        self.gate_beta = nn.Parameter(torch.tensor([2.0]))

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Initialize thresholds based on rank ranges
        # 0-10: 1, 10-32: 2, 32-64: 4
        thresholds = torch.zeros(rank)
        # if rank<64:
        #     raise ValueError("Rank must be at least 64")
        for i in range(rank):
            if i <= math.floor(rank*(0.15)): #0.15
                thresholds[i] = 1.0
            elif i <= math.floor(rank*(0.34)): #0.34
                thresholds[i] = 2.0
            else:  # i <= 63
                thresholds[i] = 4.0
        self.gate_thresholds.data = thresholds

    
    def compute_gating(self, gsd: torch.Tensor) -> torch.Tensor:
        # Support Python scalars and tensors.
        gsd = torch.as_tensor(
            gsd,
            device=self.gate_thresholds.device,
            dtype=self.gate_thresholds.dtype,
        )

        gsd = gsd.clamp(min=1e-6)
        s = torch.log10(gsd)

        # Scalar GSD -> [rank]; batch GSD -> [B, rank]
        delta = self.gate_thresholds - s.unsqueeze(-1)
        return torch.sigmoid(self.gate_beta * delta)


    def forward(self, x: torch.Tensor, gsd: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with optional GSD-based gating.

        Args:
            x: [B, seq_len, in_features] - input tensor
            gsd: [B] or scalar - ground sample distance in meters.
                  If None, gating is disabled (all gates = 1.0)

        Returns:
            output: [B, seq_len, out_features]
        """
        # Compute LoRA projection: Ax
        # x: [B, seq_len, in_features]
        # lora_A: [in_features, rank]
        # result: [B, seq_len, rank]
        lora_out = x @ self.lora_A  # [B, seq_len, rank]
        lora_out = self.dropout(lora_out)

        # Apply gating if GSD is provided
        if gsd is not None:
            h = self.compute_gating(gsd)  # [B, rank]
            # Apply gating: diag(h(s)) * Ax
            # h: [B, rank]
            if h.ndim == 1:
                # One shared GSD for the entire batch
                h = h.unsqueeze(0)
            # lora_out: [B, seq_len, rank]
            lora_out = lora_out * h.unsqueeze(1)  # [B, seq_len, rank]

        # B: [rank, out_features]
        # result: [B, seq_len, out_features]
        lora_out = lora_out @ self.lora_B

        # Apply scaling
        lora_out = lora_out * self.scaling

        return lora_out


class GatedLoRALinear(nn.Module):
    """
    Wrapper that replaces a linear layer with a Gated LoRA version.

    The original weight W is frozen, and LoRA augmentation is added:
        output = Wx + B * diag(h(s)) * Ax
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 64,
        alpha: int = 32,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        # Freeze original weights
        original_layer.weight.requires_grad = False
        if original_layer.bias is not None:
            original_layer.bias.requires_grad = False

        # Create Gated LoRA
        self.gated_lora = GatedLoRA(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        self.abs_gsd = None
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with GSD-aware LoRA.

        Args:
            x: [B, seq_len, in_features]
            gsd: [B] or scalar - ground sample distance

        Returns:
            output: [B, seq_len, out_features]
        """
        # Original frozen layer
        original_out = self.original_layer(x)

        # Gated LoRA augmentation
        lora_out = self.gated_lora(x, self.abs_gsd)
        
        return original_out + lora_out


def apply_gated_lora(
    model: nn.Module,
    target_modules: List[str],
    rank: int = 64,
    alpha: int = 32,
    dropout: float = 0.05,
) -> List[GatedLoRALinear]:
    """
    Apply Gated LoRA to specified target modules in a model.

    Args:
        model: The model to modify
        target_modules: List of module names to apply LoRA to (e.g., ["q_proj", "k_proj"])
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: Dropout probability

    Returns:
        List of GatedLoRALinear modules that were created
    """
    print("rank is " ,rank)
    gated_lora_modules = []
    absolute_gsd = getattr(model, "abs_gsd", None)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this module is one of the target modules
            module_name = name.split('.')[-1]
            if module_name in target_modules:
                # Replace with GatedLoRALinear
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]

                parent = model
                if parent_name:
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)

                # Create GatedLoRALinear wrapper
                gated_lora = GatedLoRALinear(
                    original_layer=module,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    
                )

                # Replace the module
                setattr(parent, child_name, gated_lora)
                gated_lora_modules.append(gated_lora)

                print(f"[apply_gated_lora] Applied Gated LoRA to {name}")

    return gated_lora_modules