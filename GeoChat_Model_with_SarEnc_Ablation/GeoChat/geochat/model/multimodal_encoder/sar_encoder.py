"""
sar_encoder.py
--------------
SAR encoder wrapper for GeoChat ablation study.
Replaces CLIP encoder with SAR encoder while preserving GeoChat's interface.
"""
import torch
import torch.nn as nn
import tifffile
from typing import Optional


class SARVisionTower(nn.Module):
    """
    SAR encoder wrapper that matches CLIPVisionTower interface.
    
    This wrapper loads a SwinV2-Base SAR encoder (MaRS) and processes
    single-channel SAR TIFF images to produce visual features.
    
    Architecture:
        SwinV2-Base (swinv2_base_window8_256)
        - Input: [B, 1, 512, 512] single-channel SAR
        - Output f3 stage: [B, 16, 16, 1024]
        - Flattened: [B, 256, 1024]
    
    The MLP projector operates on each patch independently, so we keep
    all 256 patches from the SAR encoder (no reshaping needed).
    """
    
    def __init__(self, encoder_checkpoint: str, delay_load=False):
        super().__init__()
        
        self.is_loaded = False
        self.encoder_checkpoint = encoder_checkpoint
        self.d_sar = 1024  # SwinV2-Base feature dimension
        self.n_visual = 256  # SAR encoder output: 16x16 = 256 patches
        
        if not delay_load:
            self.load_model()
    
    def load_model(self):
        """Load SAR encoder from checkpoint."""
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required to load SAR encoder. pip install timm")
        
        print(f"[SARVisionTower] Loading SAR encoder from {self.encoder_checkpoint}...")
        
        # Create SwinV2-Base backbone
        self.backbone = timm.create_model(
            "swinv2_base_window8_256",
            pretrained=False,
            features_only=True,
            in_chans=1,
            img_size=512,
        )
        
        # Load checkpoint
        state = torch.load(self.encoder_checkpoint, map_location="cpu", weights_only=False)
        
        if isinstance(state, dict):
            # Handle various checkpoint formats
            for key in ("model", "state_dict", "encoder"):
                if key in state:
                    state = state[key]
                    break
        
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        if missing:
            print(f"[SARVisionTower] Missing keys ({len(missing)}): {missing[:5]} ...")
        if unexpected:
            print(f"[SARVisionTower] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")
        
        # Freeze encoder
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        
        self.is_loaded = True
        print("[SARVisionTower] SAR encoder loaded successfully.")
    
    @torch.no_grad()
    def forward(self, images):
        """
        Process SAR images and return visual features.
        
        Args:
            images: Can be a list of file paths (str) or tensor [B, 1, 512, 512]
        
        Returns:
            image_features: [B, n_visual, d_sar] where n_visual=49, d_sar=1024
        """
        if type(images) is list:
            image_features = []
            for img in images:
                if isinstance(img, str):
                    # Load TIFF from file path
                    img_np = tifffile.imread(img)
                    img_tensor = torch.from_numpy(img_np).float()
                    if img_tensor.ndim == 2:
                        img_tensor = img_tensor.unsqueeze(0)
                    img_tensor = img_tensor.unsqueeze(0)  # Add batch dim
                else:
                    img_tensor = img
                
                img_tensor = img_tensor.to(device=self.device, dtype=self.dtype)
                feature = self._encode_sar(img_tensor)
                image_features.append(feature)
        else:
            # Assume tensor input [B, 1, 512, 512]
            images = images.to(device=self.device, dtype=self.dtype)
            image_features = self._encode_sar(images)
        
        return image_features
    
    def _encode_sar(self, sar_input: torch.Tensor) -> torch.Tensor:
        """
        Encode SAR input to visual features.
        
        Args:
            sar_input: [B, 1, 512, 512]
        
        Returns:
            features: [B, 256, 1024] - full SAR encoder output
        """
        # Get feature maps from backbone
        feature_maps = self.backbone(sar_input)  # list [f0, f1, f2, f3]
        f3 = feature_maps[-1]  # [B, H, W, C] = [B, 16, 16, 1024]
        
        B, H, W, C = f3.shape
        
        # Flatten spatial: [B, 16, 16, 1024] -> [B, 256, 1024]
        features = f3.reshape(B, H * W, C)
        
        return features
    
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)
    
    @property
    def dtype(self):
        return torch.float16  # Match GeoChat's FP16
    
    @property
    def device(self):
        return next(self.backbone.parameters()).device
    
    @property
    def config(self):
        # Return a simple config object for compatibility
        class Config:
            hidden_size = 1024
            image_size = 512
            patch_size = 32  # 512/16 = 32 for f3
        return Config()
    
    @property
    def hidden_size(self):
        return self.d_sar
    
    @property
    def num_patches(self):
        return self.n_visual
