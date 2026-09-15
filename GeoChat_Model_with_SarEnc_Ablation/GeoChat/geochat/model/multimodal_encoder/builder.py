import os
from .clip_encoder import CLIPVisionTower
from .sar_encoder import SARVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    
    # Check if SAR encoder should be used
    if 'sar' in vision_tower.lower() or vision_tower.endswith('.pth'):
        return SARVisionTower(vision_tower, **kwargs)
    
    # Default to CLIP for openai/laion or existing paths
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
