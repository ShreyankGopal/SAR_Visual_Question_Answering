"""
extract_adjacency_facts.py — New module for Phase 3b: Spatial Adjacency Analysis.

Implements a deterministic image-processing operation to find classes that spatially
border a target class, using 8-connectivity binary dilation.
"""

import numpy as np
from scipy import ndimage

def make_binary_mask(mask: np.ndarray, target_class: int) -> np.ndarray:
    """Create a boolean mask where mask == target_class."""
    return mask == target_class

def dilate_mask(binary_mask: np.ndarray) -> np.ndarray:
    """
    Perform one-pixel binary dilation using 8-connectivity.
    This works on arbitrary shapes and naturally respects image boundaries.
    """
    return ndimage.binary_dilation(binary_mask, structure=np.ones((3, 3)))

def get_boundary_ring(binary_mask: np.ndarray, dilated_mask: np.ndarray) -> np.ndarray:
    """
    Get only the pixels newly added by dilation (the immediate border).
    Returns a boolean mask of the boundary ring.
    """
    return dilated_mask & ~binary_mask

def compute_adjacency_for_region(mask: np.ndarray, class_map: dict, target_class_id: int) -> dict:
    """
    Given a target class ID, find all classes touching it and their proportions.
    
    Args:
        mask: uint8 array (H, W) of class labels
        class_map: dict mapping class ID to class name
        target_class_id: The ID of the class to find neighbors for
        
    Returns:
        dict: { "touching_classes": [class_names...], "class_proportions": {class_name: proportion...} }
    """
    if target_class_id not in class_map:
        return {"touching_classes": [], "class_proportions": {}}
        
    # 1. Create binary mask for the target class
    target_mask = make_binary_mask(mask, target_class_id)
    
    # Check if target class is actually present
    if not np.any(target_mask):
        return {"touching_classes": [], "class_proportions": {}}
        
    # 2. Perform 8-connectivity dilation
    dilated_mask = dilate_mask(target_mask)
    
    # 3. Get boundary ring (newly added pixels)
    boundary_ring = get_boundary_ring(target_mask, dilated_mask)
    
    # If boundary ring is empty (e.g. image is entirely one class)
    if not np.any(boundary_ring):
        return {"touching_classes": [], "class_proportions": {}}
        
    # 4. Use boundary ring to index into original mask
    touching_pixels = mask[boundary_ring]
    
    # 5. Extract unique classes and counts
    classes, counts = np.unique(touching_pixels, return_counts=True)
    
    # Filter out the target class itself (if it somehow appeared)
    valid_idx = classes != target_class_id
    classes = classes[valid_idx]
    counts = counts[valid_idx]
    
    if len(classes) == 0:
        return {"touching_classes": [], "class_proportions": {}}
        
    # Map back to class names and compute proportions
    total_touching = np.sum(counts)
    
    touching_class_names = []
    class_proportions = {}
    
    for cls_id, count in zip(classes, counts):
        if cls_id in class_map:
            cls_name = class_map[cls_id]
            touching_class_names.append(cls_name)
            class_proportions[cls_name] = float(count) / float(total_touching)
            
    # Sort touching classes by proportion (descending)
    touching_class_names = sorted(touching_class_names, key=lambda c: class_proportions[c], reverse=True)
    
    return {
        "touching_classes": touching_class_names,
        "class_proportions": class_proportions
    }
