"""
DatasetStats.py
--------------
Calculate per-dataset pixel statistics (mean, std, min, max) for train datasets.

This script reads train_config.yaml to identify all train datasets and calculates
channel-wise statistics for the images in each dataset.
"""

import json
import os
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    import tifffile
except ImportError:
    raise ImportError("tifffile is required. pip install tifffile")

try:
    from PIL import Image
except ImportError:
    raise ImportError("Pillow is required. pip install Pillow")

try:
    import torch
    torch_available = torch.cuda.is_available()
except ImportError:
    torch_available = False
    print("Warning: torch not available, CUDA cache clearing disabled")


def load_config(config_path: str):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_jsonl_records(jsonl_path: str, sample_size: int = None, seed: int = 42):
    """
    Load JSONL file into list of dictionaries.
    
    Args:
        jsonl_path: Path to JSONL file
        sample_size: If specified, randomly sample this many records
        seed: Random seed for reproducibility
    """
    records = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    
    # Apply sampling if specified
    if sample_size is not None and sample_size < len(records):
        import random
        rng = random.Random(seed)
        records = rng.sample(records, sample_size)
    
    return records


def load_image(image_path: str):
    """
    Load image based on file extension.
    
    For .tif files: load as-is (single channel SAR data)
    For .jpg/.png files: load via PIL and extract first channel
    """
    ext = os.path.splitext(image_path)[1].lower()
    
    if ext in ['.tif', '.tiff']:
        # Load TIFF as-is (single channel)
        img = tifffile.imread(image_path)
        # Ensure 2D for single channel
        if img.ndim == 3:
            img = img[0]  # Take first channel if multi-channel TIFF
        return img
    
    elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Load via PIL and extract first channel
        img = Image.open(image_path)
        img_array = np.array(img)
        if img_array.ndim == 3:
            img_array = img_array[:, :, 0]  # Take first channel
        return img_array
    
    else:
        raise ValueError(f"Unsupported image format: {ext}")


def calculate_statistics(image_paths: list, data_root: str):
    """
    Calculate channel-wise statistics for a list of image paths using Welford's algorithm.
    
    Welford's algorithm is numerically stable and uses O(1) memory (only stores running totals).
    Processes each image's pixels in vectorized batches for speed.
    
    Returns:
        dict: {'mean': float, 'std': float, 'min': float, 'max': float}
    """
    count = 0
    mean = 0.0
    M2 = 0.0  # Sum of squared differences from mean
    min_pixel = float('inf')
    max_pixel = float('-inf')
    num_images = 0
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        full_path = os.path.join(data_root, img_path)
        
        try:
            img = load_image(full_path)
            pixels = img.flatten()
            
            # Process this image's pixels using Welford's algorithm
            # Update global statistics with this batch
            n_new = len(pixels)
            n_old = count
            
            # Update count
            count += n_new
            
            # Update mean
            delta = np.sum(pixels) - mean * n_new
            mean += delta / count
            
            # Update M2 (sum of squared differences)
            # Using parallel algorithm for combining: M2_new = M2_old + sum((x - mean_new)²)
            # This is numerically stable
            M2 += np.sum((pixels - mean) ** 2)
            
            # Update min/max
            img_min = np.min(pixels)
            img_max = np.max(pixels)
            if img_min < min_pixel:
                min_pixel = img_min
            if img_max > max_pixel:
                max_pixel = img_max
            
            num_images += 1
            
            # Clear memory periodically
            if num_images % 100 == 0:
                import gc
                gc.collect()
                if torch_available:
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")
            continue
    
    # Clear memory after processing
    import gc
    gc.collect()
    if torch_available:
        torch.cuda.empty_cache()
    
    if count == 0:
        return None
    
    # Calculate final statistics
    variance = M2 / count
    std = np.sqrt(variance)
    
    stats = {
        'mean': float(mean),
        'std': float(std),
        'min': float(min_pixel),
        'max': float(max_pixel),
        'num_pixels': int(count),
        'num_images': num_images
    }
    
    return stats


def main():
    # Load configuration
    config_path = "train_config.yaml"
    print(f"Loading config from {config_path}...")
    config = load_config(config_path)
    
    c_data = config["data"]
    
    # Create output directory
    output_dir = Path("DatasetStatistics")
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Extract all train datasets and their data roots
    train_datasets = []
    
    # Get sample size from config (for primary dataset)
    train_sample_size = c_data.get("train_sample_size", None)
    sample_seed = c_data.get("sample_seed", 42)
    
    if train_sample_size:
        print(f"Sample size from config: {train_sample_size}")
        print(f"Random seed: {sample_seed}")
    
    # Primary dataset
    if "train_jsonl" in c_data:
        train_datasets.append({
            "name": "train_jsonl (primary)",
            "jsonl_path": c_data["train_jsonl"],
            "data_root": c_data["data_root"],
            "sample_size": train_sample_size,
            "seed": sample_seed
        })
    
    # Secondary datasets
    for i in range(2, 10):  # Check for train_jsonl_2, train_jsonl_3, etc.
        jsonl_key = f"train_jsonl_{i}"
        root_key = f"data_root_{i}"
        
        if jsonl_key in c_data:
            data_root = c_data.get(root_key, c_data["data_root"])
            # Use same sample size for secondary datasets
            train_datasets.append({
                "name": f"train_jsonl_{i}",
                "jsonl_path": c_data[jsonl_key],
                "data_root": data_root,
                "sample_size": train_sample_size,  # Apply same sampling
                "seed": sample_seed
            })
    
    print(f"\nFound {len(train_datasets)} train datasets to analyze:")
    for ds in train_datasets:
        print(f"  - {ds['name']}: {ds['jsonl_path']}")
    
    # Calculate statistics for each dataset
    print("\n" + "="*60)
    print("CALCULATING PIXEL STATISTICS")
    print("="*60)
    
    for ds in train_datasets:
        print(f"\nProcessing: {ds['name']}")
        print(f"JSONL: {ds['jsonl_path']}")
        print(f"Data Root: {ds['data_root']}")
        
        # Load JSONL records
        print("Loading JSONL records...")
        records = load_jsonl_records(ds["jsonl_path"], ds["sample_size"], ds["seed"])
        print(f"Total records: {len(records)}")
        
        # Extract image paths
        image_paths = []
        for record in records:
            if "image" in record:
                image_paths.append(record["image"])
        
        print(f"Total images: {len(image_paths)}")
        
        if len(image_paths) == 0:
            print("No images found, skipping...")
            continue
        
        # Calculate statistics
        print("Calculating statistics...")
        stats = calculate_statistics(image_paths, ds["data_root"])
        
        if stats:
            # Print to console
            print("\n" + "-"*50)
            print(f"Statistics for {ds['name']}:")
            print("-"*50)
            print(f"Mean:          {stats['mean']:.4f}")
            print(f"Std:           {stats['std']:.4f}")
            print(f"Min:           {stats['min']:.4f}")
            print(f"Max:           {stats['max']:.4f}")
            print(f"Num Images:    {stats['num_images']}")
            print(f"Num Pixels:    {stats['num_pixels']:,}")
            print("-"*50)
            
            # Save to file
            safe_name = ds['name'].replace(' ', '_').replace('(', '').replace(')', '')
            output_file = output_dir / f"{safe_name}_stats.txt"
            
            with open(output_file, 'w') as f:
                f.write("="*60 + "\n")
                f.write(f"Dataset Statistics: {ds['name']}\n")
                f.write("="*60 + "\n\n")
                f.write(f"JSONL Path: {ds['jsonl_path']}\n")
                f.write(f"Data Root: {ds['data_root']}\n")
                
                # Add sampling information
                if ds['sample_size']:
                    f.write(f"Sample Size: {ds['sample_size']}\n")
                    f.write(f"Random Seed: {ds['seed']}\n")
                else:
                    f.write("Sample Size: Full dataset (no sampling)\n")
                
                f.write("\nPixel Statistics:\n")
                f.write("-"*40 + "\n")
                f.write(f"Mean:          {stats['mean']:.6f}\n")
                f.write(f"Std:           {stats['std']:.6f}\n")
                f.write(f"Min:           {stats['min']:.6f}\n")
                f.write(f"Max:           {stats['max']:.6f}\n")
                f.write(f"Num Images:    {stats['num_images']}\n")
                f.write(f"Num Pixels:    {stats['num_pixels']:,}\n")
                f.write("\n" + "="*60 + "\n")
            
            print(f"Saved to: {output_file}")
        else:
            print("Failed to calculate statistics")
        
        # Clear memory after each dataset
        import gc
        gc.collect()
        if torch_available:
            torch.cuda.empty_cache()
            print("CUDA cache cleared after dataset processing")
    
    print("\n" + "="*60)
    print("STATISTICS CALCULATION COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()