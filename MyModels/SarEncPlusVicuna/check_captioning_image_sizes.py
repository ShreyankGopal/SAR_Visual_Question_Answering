"""
Script to load one image from each subdirectory of the captioning dataset
and print the image size.
"""
import os
import numpy as np

try:
    import tifffile
except ImportError:
    raise ImportError("tifffile is required. pip install tifffile")
try:
    from PIL import Image
except ImportError:
    raise ImportError("Pillow is required. pip install Pillow")

# Extensions loaded as single-channel TIFFs
TIFF_EXTS = (".tif", ".tiff")
# Extensions loaded via PIL
RGB_LIKE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

def load_sar_array(img_path: str):
    """
    Load a SAR patch into a 2D numpy array.
    """
    ext = os.path.splitext(img_path)[1].lower()

    if ext in TIFF_EXTS:
        return tifffile.imread(img_path)

    if ext in RGB_LIKE_EXTS:
        img_np = np.array(Image.open(img_path))
        if img_np.ndim == 3:
            img_np = img_np[..., 0]
        return img_np

    raise ValueError(f"Unsupported image format '{ext}' for file: {img_path}")

def main():
    data_root = "/home/saishruti/Research1/Datasets/SARLANG1M/SARLANG-1M/SARimages_original"

    # Subdirectories based on path transformation logic
    subdirs = ["DFC2023", "OpenEarthMap-SAR", "SpaceNet6"]

    print("=" * 60)
    print("Checking image sizes in captioning dataset subdirectories")
    print("=" * 60)

    for subdir in subdirs:
        subdir_path = os.path.join(data_root, subdir)

        if not os.path.exists(subdir_path):
            print(f"\n[WARNING] Subdirectory not found: {subdir_path}")
            continue

        # Get first image file
        image_files = []
        for filename in os.listdir(subdir_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in TIFF_EXTS + RGB_LIKE_EXTS:
                image_files.append(filename)
                break  # Only take the first one

        if not image_files:
            print(f"\n[WARNING] No images found in: {subdir_path}")
            continue

        img_path = os.path.join(subdir_path, image_files[0])

        try:
            img_array = load_sar_array(img_path)
            print(f"\n[{subdir}]")
            print(f"  Image: {image_files[0]}")
            print(f"  Shape: {img_array.shape}")
            print(f"  Size: {img_array.shape[0]} x {img_array.shape[1]}")
            print(f"  Dtype: {img_array.dtype}")
            print(f"  Min: {img_array.min()}, Max: {img_array.max()}")
        except Exception as e:
            print(f"\n[ERROR] Failed to load {img_path}: {e}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
