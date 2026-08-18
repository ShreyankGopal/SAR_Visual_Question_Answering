import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── OpenEarthMap 8-class colour map ────────────────────────────────────────────
# Class index : (Name, RGB)
CLASS_MAP = {
    1: ("Bareland",          (210, 180, 140)),   # tan
    2: ("Rangeland",         (144, 238, 144)),   # light green
    3: ("Developed Space",   (169, 169, 169)),   # gray
    4: ("Road",              (105, 105, 105)),   # dark gray
    5: ("Tree",              ( 34, 139,  34)),   # forest green
    6: ("Water",             ( 65, 105, 225)),   # royal blue
    7: ("Agriculture Land",  (255, 215,   0)),   # gold/yellow
    8: ("Building",          (220,  20,  60)),   # crimson
}

def label_to_rgb(label: np.ndarray) -> np.ndarray:
    """Convert a single-band label array to an RGB image."""
    h, w = label.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, (_, color) in CLASS_MAP.items():
        rgb[label == cls_id] = color
    return rgb

def visualise(sar_path: str, label_path: str, save_path: str = None):
    # ── Read SAR ──────────────────────────────────────────────────────────────
    with rasterio.open(sar_path) as src:
        sar = src.read(1).astype(np.float32)

    # Normalise to [0, 1] for display
    p2, p98 = np.percentile(sar, (2, 98))
    sar_display = np.clip((sar - p2) / (p98 - p2 + 1e-8), 0, 1)

    # ── Read Label ────────────────────────────────────────────────────────────
    with rasterio.open(label_path) as src:
        label = src.read(1)

    present_classes = np.unique(label)
    label_rgb = label_to_rgb(label)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor("#1a1a2e")

    # SAR image
    axes[0].imshow(sar_display, cmap="gray")
    axes[0].set_title("SAR Image", color="white", fontsize=14, fontweight="bold", pad=10)
    axes[0].axis("off")

    # Label image
    axes[1].imshow(label_rgb)
    axes[1].set_title("Label (RGB)", color="white", fontsize=14, fontweight="bold", pad=10)
    axes[1].axis("off")

    # Legend — only show classes present in this tile
    legend_patches = []
    for cls_id in sorted(present_classes):
        if cls_id in CLASS_MAP:
            name, color = CLASS_MAP[cls_id]
            rgb_norm = tuple(c / 255 for c in color)
            patch = mpatches.Patch(color=rgb_norm, label=f"{cls_id}: {name}")
            legend_patches.append(patch)

    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=4,
        frameon=True,
        framealpha=0.3,
        facecolor="#1a1a2e",
        edgecolor="white",
        fontsize=10,
        labelcolor="white",
        bbox_to_anchor=(0.5, 0.01),
    )

    tile_name = os.path.splitext(os.path.basename(sar_path))[0]
    fig.suptitle(
        f"OpenEarthMap — {tile_name}",
        color="white", fontsize=16, fontweight="bold", y=1.01
    )
    plt.tight_layout(rect=[0, 0.09, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved to: {save_path}")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    import random

    SAR_DIR   = "/home/saishruti/Research1/Datasets/OpenEarthMap/train/sar_images"
    LABEL_DIR = "/home/saishruti/Research1/Datasets/OpenEarthMap/train/labels"

    all_tiles = sorted(f for f in os.listdir(SAR_DIR) if f.endswith(".tif"))
    sampled   = random.sample(all_tiles, 10)

    N = len(sampled)
    fig, axes = plt.subplots(N, 2, figsize=(12, N * 5))
    fig.patch.set_facecolor("#1a1a2e")

    all_present = set()

    for row, tile in enumerate(sampled):
        sar_path   = os.path.join(SAR_DIR,   tile)
        label_path = os.path.join(LABEL_DIR, tile)

        # SAR
        with rasterio.open(sar_path) as src:
            sar = src.read(1).astype(np.float32)
        p2, p98 = np.percentile(sar, (2, 98))
        sar_display = np.clip((sar - p2) / (p98 - p2 + 1e-8), 0, 1)

        # Label
        with rasterio.open(label_path) as src:
            label = src.read(1)
        label_rgb = label_to_rgb(label)
        all_present.update(np.unique(label).tolist())

        tile_name = os.path.splitext(tile)[0]

        axes[row, 0].imshow(sar_display, cmap="gray")
        axes[row, 0].set_title(f"SAR — {tile_name}", color="white",
                               fontsize=11, fontweight="bold", pad=6)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(label_rgb)
        axes[row, 1].set_title(f"Label — {tile_name}", color="white",
                               fontsize=11, fontweight="bold", pad=6)
        axes[row, 1].axis("off")

    # Shared legend for all classes present across sampled tiles
    legend_patches = []
    for cls_id in sorted(all_present):
        if cls_id in CLASS_MAP:
            name, color = CLASS_MAP[cls_id]
            rgb_norm = tuple(c / 255 for c in color)
            legend_patches.append(mpatches.Patch(color=rgb_norm, label=f"{cls_id}: {name}"))

    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=4,
        frameon=True,
        framealpha=0.3,
        facecolor="#1a1a2e",
        edgecolor="white",
        fontsize=10,
        labelcolor="white",
        bbox_to_anchor=(0.5, 0.005),
    )

    fig.suptitle("OpenEarthMap — 5 Random Training Tiles",
                 color="white", fontsize=15, fontweight="bold", y=1.002)
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    out_path = "/home/saishruti/Research1/Shreyank_20_credit/Minor_dataset_experiments/sar_label_10tiles.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved to: {out_path}")
    plt.close(fig)
