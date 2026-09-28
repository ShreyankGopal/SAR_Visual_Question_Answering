"""geometry.py: box helpers shared by all datasets.

Conventions (verified on SIVED):
  rbox = [cx, cy, w, h, angle], pixels, long-edge (w >= h), angle in [-90, 90)
  positive angle = long edge rotated clockwise from +x in image coords (y points down)
"""
import math
import numpy as np
from PIL import Image, ImageDraw

DIRECTIONS = ["to its right", "above and to its right", "above it", "above and to its left",
              "to its left", "below and to its left", "below it", "below and to its right"]


def rbox_pts(cx, cy, w, h, angle):
    """4 corners of a rotated box."""
    t = math.radians(angle)
    R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    d = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    return d @ R.T + [cx, cy]


def poly_mask(pts, W, H):
    """Boolean mask of a polygon."""
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).polygon([tuple(p) for p in pts], fill=1)
    return np.array(m, dtype=bool)


def cell_of(x, y, grid):
    """Grid cell name containing a normalized point (right/bottom edges belong to the last cell)."""
    for name, (x1, y1, x2, y2) in grid.items():
        if x1 <= x < (x2 if x2 < 1 else 1.0001) and y1 <= y < (y2 if y2 < 1 else 1.0001):
            return name
    return None


def orientation_bin(angle):
    """Return (bin name, degrees from the nearest bin boundary)."""
    a = abs(angle)
    if a < 22.5:
        return "horizontal", 22.5 - a
    if a >= 67.5:
        return "vertical", a - 67.5
    name = "diagonal, top-left to bottom-right" if angle > 0 else "diagonal, bottom-left to top-right"
    return name, min(a - 22.5, 67.5 - a)


def direction_name(dx, dy):
    """8-way direction of (dx, dy) seen from the reference object, plus degrees from the sector edge."""
    deg = math.degrees(math.atan2(-dy, dx)) % 360      # flip y so 'above' is positive
    k = int(((deg + 22.5) % 360) // 45)
    off = abs(((deg + 22.5) % 45) - 22.5)               # 0 = sector centre
    return DIRECTIONS[k], 22.5 - off


def axial_alignment(angles_deg):
    """How aligned a set of axial angles is: 1 = all parallel, 0 = uniformly spread."""
    t = np.radians(2 * np.asarray(angles_deg))
    return float(np.abs(np.exp(1j * t).mean())) if len(t) else np.nan


def hbox_pts(x1, y1, x2, y2):
    """4 corners of an axis-aligned box."""
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=float)
