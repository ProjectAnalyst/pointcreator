#!/usr/bin/env python3
import math
from typing import List, Optional, Tuple

import cv2
import numpy as np


# -------- Image/Mask utilities --------

def binarize_mask(mask_gray: np.ndarray, threshold: int = 128) -> np.ndarray:
    if mask_gray.dtype != np.uint8:
        mask_gray = mask_gray.astype(np.uint8)
    _, bin_mask = cv2.threshold(mask_gray, threshold, 255, cv2.THRESH_BINARY)
    return (bin_mask // 255).astype(np.uint8)


def compute_columns(img_w: int, num_columns: int, band_pct: float) -> List[int]:
    margin = (1 - band_pct) / 2.0
    x_min = int(round(img_w * margin))
    x_max = int(round(img_w * (1 - margin)))
    xs = np.linspace(x_min, x_max, num_columns, dtype=int).tolist()
    return xs


def select_columns_from_base(img_w: int, base_num: int, use_num: int, band_pct: float, drop_ends: int = 0) -> List[int]:
    xs_full = compute_columns(img_w, base_num, band_pct)
    if drop_ends * 2 <= len(xs_full) and use_num == len(xs_full) - 2 * drop_ends:
        return xs_full[drop_ends: len(xs_full) - drop_ends]
    return compute_columns(img_w, use_num, band_pct)


# -------- Geometry: circles and intersections --------

def circle_from_3_points(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> Tuple[Tuple[float, float], float]:
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    temp = x2 * x2 + y2 * y2
    bc = (x1 * x1 + y1 * y1 - temp) / 2.0
    cd = (temp - x3 * x3 - y3 * y3) / 2.0
    det = (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2)
    if abs(det) < 1e-6:
        raise ValueError("Collinear points")
    cx = (bc * (y2 - y3) - cd * (y1 - y2)) / det
    cy = ((x1 - x2) * cd - (x2 - x3) * bc) / det
    r = math.hypot(cx - x1, cy - y1)
    return (cx, cy), r


def fit_circle_ransac(points: List[Tuple[float, float]], iterations: int = 800, inlier_thresh: float = 2.5, min_inliers: int = 6, seed: int = 42) -> Optional[Tuple[Tuple[float, float], float, List[int]]]:
    if len(points) < 3:
        return None
    rng = np.random.default_rng(seed)
    pts = np.asarray(points, dtype=np.float64)
    idxs = np.arange(len(points))
    best: Optional[Tuple[Tuple[float, float], float, List[int]]] = None
    for _ in range(iterations):
        sample = rng.choice(idxs, size=3, replace=False)
        try:
            center, radius = circle_from_3_points(tuple(pts[sample[0]]), tuple(pts[sample[1]]), tuple(pts[sample[2]]))
        except Exception:
            continue
        dists = np.hypot(pts[:, 0] - center[0], pts[:, 1] - center[1])
        resid = np.abs(dists - radius)
        inliers = np.where(resid <= inlier_thresh)[0].tolist()
        if len(inliers) < min_inliers:
            continue
        if best is None or len(inliers) > len(best[2]):
            best = (center, radius, inliers)
    return best


def eval_circle_y_at_x(center: Tuple[float, float], radius: float, x: float, prefer_upper: bool, img_h: int) -> Optional[float]:
    cx, cy = center
    dx = x - cx
    inside = radius * radius - dx * dx
    if inside < 0:
        return None
    dy = math.sqrt(max(0.0, inside))
    y1 = cy - dy
    y2 = cy + dy
    y = y1 if prefer_upper else y2
    if y < 0 or y >= img_h:
        return None
    return y


# -------- Masks and overlays between two circles (not concentric) --------

def build_strip_mask(shape: Tuple[int, int], upper_center: Tuple[float, float], upper_radius: float, lower_center: Tuple[float, float], lower_radius: float, x_min: int, x_max: int) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    du = np.hypot(xx - upper_center[0], yy - upper_center[1])
    dl = np.hypot(xx - lower_center[0], yy - lower_center[1])
    band = (xx >= x_min) & (xx <= x_max)
    strip = (dl <= lower_radius) & (du >= upper_radius) & band
    return strip.astype(np.uint8)


def overlay_mask_on_image(image_bgr: np.ndarray, mask01: np.ndarray, color_bgr: Tuple[int, int, int] = (0, 255, 0), alpha: float = 0.35) -> np.ndarray:
    overlay = image_bgr.copy()
    colored = np.zeros_like(image_bgr)
    colored[:, :] = color_bgr
    mask3 = np.repeat(mask01[:, :, None], 3, axis=2)
    overlay = np.where(mask3 == 1, colored, overlay)
    blended = cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)
    return blended


