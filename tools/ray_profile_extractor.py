#!/usr/bin/env python3
import argparse
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt


def binarize_mask(mask_gray: np.ndarray, threshold: int = 128) -> np.ndarray:
    if mask_gray.dtype != np.uint8:
        mask_gray = mask_gray.astype(np.uint8)
    _, bin_mask = cv2.threshold(mask_gray, threshold, 255, cv2.THRESH_BINARY)
    return (bin_mask // 255).astype(np.uint8)


def march_ray_until_transition(
    bin_mask: np.ndarray,
    origin: Tuple[float, float],
    direction: Tuple[float, float],
    target_transition: Tuple[int, int],
    max_steps: int = 4000,
    step_size: float = 0.5,
) -> Optional[Tuple[float, Tuple[float, float]]]:
    height, width = bin_mask.shape
    ox, oy = origin
    dx, dy = direction

    # Start just outside the image to ensure transition detection is consistent
    prev_val = 0 if not (0 <= int(round(ox)) < width and 0 <= int(round(oy)) < height) else int(bin_mask[int(round(oy)), int(round(ox))])
    t = 0.0
    last_inside = prev_val

    for _ in range(max_steps):
        x = ox + dx * t
        y = oy + dy * t

        ix = int(round(x))
        iy = int(round(y))

        if ix < 0 or ix >= width or iy < 0 or iy >= height:
            # out of bounds
            return None

        cur_val = int(bin_mask[iy, ix])

        if (last_inside, cur_val) == target_transition:
            dist = math.hypot(dx * t, dy * t)
            return dist, (x, y)

        last_inside = cur_val
        t += step_size

    return None


def compute_distances_for_image(
    mask_path: Path,
    output_dir: Path,
    upper_angles_deg: List[float],
    lower_angles_deg: List[float],
    origin: Tuple[float, float] = (256.0, 0.0),
    step_size: float = 0.5,
) -> Tuple[
    List[Optional[float]],  # upper distances
    List[Optional[float]],  # lower distances
    np.ndarray,
    List[Optional[Tuple[float, float]]],  # upper hit points only
    List[Optional[Tuple[float, float]]],  # lower hit points only
]:
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise RuntimeError(f"Failed to read image: {mask_path}")

    bin_mask = binarize_mask(mask_img)

    # Direction: angle relative to +Y (downward). dx = sin(theta), dy = cos(theta)
    def angle_to_dir(theta_deg: float) -> Tuple[float, float]:
        theta = math.radians(theta_deg)
        return math.sin(theta), math.cos(theta)

    upper_distances: List[Optional[float]] = []
    lower_distances: List[Optional[float]] = []
    upper_hits: List[Optional[Tuple[float, float]]] = []
    lower_hits: List[Optional[Tuple[float, float]]] = []

    for theta in upper_angles_deg:
        direction = angle_to_dir(theta)
        res_upper = march_ray_until_transition(
            bin_mask,
            origin,
            direction,
            target_transition=(0, 1),  # black -> white
            step_size=step_size,
        )
        if res_upper is None:
            upper_distances.append(None)
            upper_hits.append(None)
            continue
        dist_up, point_up = res_upper
        upper_distances.append(dist_up)
        upper_hits.append(point_up)

    # For lower set, we also compute independently for completeness (same logic)
    # But per request, these are separate angle sets sampled from the same origin.
    # We'll append in order for lower angles list.
    lower_only_distances: List[Optional[float]] = []
    for theta in lower_angles_deg:
        direction = angle_to_dir(theta)
        # Find black -> white first to ensure we are inside before looking for white -> black
        res_upper = march_ray_until_transition(
            bin_mask,
            origin,
            direction,
            target_transition=(0, 1),
            step_size=step_size,
        )
        if res_upper is None:
            lower_only_distances.append(None)
            lower_hits.append(None)
            continue
        dist_up, point_up = res_upper
        offset_origin = (point_up[0] + direction[0] * 0.5, point_up[1] + direction[1] * 0.5)
        res_lower = march_ray_until_transition(
            bin_mask,
            offset_origin,
            direction,
            target_transition=(1, 0),
            step_size=step_size,
        )
        if res_lower is None:
            lower_only_distances.append(None)
            lower_hits.append(None)
        else:
            dist_low_offset, point_low = res_lower
            absolute_dist_low = dist_up + 0.5 + dist_low_offset
            lower_only_distances.append(absolute_dist_low)
            lower_hits.append(point_low)

    # Return with separate hit lists
    return upper_distances, lower_only_distances, mask_img, upper_hits, lower_hits


def save_csv(path: Path, angles: List[float], distances: List[Optional[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("angle_deg,distance_px\n")
        for a, d in zip(angles, distances):
            f.write(f"{a},{'' if d is None else d}\n")


def save_overlay(
    mask_img: np.ndarray,
    origin: Tuple[float, float],
    angles_deg: List[float],
    points: List[Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]],
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(mask_img, cmap="gray")
    ax.scatter([origin[0]], [origin[1]], c="cyan", s=20, label="origin")

    for theta, (p_up, p_low) in zip(angles_deg, points):
        # Draw ray line for context
        theta_rad = math.radians(theta)
        dx, dy = math.sin(theta_rad), math.cos(theta_rad)
        # Extend ray to image bounds
        length = max(mask_img.shape)
        x2 = origin[0] + dx * length
        y2 = origin[1] + dy * length
        ax.plot([origin[0], x2], [origin[1], y2], color="yellow", alpha=0.25, linewidth=1)

        if p_up is not None:
            ax.scatter([p_up[0]], [p_up[1]], c="lime", s=18, marker="o", label="upper" if theta == angles_deg[0] else None)
        if p_low is not None:
            ax.scatter([p_low[0]], [p_low[1]], c="red", s=18, marker="x", label="lower" if theta == angles_deg[0] else None)

    ax.set_title("Ray intersections: upper (lime), lower (red)")
    ax.set_xlim([0, mask_img.shape[1]])
    ax.set_ylim([mask_img.shape[0], 0])
    ax.legend(loc="upper right")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def save_combined_overlay(
    mask_img: np.ndarray,
    origin: Tuple[float, float],
    upper_angles: List[float],
    upper_hits: List[Optional[Tuple[float, float]]],
    lower_angles: List[float],
    lower_hits: List[Optional[Tuple[float, float]]],
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(mask_img, cmap="gray")
    ax.scatter([origin[0]], [origin[1]], c="cyan", s=20, label="origin")

    # Draw rays for upper angles
    for idx, theta in enumerate(upper_angles):
        theta_rad = math.radians(theta)
        dx, dy = math.sin(theta_rad), math.cos(theta_rad)
        length = max(mask_img.shape)
        x2 = origin[0] + dx * length
        y2 = origin[1] + dy * length
        ax.plot([origin[0], x2], [origin[1], y2], color="yellow", alpha=0.25, linewidth=1)
        p = upper_hits[idx]
        if p is not None:
            ax.scatter([p[0]], [p[1]], c="lime", s=22, marker="o", label="upper" if idx == 0 else None)

    # Draw rays for lower angles
    for idx, theta in enumerate(lower_angles):
        theta_rad = math.radians(theta)
        dx, dy = math.sin(theta_rad), math.cos(theta_rad)
        length = max(mask_img.shape)
        x2 = origin[0] + dx * length
        y2 = origin[1] + dy * length
        ax.plot([origin[0], x2], [origin[1], y2], color="yellow", alpha=0.25, linewidth=1)
        p = lower_hits[idx]
        if p is not None:
            ax.scatter([p[0]], [p[1]], c="red", s=22, marker="x", label="lower" if idx == 0 else None)

    ax.set_title("Upper (lime) and Lower (red) ray intersections")
    ax.set_xlim([0, mask_img.shape[1]])
    ax.set_ylim([mask_img.shape[0], 0])
    ax.legend(loc="upper right")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def circle_from_3_points(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> Optional[Tuple[Tuple[float, float], float]]:
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    temp = x2 * x2 + y2 * y2
    bc = (x1 * x1 + y1 * y1 - temp) / 2.0
    cd = (temp - x3 * x3 - y3 * y3) / 2.0
    det = (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2)
    if abs(det) < 1e-6:
        return None
    cx = (bc * (y2 - y3) - cd * (y1 - y2)) / det
    cy = ((x1 - x2) * cd - (x2 - x3) * bc) / det
    r = math.hypot(cx - x1, cy - y1)
    return (cx, cy), r


def fit_circle_ransac(
    points: List[Tuple[float, float]],
    iterations: int = 1000,
    inlier_threshold: float = 2.5,
    min_inliers: int = 6,
) -> Optional[Tuple[Tuple[float, float], float, List[int]]]:
    if len(points) < 3:
        return None
    rng = np.random.default_rng(42)
    best: Optional[Tuple[Tuple[float, float], float, List[int]]] = None
    idxs = np.arange(len(points))
    pts = np.asarray(points, dtype=np.float64)
    for _ in range(iterations):
        sample = rng.choice(idxs, size=3, replace=False)
        triplet = [tuple(pts[i]) for i in sample]
        res = circle_from_3_points(triplet[0], triplet[1], triplet[2])
        if res is None:
            continue
        center, radius = res
        dists = np.hypot(pts[:, 0] - center[0], pts[:, 1] - center[1])
        residuals = np.abs(dists - radius)
        inliers = np.where(residuals <= inlier_threshold)[0].tolist()
        if len(inliers) < min_inliers:
            continue
        if best is None or len(inliers) > len(best[2]):
            best = (center, radius, inliers)
    return best


def intersect_ray_circle(
    origin: Tuple[float, float],
    direction: Tuple[float, float],
    center: Tuple[float, float],
    radius: float,
) -> Optional[Tuple[float, float]]:
    ox, oy = origin
    dx, dy = direction
    cx, cy = center
    # Solve |O + t d - C|^2 = r^2
    fx = ox - cx
    fy = oy - cy
    a = dx * dx + dy * dy
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - radius * radius
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    candidates = [t for t in (t1, t2) if t > 0]
    if not candidates:
        return None
    t = min(candidates)
    return ox + dx * t, oy + dy * t


def build_strip_polygon(
    origin: Tuple[float, float],
    angle_range: Tuple[float, float],
    center_inner: Tuple[float, float],
    radius_inner: float,
    center_outer: Tuple[float, float],
    radius_outer: float,
    step_deg: float = 2.0,
) -> List[Tuple[float, float]]:
    theta_min, theta_max = angle_range
    thetas_up = np.arange(theta_min, theta_max + 1e-6, step_deg)
    thetas_down = thetas_up[::-1]

    def angle_to_dir(theta_deg: float) -> Tuple[float, float]:
        th = math.radians(theta_deg)
        return math.sin(th), math.cos(th)

    upper_boundary: List[Tuple[float, float]] = []
    for th in thetas_up:
        d = angle_to_dir(th)
        p = intersect_ray_circle(origin, d, center_inner, radius_inner)
        if p is not None:
            upper_boundary.append(p)

    lower_boundary: List[Tuple[float, float]] = []
    for th in thetas_down:
        d = angle_to_dir(th)
        p = intersect_ray_circle(origin, d, center_outer, radius_outer)
        if p is not None:
            lower_boundary.append(p)

    if len(upper_boundary) < 2 or len(lower_boundary) < 2:
        return []
    return upper_boundary + lower_boundary


def overlay_strip_on_image(
    image_bgr: np.ndarray,
    polygon: List[Tuple[float, float]],
    color_bgr: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.35,
) -> np.ndarray:
    overlay = image_bgr.copy()
    if polygon:
        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], color_bgr)
    blended = cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)
    return blended


def build_strip_mask(
    shape: Tuple[int, int],
    upper_center: Tuple[float, float],
    upper_radius: float,
    lower_center: Tuple[float, float],
    lower_radius: float,
    x_min: int,
    x_max: int,
) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    du = np.hypot(xx - upper_center[0], yy - upper_center[1])
    dl = np.hypot(xx - lower_center[0], yy - lower_center[1])
    band = (xx >= x_min) & (xx <= x_max)
    strip = (dl <= lower_radius) & (du >= upper_radius) & band
    return strip.astype(np.uint8)


def overlay_mask_on_image(
    image_bgr: np.ndarray,
    mask01: np.ndarray,
    color_bgr: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.35,
) -> np.ndarray:
    overlay = image_bgr.copy()
    colored = np.zeros_like(image_bgr)
    colored[:, :] = color_bgr
    mask3 = np.repeat(mask01[:, :, None], 3, axis=2)
    overlay = np.where(mask3 == 1, colored, overlay)
    blended = cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)
    return blended


def find_images(root: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in exts]


def main():
    parser = argparse.ArgumentParser(description="Extract upper/lower edge distances along specified rays from masks.")
    parser.add_argument("--input_dir", required=True, type=str, help="Directory with mask images")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save CSVs and overlays")
    parser.add_argument("--max_samples", type=int, default=5, help="Process at most this many images")
    parser.add_argument("--origin_x", type=float, default=256.0, help="Ray origin x (pixels)")
    parser.add_argument("--origin_y", type=float, default=0.0, help="Ray origin y (pixels)")
    parser.add_argument("--step_size", type=float, default=0.5, help="Ray marching step (pixels)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = find_images(input_dir)
    if not images:
        print(f"No images found in {input_dir}")
        return

    # Angle sets (updated per request)
    upper_angles = [-70, -65, -60, -50, -40, -30, -20, 20, 30, 40, 50, 60, 65, 70]
    lower_angles = [-50, -45, -40, -35, -30, -25, -20, 20, 25, 30, 35, 40, 45, 50]

    processed = 0
    for img_path in images:
        try:
            upper_dists, lower_dists, mask_img, upper_points, lower_points = compute_distances_for_image(
                img_path,
                output_dir,
                upper_angles,
                lower_angles,
                origin=(args.origin_x, args.origin_y),
                step_size=args.step_size,
            )
        except Exception as e:
            print(f"[ERROR] {img_path}: {e}")
            continue

        stem = img_path.stem
        rel_parent = img_path.parent.name
        save_base = output_dir / f"{rel_parent}_{stem}"

        save_csv(Path(f"{save_base}_upper.csv"), upper_angles, upper_dists)
        save_csv(Path(f"{save_base}_lower.csv"), lower_angles, lower_dists)
        # Combined overlay with distinct angle sets
        save_combined_overlay(
            mask_img,
            origin=(args.origin_x, args.origin_y),
            upper_angles=upper_angles,
            upper_hits=upper_points,
            lower_angles=lower_angles,
            lower_hits=lower_points,
            save_path=Path(f"{save_base}_combined_overlay.png"),
        )
        # Fit RANSAC circles on ray-hit points and overlay strip on original image
        upper_hit_points = [p for p in upper_points if p is not None]
        lower_hit_points = [p for p in lower_points if p is not None]

        fit_upper = fit_circle_ransac(upper_hit_points) if len(upper_hit_points) >= 3 else None
        fit_lower = fit_circle_ransac(lower_hit_points) if len(lower_hit_points) >= 3 else None

        if fit_upper and fit_lower:
            (cux, cuy), ru, inliers_u = fit_upper
            (clx, cly), rl, inliers_l = fit_lower
            # Map mask path to original image path
            mask_str = str(img_path)
            img_str = mask_str.replace("/masks/", "/images/")
            img_str = img_str.replace("_mask.png", ".png")
            image_bgr = cv2.imread(img_str, cv2.IMREAD_COLOR)
            if image_bgr is not None:
                h, w = image_bgr.shape[:2]
                x_min = int(0.05 * w)
                x_max = int(0.95 * w)
                strip_mask = build_strip_mask(
                    shape=(h, w),
                    upper_center=(cux, cuy),
                    upper_radius=ru,
                    lower_center=(clx, cly),
                    lower_radius=rl,
                    x_min=x_min,
                    x_max=x_max,
                )
                blended = overlay_mask_on_image(image_bgr, strip_mask, color_bgr=(0, 255, 0), alpha=0.35)
                cv2.imwrite(str(Path(f"{save_base}_strip_on_image.png")), blended)

            # Save fitted circles debug overlay on mask
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.imshow(mask_img, cmap="gray")
            theta = np.linspace(0, 2 * np.pi, 360)
            ax.plot(cux + ru * np.cos(theta), cuy + ru * np.sin(theta), 'g-', alpha=0.6, label='upper circle')
            ax.plot(clx + rl * np.cos(theta), cly + rl * np.sin(theta), 'r-', alpha=0.6, label='lower circle')
            if upper_hit_points:
                uxy = np.array(upper_hit_points)
                ax.scatter(uxy[:, 0], uxy[:, 1], c='lime', s=10)
            if lower_hit_points:
                lxy = np.array(lower_hit_points)
                ax.scatter(lxy[:, 0], lxy[:, 1], c='red', s=10)
            ax.legend(loc='upper right')
            fig.tight_layout()
            fig.savefig(str(Path(f"{save_base}_fitted_circles.png")), dpi=150)
            plt.close(fig)
        print(f"Processed: {img_path}")
        processed += 1
        if processed >= args.max_samples:
            break

    print(f"Done. Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()


