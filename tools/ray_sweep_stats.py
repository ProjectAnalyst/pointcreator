#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


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

    prev_val = 0 if not (0 <= int(round(ox)) < width and 0 <= int(round(oy)) < height) else int(bin_mask[int(round(oy)), int(round(ox))])
    t = 0.0
    last_inside = prev_val
    for _ in range(max_steps):
        x = ox + dx * t
        y = oy + dy * t
        ix = int(round(x))
        iy = int(round(y))
        if ix < 0 or ix >= width or iy < 0 or iy >= height:
            return None
        cur_val = int(bin_mask[iy, ix])
        if (last_inside, cur_val) == target_transition:
            dist = math.hypot(dx * t, dy * t)
            return dist, (x, y)
        last_inside = cur_val
        t += step_size
    return None


def angle_to_dir(theta_deg: float) -> Tuple[float, float]:
    theta = math.radians(theta_deg)
    return math.sin(theta), math.cos(theta)


def compute_upper_lower_distances(
    mask_path: Path,
    upper_angles_deg: List[float],
    lower_angles_deg: List[float],
    origin: Tuple[float, float] = (256.0, 0.0),
    step_size: float = 0.5,
) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise RuntimeError(f"Failed to read image: {mask_path}")
    bin_mask = binarize_mask(mask_img)

    upper_distances: List[Optional[float]] = []
    for theta in upper_angles_deg:
        d = angle_to_dir(theta)
        res_upper = march_ray_until_transition(
            bin_mask, origin, d, target_transition=(0, 1), step_size=step_size
        )
        upper_distances.append(None if res_upper is None else res_upper[0])

    lower_distances: List[Optional[float]] = []
    for theta in lower_angles_deg:
        d = angle_to_dir(theta)
        hit_up = march_ray_until_transition(
            bin_mask, origin, d, target_transition=(0, 1), step_size=step_size
        )
        if hit_up is None:
            lower_distances.append(None)
            continue
        offset_origin = (
            hit_up[1][0] + d[0] * 0.5,
            hit_up[1][1] + d[1] * 0.5,
        )
        hit_low = march_ray_until_transition(
            bin_mask, offset_origin, d, target_transition=(1, 0), step_size=step_size
        )
        if hit_low is None:
            lower_distances.append(None)
        else:
            lower_distances.append(hit_up[0] + 0.5 + hit_low[0])

    return upper_distances, lower_distances


def find_masks(roots: List[Path]) -> List[Path]:
    files: List[Path] = []
    for root in roots:
        files.extend([p for p in root.rglob("*_mask.png") if p.is_file()])
    files.sort()
    return files


def write_distances_csv(
    out_csv: Path,
    records: List[Tuple[str, List[Optional[float]], List[Optional[float]]]],
    upper_angles: List[float],
    lower_angles: List[float],
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w") as f:
        header = ["sample"] + [f"u_{a}" for a in upper_angles] + [f"l_{a}" for a in lower_angles]
        f.write(",".join(header) + "\n")
        for sample, u, l in records:
            row = [sample]
            row += ["" if d is None else f"{d:.6f}" for d in u]
            row += ["" if d is None else f"{d:.6f}" for d in l]
            f.write(",".join(row) + "\n")


def write_stats_csv(
    out_csv: Path,
    records: List[Tuple[str, List[Optional[float]], List[Optional[float]]]],
    upper_angles: List[float],
    lower_angles: List[float],
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    total = len(records)
    u_valid = np.zeros(len(upper_angles), dtype=np.int64)
    l_valid = np.zeros(len(lower_angles), dtype=np.int64)
    for _sample, u, l in records:
        for i, d in enumerate(u):
            if d is not None:
                u_valid[i] += 1
        for i, d in enumerate(l):
            if d is not None:
                l_valid[i] += 1
    with out_csv.open("w") as f:
        f.write("set,angle_deg,valid,total,valid_ratio\n")
        for ang, cnt in zip(upper_angles, u_valid):
            ratio = cnt / total if total else 0.0
            f.write(f"upper,{ang},{cnt},{total},{ratio:.6f}\n")
        for ang, cnt in zip(lower_angles, l_valid):
            ratio = cnt / total if total else 0.0
            f.write(f"lower,{ang},{cnt},{total},{ratio:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description="Ray sweep over masks to log distances and angle validity stats")
    parser.add_argument("--roots", nargs="+", required=True, help="One or more mask root directories to scan")
    parser.add_argument("--output_dir", required=True, help="Directory to write CSV outputs")
    parser.add_argument("--upper_min", type=float, default=-80)
    parser.add_argument("--upper_max", type=float, default=80)
    parser.add_argument("--lower_min", type=float, default=-60)
    parser.add_argument("--lower_max", type=float, default=60)
    parser.add_argument("--spacing_deg", type=float, default=5.0, help="Angle spacing in degrees")
    parser.add_argument("--origin_x", type=float, default=256.0)
    parser.add_argument("--origin_y", type=float, default=0.0)
    parser.add_argument("--step_size", type=float, default=0.5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build angle grids
    def arange_deg(a, b, step):
        vals = []
        cur = a
        while cur <= b + 1e-9:
            vals.append(round(cur, 6))
            cur += step
        return vals

    upper_angles = arange_deg(args.upper_min, args.upper_max, args.spacing_deg)
    lower_angles = arange_deg(args.lower_min, args.lower_max, args.spacing_deg)

    roots = [Path(r) for r in args.roots]
    masks = find_masks(roots)
    if not masks:
        print("No mask files found.")
        return

    print(f"Found {len(masks)} masks. Sweeping angles: upper={len(upper_angles)} lower={len(lower_angles)}")

    records: List[Tuple[str, List[Optional[float]], List[Optional[float]]]] = []
    for i, m in enumerate(masks, start=1):
        try:
            u, l = compute_upper_lower_distances(
                m, upper_angles, lower_angles,
                origin=(args.origin_x, args.origin_y), step_size=args.step_size
            )
        except Exception as e:
            print(f"[ERROR] {m}: {e}")
            continue
        sample_id = f"{m.parent.name}_{m.stem}"
        records.append((sample_id, u, l))
        if i % 200 == 0:
            print(f"Processed {i}/{len(masks)}")

    write_distances_csv(output_dir / "distances.csv", records, upper_angles, lower_angles)
    write_stats_csv(output_dir / "angle_stats.csv", records, upper_angles, lower_angles)
    print(f"Done. Wrote: {output_dir / 'distances.csv'} and {output_dir / 'angle_stats.csv'}")


if __name__ == "__main__":
    main()


