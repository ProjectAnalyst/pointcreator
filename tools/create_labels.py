#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    h, w = bin_mask.shape
    ox, oy = origin
    dx, dy = direction
    prev_val = 0 if not (0 <= int(round(ox)) < w and 0 <= int(round(oy)) < h) else int(bin_mask[int(round(oy)), int(round(ox))])
    t = 0.0
    last_inside = prev_val
    for _ in range(max_steps):
        x = ox + dx * t
        y = oy + dy * t
        ix = int(round(x))
        iy = int(round(y))
        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            return None
        cur_val = int(bin_mask[iy, ix])
        if (last_inside, cur_val) == target_transition:
            dist = math.hypot(dx * t, dy * t)
            return dist, (x, y)
        last_inside = cur_val
        t += step_size
    return None


def angle_to_dir(theta_deg: float) -> Tuple[float, float]:
    th = math.radians(theta_deg)
    return math.sin(th), math.cos(th)


def compute_distances(
    mask_path: Path,
    upper_angles_deg: List[float],
    lower_angles_deg: List[float],
    origin: Tuple[float, float] = (256.0, 0.0),
    step_size: float = 0.5,
) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise RuntimeError(f"Failed to read mask: {mask_path}")
    bin_mask = binarize_mask(mask_img)

    upper_dists: List[Optional[float]] = []
    for theta in upper_angles_deg:
        d = angle_to_dir(theta)
        hit_u = march_ray_until_transition(bin_mask, origin, d, target_transition=(0, 1), step_size=step_size)
        upper_dists.append(None if hit_u is None else hit_u[0])

    lower_dists: List[Optional[float]] = []
    for theta in lower_angles_deg:
        d = angle_to_dir(theta)
        hit_u = march_ray_until_transition(bin_mask, origin, d, target_transition=(0, 1), step_size=step_size)
        if hit_u is None:
            lower_dists.append(None)
            continue
        offset_origin = (hit_u[1][0] + d[0] * 0.5, hit_u[1][1] + d[1] * 0.5)
        hit_l = march_ray_until_transition(bin_mask, offset_origin, d, target_transition=(1, 0), step_size=step_size)
        lower_dists.append(None if hit_l is None else (hit_u[0] + 0.5 + hit_l[0]))

    return upper_dists, lower_dists


def find_masks(roots: List[Path]) -> List[Path]:
    masks: List[Path] = []
    for r in roots:
        masks.extend([p for p in r.rglob("*_mask.png") if p.is_file()])
    masks.sort()
    return masks


def write_metadata(metadata_path: Path, meta: Dict) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w") as f:
        json.dump(meta, f, indent=2)


def write_index(index_path: Path, rows: List[Dict[str, str]]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    # CSV for easy reading in pandas
    header = ["sample_id", "image_path", "label_path"]
    lines = ["{}, {}, {}\n".format(r["sample_id"], r["image_path"], r["label_path"]) for r in rows]
    with index_path.open("w") as f:
        f.write(",".join(header) + "\n")
        # remove extra spaces after commas when writing lines
        for r in rows:
            f.write(f"{r['sample_id']},{r['image_path']},{r['label_path']}\n")


def main():
    parser = argparse.ArgumentParser(description="Create per-image JSON labels with ray distances and validity masks")
    parser.add_argument("--mask_roots", nargs="+", required=True, help="Mask directories to scan (*_mask.png)")
    parser.add_argument("--output_dir", required=True, help="Directory to write labels/, metadata.json, index.csv")
    parser.add_argument("--origin_x", type=float, default=256.0)
    parser.add_argument("--origin_y", type=float, default=0.0)
    parser.add_argument("--step_size", type=float, default=0.5)
    # Defaults per discussion: upper ~[-65,65], lower ~[-55,55] skipping central [-20,20] for lower
    parser.add_argument("--upper_angles", type=str, default="-65,-55,-45,-35,-25,-15,-10,-5,5,10,15,25,35,45,55,65")
    parser.add_argument("--lower_angles", type=str, default="-55,-45,-35,-25,-15,-10,10,15,25,35,45,55")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    upper_angles = [float(x) for x in args.upper_angles.split(",") if x.strip()]
    lower_angles = [float(x) for x in args.lower_angles.split(",") if x.strip()]

    masks = find_masks([Path(r) for r in args.mask_roots])
    if not masks:
        print("No masks found.")
        return

    # Save metadata.json with the angle schema and origin
    metadata = {
        "origin": {"x": args.origin_x, "y": args.origin_y},
        "upper_angles": upper_angles,
        "lower_angles": lower_angles,
        "distance_units": "pixels",
        "image_size_hint": [512, 224],
        "notes": "per-image labels contain distances in pixels and validity masks; NaNs serialized as null",
    }
    write_metadata(output_dir / "metadata.json", metadata)

    index_rows: List[Dict[str, str]] = []

    for i, m in enumerate(masks, start=1):
        try:
            u_px, l_px = compute_distances(
                m, upper_angles, lower_angles,
                origin=(args.origin_x, args.origin_y), step_size=args.step_size
            )
        except Exception as e:
            print(f"[ERROR] {m}: {e}")
            continue

        # Validity masks
        u_valid = [1 if d is not None else 0 for d in u_px]
        l_valid = [1 if d is not None else 0 for d in l_px]

        # JSON label object
        label_obj = {
            "sample_id": f"{m.parent.name}_{m.stem}",
            "image_path": str(m).replace("/masks/", "/images/").replace("_mask.png", ".png"),
            "mask_path": str(m),
            "origin": {"x": args.origin_x, "y": args.origin_y},
            "upper_dists_px": [None if d is None else float(d) for d in u_px],
            "lower_dists_px": [None if d is None else float(d) for d in l_px],
            "upper_valid_mask": u_valid,
            "lower_valid_mask": l_valid,
        }

        label_path = labels_dir / f"{m.parent.name}_{m.stem}.json"
        with label_path.open("w") as f:
            json.dump(label_obj, f, indent=2)

        index_rows.append({
            "sample_id": label_obj["sample_id"],
            "image_path": label_obj["image_path"],
            "label_path": str(label_path),
        })

        if i % 200 == 0:
            print(f"Labeled {i}/{len(masks)}")

    write_index(output_dir / "index.csv", index_rows)
    print(f"Done. Labels in {labels_dir}, metadata and index written to {output_dir}")


if __name__ == "__main__":
    main()


