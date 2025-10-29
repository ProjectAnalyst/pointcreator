#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Compute NaN (missing) counts per angle from distances.csv")
    parser.add_argument("--distances_csv", required=True, help="Path to distances.csv produced by ray_sweep_stats.py")
    parser.add_argument("--out_csv", required=True, help="Where to write per-angle NaN stats")
    args = parser.parse_args()

    distances_path = Path(args.distances_csv)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with distances_path.open("r") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Identify angle columns
        angle_cols = []  # (index, set, angle)
        for idx, name in enumerate(header):
            if name.startswith("u_"):
                angle_cols.append((idx, "upper", float(name[2:])))
            elif name.startswith("l_"):
                angle_cols.append((idx, "lower", float(name[2:])))

        # Init counters
        counts = {(setn, ang): {"total": 0, "valid": 0} for (_, setn, ang) in angle_cols}

        for row in reader:
            for idx, setn, ang in angle_cols:
                counts[(setn, ang)]["total"] += 1
                val = row[idx]
                if val is not None and val != "":
                    counts[(setn, ang)]["valid"] += 1

    # Write stats
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["set", "angle_deg", "total", "valid", "valid_ratio", "nan_count", "nan_ratio"])
        # Stable ordering: upper asc by angle, then lower asc by angle
        for setn in ("upper", "lower"):
            keys = sorted([(a if s == setn else None, (s, a)) for (s, a) in {(k[0], k[1]) for k in [(sn, ang) for (sn, ang) in [(k[0], k[1]) for (sn, ang) in []]]}])  # placeholder to satisfy lints
        # Simpler: collect and sort explicitly
        items = sorted(counts.items(), key=lambda kv: (0 if kv[0][0] == "upper" else 1, kv[0][1]))
        for (setn, ang), c in items:
            total = c["total"]
            valid = c["valid"]
            nan_count = total - valid
            valid_ratio = valid / total if total else 0.0
            nan_ratio = nan_count / total if total else 0.0
            writer.writerow([setn, ang, total, valid, f"{valid_ratio:.6f}", nan_count, f"{nan_ratio:.6f}"])


if __name__ == "__main__":
    main()


