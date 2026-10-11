#!/usr/bin/env python
"""Offline analysis for eye-to-hand AprilTag pose-consistency CSV data.

Run from repository root:
    python tests/analyze_eye_to_hand_consistency.py

Optional inputs/outputs:
    python tests/analyze_eye_to_hand_consistency.py --csv tests/data.csv --output tests/analysis.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


TESTS_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = TESTS_DIR / "tag36h11_id0_eye_to_hand_consistency.csv"
DEFAULT_FIGURE = TESTS_DIR / "eye_to_hand_consistency_analysis.png"


def load_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV contains no samples: {path}")

    data: dict[str, np.ndarray] = {}
    for name in reader.fieldnames or []:
        try:
            data[name] = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            continue
    return data


def describe(values: np.ndarray) -> str:
    return (
        f"mean={np.mean(values):.4f}, std={np.std(values, ddof=1):.4f}, "
        f"median={np.median(values):.4f}, min={np.min(values):.4f}, max={np.max(values):.4f}"
    )


def print_summary(data: dict[str, np.ndarray]) -> None:
    n = len(data["sample_index"])
    print(f"Samples: {n}")
    print(f"Z residual (mm):          {describe(data['residual_dz_mm'])}")
    print(f"Translation residual (mm): {describe(data['residual_translation_mm'])}")
    print(f"Rotation residual (deg):   {describe(data['residual_rotation_deg'])}")
    print(f"Reprojection error (px):   {describe(data['reprojection_error_px'])}")
    print(f"Decision margin:           {describe(data['decision_margin'])}")

    dz = data["residual_dz_mm"]
    z_score = np.abs(stats.zscore(dz, ddof=1)) if len(dz) > 2 else np.zeros_like(dz)
    outliers = data["sample_index"][z_score > 3.0]
    print("Z outliers (|z-score| > 3):", ", ".join(map(str, outliers.astype(int))) if len(outliers) else "none")
    print(
        "Note: residual_dz_mm is computed relative to the sample mean pose. "
        "It measures repeatability, not a fixed absolute Z calibration bias."
    )


def plot(data: dict[str, np.ndarray], output: Path) -> None:
    sample = data["sample_index"]
    dz = data["residual_dz_mm"]
    mean = float(np.mean(dz))
    std = float(np.std(dz, ddof=1))
    bins = min(12, max(5, int(np.ceil(np.sqrt(len(dz))))))

    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(2, 3)
    ax_trend = fig.add_subplot(grid[0, 0])
    ax_hist = fig.add_subplot(grid[0, 1])
    ax_qq = fig.add_subplot(grid[0, 2])
    ax_components = fig.add_subplot(grid[1, 0])
    ax_quality = fig.add_subplot(grid[1, 1])
    ax_scatter = fig.add_subplot(grid[1, 2])

    ax_trend.plot(sample, dz, "o-", color="tab:blue", label="dz")
    ax_trend.axhline(mean, color="black", linestyle="--", linewidth=1, label="mean")
    ax_trend.axhspan(mean - std, mean + std, color="tab:blue", alpha=0.15, label="mean ± 1 std")
    ax_trend.set(title="Z residual trend", xlabel="sample", ylabel="dz relative to mean (mm)")
    ax_trend.legend(fontsize=8)
    ax_trend.grid(alpha=0.35)

    ax_hist.hist(dz, bins=bins, density=True, color="tab:blue", alpha=0.75, edgecolor="white", label="samples")
    if std > 0:
        x = np.linspace(float(np.min(dz) - .3 * std), float(np.max(dz) + .3 * std), 200)
        ax_hist.plot(x, stats.norm.pdf(x, mean, std), color="tab:orange", label="normal fit")
    ax_hist.axvline(mean, color="black", linestyle="--", linewidth=1)
    ax_hist.set(title=f"Z distribution (std={std:.3f} mm)", xlabel="dz (mm)", ylabel="density")
    ax_hist.legend(fontsize=8)
    ax_hist.grid(alpha=0.35)

    stats.probplot(dz, dist="norm", plot=ax_qq)
    ax_qq.set_title("Z residual normal Q-Q plot")
    ax_qq.grid(alpha=0.35)

    for key, label in (("residual_dx_mm", "dx"), ("residual_dy_mm", "dy"), ("residual_dz_mm", "dz")):
        ax_components.plot(sample, data[key], "o-", label=label)
    ax_components.set(title="Translation residual components", xlabel="sample", ylabel="mm")
    ax_components.legend()
    ax_components.grid(alpha=0.35)

    ax_quality.plot(sample, data["reprojection_error_px"], "o-", color="tab:green", label="reprojection (px)")
    right = ax_quality.twinx()
    right.plot(sample, data["decision_margin"], "o-", color="tab:purple", label="decision margin")
    ax_quality.set(title="Detection quality", xlabel="sample", ylabel="reprojection error (px)")
    right.set_ylabel("decision margin")
    ax_quality.grid(alpha=0.35)
    lines = ax_quality.get_lines() + right.get_lines()
    ax_quality.legend(lines, [line.get_label() for line in lines], fontsize=8)

    ax_scatter.scatter(data["reprojection_error_px"], dz, c=sample, cmap="viridis", s=45)
    ax_scatter.axhline(mean, color="black", linestyle="--", linewidth=1)
    ax_scatter.set(title="Z residual vs. reprojection error", xlabel="reprojection error (px)", ylabel="dz (mm)")
    ax_scatter.grid(alpha=0.35)

    fig.suptitle(f"Eye-to-hand pose consistency analysis (n={len(sample)})", fontsize=15)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(f"Figure written to: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="input consistency CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_FIGURE, help="output PNG path")
    args = parser.parse_args()
    data = load_csv(args.csv)
    required = {
        "sample_index", "residual_dx_mm", "residual_dy_mm", "residual_dz_mm",
        "residual_translation_mm", "residual_rotation_deg", "reprojection_error_px", "decision_margin",
    }
    missing = required - data.keys()
    if missing:
        raise ValueError(f"CSV missing required numeric columns: {', '.join(sorted(missing))}")
    print_summary(data)
    plot(data, args.output)


if __name__ == "__main__":
    main()
