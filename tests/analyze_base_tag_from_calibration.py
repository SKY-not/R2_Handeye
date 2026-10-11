#!/usr/bin/env python
"""Recompute and analyze T_base_tag from camera poses and calibration_info.txt.

The calculation for each row is:
    T_base_tag = T_base_camera @ T_camera_tag

Run from the repository root:
    python tests/analyze_base_tag_from_calibration.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.spatial.transform import Rotation


TESTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = TESTS_DIR.parent
DEFAULT_CSV = TESTS_DIR / "tag36h11_id0_eye_to_hand_consistency.csv"
DEFAULT_CALIBRATION_INFO = ROOT_DIR / "results" / "eye_to_hand" / "calibration_info.txt"
DEFAULT_FIGURE = TESTS_DIR / "base_tag_recomputed_analysis.png"


def load_numeric_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV contains no data: {path}")
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in reader.fieldnames or []
    }


def load_base_camera_transform(path: Path) -> np.ndarray:
    """Read the Transform (4x4) block in calibration_info.txt as T_base_camera."""
    text = path.read_text(encoding="utf-8")
    marker = "Transform (4x4):"
    if marker not in text:
        raise ValueError(f"No 4x4 Transform block found in: {path}")

    matrix_lines = text.split(marker, maxsplit=1)[1].strip().splitlines()[:4]
    if len(matrix_lines) != 4:
        raise ValueError(f"Incomplete 4x4 Transform block in: {path}")

    rows = [
        np.fromstring(line.replace("[", " ").replace("]", " "), sep=" ")
        for line in matrix_lines
    ]
    transform = np.asarray(rows, dtype=np.float64)
    if transform.shape != (4, 4) or not np.allclose(transform[3], [0, 0, 0, 1]):
        raise ValueError(f"Invalid homogeneous transform in: {path}")
    return transform


def camera_transforms(data: dict[str, np.ndarray]) -> np.ndarray:
    sample_count = len(data["sample_index"])
    fields = {f"camera_T_{row}{col}" for row in range(4) for col in range(4)}
    missing = fields - data.keys()
    if missing:
        raise ValueError(f"CSV misses camera pose fields: {', '.join(sorted(missing))}")
    transforms = np.empty((sample_count, 4, 4), dtype=np.float64)
    for index in range(sample_count):
        transforms[index] = [
            [data[f"camera_T_{row}{col}"][index] for col in range(4)] for row in range(4)
        ]
    return transforms


def analyze(t_base_tag: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean pose, base-frame translation (mm), mean-tag rotation residuals (deg)."""
    mean = np.eye(4, dtype=np.float64)
    mean[:3, :3] = Rotation.from_matrix(t_base_tag[:, :3, :3]).mean().as_matrix()
    mean[:3, 3] = np.mean(t_base_tag[:, :3, 3], axis=0)
    mean_inverse = np.eye(4, dtype=np.float64)
    mean_inverse[:3, :3] = mean[:3, :3].T
    mean_inverse[:3, 3] = -mean_inverse[:3, :3] @ mean[:3, 3]
    residual = np.einsum("ij,njk->nik", mean_inverse, t_base_tag)
    translation_mm = (t_base_tag[:, :3, 3] - mean[:3, 3]) * 1000.0
    rotation_deg = Rotation.from_matrix(residual[:, :3, :3]).as_rotvec() * (180.0 / np.pi)
    return mean, translation_mm, rotation_deg


def print_summary(data: dict[str, np.ndarray], mean: np.ndarray, translation_mm: np.ndarray, rotation_deg: np.ndarray) -> None:
    translation_norm = np.linalg.norm(translation_mm, axis=1)
    rotation_norm = np.linalg.norm(rotation_deg, axis=1)
    dz = translation_mm[:, 2]
    print("Static visual repeatability only; this does not validate hand-eye accuracy.")
    print(f"Samples: {len(dz)}")
    print("Mean T_base_tag (recomputed):")
    print(np.array2string(mean, precision=8, suppress_small=False))
    print(f"Translation residual: mean={np.mean(translation_norm):.4f} mm, max={np.max(translation_norm):.4f} mm")
    print(f"Rotation residual:    mean={np.mean(rotation_norm):.4f} deg, max={np.max(rotation_norm):.4f} deg")
    print(f"Z residual:           mean={np.mean(dz):.4f} mm, std={np.std(dz, ddof=1):.4f} mm, range=[{np.min(dz):.4f}, {np.max(dz):.4f}] mm")
    z_score = np.abs(stats.zscore(dz, ddof=1))
    outliers = data["sample_index"][z_score > 3.0].astype(int)
    print("Z outliers (|z-score| > 3):", ", ".join(map(str, outliers)) if len(outliers) else "none")


def plot(data: dict[str, np.ndarray], translation_mm: np.ndarray, rotation_deg: np.ndarray, output: Path) -> None:
    samples = data["sample_index"]
    dz = translation_mm[:, 2]
    translation_norm = np.linalg.norm(translation_mm, axis=1)
    rotation_norm = np.linalg.norm(rotation_deg, axis=1)
    mean = float(np.mean(dz))
    std = float(np.std(dz, ddof=1))
    bins = min(12, max(5, int(np.ceil(np.sqrt(len(dz))))))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    trend, hist, qq, components, residuals, quality = axes.ravel()

    trend.plot(samples, dz, "o-", color="tab:blue")
    trend.axhline(mean, color="black", linestyle="--", linewidth=1, label="mean")
    trend.axhspan(mean - std, mean + std, color="tab:blue", alpha=0.15, label="mean +/- 1 std")
    trend.set(title="Recomputed base-Z residual trend", xlabel="sample", ylabel="dz (mm)")
    trend.legend(fontsize=8)
    trend.grid(alpha=0.35)

    hist.hist(dz, bins=bins, density=True, color="tab:blue", alpha=0.75, edgecolor="white", label="samples")
    x = np.linspace(float(np.min(dz) - 0.3 * std), float(np.max(dz) + 0.3 * std), 200)
    hist.plot(x, stats.norm.pdf(x, mean, std), color="tab:orange", label="normal fit")
    hist.set(title=f"Base-Z distribution (std={std:.3f} mm)", xlabel="dz (mm)", ylabel="density")
    hist.legend(fontsize=8)
    hist.grid(alpha=0.35)

    stats.probplot(dz, dist="norm", plot=qq)
    qq.set_title("Base-Z normal Q-Q plot")
    qq.grid(alpha=0.35)

    for values, label in zip(translation_mm.T, ("dx", "dy", "dz")):
        components.plot(samples, values, "o-", label=label)
    components.set(title="Recomputed translation components", xlabel="sample", ylabel="mm")
    components.legend()
    components.grid(alpha=0.35)

    residuals.plot(samples, translation_norm, "o-", label="translation (mm)")
    residuals_right = residuals.twinx()
    residuals_right.plot(samples, rotation_norm, "o-", color="tab:orange", label="rotation (deg)")
    residuals.set(title="Recomputed residual norms", xlabel="sample", ylabel="translation (mm)")
    residuals_right.set_ylabel("rotation (deg)")
    lines = residuals.get_lines() + residuals_right.get_lines()
    residuals.legend(lines, [line.get_label() for line in lines], fontsize=8)
    residuals.grid(alpha=0.35)

    quality.scatter(data["reprojection_error_px"], dz, c=samples, cmap="viridis", s=45)
    quality.axhline(mean, color="black", linestyle="--", linewidth=1)
    quality.set(title="Base-Z residual vs reprojection error", xlabel="reprojection error (px)", ylabel="dz (mm)")
    quality.grid(alpha=0.35)

    fig.suptitle(f"T_base_tag recomputed from calibration_info.txt (n={len(samples)})", fontsize=15)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(f"Figure written to: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--calibration-info", type=Path, default=DEFAULT_CALIBRATION_INFO)
    parser.add_argument("--output", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args()

    data = load_numeric_csv(args.csv)
    t_base_camera = load_base_camera_transform(args.calibration_info)
    t_base_tag = np.einsum("ij,njk->nik", t_base_camera, camera_transforms(data))
    mean, translation_mm, rotation_deg = analyze(t_base_tag)
    print(f"Using T_base_camera from: {args.calibration_info}")
    print_summary(data, mean, translation_mm, rotation_deg)
    plot(data, translation_mm, rotation_deg, args.output)


if __name__ == "__main__":
    main()
