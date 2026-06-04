#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check eye-to-hand AprilTag pose consistency for tag36h11 id=0.

Keys:
  Enter: capture the current valid tag pose
  Space: finish capture, write CSV, and show analysis plots
  Esc: quit without analysis
"""

from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from camera.realsense import RealSenseCamera


TAG_FAMILY = "tag36h11"
TARGET_TAG_ID = 0
TAG_SIZE_M = 0.010
DECISION_MARGIN_THRESHOLD = 20.0
AXIS_LENGTH_M = 0.03
EXAMPLE_POINT_IN_TAG_M = np.array([0.02, 0.0, 0.0], dtype=np.float64)
HANDEYE_FILE = os.path.join(ROOT, "results", "eye_to_hand", "handeye_transform.txt")
CSV_FILE = os.path.join(ROOT, "tests", "tag36h11_id0_eye_to_hand_consistency.csv")


@dataclass
class Sample:
    index: int
    timestamp: float
    decision_margin: float
    reprojection_error_px: float
    t_camera_tag: np.ndarray
    t_base_tag: np.ndarray


@dataclass
class Analysis:
    mean_transform: np.ndarray
    residual_components_mm_deg: np.ndarray
    translation_residual_mm: np.ndarray
    rotation_residual_deg: np.ndarray


def create_apriltag_detector() -> Any:
    try:
        from pyapriltags import Detector
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to import pyapriltags: {exc}") from exc

    return Detector(
        families=TAG_FAMILY,
        nthreads=2,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )


def load_transform(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Hand-eye result file not found: {path}")
    transform = np.loadtxt(path, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got {transform.shape}: {path}")
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = transform[:3, :3].T
    inv[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return inv


def transform_to_rvec_pose(transform: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    return np.concatenate([transform[:3, 3], rvec.reshape(3)], axis=0)


def detection_to_transform(detection: Any) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(detection.pose_R, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(detection.pose_t, dtype=np.float64).reshape(3)
    return transform


def detect_target(
    detector: Any,
    image_bgr: np.ndarray,
    camera_matrix: np.ndarray,
) -> tuple[list[Any], Optional[Any]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    detections = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=(fx, fy, cx, cy),
        tag_size=TAG_SIZE_M,
    )

    target: Optional[Any] = None
    for detection in detections:
        tag_id = int(getattr(detection, "tag_id", -1))
        margin = float(getattr(detection, "decision_margin", 0.0))
        has_pose = getattr(detection, "pose_R", None) is not None and getattr(detection, "pose_t", None) is not None
        if tag_id == TARGET_TAG_ID and margin >= DECISION_MARGIN_THRESHOLD and has_pose:
            target = detection
            break

    return detections, target


def tag_object_points() -> np.ndarray:
    half = TAG_SIZE_M / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def reprojection_error_px(detection: Any, camera_matrix: np.ndarray) -> float:
    corners = np.asarray(detection.corners, dtype=np.float64).reshape(4, 2)
    pose_r = np.asarray(detection.pose_R, dtype=np.float64).reshape(3, 3)
    pose_t = np.asarray(detection.pose_t, dtype=np.float64).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(pose_r)
    projected, _ = cv2.projectPoints(
        tag_object_points(),
        rvec,
        pose_t,
        camera_matrix,
        np.zeros((5, 1), dtype=np.float64),
    )
    projected = projected.reshape(4, 2)
    return float(np.mean(np.linalg.norm(projected - corners, axis=1)))


def draw_detection_overlay(
    image: np.ndarray,
    detection: Any,
    camera_matrix: np.ndarray,
    is_target: bool,
) -> None:
    corners = np.asarray(detection.corners, dtype=np.float64).reshape(4, 2)
    center = np.asarray(detection.center, dtype=np.float64).reshape(2)
    color = (0, 255, 0) if is_target else (0, 180, 255)

    cv2.polylines(image, [corners.astype(np.int32).reshape(-1, 1, 2)], True, color, 2)
    cv2.circle(image, tuple(center.astype(np.int32)), 4, color, -1)
    cv2.putText(
        image,
        f"id={int(detection.tag_id)} dm={float(detection.decision_margin):.1f}",
        tuple((center + np.array([8.0, -8.0])).astype(np.int32)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )

    if getattr(detection, "pose_R", None) is None or getattr(detection, "pose_t", None) is None:
        return

    pose_r = np.asarray(detection.pose_R, dtype=np.float64).reshape(3, 3)
    pose_t = np.asarray(detection.pose_t, dtype=np.float64).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(pose_r)
    axis_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [AXIS_LENGTH_M, 0.0, 0.0],
            [0.0, AXIS_LENGTH_M, 0.0],
            [0.0, 0.0, AXIS_LENGTH_M],
        ],
        dtype=np.float64,
    )
    image_points, _ = cv2.projectPoints(
        axis_points,
        rvec,
        pose_t,
        camera_matrix,
        np.zeros((5, 1), dtype=np.float64),
    )
    points = image_points.reshape(-1, 2).astype(np.int32)
    origin = tuple(points[0])
    cv2.line(image, origin, tuple(points[1]), (0, 0, 255), 2)
    cv2.line(image, origin, tuple(points[2]), (0, 255, 0), 2)
    cv2.line(image, origin, tuple(points[3]), (255, 0, 0), 2)


def draw_text_lines(image: np.ndarray, lines: list[str]) -> None:
    for i, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (20, 30 + 28 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def mean_transform(transforms: list[np.ndarray]) -> np.ndarray:
    rotations = Rotation.from_matrix(np.stack([transform[:3, :3] for transform in transforms], axis=0))
    translation = np.mean(np.stack([transform[:3, 3] for transform in transforms], axis=0), axis=0)

    mean = np.eye(4, dtype=np.float64)
    mean[:3, :3] = rotations.mean().as_matrix()
    mean[:3, 3] = translation
    return mean


def analyze_samples(samples: list[Sample]) -> Analysis:
    mean = mean_transform([sample.t_base_tag for sample in samples])
    mean_inv = invert_transform(mean)

    components: list[np.ndarray] = []
    translation_norms: list[float] = []
    rotation_norms: list[float] = []

    for sample in samples:
        residual = mean_inv @ sample.t_base_tag
        translation_mm = residual[:3, 3] * 1000.0
        rotation_vec_deg = Rotation.from_matrix(residual[:3, :3]).as_rotvec() * (180.0 / np.pi)
        components.append(np.concatenate([translation_mm, rotation_vec_deg], axis=0))
        translation_norms.append(float(np.linalg.norm(translation_mm)))
        rotation_norms.append(float(np.linalg.norm(rotation_vec_deg)))

    return Analysis(
        mean_transform=mean,
        residual_components_mm_deg=np.stack(components, axis=0),
        translation_residual_mm=np.asarray(translation_norms, dtype=np.float64),
        rotation_residual_deg=np.asarray(rotation_norms, dtype=np.float64),
    )


def matrix_fields(prefix: str) -> list[str]:
    return [f"{prefix}_T_{row}{col}" for row in range(4) for col in range(4)]


def matrix_values(matrix: np.ndarray) -> list[float]:
    return [float(matrix[row, col]) for row in range(4) for col in range(4)]


def write_csv(samples: list[Sample], analysis: Analysis, path: str) -> None:
    fieldnames = [
        "sample_index",
        "timestamp",
        "decision_margin",
        "reprojection_error_px",
        "camera_tx",
        "camera_ty",
        "camera_tz",
        "camera_rvec_x",
        "camera_rvec_y",
        "camera_rvec_z",
        *matrix_fields("camera"),
        "base_tx",
        "base_ty",
        "base_tz",
        "base_rvec_x",
        "base_rvec_y",
        "base_rvec_z",
        *matrix_fields("base"),
        "residual_dx_mm",
        "residual_dy_mm",
        "residual_dz_mm",
        "residual_translation_mm",
        "residual_drx_deg",
        "residual_dry_deg",
        "residual_drz_deg",
        "residual_rotation_deg",
    ]

    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for row_index, sample in enumerate(samples):
            camera_pose = transform_to_rvec_pose(sample.t_camera_tag)
            base_pose = transform_to_rvec_pose(sample.t_base_tag)
            residual = analysis.residual_components_mm_deg[row_index]

            row = {
                "sample_index": sample.index,
                "timestamp": sample.timestamp,
                "decision_margin": sample.decision_margin,
                "reprojection_error_px": sample.reprojection_error_px,
                "camera_tx": camera_pose[0],
                "camera_ty": camera_pose[1],
                "camera_tz": camera_pose[2],
                "camera_rvec_x": camera_pose[3],
                "camera_rvec_y": camera_pose[4],
                "camera_rvec_z": camera_pose[5],
                "base_tx": base_pose[0],
                "base_ty": base_pose[1],
                "base_tz": base_pose[2],
                "base_rvec_x": base_pose[3],
                "base_rvec_y": base_pose[4],
                "base_rvec_z": base_pose[5],
                "residual_dx_mm": residual[0],
                "residual_dy_mm": residual[1],
                "residual_dz_mm": residual[2],
                "residual_translation_mm": analysis.translation_residual_mm[row_index],
                "residual_drx_deg": residual[3],
                "residual_dry_deg": residual[4],
                "residual_drz_deg": residual[5],
                "residual_rotation_deg": analysis.rotation_residual_deg[row_index],
            }
            row.update(dict(zip(matrix_fields("camera"), matrix_values(sample.t_camera_tag))))
            row.update(dict(zip(matrix_fields("base"), matrix_values(sample.t_base_tag))))
            writer.writerow(row)


def draw_frame_3d(
    axis: Any,
    transform: np.ndarray,
    label: str,
    length: float,
    linewidth: float = 1.5,
    alpha: float = 1.0,
) -> None:
    origin = transform[:3, 3]
    axes = transform[:3, :3]
    colors = ("r", "g", "b")
    for i, color in enumerate(colors):
        end = origin + axes[:, i] * length
        axis.plot(
            [origin[0], end[0]],
            [origin[1], end[1]],
            [origin[2], end[2]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )
    axis.text(origin[0], origin[1], origin[2], label, fontsize=9)


def set_axes_equal(axis: Any, points: np.ndarray) -> None:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 0.03)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def show_analysis(samples: list[Sample], analysis: Analysis) -> None:
    sample_ids = np.asarray([sample.index for sample in samples], dtype=np.int64)
    residual = analysis.residual_components_mm_deg
    reprojection = np.asarray([sample.reprojection_error_px for sample in samples], dtype=np.float64)
    margins = np.asarray([sample.decision_margin for sample in samples], dtype=np.float64)
    points = np.stack([sample.t_base_tag[:3, 3] for sample in samples], axis=0)

    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(2, 4)

    ax_translation = fig.add_subplot(grid[0, 0])
    ax_rotation = fig.add_subplot(grid[0, 1])
    ax_components_t = fig.add_subplot(grid[0, 2])
    ax_components_r = fig.add_subplot(grid[0, 3])
    ax_reprojection = fig.add_subplot(grid[1, 0])
    ax_margin = fig.add_subplot(grid[1, 1])
    ax_3d = fig.add_subplot(grid[1, 2:4], projection="3d")

    ax_translation.plot(sample_ids, analysis.translation_residual_mm, marker="o")
    ax_translation.set_title("Translation residual")
    ax_translation.set_xlabel("sample")
    ax_translation.set_ylabel("mm")
    ax_translation.grid(True)

    ax_rotation.plot(sample_ids, analysis.rotation_residual_deg, marker="o", color="tab:orange")
    ax_rotation.set_title("Rotation residual")
    ax_rotation.set_xlabel("sample")
    ax_rotation.set_ylabel("deg")
    ax_rotation.grid(True)

    ax_components_t.plot(sample_ids, residual[:, 0], marker="o", label="dx")
    ax_components_t.plot(sample_ids, residual[:, 1], marker="o", label="dy")
    ax_components_t.plot(sample_ids, residual[:, 2], marker="o", label="dz")
    ax_components_t.set_title("Translation components")
    ax_components_t.set_xlabel("sample")
    ax_components_t.set_ylabel("mm")
    ax_components_t.legend()
    ax_components_t.grid(True)

    ax_components_r.plot(sample_ids, residual[:, 3], marker="o", label="dRx")
    ax_components_r.plot(sample_ids, residual[:, 4], marker="o", label="dRy")
    ax_components_r.plot(sample_ids, residual[:, 5], marker="o", label="dRz")
    ax_components_r.set_title("Rotation components")
    ax_components_r.set_xlabel("sample")
    ax_components_r.set_ylabel("deg")
    ax_components_r.legend()
    ax_components_r.grid(True)

    ax_reprojection.plot(sample_ids, reprojection, marker="o", color="tab:green")
    ax_reprojection.set_title("Reprojection error")
    ax_reprojection.set_xlabel("sample")
    ax_reprojection.set_ylabel("px")
    ax_reprojection.grid(True)

    ax_margin.plot(sample_ids, margins, marker="o", color="tab:purple")
    ax_margin.axhline(DECISION_MARGIN_THRESHOLD, color="tab:red", linestyle="--", linewidth=1)
    ax_margin.set_title("Decision margin")
    ax_margin.set_xlabel("sample")
    ax_margin.grid(True)

    ax_3d.scatter(points[:, 0], points[:, 1], points[:, 2], c="tab:gray", s=30, label="samples")
    mean = analysis.mean_transform
    ax_3d.scatter([mean[0, 3]], [mean[1, 3]], [mean[2, 3]], c="k", s=60, label="mean")
    draw_frame_3d(ax_3d, np.eye(4, dtype=np.float64), "base", 0.05, linewidth=2.5)
    for i, sample in enumerate(samples):
        draw_frame_3d(ax_3d, sample.t_base_tag, f"s{i}", 0.012, linewidth=0.8, alpha=0.35)
    draw_frame_3d(ax_3d, mean, "tag_mean", 0.035, linewidth=2.5)

    example = mean.copy()
    example[:3, 3] = mean[:3, 3] + mean[:3, :3] @ EXAMPLE_POINT_IN_TAG_M
    draw_frame_3d(ax_3d, example, "example", 0.025, linewidth=2.0)

    all_points = np.vstack([points, np.zeros((1, 3)), mean[:3, 3], example[:3, 3]])
    set_axes_equal(ax_3d, all_points)
    ax_3d.set_title("T_base_tag samples")
    ax_3d.set_xlabel("base x (m)")
    ax_3d.set_ylabel("base y (m)")
    ax_3d.set_zlabel("base z (m)")
    ax_3d.legend(loc="best")

    fig.suptitle(
        "tag36h11 id=0 eye-to-hand consistency "
        f"(n={len(samples)}, mean trans={np.mean(analysis.translation_residual_mm):.2f} mm, "
        f"mean rot={np.mean(analysis.rotation_residual_deg):.3f} deg)"
    )
    fig.tight_layout()
    plt.show()


def print_summary(samples: list[Sample], analysis: Analysis) -> None:
    print("\nCaptured samples:", len(samples))
    print("Mean T_base_tag:")
    print(np.array2string(analysis.mean_transform, precision=8, suppress_small=False))
    print("\nTranslation residual (mm):")
    print(
        f"  mean={np.mean(analysis.translation_residual_mm):.3f}, "
        f"std={np.std(analysis.translation_residual_mm):.3f}, "
        f"max={np.max(analysis.translation_residual_mm):.3f}"
    )
    print("Rotation residual (deg):")
    print(
        f"  mean={np.mean(analysis.rotation_residual_deg):.4f}, "
        f"std={np.std(analysis.rotation_residual_deg):.4f}, "
        f"max={np.max(analysis.rotation_residual_deg):.4f}"
    )
    reprojection = np.asarray([sample.reprojection_error_px for sample in samples], dtype=np.float64)
    print("Reprojection error (px):")
    print(f"  mean={np.mean(reprojection):.3f}, std={np.std(reprojection):.3f}, max={np.max(reprojection):.3f}")
    print(f"\nCSV written to: {CSV_FILE}")


def main() -> None:
    t_base_camera = load_transform(HANDEYE_FILE)
    detector = create_apriltag_detector()
    samples: list[Sample] = []

    camera = RealSenseCamera()
    camera.connect()
    if camera.intrinsics is None or camera.dist_coeffs is None:
        camera.disconnect()
        raise RuntimeError("Camera intrinsics or distortion coefficients are unavailable.")

    raw_intrinsics = np.asarray(camera.intrinsics, dtype=np.float64)
    raw_distortion = np.asarray(camera.dist_coeffs, dtype=np.float64).reshape(-1, 1)
    optimal_intrinsics: Optional[np.ndarray] = None
    latest_target: Optional[Any] = None
    latest_t_camera_tag: Optional[np.ndarray] = None
    latest_t_base_tag: Optional[np.ndarray] = None
    latest_reprojection_error = float("nan")

    print("\nEye-to-hand tag pose consistency test")
    print(f"  tag: {TAG_FAMILY}, id={TARGET_TAG_ID}, size={TAG_SIZE_M * 1000.0:.1f} mm")
    print(f"  hand-eye: {HANDEYE_FILE}")
    print("  keys: Enter=capture, Space=finish+analyze, Esc=quit")

    try:
        while True:
            color_bgr, _ = camera.get_data()
            height, width = color_bgr.shape[:2]
            if optimal_intrinsics is None:
                optimal_intrinsics, _ = cv2.getOptimalNewCameraMatrix(
                    raw_intrinsics,
                    raw_distortion,
                    (width, height),
                    1.0,
                    (width, height),
                )

            assert optimal_intrinsics is not None
            undistorted = cv2.undistort(color_bgr, raw_intrinsics, raw_distortion, None, optimal_intrinsics)
            detections, target = detect_target(detector, undistorted, optimal_intrinsics)

            latest_target = target
            latest_t_camera_tag = None
            latest_t_base_tag = None
            latest_reprojection_error = float("nan")

            for detection in detections:
                draw_detection_overlay(
                    undistorted,
                    detection,
                    optimal_intrinsics,
                    is_target=target is not None and detection is target,
                )

            if target is not None:
                latest_t_camera_tag = detection_to_transform(target)
                latest_t_base_tag = t_base_camera @ latest_t_camera_tag
                latest_reprojection_error = reprojection_error_px(target, optimal_intrinsics)

            lines = [
                f"target: {TAG_FAMILY} id={TARGET_TAG_ID} size={TAG_SIZE_M * 1000.0:.1f}mm",
                f"samples: {len(samples)}   keys: Enter=capture, Space=analyze, Esc=quit",
                "status: TARGET OK" if target is not None else "status: NO VALID TARGET",
            ]
            if latest_t_base_tag is not None:
                pose = transform_to_rvec_pose(latest_t_base_tag)
                lines.append(f"T_base_tag t(m): x={pose[0]:+.4f} y={pose[1]:+.4f} z={pose[2]:+.4f}")
                lines.append(f"rvec(rad): rx={pose[3]:+.4f} ry={pose[4]:+.4f} rz={pose[5]:+.4f}")
                lines.append(f"repr={latest_reprojection_error:.3f}px dm={float(target.decision_margin):.1f}")

            draw_text_lines(undistorted, lines)
            cv2.imshow("tag36h11 id=0 eye-to-hand consistency", undistorted)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                print("Quit without analysis.")
                break

            if key in (13, 10):
                if latest_target is None or latest_t_camera_tag is None or latest_t_base_tag is None:
                    print("No valid target to capture.")
                    continue

                sample = Sample(
                    index=len(samples),
                    timestamp=time.time(),
                    decision_margin=float(latest_target.decision_margin),
                    reprojection_error_px=latest_reprojection_error,
                    t_camera_tag=latest_t_camera_tag.copy(),
                    t_base_tag=latest_t_base_tag.copy(),
                )
                samples.append(sample)
                print(
                    f"Captured sample {sample.index}: "
                    f"T_base_tag translation={sample.t_base_tag[:3, 3]}, "
                    f"repr={sample.reprojection_error_px:.3f}px"
                )

            if key == 32:
                if len(samples) < 2:
                    print("Need at least 2 samples for consistency analysis.")
                    continue
                analysis = analyze_samples(samples)
                write_csv(samples, analysis, CSV_FILE)
                print_summary(samples, analysis)
                cv2.destroyWindow("tag36h11 id=0 eye-to-hand consistency")
                show_analysis(samples, analysis)
                break
    finally:
        camera.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
