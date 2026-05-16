#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Checkerboard corner detection for calibration data collection."""

from typing import Optional, Tuple

import cv2
import numpy as np


class CheckerboardExtractor:
    """Detect checkerboard inner corners."""

    def __init__(self, checkerboard_size: Tuple[int, int] = (5, 5), square_size: float = 0.024) -> None:
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        self.refine_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )

    def detect_corners(
        self,
        gray_image: np.ndarray,
        refine: bool = True,
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect checkerboard corners, optionally refined to sub-pixel precision."""
        success, corners = cv2.findChessboardCorners(
            gray_image,
            self.checkerboard_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if not success:
            return False, None, None

        if refine:
            corners_refined = cv2.cornerSubPix(
                gray_image,
                corners,
                (5, 5),
                (-1, -1),
                self.refine_criteria,
            )
            return True, corners, corners_refined

        return True, corners, corners
