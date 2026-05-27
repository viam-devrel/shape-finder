from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

# OpenCV HSV ranges: H in [0, 179], S and V in [0, 255].
# Each color maps to one or more hue bands (red wraps around 0/180).
COLOR_HUE_RANGES: Dict[str, List[Tuple[int, int]]] = {
    "red": [(0, 10), (170, 179)],
    "orange": [(11, 25)],
    "yellow": [(26, 34)],
    "green": [(35, 85)],
    "blue": [(86, 130)],
    "purple": [(131, 160)],
}

S_MIN = 80
V_MIN = 50
MIN_AREA = 500
EPSILON_FACTOR = 0.04
SQUARE_TOL = 0.15


@dataclass
class ShapeDetection:
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    label: str
    confidence: float


def detect_shapes(rgb: np.ndarray) -> List[ShapeDetection]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat_mask = cv2.inRange(hsv, (0, S_MIN, V_MIN), (179, 255, 255))

    kernel = np.ones((5, 5), np.uint8)
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, kernel)
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    detections: List[ShapeDetection] = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_AREA:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, EPSILON_FACTOR * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        x, y, w, h = cv2.boundingRect(approx)
        if h == 0:
            continue
        aspect = w / h
        shape = (
            "square"
            if (1 - SQUARE_TOL) <= aspect <= (1 + SQUARE_TOL)
            else "rectangle"
        )

        color, confidence = _predominant_color(hsv, sat_mask, x, y, w, h)
        detections.append(
            ShapeDetection(x, y, x + w, y + h, f"{shape}-{color}", confidence)
        )
    return detections


def _predominant_color(
    hsv: np.ndarray, sat_mask: np.ndarray, x: int, y: int, w: int, h: int
) -> Tuple[str, float]:
    roi_hsv = hsv[y : y + h, x : x + w]
    roi_mask = sat_mask[y : y + h, x : x + w]
    hue = roi_hsv[:, :, 0][roi_mask > 0]
    total = int(hue.size)
    if total == 0:
        return "unknown", 0.0

    best_color = "unknown"
    best_count = 0
    for name, bands in COLOR_HUE_RANGES.items():
        count = 0
        for low, high in bands:
            count += int(np.count_nonzero((hue >= low) & (hue <= high)))
        if count > best_count:
            best_count = count
            best_color = name

    return best_color, best_count / total
