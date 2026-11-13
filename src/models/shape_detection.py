from __future__ import annotations

import math
from dataclasses import dataclass, field
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
    "purple": [(131, 169)],
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
    # Image coordinates (y down): positive is clockwise on screen.
    # Rectangle: long side vs +x, in (-90, 90]. Square: in (-45, 45].
    angle_deg: float = 0.0
    corners: List[Tuple[float, float]] = field(default_factory=list)


def detect_shapes(rgb: np.ndarray) -> List[ShapeDetection]:
    if rgb.dtype != np.uint8:
        raise ValueError(f"detect_shapes expects uint8 RGB, got {rgb.dtype}")
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

        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        if min(rw, rh) == 0:
            continue
        is_square = max(rw, rh) / min(rw, rh) <= 1 + SQUARE_TOL
        shape = "square" if is_square else "rectangle"
        corners = cv2.boxPoints(rect)

        x, y, w, h = cv2.boundingRect(approx)
        shape_mask = np.zeros_like(sat_mask)
        cv2.drawContours(shape_mask, [contour], -1, 255, thickness=cv2.FILLED)
        color, confidence = _predominant_color(
            hsv, cv2.bitwise_and(shape_mask, sat_mask)
        )
        detections.append(
            ShapeDetection(
                x,
                y,
                x + w,
                y + h,
                f"{shape}-{color}",
                confidence,
                angle_deg=_fold_angle(corners, is_square),
                corners=[(float(px), float(py)) for px, py in corners],
            )
        )
    return detections


def _fold_angle(corners: np.ndarray, is_square: bool) -> float:
    # Use the box's long edge rather than minAreaRect's own angle, whose range
    # differs between OpenCV versions.
    e1 = corners[1] - corners[0]
    e2 = corners[2] - corners[1]
    edge = e1 if np.hypot(*e1) >= np.hypot(*e2) else e2
    angle = math.degrees(math.atan2(float(edge[1]), float(edge[0])))
    period = 90.0 if is_square else 180.0
    angle %= period
    if angle > period / 2:
        angle -= period
    return angle


def _predominant_color(hsv: np.ndarray, mask: np.ndarray) -> Tuple[str, float]:
    hue = hsv[:, :, 0][mask > 0]
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
