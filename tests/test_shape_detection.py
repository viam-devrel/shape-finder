import cv2
import numpy as np
import pytest

from models.shape_detection import detect_shapes


def gray_image(h=200, w=200):
    return np.full((h, w, 3), 128, dtype=np.uint8)  # neutral gray, RGB


def test_all_gray_image_has_no_detections():
    assert detect_shapes(gray_image()) == []


def filled_rect(color_rgb, x, y, w, h, canvas=(240, 240)):
    img = np.full((canvas[0], canvas[1], 3), 128, dtype=np.uint8)
    cv2.rectangle(img, (x, y), (x + w, y + h), color_rgb, thickness=-1)
    return img


def test_detects_red_square():
    img = filled_rect((220, 20, 20), x=40, y=40, w=80, h=80)
    dets = detect_shapes(img)

    assert len(dets) == 1
    d = dets[0]
    assert d.label == "square-red"
    assert abs(d.x_min - 40) <= 3
    assert abs(d.y_min - 40) <= 3
    assert abs(d.x_max - 120) <= 3
    assert abs(d.y_max - 120) <= 3
    assert d.confidence > 0.9


def test_detects_blue_rectangle():
    img = filled_rect((20, 20, 220), x=30, y=60, w=140, h=60)
    dets = detect_shapes(img)

    assert len(dets) == 1
    assert dets[0].label == "rectangle-blue"


def test_detects_two_shapes_of_different_colors():
    img = np.full((240, 320, 3), 128, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (100, 100), (220, 20, 20), -1)   # red square
    cv2.rectangle(img, (160, 40), (300, 100), (20, 180, 20), -1)  # green rect
    labels = sorted(d.label for d in detect_shapes(img))
    assert labels == ["rectangle-green", "square-red"]


def test_ignores_tiny_shapes():
    # 10x10 = 100 px area, below MIN_AREA (500).
    img = filled_rect((220, 20, 20), x=20, y=20, w=10, h=10)
    assert detect_shapes(img) == []


def test_rejects_non_uint8_input():
    img = np.zeros((50, 50, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="uint8"):
        detect_shapes(img)


def test_classifies_color_in_former_hue_gap():
    # RGB(200, 58, 150) converts to OpenCV hue ~161, which previously fell
    # in the 161-169 gap (purple ended at 160, red wrap started at 170).
    img = filled_rect((200, 58, 150), x=40, y=40, w=80, h=80)
    dets = detect_shapes(img)
    assert len(dets) == 1
    # Previously labeled "square-unknown"; now should be "square-purple".
    assert not dets[0].label.endswith("-unknown")
    assert dets[0].label == "square-purple"


RED = (220, 20, 20)


def rotated_rect(color_rgb, w, h, angle, center=(150, 150), canvas=(300, 300)):
    img = np.full((canvas[0], canvas[1], 3), 128, dtype=np.uint8)
    pts = cv2.boxPoints((center, (w, h), angle)).astype(np.int32)
    cv2.fillPoly(img, [pts], color_rgb)
    return img


def only_detection(img):
    dets = detect_shapes(img)
    assert len(dets) == 1
    return dets[0]


def angle_close(a, b, period=180.0, tol=1.0):
    # Angles near the fold limits may come out at either end of the range.
    d = (a - b) % period
    return min(d, period - d) < tol


def test_rotated_rectangle_reports_angle_and_corners():
    d = only_detection(rotated_rect(RED, 120, 50, 30))
    assert d.label == "rectangle-red"
    assert angle_close(d.angle_deg, 30)
    assert len(d.corners) == 4


def test_rectangle_at_45_degrees_is_not_a_square():
    d = only_detection(rotated_rect(RED, 120, 50, 45))
    assert d.label == "rectangle-red"
    assert angle_close(d.angle_deg, 45)


def test_square_at_45_degrees_folds_to_quarter_turn():
    d = only_detection(rotated_rect(RED, 80, 80, 45))
    assert d.label == "square-red"
    assert -45 < d.angle_deg <= 45
    assert angle_close(d.angle_deg, 45, period=90.0)


def test_axis_aligned_rectangle_angles():
    wide = only_detection(rotated_rect(RED, 120, 50, 0))
    tall = only_detection(rotated_rect(RED, 50, 120, 0))
    assert angle_close(wide.angle_deg, 0)
    assert angle_close(tall.angle_deg, 90)
    assert -90 < tall.angle_deg <= 90


def test_color_sampled_from_shape_not_bounding_box():
    # The green square sits inside the 45° red rectangle's bounding box,
    # about 15 px clear of the rectangle itself (enough that the 5x5
    # morphology doesn't merge them).
    img = rotated_rect(RED, 160, 40, 45)
    cv2.rectangle(img, (175, 80), (220, 125), (20, 180, 20), -1)
    dets = sorted(detect_shapes(img), key=lambda d: d.label)
    assert [d.label for d in dets] == ["rectangle-red", "square-green"]
    assert dets[0].confidence > 0.95
