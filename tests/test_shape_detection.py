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
