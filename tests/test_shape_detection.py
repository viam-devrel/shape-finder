import cv2
import numpy as np

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
