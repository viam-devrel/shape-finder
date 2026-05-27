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
