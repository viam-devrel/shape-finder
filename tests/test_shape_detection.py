import cv2
import numpy as np

from models.shape_detection import detect_shapes


def gray_image(h=200, w=200):
    return np.full((h, w, 3), 128, dtype=np.uint8)  # neutral gray, RGB


def test_all_gray_image_has_no_detections():
    assert detect_shapes(gray_image()) == []
