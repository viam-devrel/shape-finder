import asyncio

import cv2
import numpy as np
import pytest
from PIL import Image
from viam.media.utils.pil import pil_to_viam_image
from viam.media.video import CameraMimeType, ViamImage

from models.detector import Detector, _to_proto, _viam_image_to_rgb
from models.shape_detection import ShapeDetection


def test_to_proto_maps_fields():
    sd = ShapeDetection(1, 2, 3, 4, "square-red", 0.95)
    d = _to_proto(sd)
    assert (d.x_min, d.y_min, d.x_max, d.y_max) == (1, 2, 3, 4)
    assert d.class_name == "square-red"
    assert abs(d.confidence - 0.95) < 1e-6


def test_viam_image_to_rgb_roundtrip():
    arr = np.zeros((10, 12, 3), dtype=np.uint8)
    arr[:, :, 0] = 200  # red channel
    viam_img = pil_to_viam_image(Image.fromarray(arr), CameraMimeType.PNG)
    rgb = _viam_image_to_rgb(viam_img)
    assert rgb.shape == (10, 12, 3)
    assert int(rgb[0, 0, 0]) == 200  # PNG is lossless


def test_get_detections_on_synthetic_image():
    arr = np.full((240, 240, 3), 128, dtype=np.uint8)
    cv2.rectangle(arr, (40, 40), (120, 120), (220, 20, 20), -1)  # red square
    viam_img = pil_to_viam_image(Image.fromarray(arr), CameraMimeType.PNG)

    det = Detector("test-detector")
    results = asyncio.run(det.get_detections(viam_img))

    assert len(results) == 1
    assert results[0].class_name == "square-red"


def test_get_properties_reports_detections_only():
    det = Detector("test-detector")
    props = asyncio.run(det.get_properties())
    assert props.detections_supported is True
    assert props.classifications_supported is False
    assert props.object_point_clouds_supported is False


def test_get_detections_returns_empty_on_undecodable_image():
    bad = ViamImage(b"not-an-image", CameraMimeType.JPEG)
    det = Detector("test-detector")
    assert asyncio.run(det.get_detections(bad)) == []


def _red_square_image():
    arr = np.full((240, 240, 3), 128, dtype=np.uint8)
    cv2.rectangle(arr, (40, 40), (120, 120), (220, 20, 20), -1)
    return pil_to_viam_image(Image.fromarray(arr), CameraMimeType.PNG)


class _FakeCamera:
    """Minimal stand-in exposing the current Camera.get_images() contract."""

    def __init__(self, images):
        self._images = images

    async def get_images(self, *args, **kwargs):
        return self._images, None


def _detector_with_camera(images):
    det = Detector("test-detector")
    det.camera = _FakeCamera(images)
    det.camera_name = "cam1"
    return det


def test_get_detections_from_camera_uses_get_images():
    det = _detector_with_camera([_red_square_image()])
    results = asyncio.run(det.get_detections_from_camera("cam1"))
    assert len(results) == 1
    assert results[0].class_name == "square-red"


def test_capture_all_from_camera_returns_image_and_detections():
    det = _detector_with_camera([_red_square_image()])
    result = asyncio.run(
        det.capture_all_from_camera(
            "cam1", return_image=True, return_detections=True
        )
    )
    assert result.image is not None
    assert len(result.detections) == 1
    assert result.detections[0].class_name == "square-red"


def test_camera_method_errors_when_no_images_returned():
    det = _detector_with_camera([])
    with pytest.raises(ValueError, match="no images"):
        asyncio.run(det.get_detections_from_camera("cam1"))
