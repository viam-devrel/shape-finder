import numpy as np
from PIL import Image
from viam.media.utils.pil import pil_to_viam_image
from viam.media.video import CameraMimeType

from models.detector import _to_proto, _viam_image_to_rgb
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
