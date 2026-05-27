from google.protobuf.struct_pb2 import Struct
from viam.proto.app.robot import ComponentConfig

from models.detector import Detector


def _config(attrs: dict) -> ComponentConfig:
    s = Struct()
    s.update(attrs)
    return ComponentConfig(attributes=s)


def test_validate_requires_camera_when_set():
    required, optional = Detector.validate_config(_config({"camera_name": "cam1"}))
    assert required == ["cam1"]
    assert optional == []


def test_validate_no_camera_when_absent():
    required, optional = Detector.validate_config(_config({}))
    assert required == []
    assert optional == []


import asyncio

import pytest


def test_camera_method_errors_without_camera():
    det = Detector("test-detector")
    det.camera = None
    det.camera_name = ""
    with pytest.raises(ValueError, match="camera_name"):
        asyncio.run(det.get_detections_from_camera("whatever"))
