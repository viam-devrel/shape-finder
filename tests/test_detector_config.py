import asyncio

import pytest
from google.protobuf.struct_pb2 import Struct
from viam.components.camera import Camera
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


def test_camera_method_errors_without_camera():
    det = Detector("test-detector")
    det.camera = None
    det.camera_name = ""
    with pytest.raises(ValueError, match="camera_name"):
        asyncio.run(det.get_detections_from_camera("whatever"))


def test_new_initializes_state_without_camera():
    det = Detector.new(_config({}), {})
    assert det.camera is None
    assert det.camera_name == ""
    with pytest.raises(ValueError, match="camera_name"):
        asyncio.run(det.get_detections_from_camera("cam1"))


def test_new_wires_configured_camera():
    deps = {Camera.get_resource_name("cam1"): object()}
    det = Detector.new(_config({"camera_name": "cam1"}), deps)
    assert det.camera_name == "cam1"
    assert det.camera is not None


def test_resolve_camera_rejects_mismatched_name():
    deps = {Camera.get_resource_name("cam1"): object()}
    det = Detector.new(_config({"camera_name": "cam1"}), deps)
    with pytest.raises(ValueError, match="does not match"):
        asyncio.run(det.get_detections_from_camera("other-cam"))
