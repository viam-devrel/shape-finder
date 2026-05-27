# Shape Finder Detector Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the `devrel:shape-finder:detector` Viam vision service so it finds colored squares and rectangles in an image with OpenCV and returns them as `Detection`s labeled `"<shape>-<color>"`.

**Architecture:** A pure, framework-free `detect_shapes(rgb)` function (OpenCV + numpy) does all the vision work and is unit-tested with synthetic images. The Viam `Detector` class is a thin adapter: it decodes `ViamImage` to RGB, calls `detect_shapes`, and maps results to `Detection` protos. Camera-based methods use an optional `camera_name` dependency.

**Tech Stack:** Python 3.11, Viam Python SDK 0.76.0, OpenCV (`opencv-python-headless`), numpy, pytest.

**Design doc:** `docs/plans/2026-05-26-shape-finder-detector-design.md`

**Conventions:**
- Source lives under `src/`; `main.py` imports as `from models.detector import Detector`, so intra-package imports use the `models.` prefix.
- Run Python/tests with the module venv: `venv/bin/python`.
- Commit after each task.

---

## Task 0: Dependencies and test setup

**Files:**
- Modify: `requirements.txt`
- Create: `tests/conftest.py`

**Step 1: Add runtime deps to `requirements.txt`**

Final contents:

```
viam-sdk==0.76.0

typing-extensions
numpy
opencv-python-headless
```

**Step 2: Install deps + pytest into the venv**

Run:
```bash
./setup.sh
venv/bin/python -m pip install -q numpy opencv-python-headless pytest
```
Expected: installs succeed. (pytest is a dev-only tool, intentionally kept out of `requirements.txt` so the deployed module bundle stays lean.)

**Step 3: Make `src/` importable in tests**

Create `tests/conftest.py`:

```python
import sys
from pathlib import Path

# Allow `from models.shape_detection import ...` in tests, matching how
# main.py imports modules at runtime.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
```

**Step 4: Verify pytest collects nothing yet (sanity)**

Run: `venv/bin/python -m pytest -q`
Expected: "no tests ran" (exit ok), no import errors.

**Step 5: Commit**

```bash
git add requirements.txt tests/conftest.py
git commit -m "chore: add opencv/numpy deps and pytest setup"
```

---

## Task 1: `detect_shapes` returns empty on a non-colored image

**Files:**
- Create: `src/models/shape_detection.py`
- Test: `tests/test_shape_detection.py`

**Step 1: Write the failing test**

```python
import numpy as np

from models.shape_detection import detect_shapes


def gray_image(h=200, w=200):
    return np.full((h, w, 3), 128, dtype=np.uint8)  # neutral gray, RGB


def test_all_gray_image_has_no_detections():
    assert detect_shapes(gray_image()) == []
```

**Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_shape_detection.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'models.shape_detection'`.

**Step 3: Write minimal implementation**

Create `src/models/shape_detection.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
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
    "purple": [(131, 160)],
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


def detect_shapes(rgb: np.ndarray) -> List[ShapeDetection]:
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

        x, y, w, h = cv2.boundingRect(approx)
        if h == 0:
            continue
        aspect = w / h
        shape = (
            "square"
            if (1 - SQUARE_TOL) <= aspect <= (1 + SQUARE_TOL)
            else "rectangle"
        )

        color, confidence = _predominant_color(hsv, sat_mask, x, y, w, h)
        detections.append(
            ShapeDetection(x, y, x + w, y + h, f"{shape}-{color}", confidence)
        )
    return detections


def _predominant_color(
    hsv: np.ndarray, sat_mask: np.ndarray, x: int, y: int, w: int, h: int
) -> Tuple[str, float]:
    roi_hsv = hsv[y : y + h, x : x + w]
    roi_mask = sat_mask[y : y + h, x : x + w]
    hue = roi_hsv[:, :, 0][roi_mask > 0]
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
```

**Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_shape_detection.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/models/shape_detection.py tests/test_shape_detection.py
git commit -m "feat: detect_shapes skeleton returns no detections on blank image"
```

---

## Task 2: Detect a red square

**Files:**
- Test: `tests/test_shape_detection.py`

**Step 1: Add the failing test**

```python
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
    # Bounding box within a few px of the drawn rectangle.
    assert abs(d.x_min - 40) <= 3
    assert abs(d.y_min - 40) <= 3
    assert abs(d.x_max - 120) <= 3
    assert abs(d.y_max - 120) <= 3
    assert d.confidence > 0.9
```

Add `import cv2` to the test file imports if not present.

**Step 2: Run test**

Run: `venv/bin/python -m pytest tests/test_shape_detection.py::test_detects_red_square -q`
Expected: PASS (implementation from Task 1 already supports this).

If it FAILS, debug with superpowers:systematic-debugging — likely culprits are the `S_MIN`/`V_MIN` thresholds vs. the chosen RGB, or `MIN_AREA`. Do not loosen assertions to force a pass.

**Step 3: Commit**

```bash
git add tests/test_shape_detection.py
git commit -m "test: detect red square"
```

---

## Task 3: Distinguish rectangle from square, and a second color

**Files:**
- Test: `tests/test_shape_detection.py`

**Step 1: Add the failing test**

```python
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
```

**Step 2: Run tests**

Run: `venv/bin/python -m pytest tests/test_shape_detection.py -q`
Expected: PASS.

**Step 3: Commit**

```bash
git add tests/test_shape_detection.py
git commit -m "test: distinguish rectangle vs square and classify multiple colors"
```

---

## Task 4: Ignore shapes below MIN_AREA

**Files:**
- Test: `tests/test_shape_detection.py`

**Step 1: Add the failing test**

```python
def test_ignores_tiny_shapes():
    # 10x10 = 100 px area, below MIN_AREA (500).
    img = filled_rect((220, 20, 20), x=20, y=20, w=10, h=10)
    assert detect_shapes(img) == []
```

**Step 2: Run test**

Run: `venv/bin/python -m pytest tests/test_shape_detection.py::test_ignores_tiny_shapes -q`
Expected: PASS.

**Step 3: Commit**

```bash
git add tests/test_shape_detection.py
git commit -m "test: ignore shapes below minimum area"
```

---

## Task 5: Detection proto mapping + image decode helpers

**Files:**
- Modify: `src/models/detector.py` (add module-level helpers above the class)
- Test: `tests/test_detector_helpers.py`

**Step 1: Write the failing test**

```python
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
```

**Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_detector_helpers.py -q`
Expected: FAIL — `ImportError: cannot import name '_to_proto'`.

**Step 3: Add helpers to `src/models/detector.py`**

Add these imports near the top (keep existing imports):

```python
import numpy as np
from viam.media.utils.pil import viam_to_pil_image
```

Add module-level helpers above the `Detector` class:

```python
from models.shape_detection import ShapeDetection, detect_shapes


def _viam_image_to_rgb(image: ViamImage) -> np.ndarray:
    pil = viam_to_pil_image(image)
    return np.array(pil.convert("RGB"))


def _to_proto(sd: ShapeDetection) -> Detection:
    return Detection(
        x_min=sd.x_min,
        y_min=sd.y_min,
        x_max=sd.x_max,
        y_max=sd.y_max,
        confidence=sd.confidence,
        class_name=sd.label,
    )
```

**Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_detector_helpers.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/models/detector.py tests/test_detector_helpers.py
git commit -m "feat: add ViamImage->RGB and ShapeDetection->Detection helpers"
```

---

## Task 6: Implement `get_detections` and `get_properties`

**Files:**
- Modify: `src/models/detector.py`
- Test: `tests/test_detector_helpers.py`

**Step 1: Write the failing test**

```python
import asyncio


def test_get_detections_on_synthetic_image():
    import cv2
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
```

Add `from models.detector import Detector` to the test imports.

**Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_detector_helpers.py -q`
Expected: FAIL — `get_detections` raises `NotImplementedError`.

> Note: `Detector("test-detector")` constructs the resource directly (no `reconfigure`), which is fine for `get_detections`/`get_properties` since neither touches config or the camera.

**Step 3: Replace the method bodies in `src/models/detector.py`**

`get_detections`:

```python
    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        try:
            rgb = _viam_image_to_rgb(image)
        except Exception as exc:  # noqa: BLE001 - never crash the service on a bad frame
            self.logger.error(f"Failed to decode image: {exc}")
            return []
        return [_to_proto(sd) for sd in detect_shapes(rgb)]
```

`get_properties`:

```python
    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> Vision.Properties:
        return Vision.Properties(
            classifications_supported=False,
            detections_supported=True,
            object_point_clouds_supported=False,
        )
```

**Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_detector_helpers.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/models/detector.py tests/test_detector_helpers.py
git commit -m "feat: implement get_detections and get_properties"
```

---

## Task 7: Camera config — `validate_config` and `reconfigure`

**Files:**
- Modify: `src/models/detector.py`
- Test: `tests/test_detector_config.py`

**Step 1: Write the failing test**

```python
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
```

**Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_detector_config.py -q`
Expected: FAIL — current `validate_config` returns `([], [])` unconditionally, so `test_validate_requires_camera_when_set` fails.

**Step 3: Implement config handling in `src/models/detector.py`**

Add imports:

```python
from typing import cast
from viam.components.camera import Camera
```

Replace `validate_config`:

```python
    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        fields = config.attributes.fields
        camera_name = (
            fields["camera_name"].string_value if "camera_name" in fields else ""
        )
        if camera_name:
            return [camera_name], []
        return [], []
```

Add a `reconfigure` method (place after `new`):

```python
    def reconfigure(
        self,
        config: ComponentConfig,
        dependencies: Mapping[ResourceName, ResourceBase],
    ) -> None:
        fields = config.attributes.fields
        self.camera_name = (
            fields["camera_name"].string_value if "camera_name" in fields else ""
        )
        self.camera: Optional[Camera] = None
        if self.camera_name:
            self.camera = cast(
                Camera, dependencies[Camera.get_resource_name(self.camera_name)]
            )
```

**Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_detector_config.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/models/detector.py tests/test_detector_config.py
git commit -m "feat: parse camera_name config and wire camera dependency"
```

---

## Task 8: Camera-based methods

**Files:**
- Modify: `src/models/detector.py`
- Test: `tests/test_detector_config.py`

**Step 1: Write the failing test (no-camera error path)**

```python
import asyncio
import pytest


def test_camera_method_errors_without_camera():
    det = Detector("test-detector")
    det.camera = None
    det.camera_name = ""
    with pytest.raises(ValueError, match="camera_name"):
        asyncio.run(det.get_detections_from_camera("whatever"))
```

**Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_detector_config.py::test_camera_method_errors_without_camera -q`
Expected: FAIL — currently raises `NotImplementedError`, not `ValueError`.

**Step 3: Implement the camera methods in `src/models/detector.py`**

Add imports:

```python
from viam.services.vision import CaptureAllResult
```
(`CaptureAllResult` is already exported by `from viam.services.vision import *`, but import it explicitly for clarity.)

Add a private resolver and replace the two camera methods:

```python
    def _resolve_camera(self, camera_name: str) -> Camera:
        if self.camera is None:
            raise ValueError(
                "No camera configured; set 'camera_name' in the service config "
                "to use camera-based methods"
            )
        return self.camera

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        camera = self._resolve_camera(camera_name)
        image = await camera.get_image()
        return await self.get_detections(image)

    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> CaptureAllResult:
        camera = self._resolve_camera(camera_name)
        image = await camera.get_image()
        result = CaptureAllResult()
        if return_image:
            result.image = image
        if return_detections:
            result.detections = await self.get_detections(image)
        return result
```

**Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_detector_config.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/models/detector.py tests/test_detector_config.py
git commit -m "feat: implement camera-based detection methods"
```

---

## Task 9: Full suite, lint check, and review

**Files:** none (verification only)

**Step 1: Run the whole test suite**

Run: `venv/bin/python -m pytest -q`
Expected: all tests PASS.

**Step 2: Byte-compile the source to catch syntax/import errors**

Run: `venv/bin/python -m compileall src`
Expected: no errors.

**Step 3: Confirm the module imports cleanly**

Run: `cd src && ../venv/bin/python -c "import main"`
Expected: no output, exit 0. (Confirms `main.py` -> `models.detector` -> `models.shape_detection` import chain is intact.)

**Step 4: Request code review**

Use the superpowers:requesting-code-review skill to review the diff against `docs/plans/2026-05-26-shape-finder-detector-design.md`.

**Step 5: Commit any review fixes, then finish**

Use superpowers:finishing-a-development-branch to decide how to integrate.

---

## Manual verification (optional, after automated tests pass)

Reload the module onto a machine with a camera and a configured `shape-finder:detector` service (with `camera_name` set), then open the service's **Test** panel in the Viam app and confirm bounding boxes with `"<shape>-<color>"` labels appear over colored squares/rectangles in view.
