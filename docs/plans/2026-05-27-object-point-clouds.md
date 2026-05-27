# Object Point Clouds Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `get_object_point_clouds` on `devrel:shape-finder:detector` so it returns one `PointCloudObject` (point subset + 3D AABB geometry labeled `"<shape>-<color>"`) per detected colored shape, by projecting the source camera's point cloud into image space and selecting points inside each 2D box.

**Architecture:** A new pure, framework-free module `src/models/point_cloud.py` does PCD I/O (via `pypcd4`), pinhole projection, box selection, and AABB computation — all numpy, all unit-testable. The `Detector.get_object_point_clouds` method is a thin adapter: resolve camera, fetch frame + cloud + intrinsics, run the existing `detect_shapes`, then assemble `PointCloudObject`s. OpenCV's role is unchanged (the 2D detection).

**Tech Stack:** Python 3.11, Viam Python SDK 0.76.0, OpenCV, numpy, `pypcd4`, pytest.

**Design doc:** `docs/plans/2026-05-27-object-point-clouds-design.md`

**Conventions:**
- Source under `src/`; intra-package imports use the `models.` prefix.
- Run Python/tests with `venv/bin/python`. Commit after each task.
- Points are treated as millimeters (Viam convention); `dims_mm`/`Pose` need no scaling.

**Verified facts (viam-sdk 0.76.0 + pypcd4 1.4.3):**
- `Camera.get_point_cloud()` -> `(bytes, str)`; `Camera.get_properties()` -> response with `.supports_pcd` and `.intrinsic_parameters` (`focal_x_px, focal_y_px, center_x_px, center_y_px, width_px, height_px`).
- `PointCloudObject(point_cloud: bytes, geometries: GeometriesInFrame)`; `GeometriesInFrame(reference_frame: str, geometries: [Geometry])`; `Geometry(center: Pose, box: RectangularPrism, label: str)`; `RectangularPrism(dims_mm: Vector3)`; `Pose(x,y,z,o_x,o_y,o_z,theta)`.
- pypcd4: parse with `PointCloud.from_fileobj(io.BytesIO(b)).numpy(("x","y","z"))`; build with `PointCloud.from_xyz_points(arr_float32)`; encode by `pc.save(io.BytesIO())` then `.getvalue()`. Round-trip confirmed.

---

## Task 0: Add `pypcd4` dependency

**Files:**
- Modify: `requirements.txt`

**Step 1: Append `pypcd4` to `requirements.txt`**

Final contents:

```
viam-sdk==0.76.0

typing-extensions
numpy
opencv-python-headless
Pillow
pypcd4
```

**Step 2: Install**

Run: `venv/bin/python -m pip install -q pypcd4`
Expected: succeeds (already present from design verification; idempotent).

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add pypcd4 for point cloud I/O"
```

---

## Task 1: PCD parse/encode round-trip (`point_cloud.py`)

**Files:**
- Create: `src/models/point_cloud.py`
- Test: `tests/test_point_cloud.py`

**Step 1: Write the failing test**

```python
import numpy as np

from models.point_cloud import encode_pcd, parse_pcd


def test_pcd_encode_parse_roundtrip():
    pts = np.array([[1.0, 2.0, 3.0], [-400.0, -400.0, 1000.0]], dtype=np.float64)
    data = encode_pcd(pts)
    assert isinstance(data, (bytes, bytearray))
    out = parse_pcd(data)
    assert out.shape == (2, 3)
    assert np.allclose(out, pts, atol=1e-3)
```

**Step 2: Run, confirm FAIL**

Run: `venv/bin/python -m pytest tests/test_point_cloud.py -q`
Expected: `ModuleNotFoundError: No module named 'models.point_cloud'`.

**Step 3: Create `src/models/point_cloud.py`**

```python
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from pypcd4 import PointCloud

# A coplanar or single-point cluster would yield a zero-thickness box, which is
# not a useful obstacle geometry; clamp each dimension to at least this (mm).
MIN_DIM_MM = 1.0


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


def parse_pcd(pcd_bytes: bytes) -> np.ndarray:
    """Parse PCD bytes into an (N, 3) float64 array of XYZ points."""
    pc = PointCloud.from_fileobj(io.BytesIO(pcd_bytes))
    return np.asarray(pc.numpy(("x", "y", "z")), dtype=np.float64)


def encode_pcd(points_xyz: np.ndarray) -> bytes:
    """Encode an (N, 3) XYZ array into binary PCD bytes."""
    pc = PointCloud.from_xyz_points(np.asarray(points_xyz, dtype=np.float32))
    buffer = io.BytesIO()
    pc.save(buffer)
    return buffer.getvalue()
```

**Step 4: Run, confirm PASS**

Run: `venv/bin/python -m pytest tests/test_point_cloud.py -q`

**Step 5: Commit**

```bash
git add src/models/point_cloud.py tests/test_point_cloud.py
git commit -m "feat: PCD parse/encode helpers via pypcd4"
```

---

## Task 2: Pinhole projection (`project_to_pixels`)

**Files:**
- Modify: `src/models/point_cloud.py`
- Test: `tests/test_point_cloud.py`

**Step 1: Add the failing test**

```python
from models.point_cloud import Intrinsics, project_to_pixels


def test_project_to_pixels_pinhole():
    intr = Intrinsics(fx=100.0, fy=100.0, cx=120.0, cy=120.0)
    pts = np.array(
        [
            [0.0, 0.0, 1000.0],     # on axis -> (cx, cy)
            [-400.0, -400.0, 1000.0],  # -> (120 - 40, 120 - 40) = (80, 80)
            [0.0, 0.0, -5.0],       # behind camera -> invalid
        ],
        dtype=np.float64,
    )
    u, v, valid = project_to_pixels(pts, intr)
    assert valid.tolist() == [True, True, False]
    assert np.isclose(u[0], 120.0) and np.isclose(v[0], 120.0)
    assert np.isclose(u[1], 80.0) and np.isclose(v[1], 80.0)
```

**Step 2: Run, confirm FAIL** (`ImportError: cannot import name 'project_to_pixels'`).

Run: `venv/bin/python -m pytest tests/test_point_cloud.py::test_project_to_pixels_pinhole -q`

**Step 3: Add to `src/models/point_cloud.py`**

```python
def project_to_pixels(
    points_xyz: np.ndarray, intrinsics: Intrinsics
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project (N, 3) camera-frame points to pixel coords via a pinhole model.

    Returns (u, v, valid). `valid` marks points in front of the camera (Z > 0);
    u/v for invalid points are computed with a safe Z and should be ignored.
    """
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]
    valid = z > 0
    safe_z = np.where(valid, z, 1.0)
    u = intrinsics.fx * (x / safe_z) + intrinsics.cx
    v = intrinsics.fy * (y / safe_z) + intrinsics.cy
    return u, v, valid
```

**Step 4: Run, confirm PASS.**

**Step 5: Commit**

```bash
git add src/models/point_cloud.py tests/test_point_cloud.py
git commit -m "feat: pinhole projection of points to pixels"
```

---

## Task 3: Box selection (`points_in_box`)

**Files:**
- Modify: `src/models/point_cloud.py`
- Test: `tests/test_point_cloud.py`

**Step 1: Add the failing test**

```python
from models.point_cloud import points_in_box


def test_points_in_box_edges():
    u = np.array([40.0, 80.0, 120.0, 10.0])
    v = np.array([40.0, 80.0, 80.0, 80.0])
    valid = np.array([True, True, True, False])
    # box = (x_min, y_min, x_max, y_max); x_min inclusive, x_max exclusive
    mask = points_in_box(u, v, valid, (40, 40, 120, 120))
    # idx0: on min edge -> in; idx1: inside; idx2: on max edge -> out;
    # idx3: invalid -> out
    assert mask.tolist() == [True, True, False, False]
```

**Step 2: Run, confirm FAIL.**

**Step 3: Add to `src/models/point_cloud.py`**

```python
def points_in_box(
    u: np.ndarray,
    v: np.ndarray,
    valid: np.ndarray,
    box: Tuple[int, int, int, int],
) -> np.ndarray:
    """Boolean mask for points whose pixel lands inside `box`.

    `box` is (x_min, y_min, x_max, y_max); mins inclusive, maxs exclusive.
    """
    x_min, y_min, x_max, y_max = box
    return valid & (u >= x_min) & (u < x_max) & (v >= y_min) & (v < y_max)
```

**Step 4: Run, confirm PASS.**

**Step 5: Commit**

```bash
git add src/models/point_cloud.py tests/test_point_cloud.py
git commit -m "feat: select points whose pixel falls inside a 2D box"
```

---

## Task 4: Axis-aligned bounding box (`aabb_from_points`)

**Files:**
- Modify: `src/models/point_cloud.py`
- Test: `tests/test_point_cloud.py`

**Step 1: Add the failing test**

```python
from models.point_cloud import aabb_from_points


def test_aabb_center_dims_with_clamp():
    pts = np.array(
        [
            [-420.0, -420.0, 1000.0],
            [-380.0, -380.0, 1000.0],  # all share Z -> zero-thickness in Z
        ],
        dtype=np.float64,
    )
    center, dims = aabb_from_points(pts)
    assert np.allclose(center, [-400.0, -400.0, 1000.0])
    assert np.isclose(dims[0], 40.0) and np.isclose(dims[1], 40.0)
    assert np.isclose(dims[2], 1.0)  # clamped from 0 to MIN_DIM_MM
```

**Step 2: Run, confirm FAIL.**

**Step 3: Add to `src/models/point_cloud.py`**

```python
def aabb_from_points(
    points_xyz: np.ndarray,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Return (center_xyz, dims_xyz) of the axis-aligned bounding box.

    Each dimension is clamped to at least MIN_DIM_MM so a coplanar or
    single-point cluster never produces a zero-dimension (invalid) box.
    """
    mins = points_xyz.min(axis=0)
    maxs = points_xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    dims = np.maximum(maxs - mins, MIN_DIM_MM)
    return tuple(center.tolist()), tuple(dims.tolist())
```

**Step 4: Run, confirm PASS.**

**Step 5: Commit**

```bash
git add src/models/point_cloud.py tests/test_point_cloud.py
git commit -m "feat: axis-aligned bounding box from points with min-dim clamp"
```

---

## Task 5: `get_object_point_clouds` + properties flip (`detector.py`)

**Files:**
- Modify: `src/models/detector.py`
- Test: `tests/test_detector_pointcloud.py` (create)

**Step 1: Write the failing tests**

Create `tests/test_detector_pointcloud.py`:

```python
import asyncio
import io

import cv2
import numpy as np
import pytest
from PIL import Image
from viam.media.utils.pil import pil_to_viam_image
from viam.media.video import CameraMimeType
from viam.proto.component.camera import GetPropertiesResponse, IntrinsicParameters

from models.detector import Detector
from models.point_cloud import encode_pcd, parse_pcd

# Intrinsics chosen so that points near (-400, -400, 1000) project to ~(80, 80),
# which is inside the red square's box (40,40)-(120,120) on a 240x240 image.
_INTR = IntrinsicParameters(
    width_px=240, height_px=240,
    focal_x_px=100.0, focal_y_px=100.0,
    center_x_px=120.0, center_y_px=120.0,
)


def _red_square_image():
    arr = np.full((240, 240, 3), 128, dtype=np.uint8)
    cv2.rectangle(arr, (40, 40), (120, 120), (220, 20, 20), -1)
    return pil_to_viam_image(Image.fromarray(arr), CameraMimeType.PNG)


def _cloud_bytes():
    # 4 points that project inside the box, 1 outside (projects to (120,120) =
    # x_max, excluded), 1 behind the camera (invalid).
    pts = np.array(
        [
            [-400.0, -400.0, 1000.0],
            [-380.0, -400.0, 1000.0],
            [-400.0, -380.0, 1000.0],
            [-420.0, -420.0, 1000.0],
            [0.0, 0.0, 1000.0],     # -> (120,120), on exclusive max edge -> out
            [0.0, 0.0, -50.0],      # behind camera -> out
        ],
        dtype=np.float64,
    )
    return encode_pcd(pts)


class _FakePcdCamera:
    def __init__(self, supports_pcd=True, intrinsics=_INTR):
        self._props = GetPropertiesResponse(
            supports_pcd=supports_pcd, intrinsic_parameters=intrinsics
        )

    async def get_properties(self, *a, **k):
        return self._props

    async def get_images(self, *a, **k):
        return [_red_square_image()], None

    async def get_point_cloud(self, *a, **k):
        return _cloud_bytes(), "pointcloud/pcd"


def _detector(camera):
    det = Detector("test-detector")
    det.camera = camera
    det.camera_name = "cam1"
    return det


def test_object_point_clouds_returns_one_object_per_shape():
    det = _detector(_FakePcdCamera())
    objs = asyncio.run(det.get_object_point_clouds("cam1"))

    assert len(objs) == 1
    geoms = objs[0].geometries.geometries
    assert len(geoms) == 1
    assert geoms[0].label == "square-red"
    # Only the 4 in-box, in-front points were selected.
    assert parse_pcd(objs[0].point_cloud).shape == (4, 3)
    # Z dimension clamped (all points share Z=1000).
    assert np.isclose(geoms[0].box.dims_mm.z, 1.0)


def test_errors_when_camera_lacks_pcd_support():
    det = _detector(_FakePcdCamera(supports_pcd=False))
    with pytest.raises(ValueError, match="point cloud"):
        asyncio.run(det.get_object_point_clouds("cam1"))


def test_properties_reports_object_point_clouds_supported():
    det = Detector("test-detector")
    props = asyncio.run(det.get_properties())
    assert props.object_point_clouds_supported is True
```

**Step 2: Run, confirm FAIL**

Run: `venv/bin/python -m pytest tests/test_detector_pointcloud.py -q`
Expected: `get_object_point_clouds` raises `NotImplementedError`; the properties test fails (currently `False`).

**Step 3: Implement in `src/models/detector.py`**

Extend the common-proto import (the scaffold has `from viam.proto.common import PointCloudObject, ResourceName`):

```python
from viam.proto.common import (
    GeometriesInFrame,
    Geometry,
    PointCloudObject,
    Pose,
    RectangularPrism,
    ResourceName,
    Vector3,
)
```

Add the point-cloud helpers import alongside the existing `models.shape_detection` import:

```python
from models.point_cloud import (
    Intrinsics,
    aabb_from_points,
    encode_pcd,
    parse_pcd,
    points_in_box,
    project_to_pixels,
)
```

Replace the `get_object_point_clouds` body:

```python
    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[PointCloudObject]:
        camera = self._resolve_camera(camera_name)
        props = await camera.get_properties()
        intr = props.intrinsic_parameters
        if not props.supports_pcd or intr.focal_x_px == 0 or intr.focal_y_px == 0:
            raise ValueError(
                "Camera does not support point clouds with usable intrinsics; "
                "get_object_point_clouds requires supports_pcd and non-zero "
                "focal lengths"
            )

        image = await self._grab_frame(camera)
        pcd_bytes, _ = await camera.get_point_cloud()
        detections = detect_shapes(_viam_image_to_rgb(image))
        if not detections:
            return []

        try:
            points = parse_pcd(pcd_bytes)
        except Exception as exc:  # noqa: BLE001 - surface a clear gRPC error
            raise ValueError(f"Failed to parse point cloud: {exc}") from exc
        if points.size == 0:
            return []

        intrinsics = Intrinsics(
            intr.focal_x_px, intr.focal_y_px, intr.center_x_px, intr.center_y_px
        )
        u, v, valid = project_to_pixels(points, intrinsics)

        objects: List[PointCloudObject] = []
        for det in detections:
            box = (det.x_min, det.y_min, det.x_max, det.y_max)
            subset = points[points_in_box(u, v, valid, box)]
            if subset.shape[0] == 0:
                continue
            center, dims = aabb_from_points(subset)
            geometry = Geometry(
                center=Pose(
                    x=center[0], y=center[1], z=center[2],
                    o_x=0, o_y=0, o_z=1, theta=0,
                ),
                box=RectangularPrism(
                    dims_mm=Vector3(x=dims[0], y=dims[1], z=dims[2])
                ),
                label=det.label,
            )
            objects.append(
                PointCloudObject(
                    point_cloud=encode_pcd(subset),
                    geometries=GeometriesInFrame(
                        reference_frame=self.camera_name, geometries=[geometry]
                    ),
                )
            )
        return objects
```

Flip `get_properties`:

```python
        return Vision.Properties(
            classifications_supported=False,
            detections_supported=True,
            object_point_clouds_supported=True,
        )
```

Leave `get_classifications*`, `do_command`, `get_status` raising `NotImplementedError`.

**Step 4: Run, confirm PASS**

Run: `venv/bin/python -m pytest tests/test_detector_pointcloud.py -q`
Then the full suite: `venv/bin/python -m pytest -q`

**Step 5: Commit**

```bash
git add src/models/detector.py tests/test_detector_pointcloud.py
git commit -m "feat: implement get_object_point_clouds by projecting boxes onto the cloud"
```

---

## Task 6: Docs, verification, and review

**Files:**
- Modify: `devrel_shape-finder_detector.md`

**Step 1: Document the new capability**

In `devrel_shape-finder_detector.md`, add a short section after the behavior description noting that the service also implements `GetObjectPointClouds`: for a `camera_name` whose camera reports `supports_pcd` and provides intrinsics, it returns one point-cloud object per detected shape (the points inside the shape's 2D box, with a 3D axis-aligned bounding box labeled `"<shape>-<color>"`). Note the assumption that the cloud is registered to the color image and expressed in millimeters.

**Step 2: Full verification**

Run, expecting all green / clean:
```bash
venv/bin/python -m pytest -q
venv/bin/python -m compileall -q src
cd src && ../venv/bin/python -c "import main" && cd ..
```

**Step 3: Commit docs**

```bash
git add devrel_shape-finder_detector.md
git commit -m "docs: document get_object_point_clouds support"
```

**Step 4: Review**

Use superpowers:requesting-code-review against `docs/plans/2026-05-27-object-point-clouds-design.md`, then superpowers:finishing-a-development-branch.

---

## Manual verification (optional)

On a machine with an RGBD camera that reports `supports_pcd` and intrinsics (e.g. a RealSense), configure the `shape-finder:detector` with that `camera_name`, then call `GetObjectPointClouds` (e.g. via the SDK or the app) and confirm one labeled, boxed point-cloud object appears per colored shape in view.
