# Rotated Shapes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Classify square vs rectangle from each shape's rotated rectangle, report its angle and corners, sample colour from the shape itself, and expose all of it through a `detect` DoCommand.

**Architecture:** `detect_shapes` in `src/models/shape_detection.py` fits `cv2.minAreaRect` to each contour and adds `angle_deg` and `corners` to `ShapeDetection` (defaults keep existing construction working). `Detector` in `src/models/detector.py` gets a shared `_detect(image)` helper used by both `get_detections` and the new `do_command`.

**Tech Stack:** Python 3.11, OpenCV 4.13, numpy, Viam Python SDK, pytest.

**Design doc:** `docs/plans/2026-11-12-rotated-shapes-design.md`

**Conventions:**
- Source is under `src/`; imports use the `models.` prefix (`tests/conftest.py` puts `src` on the path).
- Run tests with `venv/bin/python -m pytest -q` from the repo root. The baseline is 21 passing.
- Commit after each task. End each commit message with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_012FAJrBigYipqTBAmfmVZx1
  ```

**Verified in the venv beforehand (OpenCV 4.13):**
- `cv2.boxPoints(((cx, cy), (w, h), angle))` + `cv2.fillPoly` draws a rotated rectangle; the angle folded from the long edge of the detected box matches the drawn one (30 → 29.98, 45 → 45.0, 0 → 0.0, tall 50×120 → 90.0, -30 → -29.98).
- A 120×50 rectangle at 45° has a 118×118 bounding box, so today it is labelled `square-…`.
- The neighbour fixture in Task 2 today gives `square-red` with confidence 0.77.

---

## Task 1: Rotated rect: classification, angle, corners

**Files:**
- Modify: `src/models/shape_detection.py`
- Test: `tests/test_shape_detection.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_shape_detection.py`:

```python
RED = (220, 20, 20)


def rotated_rect(color_rgb, w, h, angle, center=(150, 150), canvas=(300, 300)):
    img = np.full((canvas[0], canvas[1], 3), 128, dtype=np.uint8)
    pts = cv2.boxPoints((center, (w, h), angle)).astype(np.int32)
    cv2.fillPoly(img, [pts], color_rgb)
    return img


def only_detection(img):
    dets = detect_shapes(img)
    assert len(dets) == 1
    return dets[0]


def angle_close(a, b, period=180.0, tol=1.0):
    # Angles near the fold limits may come out at either end of the range.
    d = (a - b) % period
    return min(d, period - d) < tol


def test_rotated_rectangle_reports_angle_and_corners():
    d = only_detection(rotated_rect(RED, 120, 50, 30))
    assert d.label == "rectangle-red"
    assert angle_close(d.angle_deg, 30)
    assert len(d.corners) == 4


def test_rectangle_at_45_degrees_is_not_a_square():
    d = only_detection(rotated_rect(RED, 120, 50, 45))
    assert d.label == "rectangle-red"
    assert angle_close(d.angle_deg, 45)


def test_square_at_45_degrees_folds_to_quarter_turn():
    d = only_detection(rotated_rect(RED, 80, 80, 45))
    assert d.label == "square-red"
    assert -45 < d.angle_deg <= 45
    assert angle_close(d.angle_deg, 45, period=90.0)


def test_axis_aligned_rectangle_angles():
    wide = only_detection(rotated_rect(RED, 120, 50, 0))
    tall = only_detection(rotated_rect(RED, 50, 120, 0))
    assert angle_close(wide.angle_deg, 0)
    assert angle_close(tall.angle_deg, 90)
    assert -90 < tall.angle_deg <= 90
```

- [ ] **Step 2: Run them to verify they fail**

Run: `venv/bin/python -m pytest -q tests/test_shape_detection.py`
Expected: the 4 new tests FAIL (`AttributeError: 'ShapeDetection' object has no attribute 'angle_deg'`, and `square-red` != `rectangle-red` for the 45° rectangle). The existing 7 pass.

- [ ] **Step 3: Implement**

In `src/models/shape_detection.py`:

Change the imports at the top to:

```python
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np
```

Replace the `ShapeDetection` dataclass with:

```python
@dataclass
class ShapeDetection:
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    label: str
    confidence: float
    # Image coordinates (y down): positive is clockwise on screen.
    # Rectangle: long side vs +x, in (-90, 90]. Square: in (-45, 45].
    angle_deg: float = 0.0
    corners: List[Tuple[float, float]] = field(default_factory=list)
```

Replace the body of the `for contour in contours:` loop with:

```python
    for contour in contours:
        if cv2.contourArea(contour) < MIN_AREA:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, EPSILON_FACTOR * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        if min(rw, rh) == 0:
            continue
        is_square = max(rw, rh) / min(rw, rh) <= 1 + SQUARE_TOL
        shape = "square" if is_square else "rectangle"
        corners = cv2.boxPoints(rect)

        x, y, w, h = cv2.boundingRect(approx)
        color, confidence = _predominant_color(hsv, sat_mask, x, y, w, h)
        detections.append(
            ShapeDetection(
                x,
                y,
                x + w,
                y + h,
                f"{shape}-{color}",
                confidence,
                angle_deg=_fold_angle(corners, is_square),
                corners=[(float(px), float(py)) for px, py in corners],
            )
        )
    return detections
```

Add this function below `detect_shapes`:

```python
def _fold_angle(corners: np.ndarray, is_square: bool) -> float:
    # Use the box's long edge rather than minAreaRect's own angle, whose range
    # differs between OpenCV versions.
    e1 = corners[1] - corners[0]
    e2 = corners[2] - corners[1]
    edge = e1 if np.hypot(*e1) >= np.hypot(*e2) else e2
    angle = math.degrees(math.atan2(float(edge[1]), float(edge[0])))
    period = 90.0 if is_square else 180.0
    angle %= period
    if angle > period / 2:
        angle -= period
    return angle
```

- [ ] **Step 4: Run all tests**

Run: `venv/bin/python -m pytest -q`
Expected: 25 passed.

- [ ] **Step 5: Commit**

```bash
git add src/models/shape_detection.py tests/test_shape_detection.py
git commit -m "feat: classify shapes from rotated rect and report angle and corners"
```

---

## Task 2: Sample colour from the shape, not its bounding box

**Files:**
- Modify: `src/models/shape_detection.py`
- Test: `tests/test_shape_detection.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_shape_detection.py`:

```python
def test_color_sampled_from_shape_not_bounding_box():
    # The green square sits inside the 45° red rectangle's bounding box,
    # about 15 px clear of the rectangle itself (enough that the 5x5
    # morphology doesn't merge them).
    img = rotated_rect(RED, 160, 40, 45)
    cv2.rectangle(img, (175, 80), (220, 125), (20, 180, 20), -1)
    dets = sorted(detect_shapes(img), key=lambda d: d.label)
    assert [d.label for d in dets] == ["rectangle-red", "square-green"]
    assert dets[0].confidence > 0.95
```

- [ ] **Step 2: Run it to verify it fails**

Run: `venv/bin/python -m pytest -q tests/test_shape_detection.py::test_color_sampled_from_shape_not_bounding_box`
Expected: FAIL on the confidence assertion (about 0.77: green pixels in the red rectangle's bounding box are counted).

- [ ] **Step 3: Implement**

In `detect_shapes`, replace:

```python
        color, confidence = _predominant_color(hsv, sat_mask, x, y, w, h)
```

with:

```python
        shape_mask = np.zeros_like(sat_mask)
        cv2.drawContours(shape_mask, [contour], -1, 255, thickness=cv2.FILLED)
        color, confidence = _predominant_color(
            hsv, cv2.bitwise_and(shape_mask, sat_mask)
        )
```

Replace the signature and first three lines of `_predominant_color`:

```python
def _predominant_color(
    hsv: np.ndarray, sat_mask: np.ndarray, x: int, y: int, w: int, h: int
) -> Tuple[str, float]:
    roi_hsv = hsv[y : y + h, x : x + w]
    roi_mask = sat_mask[y : y + h, x : x + w]
    hue = roi_hsv[:, :, 0][roi_mask > 0]
```

with:

```python
def _predominant_color(hsv: np.ndarray, mask: np.ndarray) -> Tuple[str, float]:
    hue = hsv[:, :, 0][mask > 0]
```

The rest of `_predominant_color` is unchanged.

- [ ] **Step 4: Run all tests**

Run: `venv/bin/python -m pytest -q`
Expected: 26 passed.

- [ ] **Step 5: Commit**

```bash
git add src/models/shape_detection.py tests/test_shape_detection.py
git commit -m "fix: sample shape colour from its contour, not its bounding box"
```

---

## Task 3: `detect` DoCommand

**Files:**
- Modify: `src/models/detector.py`
- Test: `tests/test_detector_helpers.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_detector_helpers.py`:

```python
def _rotated_rect_image():
    arr = np.full((300, 300, 3), 128, dtype=np.uint8)
    pts = cv2.boxPoints(((150, 150), (120, 50), 30)).astype(np.int32)
    cv2.fillPoly(arr, [pts], (220, 20, 20))
    return pil_to_viam_image(Image.fromarray(arr), CameraMimeType.PNG)


def test_do_command_detect_returns_angle_and_corners():
    det = _detector_with_camera([_rotated_rect_image()])
    result = asyncio.run(det.do_command({"command": "detect"}))

    (d,) = result["detections"]
    assert d["label"] == "rectangle-red"
    assert abs(d["angle_deg"] - 30) < 1
    assert {"x_min", "y_min", "x_max", "y_max", "confidence"} <= d.keys()
    assert len(d["corners"]) == 4
    assert all(type(v) is float for corner in d["corners"] for v in corner)
    dict_to_struct(result)  # what the SDK does to send it; must not raise


def test_do_command_accepts_matching_camera():
    det = _detector_with_camera([_rotated_rect_image()])
    result = asyncio.run(det.do_command({"command": "detect", "camera": "cam1"}))
    assert len(result["detections"]) == 1


def test_do_command_rejects_mismatched_camera():
    det = _detector_with_camera([_rotated_rect_image()])
    with pytest.raises(ValueError, match="does not match"):
        asyncio.run(det.do_command({"command": "detect", "camera": "other"}))


def test_do_command_rejects_unknown_command():
    det = _detector_with_camera([_rotated_rect_image()])
    with pytest.raises(ValueError, match="bogus"):
        asyncio.run(det.do_command({"command": "bogus"}))


def test_do_command_returns_empty_on_undecodable_frame():
    det = _detector_with_camera([ViamImage(b"not-an-image", CameraMimeType.JPEG)])
    assert asyncio.run(det.do_command({"command": "detect"})) == {"detections": []}
```

And add to the imports at the top of the file:

```python
from viam.utils import dict_to_struct
```

- [ ] **Step 2: Run them to verify they fail**

Run: `venv/bin/python -m pytest -q tests/test_detector_helpers.py`
Expected: the 5 new tests FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement**

In `src/models/detector.py`, add near the top:

```python
from dataclasses import asdict
```

Add this method to `Detector`, just above `get_detections`:

```python
    def _detect(self, image: ViamImage) -> List[ShapeDetection]:
        try:
            rgb = _viam_image_to_rgb(image)
        except Exception as exc:  # noqa: BLE001 - never crash the service on a bad frame
            self.logger.error(f"Failed to decode image: {exc}")
            return []
        return detect_shapes(rgb)
```

Replace the body of `get_detections` with:

```python
        return [_to_proto(sd) for sd in self._detect(image)]
```

Replace the body of `do_command` with:

```python
        cmd = command.get("command")
        if cmd != "detect":
            raise ValueError(f"Unknown command {cmd!r}; supported: 'detect'")
        camera = self._resolve_camera(str(command.get("camera", "")))
        image = await self._grab_frame(camera)
        return {"detections": [asdict(sd) for sd in self._detect(image)]}
```

- [ ] **Step 4: Run all tests**

Run: `venv/bin/python -m pytest -q`
Expected: 31 passed.

- [ ] **Step 5: Commit**

```bash
git add src/models/detector.py tests/test_detector_helpers.py
git commit -m "feat: add detect DoCommand returning angle and corners"
```

---

## Task 4: Document the behaviour

**Files:**
- Modify: `devrel_shape-finder_detector.md`

- [ ] **Step 1: Update the shape and colour bullets**

Replace:

```markdown
- **Shapes:** four-sided convex contours are classified as `square` (aspect
  ratio within ±15% of 1:1) or `rectangle`.
- **Colors:** the predominant hue within each detection's region of interest is
  matched against built-in HSV ranges for `red`, `orange`, `yellow`, `green`,
  `blue`, and `purple`. A region that matches no range is labeled `unknown`.
```

with:

```markdown
- **Shapes:** four-sided convex contours are classified as `square` (sides of
  the shape's rotated rectangle within 15% of each other) or `rectangle`, so
  rotated shapes are labelled correctly. The returned bounding box is still
  axis-aligned.
- **Colors:** the predominant hue within each shape's outline is matched
  against built-in HSV ranges for `red`, `orange`, `yellow`, `green`, `blue`,
  and `purple`. A shape that matches no range is labeled `unknown`.
```

- [ ] **Step 2: Add a DoCommand section at the end of the file**

````markdown
## DoCommand

### `detect`

Grabs a frame from the configured camera and returns each detection with its
rotation, which the standard `Detection` type can't carry.

```json
{"command": "detect", "camera": "my-camera"}
```

`camera` is optional; if given it must match `camera_name`.

Response:

```json
{"detections": [{
  "label": "rectangle-red", "confidence": 0.98,
  "x_min": 10, "y_min": 20, "x_max": 110, "y_max": 90,
  "angle_deg": 30.0,
  "corners": [[33.5, 90.2], [137.4, 30.2], [162.4, 73.5], [58.5, 133.5]]
}]}
```

- `angle_deg` is in image coordinates (x right, y down), so positive is
  clockwise on screen. For a rectangle it is the long side's angle against the
  x axis, in (-90, 90]. A square has no long side, so its angle is in
  (-45, 45].
- `corners` are the four corners of the shape's rotated rectangle, in pixels.
````

- [ ] **Step 3: Commit**

```bash
git add devrel_shape-finder_detector.md
git commit -m "docs: document rotated shapes and the detect DoCommand"
```

---

## After the tasks

Check the change against the viam-mujoco-sim vision pick cell (8/8 sorted) before releasing, as the improvements doc asks. That cell needs the sim, so it's a manual step for the user.
