# Design: `devrel:shape-finder:detector` vision service

Date: 2026-05-26
Status: Approved

## Goal

Implement the scaffolded Viam vision service (`devrel:shape-finder`, model
`detector`) so it finds squares and rectangles of various colors in an image
using OpenCV, returning them as `Detection`s with bounding boxes. Each detection
is labeled `"<shape>-<color>"` (e.g. `square-red`, `rectangle-blue`), where the
color is the predominant hue within the bounding-box region of interest, mapped
to named HSV hue ranges.

This is a rapid prototype meant for quick iteration. Color ranges and detection
thresholds are hardcoded constants for now.

## 1. Configuration (minimal)

One optional attribute: `camera_name`, needed only by the camera-based methods.

- `validate_config`: returns `([camera_name], [])` when `camera_name` is set
  (declaring it a required dependency), otherwise `([], [])`.
- `reconfigure`: reads `camera_name` and stores the `Camera` client from
  dependencies when present.
- `get_detections(image)` requires no configuration. Camera-based methods raise
  a clear error when `camera_name` was not configured.

All other tuning (color ranges, thresholds) lives in hardcoded module constants.

## 2. Detection algorithm (pure, `src/models/shape_detection.py`)

A framework-free function `detect_shapes(rgb: np.ndarray) -> list[ShapeDetection]`
so it is unit-testable without the Viam SDK. `ShapeDetection` is a small
dataclass: `x_min, y_min, x_max, y_max, label, confidence`.

Pipeline (Approach B — color-agnostic shape detection, then classify ROI color):

1. RGB -> HSV via `cv2.cvtColor`.
2. **Saturation mask**: `cv2.inRange(hsv, (0, S_MIN, V_MIN), (179, 255, 255))`
   isolates colorful regions from a dull / white / black background.
   Morphological open + close to denoise.
3. `cv2.findContours` on the mask.
4. Per contour: skip if `area < MIN_AREA`; `approxPolyDP(eps=0.04*perimeter)`;
   keep if 4 vertices **and** convex. `boundingRect` -> `(x, y, w, h)`.
5. **Shape**: `aspect = w / h`; within `[1 - SQUARE_TOL, 1 + SQUARE_TOL]` ->
   `square`, otherwise `rectangle`.
6. **Color**: among the *saturated* pixels inside the bounding box, count how
   many fall into each color's hue range; the max-count color wins.
   `confidence = winning_count / total_saturated_in_box`. If no range matches,
   color is `unknown`.
7. Label = `"<shape>-<color>"`.

### Constants

- `COLOR_RANGES`: red (wraps `0-10` and `170-179`), orange, yellow, green, blue,
  purple. (OpenCV hue is 0-179.)
- `S_MIN` ~= 80, `V_MIN` ~= 50
- `MIN_AREA` ~= 500
- `EPSILON_FACTOR` = 0.04
- `SQUARE_TOL` ~= 0.15 (±15% aspect ratio counts as square)

## 3. Vision API methods (`src/models/detector.py`)

Thin wrappers over the pure function:

- **`get_detections(image)`**: `viam_to_pil_image(image)` ->
  `np.array(...convert("RGB"))` -> `detect_shapes(...)` -> map each
  `ShapeDetection` to a `Detection` proto
  (`x_min, y_min, x_max, y_max, confidence, class_name`).
- **`get_detections_from_camera(camera_name)`**: grab a frame from the
  configured camera, then the same path as above.
- **`capture_all_from_camera(..., return_image, return_detections, ...)`**: grab
  one frame, return a `CaptureAllResult` populated per the flags (so it works in
  the app Test panel and with data capture).
- **`get_properties()`**: `detections_supported=True`,
  `classifications_supported=False`, `object_point_clouds_supported=False`.
- Classification and point-cloud methods remain `NotImplementedError`.

## 4. Dependencies

Add to `requirements.txt`: `opencv-python-headless` (no GUI / X11 deps — correct
for a headless module) and `numpy`.

## 5. Error handling

- Image decode failure: log the error, return `[]` (do not crash the service).
- Camera method called with no `camera_name` configured: raise `ValueError` with
  a clear message (also logged).
- Guard `h == 0` and empty / all-zero ROIs in the color step to avoid
  div-by-zero (fall back to `unknown`, `confidence=0`).

## 6. Testing

`tests/test_detector.py` against synthetic images drawn with `cv2.rectangle` on a
neutral-gray background, targeting the pure `detect_shapes` function (fast, no
Viam runtime):

- Red square + blue rectangle -> two detections labeled `square-red` /
  `rectangle-blue`, bounding boxes within a few px of where they were drawn.
- A shape below `MIN_AREA` is ignored.
- An all-gray image -> zero detections.
