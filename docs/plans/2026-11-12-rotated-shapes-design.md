# Rotated shapes and their angle — design

Date: 2026-11-12. Status: approved. Item 1 of
`2026-11-12-shape-finder-improvements.md`.

## Problem

`detect_shapes` labels square vs rectangle from `cv2.boundingRect`, the
axis-aligned box. A rectangle turned about 45° has a near-square bounding box,
so it is labelled `square-…`. Colour is sampled from that same box, which for a
rotated shape can be about twice the shape's area and can take in a neighbour's
pixels. Nothing reports which way a shape is turned, though a pick needs it to
align the gripper.

## Consumer

User code (e.g. `vision_pick.py` in viam-labs/viam-mujoco-sim) calls the
service's DoCommand and turns `angle_deg` into gripper yaw. No change to the
standard vision API; the `Detection` proto has no field for rotation.

## Detection (`src/models/shape_detection.py`)

- Keep the existing filter: area ≥ `MIN_AREA`, `approxPolyDP` gives 4
  vertices, convex.
- Fit `cv2.minAreaRect` to the raw `contour` (also used for the colour mask);
  `cv2.boundingRect` keeps using `approx`, as today. Let `w, h` be the rotated
  rect's sides; classify `square` when `max(w, h) / min(w, h) <= 1 + SQUARE_TOL`,
  else `rectangle`. Long over short, so it no longer depends on orientation.
  Today's rule accepted `w/h` in [0.85, 1.15], which for tall shapes reached
  1/0.85 ≈ 1.18; the new rule is 1.15 either way. Intended.
- The axis-aligned box (`x_min … y_max`) is unchanged. `Detection` labels change
  for rotated shapes and near-threshold tall ones, and confidence can change
  slightly because colour is sampled from the contour (below).
- `ShapeDetection` gains:
  - `angle_deg: float = 0.0`
  - `corners: List[Tuple[float, float]] = field(default_factory=list)` — the
    four `cv2.boxPoints` of the rotated rect, in pixel coordinates, as Python
    floats.

  Defaults keep positional construction (e.g. `test_to_proto_maps_fields`)
  working.
- **Angle convention.** Image coordinates (x right, y down), so a positive
  angle is clockwise on screen.
  - Rectangle: angle of the long side against the +x axis, folded to
    (-90, 90].
  - Square: no long side; angle is periodic every 90°, folded to (-45, 45].
  - Computed from `boxPoints` (the edge vector of the long side via `atan2`)
    rather than `minAreaRect`'s own angle, whose range differs between OpenCV
    versions.
- **Colour.** `_predominant_color` samples hue from the filled contour
  (`cv2.drawContours` with `thickness=cv2.FILLED` onto a zero mask, ANDed with
  the saturation mask) instead of the bounding box.

## DoCommand (`src/models/detector.py`)

Request:

```json
{"command": "detect", "camera": "cam"}
```

`camera` is optional; it is passed to `_resolve_camera`, so it must match
`camera_name` if given, and the existing "no camera configured" error applies.

Response:

```json
{"detections": [{
  "label": "rectangle-red", "confidence": 0.98,
  "x_min": 10, "y_min": 20, "x_max": 110, "y_max": 90,
  "angle_deg": 30.0,
  "corners": [[x, y], [x, y], [x, y], [x, y]]
}]}
```

Any other `command` (or none) raises `ValueError` naming it. A frame that fails
to decode logs and returns `{"detections": []}`, matching `get_detections`.

## Tests

Synthetic images drawn with `cv2.fillPoly` / `cv2.boxPoints`, in the style of
`tests/test_shape_detection.py`:

Angles near the fold limits may come out at either end, so compare modulo
the period (180° for rectangles, 90° for squares).

- 120×50 rectangle drawn at 30° → `rectangle-…`, `angle_deg` within 1° of 30;
- 120×50 rectangle at 45° → `rectangle-…`, not `square-…` (the bug);
- square at 45° → `square-…`, `abs(abs(angle_deg) - 45) < 1`;
- axis-aligned wide rectangle → angle within 1° of 0; tall rectangle →
  `min(abs(a - 90), abs(a + 90)) < 1`;
- a rotated shape whose bounding box overlaps a different-coloured neighbour
  (gap ≥ 8 px, so the 5×5 morphology doesn't merge them) → both detected, each
  with its own colour;
- existing tests keep passing.

DoCommand, using `_FakeCamera` from `tests/test_detector_helpers.py`:

- `detect` returns the fields above for a synthetic frame, with `corners`
  values as Python floats;
- an unknown command raises `ValueError`.

## Out of scope

- An image argument on the DoCommand (it uses the configured camera only).
- Exposing the angle through the standard vision API or point clouds (item 3,
  on hold).
- Configurable thresholds (item 2) and new shapes (roadmap).
