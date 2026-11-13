# Model devrel:shape-finder:detector

A vision service that detects colored squares and rectangles in an image using
OpenCV. Each detection is returned as a bounding box with a class name of the
form `"<shape>-<color>"` — for example `square-red` or `rectangle-blue`.

- **Shapes:** four-sided convex contours are classified as `square` (sides of
  the shape's rotated rectangle within 15% of each other) or `rectangle`, so
  rotated shapes are labelled correctly. The returned bounding box is still
  axis-aligned.
- **Colors:** the predominant hue within each shape's outline is matched
  against built-in HSV ranges for `red`, `orange`, `yellow`, `green`, `blue`,
  and `purple`. A shape that matches no range is labeled `unknown`.

Color ranges and detection thresholds (minimum area, square tolerance, etc.) are
currently fixed in the implementation and are not configurable.

## Configuration

The following attribute template can be used to configure this model:

```json
{
  "camera_name": <string>
}
```

`camera_name` is only required for the camera-based methods
(`GetDetectionsFromCamera` and `CaptureAllFromCamera`). `GetDetections`, which
takes an image directly, works without any configuration.

### Attributes

The following attributes are available for this model:

| Name          | Type   | Inclusion | Description                                                                                                   |
|---------------|--------|-----------|---------------------------------------------------------------------------------------------------------------|
| `camera_name` | string | Optional  | Name of the camera component to pull frames from. Required to use `GetDetectionsFromCamera` / `CaptureAllFromCamera`; configured as a dependency when set. |

### Example Configuration

```json
{
  "camera_name": "my-camera"
}
```

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
