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

An unknown or missing `command`, a mismatched `camera`, or no `camera_name`
configured raises an error. A frame that can't be decoded returns
`{"detections": []}`.

Response:

```json
{"detections": [{
  "label": "rectangle-red", "confidence": 1.0,
  "x_min": 87, "y_min": 100, "x_max": 213, "y_max": 200,
  "angle_deg": 30.0,
  "corners": [[189.3, 201.6], [84.5, 141.1], [109.7, 97.3], [214.6, 157.8]]
}]}
```

- `angle_deg` is in image coordinates (x right, y down), so positive is
  clockwise on screen. For a rectangle it is the long side's angle against the
  x axis, in (-90, 90]. A square has no long side, so its angle is in
  (-45, 45].
- `corners` are the four corners of the shape's rotated rectangle, in pixels.
