# Model devrel:shape-finder:detector

A vision service that detects colored squares and rectangles in an image using
OpenCV. Each detection is returned as a bounding box with a class name of the
form `"<shape>-<color>"` — for example `square-red` or `rectangle-blue`.

- **Shapes:** four-sided convex contours are classified as `square` (aspect
  ratio within ±15% of 1:1) or `rectangle`.
- **Colors:** the predominant hue within each detection's region of interest is
  matched against built-in HSV ranges for `red`, `orange`, `yellow`, `green`,
  `blue`, and `purple`. A region that matches no range is labeled `unknown`.

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
