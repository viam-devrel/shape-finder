# Design: `get_object_point_clouds` for `devrel:shape-finder:detector`

Date: 2026-05-27
Status: Approved

## Goal

Implement the `get_object_point_clouds` method of the vision service so that,
for a source camera that supports point clouds, it returns one
`PointCloudObject` per detected colored shape. Each object holds the subset of
the camera's point cloud that falls within the shape's 2D bounding box, plus a
3D axis-aligned bounding-box geometry labeled `"<shape>-<color>"`.

The 2D shapes come from the existing OpenCV `detect_shapes` path. The new work
is 3D: projecting the camera's point cloud into image space using the camera
intrinsics, then selecting the points that land inside each 2D box.

## Library choice: pypcd4 (not open3d)

PCD parsing/encoding uses **`pypcd4`** (PyPI, 1.4.x): a lightweight, numpy-native
PCD reader/writer. It is preferred over open3d because the module targets
`linux/arm64` and `darwin/arm64`, where open3d's large native wheels are
problematic. The exact from-bytes / to-bytes calls (possibly via `io.BytesIO`)
are to be pinned during implementation.

Add `pypcd4` to `requirements.txt`.

## Method flow: `get_object_point_clouds(camera_name)`

1. Resolve the camera (reuse `_resolve_camera`).
2. `props = await camera.get_properties()`. If `not props.supports_pcd` OR the
   intrinsics are unusable (`focal_x_px == 0` or `focal_y_px == 0`), raise a
   clear `ValueError`.
3. Grab a color frame (`_grab_frame` -> `get_images`) and the cloud
   (`await camera.get_point_cloud()` -> PCD bytes).
4. `detect_shapes(rgb)` -> 2D boxes (existing OpenCV path).
5. Parse PCD -> `Nx3` XYZ (plus optional RGB) via pypcd4.
6. Project all points to pixels with the intrinsics; keep `Z > 0`.
7. For each box, select the points whose `(u, v)` lands inside it; build a
   `PointCloudObject`.

## Projection & point selection (pure, `src/models/point_cloud.py`)

Framework-free numpy functions, unit-testable without a robot.

**Project points to pixels** with `fx=focal_x_px, fy=focal_y_px,
cx=center_x_px, cy=center_y_px` and points as an `Nx3` array in the camera
optical frame (+X right, +Y down, +Z forward):

```
valid = Z > 0
u = fx * (X / Z) + cx      # computed where valid
v = fy * (Y / Z) + cy
```

**Select per box** for a `ShapeDetection` box `(x_min, y_min, x_max, y_max)` in
color-image pixels:

```
in_box = valid & (u >= x_min) & (u < x_max) & (v >= y_min) & (v < y_max)
subset = points[in_box]    # slice ALL columns so RGB (if present) rides along
```

Boxes with an empty (or near-empty) `subset` produce no object.

**Assumptions:**
- The point cloud is registered to the color frame, and the intrinsics
  correspond to the image returned by `get_images`.
- Points are in **millimeters** (Viam convention), so `dims_mm`/`Pose` need no
  scaling. A meters-emitting camera would be off by 1000x; a scale config knob
  is explicitly out of scope for this prototype.

## Geometry (3D axis-aligned bounding box)

From the selected `subset` XYZ:

```
mins = subset_xyz.min(axis=0);  maxs = subset_xyz.max(axis=0)
center = (mins + maxs) / 2
dims   = maxs - mins        # clamp each component to >= 1.0 mm so a coplanar or
                            # single-point cluster never yields a zero-dim box
```

Assemble:

```python
Geometry(
    center=Pose(x=cx, y=cy, z=cz, o_x=0, o_y=0, o_z=1, theta=0),  # identity orientation
    box=RectangularPrism(dims_mm=Vector3(x=dx, y=dy, z=dz)),
    label=f"{shape}-{color}",
)
GeometriesInFrame(reference_frame=camera_name, geometries=[geometry])
PointCloudObject(point_cloud=<subset re-encoded to PCD via pypcd4>, geometries=gif)
```

Relevant proto facts (verified against viam-sdk 0.76.0):
- `PointCloudObject` fields: `point_cloud` (bytes), `geometries`
  (`GeometriesInFrame`).
- `GeometriesInFrame`: `reference_frame` (str), `geometries` (repeated
  `Geometry`).
- `Geometry`: `center` (`Pose`), shape oneof incl. `box` (`RectangularPrism`),
  `label`.
- `RectangularPrism`: `dims_mm` (`Vector3`).
- `Pose`: `x, y, z, o_x, o_y, o_z, theta`.
- `IntrinsicParameters`: `width_px, height_px, focal_x_px, focal_y_px,
  center_x_px, center_y_px`.

## Error handling

- No PCD support / unusable intrinsics -> `ValueError`.
- Empty cloud, or no box receives any points -> return `[]` (nothing detected,
  not an error).
- PCD parse/encode failure -> `ValueError` with context (propagates as a clean
  gRPC error).
- Flip `get_properties` -> `object_point_clouds_supported=True`.

## Testing

Pure functions tested with synthetic data (no robot):
- `project_to_pixels`: known points + intrinsics -> expected pixel coordinates
  and `valid` mask (incl. `Z <= 0` rejected).
- `points_in_box`: mask correctness at box edges.
- `aabb_from_points`: center/dims, including the 1mm clamp on a coplanar
  cluster.
- pypcd4 encode -> parse XYZ round-trip.

Integration with a fake camera:
- Two separated 3D point clusters that project into two detected boxes -> two
  `PointCloudObject`s with the correct labels and AABBs.
- `supports_pcd == False` (or zero intrinsics) -> `ValueError`.
- `get_properties` reports `object_point_clouds_supported == True`.

## Out of scope (YAGNI)

- Meters/units scaling config.
- Oriented (non-axis-aligned) bounding boxes or clustering within a box.
- Distortion handling (`distortion_parameters` ignored; pinhole only).
