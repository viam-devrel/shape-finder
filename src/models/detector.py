from dataclasses import asdict
from typing import ClassVar, List, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
from typing_extensions import Self
from viam.media.utils.pil import viam_to_pil_image
from viam.media.video import ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.proto.service.vision import Classification, Detection
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.components.camera import Camera
from viam.services.vision import CaptureAllResult, Vision
from viam.utils import ValueTypes

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


class Detector(Vision, EasyResource):
    # To enable debug-level logging, either run viam-server with the --debug option,
    # or configure your resource/machine to display debug logs.
    MODEL: ClassVar[Model] = Model(ModelFamily("devrel", "shape-finder"), "detector")

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        """This method creates a new instance of this Vision service.
        The default implementation sets the name from the `config` parameter.

        Args:
            config (ComponentConfig): The configuration for this resource
            dependencies (Mapping[ResourceName, ResourceBase]): The dependencies (both required and optional)

        Returns:
            Self: The resource
        """
        instance = super().new(config, dependencies)
        instance.reconfigure(config, dependencies)
        return instance

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        """This method allows you to validate the configuration object received from the machine,
        as well as to return any required dependencies or optional dependencies based on that `config`.

        Args:
            config (ComponentConfig): The configuration for this resource

        Returns:
            Tuple[Sequence[str], Sequence[str]]: A tuple where the
                first element is a list of required dependencies and the
                second element is a list of optional dependencies
        """
        fields = config.attributes.fields
        camera_name = (
            fields["camera_name"].string_value if "camera_name" in fields else ""
        )
        if camera_name:
            return [camera_name], []
        return [], []

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

    def _resolve_camera(self, camera_name: str) -> Camera:
        if self.camera is None:
            raise ValueError(
                "No camera configured; set 'camera_name' in the service config "
                "to use camera-based methods"
            )
        if camera_name and camera_name != self.camera_name:
            raise ValueError(
                f"Requested camera '{camera_name}' does not match the configured "
                f"camera '{self.camera_name}'"
            )
        return self.camera

    async def _grab_frame(self, camera: Camera) -> ViamImage:
        # `get_images` returns images from all of a camera's imagers; for shape
        # detection we use the first one. (The older `get_image` was removed.)
        images, _ = await camera.get_images()
        if not images:
            raise ValueError(f"Camera '{self.camera_name}' returned no images")
        return images[0]

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
        image = await self._grab_frame(camera)
        result = CaptureAllResult()
        if return_image:
            result.image = image
        if return_detections:
            result.detections = await self.get_detections(image)
        return result

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        camera = self._resolve_camera(camera_name)
        image = await self._grab_frame(camera)
        return await self.get_detections(image)

    def _detect(self, image: ViamImage) -> List[ShapeDetection]:
        try:
            rgb = _viam_image_to_rgb(image)
        except Exception as exc:  # noqa: BLE001 - never crash the service on a bad frame
            self.logger.error(f"Failed to decode image: {exc}")
            return []
        return detect_shapes(rgb)

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        return [_to_proto(sd) for sd in self._detect(image)]

    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        self.logger.error("`get_classifications_from_camera` is not implemented")
        raise NotImplementedError()

    async def get_classifications(
        self,
        image: ViamImage,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        self.logger.error("`get_classifications` is not implemented")
        raise NotImplementedError()

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[PointCloudObject]:
        self.logger.error("`get_object_point_clouds` is not implemented")
        raise NotImplementedError()

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

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        cmd = command.get("command")
        if cmd != "detect":
            raise ValueError(f"Unknown command {cmd!r}; supported: 'detect'")
        camera = self._resolve_camera(str(command.get("camera", "")))
        image = await self._grab_frame(camera)
        return {"detections": [asdict(sd) for sd in self._detect(image)]}

    async def get_status(
        self, *, timeout: Optional[float] = None, **kwargs
    ) -> Mapping[str, ValueTypes]:
        self.logger.error("`get_status` is not implemented")
        raise NotImplementedError()
