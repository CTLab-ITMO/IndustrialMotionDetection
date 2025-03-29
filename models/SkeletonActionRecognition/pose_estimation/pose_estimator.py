import numpy as np
import openvino as ov
from pathlib import Path
from typing import Optional, Union, Dict, Any
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PoseEstimator:
    """Class for human pose estimation using YOLO and OpenVINO."""

    SUPPORTED_MODELS = [
        "yolo11n-pose", "yolo11s-pose", "yolo11m-pose", "yolo11l-pose", "yolo11x-pose",
        "yolov8n-pose", "yolov8s-pose", "yolov8m-pose", "yolov8l-pose", "yolov8x-pose",
    ]

    DEFAULT_OPENVINO_DIR = Path("pose_estimation/yolo11m-pose_openvino_model")

    def __init__(self, model_config: Union[str, Path], openvino_dir: Optional[Path] = None):
        """Initialize PoseEstimator.

        Args:
            model_config: Path to YOLO model configuration or name of a predefined model
            openvino_dir: Directory for saving OpenVINO models (default: 'openvino_models')
        """
        self._validate_model_name(model_config)
        self.pose_model = YOLO(model_config)
        self.model_config = model_config
        self.openvino_dir = openvino_dir or self.DEFAULT_OPENVINO_DIR
        self.openvino_model_path = None
        self._init_openvino_path()

    def _validate_model_name(self, model_name: str) -> None:
        """Check if the specified model is supported."""
        if isinstance(model_name, str) and model_name not in self.SUPPORTED_MODELS:
            logger.warning(f"Model {model_name} is not in the list of officially supported models. "
                           f"Supported models are: {', '.join(self.SUPPORTED_MODELS)}")

    def _init_openvino_path(self) -> None:
        """Initialize path for OpenVINO model."""
        model_name = Path(self.model_config).stem if isinstance(self.model_config, (str, Path)) else self.model_config
        self.openvino_model_path = self.openvino_dir / f"{model_name}.xml"
        self.openvino_dir.mkdir(parents=True, exist_ok=True)

    def predict(self, image: np.ndarray, **kwargs) -> Any:
        """Perform pose prediction on an image.

        Args:
            image: Input image as numpy array
            **kwargs: Additional arguments for YOLO predict

        Returns:
            Image with drawn keypoints
        """
        try:
            keypoints = self.pose_model(image, **kwargs)
            annotated_image = keypoints[0].plot()[:, :, ::1]
            return keypoints, annotated_image
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def convert_to_openvino(self, model_name: Optional[str] = None, **export_kwargs) -> None:
        """Convert the model to OpenVINO format.

        Args:
            model_name: Model name (if None, uses name from model_config)
            export_kwargs: Additional export parameters
        """
        if model_name is None:
            model_name = Path(self.model_config).stem if isinstance(self.model_config,
                                                                    (str, Path)) else self.model_config

        if not self.openvino_model_path.exists():
            logger.info(f"Converting model to OpenVINO format, saving to {self.openvino_model_path}")
            default_kwargs = {
                "format": "openvino",
                "dynamic": True,
                "half": True,
                "imgsz": 640,
                "batch": 1,
            }
            kwargs = {**default_kwargs, **export_kwargs}
            self.pose_model.export(**kwargs)
            logger.info("Model conversion completed successfully")
        else:
            logger.info("OpenVINO model already exists, skipping conversion")

    @staticmethod
    def _setup_ov_config(device: str) -> Dict[str, Any]:
        """Configure OpenVINO settings for the specified device.

        Args:
            device: Target device (CPU, GPU, etc.)

        Returns:
            Dictionary with OpenVINO configuration
        """
        ov_config = {}
        if device.lower() == "gpu":
            ov_config = {
                "GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES",
                "PERFORMANCE_HINT": "THROUGHPUT",
            }
        return ov_config

    @staticmethod
    def _init_predictor(pose_model: YOLO, predictor_args: Dict[str, Any]) -> None:
        """Initialize predictor for YOLO model.

        Args:
            pose_model: YOLO model instance
            predictor_args: Arguments for predictor initialization
        """
        if pose_model.predictor is None:
            custom_args = {
                "conf": 0.60,
                "batch": 1,
                "save": False,
                "mode": "predict",
            }
            args = {**pose_model.overrides, **custom_args, **predictor_args}
            pose_model.predictor = pose_model._smart_load("predictor")(overrides=args, _callbacks=pose_model.callbacks)
            pose_model.predictor.setup_model(model=pose_model.model)

    def predict_openvino(self,
                         image: np.ndarray,
                         device: str = "AUTO",
                         predictor_args: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Perform prediction using OpenVINO.

        Args:
            image: Input image
            device: Inference device (CPU, GPU, AUTO)
            predictor_args: Additional arguments for predictor

        Returns:
            Prediction results
        """
        if not self.openvino_model_path.exists():
            self.convert_to_openvino()

        core = ov.Core()
        device = self._select_device(core, device)

        try:
            # Load and compile model
            pose_ov_model = core.read_model(self.openvino_model_path)

            if device == "CPU":
                pose_ov_model.reshape({0: [1, 3, 640, 640]})

            ov_config = self._setup_ov_config(device)
            compiled_model = core.compile_model(pose_ov_model, device, ov_config)

            # Initialize YOLO model with OpenVINO
            pose_model = YOLO(self.openvino_model_path.parent, task="pose")
            self._init_predictor(pose_model, predictor_args or {})

            # Bind compiled OpenVINO model
            pose_model.predictor.model.ov_compiled_model = compiled_model
            keypoints = pose_model(image)
            annotated_image = keypoints[0].plot()[:, :, ::-1]
            return keypoints, annotated_image
        except Exception as e:
            logger.error(f"OpenVINO prediction failed: {e}")
            raise

    @staticmethod
    def _select_device(core: ov.Core, device: str) -> str:
        """Select available device for OpenVINO.

        Args:
            core: OpenVINO Core instance
            device: Requested device

        Returns:
            Available device name
        """
        available_devices = core.available_devices
        device = device.upper()

        if device == "AUTO":
            if "GPU" in available_devices:
                return "GPU"
            return "CPU"

        if device not in available_devices:
            logger.warning(f"Requested device {device} not available. Available devices: {available_devices}")
            return available_devices[0]

        return device

    @property
    def is_openvino_converted(self) -> bool:
        """Check if the model has been converted to OpenVINO.

        Returns:
            True if OpenVINO model exists, False otherwise
        """
        return self.openvino_model_path.exists()
