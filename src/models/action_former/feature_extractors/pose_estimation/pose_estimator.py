import numpy as np
import openvino as ov
from pathlib import Path
from ultralytics import YOLO
from typing import Union, Any
from logging_config import setup_logging

logger = setup_logging(module_name=__name__)


class PoseEstimator:
    """Simplified human pose estimation using YOLO."""

    SUPPORTED_MODELS = [
        "yolo11n-pose", "yolo11s-pose", "yolo11m-pose", "yolo11l-pose", "yolo11x-pose",
        "yolov8n-pose", "yolov8s-pose", "yolov8m-pose", "yolov8l-pose", "yolov8x-pose",
    ]

    def __init__(self, model: Union[str, Path]):
        """Initialize with model path/name and optional OpenVINO directory."""
        if isinstance(model, str) and model not in self.SUPPORTED_MODELS:
            logger.warning(f"Model {model} not in supported list: {', '.join(self.SUPPORTED_MODELS)}")

        self.model = YOLO(model)

        self.ov_path = Path("models/yolo-pose_openvino_model") / f"{Path(model).stem}.xml"
        self.ov_path.parent.mkdir(parents=True, exist_ok=True)

        self.onnx_path = Path("models/yolo-pose_onnx_model") / f"{Path(model).stem}.onnx"
        self.onnx_path.parent.mkdir(parents=True, exist_ok=True)

    def predict(self, image: np.ndarray, optimization_type: str, **kwargs) -> Any:
        """Predict poses and return keypoints with annotated image.

        :param image: input image
        :param optimization_type: type of optimization like ONNX or OpenVINO or just YOLO
        """
        try:
            results = None
            if optimization_type is None or optimization_type == 'simple':
                results = self.model(image, **kwargs)
            elif optimization_type == 'ONNX':
                if not self.onnx_path.exists():
                    self.model.export(format="onnx", dynamic=True, simplify=True, name=str(self.onnx_path))
                model = YOLO(str(self.onnx_path))
                results = model(image, **kwargs)
            elif optimization_type == 'OpenVINO':
                results = self.predict_openvino(image, **kwargs)
            return results, results[0].plot()[:, :, ::1]

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def to_openvino(self, **kwargs) -> None:
        """Convert model to OpenVINO format if not already converted."""
        if not self.ov_path.exists():
            self.model.export(format="openvino", dynamic=True, half=True, imgsz=640, **kwargs)
        else:
            logger.info("OpenVINO model exists, skipping conversion")

    def predict_openvino(self, image: np.ndarray, device: str = "AUTO", **kwargs) -> np.ndarray:
        """Predict using OpenVINO with automatic device selection."""
        self.to_openvino()

        core = ov.Core()
        device = "GPU" if device.upper() == "AUTO" and "GPU" in core.available_devices else "CPU"

        try:
            model = core.read_model(self.ov_path)
            if device == "CPU":
                model.reshape({0: [1, 3, 640, 640]})

            compiled = core.compile_model(model, device)
            yolo = YOLO(self.ov_path.parent, task="pose")

            if not yolo.predictor:
                yolo.predictor = yolo._smart_load("predictor")(overrides={
                    **yolo.overrides,
                    "conf": 0.6,
                    "batch": 1,
                    "save": False,
                    **kwargs
                }, _callbacks=yolo.callbacks)
                yolo.predictor.setup_model(model=yolo.model)

            yolo.predictor.model.ov_compiled_model = compiled
            results = yolo(image)
            return results
        except Exception as e:
            logger.error(f"OpenVINO prediction failed: {e}")
            raise
