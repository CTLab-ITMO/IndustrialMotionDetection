import cv2
import numpy as np
import openvino as ov
from pathlib import Path
from ultralytics import YOLO

"""
model_id = [
    "yolo11n-pose",
    "yolo11s-pose",
    "yolo11m-pose",
    "yolo11l-pose",
    "yolo11x-pose",
    "yolov8n-pose",
    "yolov8s-pose",
    "yolov8m-pose",
    "yolov8l-pose",
    "yolov8x-pose",
]
"""


class PoseEstimator:
    def __init__(self, model_config):
        self.pose_model = YOLO(model_config)
        self.model_config = model_config
        self.openvino_model_path = None
        self.openvino_path = None

    def predict(self, image: np.ndarray) -> np.ndarray:
        # pose_model.predict(frame)[0].keypoints.xy[i] -- тут содержится скелетные точки i-го человека
        # xy[i][j] возвращает тензор
        # кисти 9 и 10
        # res = pose_model.predict(frame)[0].keypoints.xy[0][0]

        result = self.pose_model(image)[0].plot()[:, :, ::1]
        return result

    def convert_to_openvino(self, model_name="yolo11m-pose"):
        # TODO:: прописать сюда глобальный путь
        self.openvino_path = Path(f"{model_name}_openvino_model/{model_name}.xml")
        if not self.openvino_path.exists():
            self.pose_model.export(format="openvino", dynamic=True, half=True)

    def predict_openvino(self, device, image: np.ndarray) -> np.ndarray:
        core = ov.Core()
        pose_ov_model = core.read_model(self.openvino_path)
        ov_config = {}

        if "cpu" == device:
            pose_ov_model.reshape({0: [1, 3, 640, 640]})
        if "gpu" == device or ("auto" == device and "gpu" == core.available_devices):
            ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}

        pose_compiled_model = core.compile_model(pose_ov_model, device, ov_config)
        pose_model = YOLO(self.openvino_path.parent, task="pose")

        if pose_model.predictor is None:
            custom = {"conf": 0.60, "batch": 1, "save": False, "mode": "predict"}
            args = {**pose_model.overrides, **custom}
            pose_model.predictor = pose_model._smart_load("predictor")(overrides=args, _callbacks=pose_model.callbacks)
            pose_model.predictor.setup_model(model=pose_model.model)

        pose_model.predictor.model.ov_compiled_model = pose_compiled_model
        result = pose_model(image)
        return result
