import cv2
import sys
import torch
import numpy as np
from matplotlib import pyplot as plt

directory_path = '/Users/anastasiaspileva/PycharmProjects/ActionRecognition/depth_estimation/Depth-Anything-V2'
sys.path.append(directory_path)
from depth_anything_v2.dpt import DepthAnythingV2


# TODO:: сделать препроцессинг
# TODO:: сделать логгирование

class DepthEstimator:
    def __init__(self, model_config, model_path, device):
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        self.model_config = model_configs[model_config]
        self.model_path = model_path
        self.device = device
        self._load_model(device)

    def _load_model(self, device):
        self.model = DepthAnythingV2(**self.model_config)
        self.model.load_state_dict(torch.load(self.model_path, map_location=device))

    def predict(self, image: np.ndarray):
        self.model.to(self.device).eval()
        depth = self.model.infer_image(image)
        return depth
