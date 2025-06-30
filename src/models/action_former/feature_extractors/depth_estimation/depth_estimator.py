import torch
import numpy as np
from pathlib import Path
from typing import Union
from logging_config import setup_logging
from .Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2

logger = setup_logging(module_name=__name__)


class DepthEstimator:
    """A class for estimating depth using DepthAnythingV2."""

    MODEL_CONFIGS = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    def __init__(self,
                 model_config: str,
                 model_path: Union[str, Path],
                 device: Union[str, torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialization of the DepthEstimator.

        :param model_config: Model configuration ('vitz', 'vitz', 'vitz', 'vitz')
        :param model_path: Path to the model weights file
        :param device: A device for computing ('cpu', 'cuda', torch.device)
        """
        self._validate_model_config(model_config)
        self.model_config = self.MODEL_CONFIGS[model_config]
        self.model_path = Path(model_path)
        self.device = torch.device(device) if isinstance(device, str) else device
        self._validate_paths()
        self.model = self._load_model()

    def _validate_model_config(self, config: str) -> None:
        """Checks the validity of the model configuration."""
        if config not in self.MODEL_CONFIGS:
            raise FileExistsError

    def _validate_paths(self) -> None:
        """Checks the existence of the model file."""
        if not self.model_path.exists():
            raise FileExistsError

    def _load_model(self):
        """Loads and initializes the model."""
        try:
            model = DepthAnythingV2(**self.model_config)
            state_dict = torch.load(self.model_path, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']

            model.load_state_dict(state_dict)
            return model
        except Exception as e:
            raise e

    def predict(self,
                image: np.ndarray) -> Union[np.ndarray, torch.Tensor]:
        """Performs depth prediction for the input image.

        :param image: Input image (H, W, 3)

        :return: Depth map (H, W)
        """
        try:
            self.model.to(self.device).eval()
            depth = self.model.infer_image(image)
            return depth
        except Exception as e:
            logger.error(f"Failed to predict depth: {e}")
            raise
