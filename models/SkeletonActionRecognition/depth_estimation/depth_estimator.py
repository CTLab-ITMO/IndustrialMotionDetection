import cv2
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Union
from matplotlib import pyplot as plt

with open("configs/models.yaml", "r") as f:
    config = yaml.safe_load(f)

DEPTH_ANYTHING_DIR = Path(config["depth_estimation"]["model_path"])
sys.path.append(str(DEPTH_ANYTHING_DIR))
try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError as e:
    raise


class DepthEstimator:
    """Класс для оценки глубины с использованием DepthAnythingV2."""

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
        """Инициализация DepthEstimator.

        Args:
            model_config: Конфигурация модели ('vits', 'vitb', 'vitl', 'vitg')
            model_path: Путь к файлу с весами модели
            device: Устройство для вычислений ('cpu', 'cuda', torch.device)
        """
        self._validate_model_config(model_config)
        self.model_config = self.MODEL_CONFIGS[model_config]
        self.model_path = Path(model_path)
        self.device = torch.device(device) if isinstance(device, str) else device

        self._validate_paths()
        self.model = self._load_model()

    def _validate_model_config(self, config: str) -> None:
        """Проверяет валидность конфигурации модели."""
        if config not in self.MODEL_CONFIGS:
            raise ValueError(f"Invalid model config '{config}'. Available: {list(self.MODEL_CONFIGS.keys())}")

    def _validate_paths(self) -> None:
        """Проверяет существование файла модели."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

    def _load_model(self) -> DepthAnythingV2:
        """Загружает и инициализирует модель."""
        try:
            model = DepthAnythingV2(**self.model_config)
            state_dict = torch.load(self.model_path, map_location='cpu')  # Всегда загружаем на CPU сначала

            # Совместимость с разными форматами сохраненных моделей
            if 'model' in state_dict:
                state_dict = state_dict['model']

            model.load_state_dict(state_dict)
            return model
        except Exception as e:
            raise

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Препроцессинг входного изображения.

        Args:
            image: Входное изображение (H, W, 3) в формате BGR или RGB

        Returns:
            Нормализованный тензор (1, 3, H, W)
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(image)}")

        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image.astype(np.float32) / 255.0
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]  # ImageNet нормализация
        return image

    def predict(self,
                image: np.ndarray,
                return_numpy: bool = True,
                visualize: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """Выполняет предсказание глубины для входного изображения.

        Args:
            image: Входное изображение (H, W, 3)
            return_numpy: Возвращать результат как numpy array (True) или torch tensor (False)
            visualize: Визуализировать результат

        Returns:
            Карта глубины (H, W)
        """
        try:
            self.model.to(self.device).eval()
            depth = self.model.infer_image(image)
            return depth

        except Exception as e:
            raise

    def _visualize_depth(self, depth: torch.Tensor, original_image: np.ndarray) -> None:
        """Визуализирует карту глубины."""
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        depth_np = depth.numpy()
        plt.imshow(depth_np, cmap='plasma')
        plt.colorbar()
        plt.title("Depth Map")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    def __del__(self):
        """Очистка ресурсов при удалении объекта."""
        if hasattr(self, 'model'):
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
