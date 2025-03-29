import cv2
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Segmenter:
    """Класс для семантической сегментации с использованием SAM2."""

    def __init__(self,
                 model_path: Union[str, Path],
                 config_path: Union[str, Path],
                 device: str = "cuda"):
        """Инициализация сегментатора.

        Args:
            model_path: Путь к файлу с весами модели
            config_path: Путь к конфигурационному файлу модели
            device: Устройство для выполнения вычислений ("cpu" или "cuda")
        """
        self.model_path = Path(model_path)
        self.model_cfg = Path(config_path)
        self.device = device
        self.scores = None
        self.logits = None
        self.current_image = None

        self._validate_paths()
        self._load_model()
        logger.info(f"Segmenter initialized with model: {self.model_path}, device: {self.device}")

    def _validate_paths(self) -> None:
        """Проверяет существование файлов модели и конфигурации."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        if not self.model_cfg.exists():
            raise FileNotFoundError(f"Config file not found at {self.model_cfg}")

    def _load_model(self) -> None:
        """Загружает и инициализирует модель SAM2."""
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as e:
            logger.error(f"Failed to import SAM2 modules: {e}")
            raise

        try:
            logger.info("Loading SAM2 model...")
            sam2_model = build_sam2(str(self.model_cfg), str(self.model_path), device=self.device)
            self.predictor = SAM2ImagePredictor(sam2_model)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Препроцессинг входного изображения.

        Args:
            image: Входное изображение (H, W, 3) в формате BGR или RGB

        Returns:
            Преобразованное изображение (H, W, 3) в формате RGB
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(image)}")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be 3-channel (H, W, 3)")

        # Конвертация BGR в RGB если нужно
        if image.dtype == np.uint8:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def configure_params(self,
                         image: np.ndarray,
                         points: Optional[np.ndarray] = None,
                         labels: Optional[np.ndarray] = None) -> None:
        """Настраивает параметры сегментации и выбирает лучшие маски.

        Args:
            image: Входное изображение (H, W, 3)
            points: Координаты точек подсказок (N, 2)
            labels: Метки точек (1 - объект, 0 - фон)
        """
        try:
            image = self.preprocess_image(image)
            self.current_image = image

            logger.info("Configuring segmentation parameters...")
            self.predictor.set_image(image)

            masks, scores, logits = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True,
            )

            # Сортировка масок по качеству
            sorted_ind = np.argsort(scores)[::-1]
            self.scores = scores[sorted_ind]
            self.logits = logits[sorted_ind]

            logger.info(f"Found {len(scores)} masks, best score: {self.scores[0]:.2f}")
        except Exception as e:
            logger.error(f"Failed to configure parameters: {e}")
            raise

    def predict(self,
                image: Optional[np.ndarray] = None,
                points: Optional[np.ndarray] = None,
                labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Выполняет сегментацию изображения.

        Args:
            image: Входное изображение (если None, используется последнее)
            points: Координаты точек подсказок (N, 2)
            labels: Метки точек (1 - объект, 0 - фон)

        Returns:
            Кортеж (image, masks, scores)
        """
        try:
            if image is not None:
                image = self.preprocess_image(image)
                self.current_image = image
                self.predictor.set_image(image)
            elif self.current_image is None:
                raise ValueError("No image provided and no previous image available")

            if self.logits is None:
                logger.warning("No preconfigured masks found, running initial configuration")
                self.configure_params(self.current_image, points, labels)

            # Используем лучшую маску из предыдущего шага
            mask_input = self.logits[np.argmax(self.scores), :, :]

            logger.info("Running segmentation prediction...")
            masks, scores, _ = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                mask_input=mask_input[None, :, :],
                multimask_output=False,
            )

            logger.info(f"Segmentation completed with score: {scores[0]:.2f}")
            return self.current_image, masks, scores
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            raise

    @staticmethod
    def visualize_results(image: np.ndarray,
                          masks: np.ndarray,
                          alpha: float = 0.5) -> None:
        """Визуализирует результаты сегментации.

        Args:
            image: Исходное изображение
            masks: Маски сегментации
            alpha: Прозрачность наложения
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(15, 5))

        # Оригинальное изображение
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')

        # Маска
        plt.subplot(1, 3, 2)
        plt.imshow(masks[0], cmap='gray')
        plt.title("Segmentation Mask")
        plt.axis('off')

        # Наложение
        plt.subplot(1, 3, 3)
        plt.imshow(image)
        plt.imshow(masks[0], alpha=alpha, cmap='jet')
        plt.title("Overlay")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    def __del__(self):
        """Очистка ресурсов."""
        if hasattr(self, 'predictor'):
            del self.predictor
        logger.info("Segmenter resources released")
