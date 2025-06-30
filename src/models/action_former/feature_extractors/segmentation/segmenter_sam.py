import cv2
import numpy as np
from pathlib import Path
from logging_config import setup_logging
from typing import Optional, Tuple, Union

logger = setup_logging(module_name=__name__)


class SamSegmenter:
    """A class for semantic segmentation using SAMv2."""

    def __init__(self,
                 model_path: Union[str, Path],
                 config_path: Union[str, Path],
                 device: str = "cuda"):
        """Initialize segmentation model.

        :param model_path: Path to the model weights file
        :param config_path: Path to the model configuration file
        :param device: A device for performing calculations ("cpu" or "cuda")
        """
        self.model_path = Path(model_path)
        self.model_cfg = Path(config_path)
        self.device = device
        self.scores = None
        self.logits = None
        self.current_image = None

        self._validate_paths()
        self._load_model()

    def _validate_paths(self) -> None:
        """Checks the existence of the model and configuration files."""
        if not self.model_path.exists():
            logger.error(FileNotFoundError(f"Model file not found at {self.model_path}"))
            raise

        if not self.model_cfg.exists():
            logger.error(FileNotFoundError(f"Config file not found at {self.model_cfg}"))
            raise

    def _load_model(self) -> None:
        """Loads and initializes the SAM 2 model."""
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as e:
            logger.error(f"Failed to import SAM2 modules: {e}")
            raise

        try:
            sam2_model = build_sam2(str(self.model_cfg), str(self.model_path), device=self.device)
            self.predictor = SAM2ImagePredictor(sam2_model)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    @staticmethod
    def preprocess_image(image: np.ndarray) -> np.ndarray:
        """Preprocessing of the input image.

        :param image: input image (H, W, 3) в of format in BGR or RGB

        :return: Converted image (H, W, 3) in RGB format
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(image)}")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be 3-channel (H, W, 3)")

        if image.dtype == np.uint8:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def configure_params(self,
                         image: np.ndarray,
                         points: Optional[np.ndarray] = None,
                         labels: Optional[np.ndarray] = None) -> None:
        """Adjusts the segmentation parameters and selects the best masks.

        :param image: Input image (H, W, 3)
        :param points: Coordinates of the hint points (N, 2)
        :param labels: Point labels (1 - object, 0 - background)
        """
        try:
            image = self.preprocess_image(image)
            self.current_image = image
            self.predictor.set_image(image)
            masks, scores, logits = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True,
            )

            sorted_ind = np.argsort(scores)[::-1]
            self.scores = scores[sorted_ind]
            self.logits = logits[sorted_ind]
        except Exception as e:
            logger.error(f"Failed to get best mask: {e}")
            raise

    def predict(self,
                image: Optional[np.ndarray] = None,
                points: Optional[np.ndarray] = None,
                labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Performs image segmentation.

        :param image: Input image (if None, the last one is used)
        :param points: Coordinates of the hint points (N, 2)
        :param labels: Point labels (1 - object, 0 - background)

        :return: Tuple (image, masks, scores)
        """
        try:
            if image is not None:
                image = self.preprocess_image(image)
                self.current_image = image
                self.predictor.set_image(image)
            elif self.current_image is None:
                logger.error(ValueError("No image provided and no previous image available"))
                raise

            if self.logits is None:
                self.configure_params(self.current_image, points, labels)

            mask_input = self.logits[np.argmax(self.scores), :, :]
            masks, scores, _ = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                mask_input=mask_input[None, :, :],
                multimask_output=False,
            )
            return self.current_image, masks, scores
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            raise
