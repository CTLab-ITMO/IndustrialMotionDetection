import torch
import logging
import numpy as np
from PIL import Image
from typing import List, Union, Optional, Tuple
from torchvision import transforms
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation


class ClipSegmentation:
    """
    A class for image segmentation using CLIP and segmentation models.
    This class allows segmenting objects in images based on text prompts,
    selecting the most prominent object from multiple prompts, and returning
    either the object mask or an empty mask if no objects are found.
    """

    def __init__(self, clip_seg_config: str, threshold: float = 0.4):
        """
        Initialize the segmentation pipeline with CLIP and segmentation model.

        Args:
            clip_seg_config: Pretrained model configuration for CLIP text and image processing
            threshold: Confidence threshold for mask binarization (default: 0.4)
        """
        self.clip_segmenter = CLIPSegProcessor.from_pretrained(clip_seg_config)
        self.seg_model = CLIPSegForImageSegmentation.from_pretrained(clip_seg_config)
        self.threshold = threshold

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Image preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.ToTensor()
        ])

    def preprocess_image(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocess the input image for model inference.

        Args:
            image: Input image (PIL Image or numpy array)

        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if image.mode != "RGB":
            image = image.convert("RGB")

        return self.preprocess(image)

    def predict_masks(self, image: Union[Image.Image, np.ndarray], prompts: List[str]) -> np.ndarray:
        """
        Generate segmentation masks for all provided prompts.

        Args:
            image: Input image to segment
            prompts: List of text prompts describing objects to segment

        Returns:
            np.ndarray: Array of segmentation masks (shape: [num_prompts, height, width])
        """
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image)

            # Prepare model inputs
            inputs = self.clip_segmenter(
                text=prompts,
                images=[processed_image] * len(prompts),
                padding="max_length",
                return_tensors="pt"
            )

            # Run model inference
            with torch.no_grad():
                outputs = self.seg_model(**inputs)

            # Post-process predictions
            preds = outputs.logits.unsqueeze(1)
            masks = [(torch.sigmoid(preds[i][0]) > self.threshold).float()
                     for i in range(len(prompts))]
            masks = torch.stack(masks).numpy()
            return masks

        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise

    def find_best_object(self, image: Union[Image.Image, np.ndarray], prompts: List[str]) -> Tuple[
        Optional[np.ndarray], Optional[int]]:
        """
        Identifies the most prominent object from the list of prompts.

        Args:
            image: Input image to analyze
            prompts: List of text prompts describing potential objects

        Returns:
            Tuple: (mask of best matching object, index of matching prompt)
                   or (None, None) if no objects found
        """
        masks = self.predict_masks(image, prompts)

        # Calculate area of each mask (number of positive pixels)
        mask_areas = [np.sum(mask) for mask in masks]

        # Find mask with largest area
        best_idx = np.argmax(mask_areas)
        best_area = mask_areas[best_idx]

        # Return None if no objects detected
        if best_area == 0:
            return None, None

        return masks[best_idx], best_idx

    def predict(self, image: Union[Image.Image, np.ndarray], prompts: List[str]) -> np.ndarray:
        """
        Main prediction method that returns either:
        - The mask of the most prominent object matching the prompts, or
        - An empty mask if no objects are detected

        Args:
            image: Input image to segment
            prompts: List of text prompts describing objects to find

        Returns:
            np.ndarray: Segmentation mask (2D array with same dimensions as input image)
        """
        best_mask, best_idx = self.find_best_object(image, prompts)

        if best_mask is None:
            thing = 'nothing'
            if isinstance(image, np.ndarray):
                h, w = image.shape[:2]
            else:
                w, h = image.size
            return np.zeros((h, w), dtype=np.float32), thing
        return best_mask, prompts[best_idx]
