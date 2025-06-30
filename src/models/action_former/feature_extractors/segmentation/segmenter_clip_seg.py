import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from logging_config import setup_logging
from typing import List, Union, Optional, Tuple
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

logger = setup_logging(module_name=__name__)


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

        :param clip_seg_config: Pretrained model configuration for CLIP text and image processing
        :param threshold: Confidence threshold for mask binarization (default: 0.4)
        """
        self.clip_seg_processor = CLIPSegProcessor.from_pretrained(clip_seg_config)
        self.seg_model = CLIPSegForImageSegmentation.from_pretrained(clip_seg_config)
        self.threshold = threshold
        self.preprocess = transforms.Compose([
            transforms.ToTensor()
        ])

    def predict_masks(self, image: Union[Image.Image, np.ndarray], prompts: List[str]) -> np.ndarray:
        """
        Generate segmentation masks for all provided prompts.

        :param image: Input image to segment
        :param prompts: List of text prompts describing objects to segment

        :return np.ndarray: Array of segmentation masks (shape: [num_prompts, height, width])
        """
        try:
            masks = []
            for prompt in prompts:
                inputs = self.clip_seg_processor(text=[prompt],
                                                 images=[image],
                                                 padding="max_length",
                                                 return_tensors="pt")
                with torch.no_grad():
                    outputs = self.seg_model(**inputs)
                preds = outputs.logits.unsqueeze(1)
                mask = torch.sigmoid(preds[0][0]).numpy()
                mask = np.array((mask > self.threshold), dtype=np.uint8)
                masks.append(mask)

            return np.stack(masks)
        except Exception as e:
            raise e

    def find_best_object(self, image: Union[Image.Image, np.ndarray], prompts: List[str]) -> Tuple[
        Optional[np.ndarray], Optional[int]]:
        """
        Identifies the most prominent object from the list of prompts.

        :param image: Input image to analyze
        :param prompts: List of text prompts describing potential objects

        :return Tuple: (mask of best matching object, index of matching prompt)
                   or (None, None) if no objects found
        """
        try:
            masks = self.predict_masks(image, prompts)
            mask_areas = [np.sum(mask) for mask in masks]
            best_idx = np.argmax(mask_areas)
            best_area = mask_areas[best_idx]
            if best_area == 0:
                return None, None

            return masks[best_idx], best_idx
        except Exception as e:
            raise e

    def predict(self, image: Union[Image.Image, np.ndarray], prompts: List[str]) -> np.ndarray:
        """
        Main prediction method that returns either:
        - The mask of the most prominent object matching the prompts, or
        - An empty mask if no objects are detected

        :param image: Input image to segment
        :param prompts: List of text prompts describing objects to find

        :return np.ndarray: Segmentation mask (2D array with same dimensions as input image)
        """
        try:
            best_mask, best_idx = self.find_best_object(image, prompts)
            if best_mask is None:
                thing = 'nothing'
                if isinstance(image, np.ndarray):
                    h, w = image.shape[:2]
                else:
                    w, h = image.size
                return np.zeros((h, w), dtype=np.float32), thing
            return best_mask, prompts[best_idx]
        except Exception as e:
            logger.error(f"Error during prediction mask of segmentation: {str(e)}")
