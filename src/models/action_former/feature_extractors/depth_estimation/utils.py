import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """Preprocessing of the input image.

    :param image: Input image (H, W, 3) in BGR or RGB format

    :return: Normalized tensor (1, 3, H, W)
    """
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image.astype(np.float32) / 255.0
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return image


def _visualize_depth(depth: torch.Tensor, original_image: np.ndarray) -> None:
    """Visualizes a depth map."""
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
