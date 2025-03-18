import torch
import numpy as np
from PIL import Image
from utils import show_masks
from segmentation import Segmenter

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

image = Image.open('/Users/anastasiaspileva/PycharmProjects/ActionRecognition/data/frames2/frame_0096.jpg')
image = np.array(image.convert("RGB"))

sam2_checkpoint = "/Users/anastasiaspileva/PycharmProjects/ActionRecognition/segmentation/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

input_points = np.array([[380, 410], [1100, 400], [740, 190], [800, 600]])
input_labels = np.array([1, 1, 1, 1])

seg = Segmenter(model_path=sam2_checkpoint, config_path=model_cfg, device=device)
seg.configure_params(image=image, points=input_points, labels=input_labels)

image, masks, scores = seg.predict(image, points=input_points, labels=input_labels)
show_masks(image, masks, scores, point_coords=input_points, input_labels=input_labels)
