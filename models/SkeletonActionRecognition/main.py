import cv2
import torch
import numpy as np
from PIL import Image

from interaction_analysis.object_interaction import get_yolo_depth_result, check_points_in_mask
from segmentation import Segmenter
from pose_estimation.inference_pose import PoseEstimator
from depth_estimation.depth_estimation import DepthEstimator

MODEL_PATH_DEPTH = '/Users/anastasiaspileva/PycharmProjects/ActionRecognition/depth_estimation/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth'
MODEL_PATH_YOLO = 'yolo11m-pose.pt'
MODEL_PARH_SEG = '/Users/anastasiaspileva/PycharmProjects/ActionRecognition/segmentation/checkpoints/sam2.1_hiera_large.pt'
SEG_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

IMAGE_PATH = '/Users/anastasiaspileva/PycharmProjects/ActionRecognition/data/frames2/frame_0096.jpg'
OUTPUT_PATH = '/Users/anastasiaspileva/Desktop/output'

THRESHOLD_DISTANCE = 10

# сегментация
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

image = Image.open(IMAGE_PATH)
image = np.array(image.convert("RGB"))

input_points = np.array([[380, 410], [1100, 400], [740, 190], [800, 600]])
input_labels = np.array([1, 1, 1, 1])

seg_model = Segmenter(model_path=MODEL_PARH_SEG, config_path=SEG_CONFIG, device=device)
seg_model.configure_params(image=image, points=input_points, labels=input_labels)

_, masks, scores = seg_model.predict(image, points=input_points, labels=input_labels)
masks = np.squeeze(masks, axis=0)

# получение 3d точек человека
yolo_model = PoseEstimator(MODEL_PATH_YOLO)
depth_model = DepthEstimator(model_config='vits', model_path=MODEL_PATH_DEPTH, device='cpu')
points_3d = get_yolo_depth_result(depth_model.model, yolo_model.pose_model, IMAGE_PATH)
answer = check_points_in_mask(points_3d, masks, THRESHOLD_DISTANCE)

# Параметры текста
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.0
font_color = (255, 255, 255)
thickness = 2

position = (10, 50)

image = cv2.imread(IMAGE_PATH)
if not answer:
    text = "not holding on to the ladder"
    cv2.putText(image, text, position, font, font_scale, font_color, thickness,
                cv2.LINE_AA)
else:
    text = "holding on to the ladder"
    cv2.putText(image, text, position, font, font_scale, font_color, thickness,
                cv2.LINE_AA)
cv2.imwrite(OUTPUT_PATH + '_result_holding.jpg', image)
