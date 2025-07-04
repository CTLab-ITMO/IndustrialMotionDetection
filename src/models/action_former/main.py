import cv2
import torch
import argparse
import logging
import warnings
from typing import Dict
from statistics import mode

from interaction_analysis.baseline import PoseMaskAnalyzer
from utils.utils import annotate_image, preprocess_sequence
from interaction_analysis.action_recognizer.model.action_former import ActionFormer
from utils.model_handlers import run_segmentation_model, run_depth_model, run_keypoints_model
from utils.config import MODEL_CONFIGS, SEQUENCE_LENGTH, ACTIONS_WITH_OBJECTS, USE_DEPTH, MASK
from interaction_analysis.action_recognizer.model.action_dataset import MultiModalActionDataset

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def process_action_former_prediction(model: ActionFormer, inputs: Dict[str, torch.Tensor]) -> str:
    """Prediction based on action former model.

    :param model: Neural network model
    :param inputs: Input data

    :return: Predicted class
    """
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.item()


def get_action_by_action_former(video_source: str) -> None:
    """Action recognition pipeline using baseline ActionFormer model.

    :param video_source: Path to video file or stream.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ActionFormer(
        num_classes=5,
        adj_matrix=MultiModalActionDataset.get_adj_matrix()
    )
    model.load_state_dict(torch.load(MODEL_CONFIGS["action_former"]))
    model = model.to(device)
    model.eval()

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logger.error("Error opening the video stream")
        return

    sequence_buffer = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            keypoints, annot_frame = run_keypoints_model(frame)
            mask, _ = run_segmentation_model(frame, list(ACTIONS_WITH_OBJECTS.keys()))

            if len(sequence_buffer) >= SEQUENCE_LENGTH:
                sequence_buffer.pop(0)
            sequence_buffer.append({'keypoints': keypoints, 'mask': mask})

            if len(sequence_buffer) == SEQUENCE_LENGTH:
                inputs = preprocess_sequence(sequence_buffer)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                action_idx = process_action_former_prediction(model, inputs)
                annot_frame = annotate_image(annot_frame, str(action_idx))

            cv2.imshow('Action Recognition', annot_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            logger.error(f"Error processing frame: {e}")

    cap.release()
    cv2.destroyAllWindows()


def get_action_by_base_model(video_source: str, forbidden_territory: bool) -> None:
    """Action recognition pipeline using baseline PoseMaskAnalyzer model.

    :param video_source: Path to video file or stream.
    """
    model = PoseMaskAnalyzer(use_depth=USE_DEPTH)
    labels_buffer = []

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logger.error("Cannot open video file")
        return

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        depth_map = run_depth_model(image) if USE_DEPTH else None
        keypoints, annotated_image = run_keypoints_model(image)
        try:
            if forbidden_territory:
                mask = MASK
                action = 'not safe walking' if model.predict(image, mask, keypoints, depth_map) else 'do nothing'
            else:
                mask, thing = run_segmentation_model(image, list(ACTIONS_WITH_OBJECTS.keys()))
                action = 'do nothing'
                if thing != 'nothing' and mask is not None and keypoints is not None:
                    if model.predict(image, mask, keypoints, depth_map):
                        action = ACTIONS_WITH_OBJECTS[thing]

            labels_buffer.append(action)
            if len(labels_buffer) >= 7:
                annotated_image = annotate_image(annotated_image, mode(labels_buffer))
                labels_buffer = []
            cv2.imshow('Action Recognition', annotated_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            logger.error(f"Error processing frame: {e}")

    cap.release()
    cv2.destroyAllWindows()


def get_action(video_address: str, model_type: str, forbidden_territory: bool) -> None:
    if model_type == 'base_model' or forbidden_territory:
        get_action_by_base_model(video_address, forbidden_territory)
    else:
        get_action_by_action_former(video_address)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Action recognition system')
    parser.add_argument('--video_address', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--forbidden_territory', type=bool, required=False)
    args = parser.parse_args()

    get_action(args.video_address, args.model, args.forbidden_territory)
