import os
import yaml
import numpy as np
import pandas as pd
from statistics import mode
from main import get_action
from sklearn.metrics import average_precision_score


def get_video_predictions(video_folders, get_action_func):
    """
    Generates model predictions for each video by aggregating frame-level predictions.

    Args:
        video_folders (list): List of paths to video folders (e.g., ["video1", "video2"]).
        get_action_func (function): Function that returns a prediction for a given frame.

    Returns:
        dict: Dictionary {video_name: predicted_class}.
    """
    predictions = {}

    for video in video_folders:
        frames_dir = f"{video}/"
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])

        frame_predictions = []
        for frame_file in frame_files:
            frame_path = os.path.join(frames_dir, frame_file)
            action = get_action_func(frame_path)
            frame_predictions.append(action)

        final_prediction = mode(frame_predictions)

        predictions[video] = final_prediction

    return predictions


def calculate_map(ground_truth, predictions, classes=None):
    """
    Computes mean Average Precision (mAP) for video action classification.

    Args:
        ground_truth (dict): Dictionary {video_name: true_class}.
        predictions (dict): Dictionary {video_name: predicted_class}.
        classes (list, optional): List of classes. If None, uses unique classes from ground_truth.

    Returns:
        float: mAP score.
    """
    if classes is None:
        classes = list(set(ground_truth.values()))

    ap_scores = []

    for cls in classes:
        y_true = []
        y_scores = []

        for video, true_label in ground_truth.items():
            pred = predictions.get(video, None)
            if pred is None:
                continue

            y_true.append(1 if true_label == cls else 0)
            y_scores.append(1 if pred == cls else 0)

        if sum(y_true) == 0:
            continue

        ap = average_precision_score(y_true, y_scores)
        ap_scores.append(ap)

    return np.mean(ap_scores) if ap_scores else 0.0


if __name__ == "__main__":
    """Calculates mAP for a single action."""
    with open("configs/models.yaml", "r") as f:
        config = yaml.safe_load(f)

    video_path = 'data/'
    video_folders = [os.path.join(video_path, video_name) for video_name in os.listdir(video_path)]
    df = pd.read_csv(config["dataset"]["all_labels_path"])
    df["video_path"] = df["video_path"].str.strip()
    df["action_category"] = df["action_category"].str.lower()

    ground_truth = dict(zip(df["video_path"], df["action_category"]))

    predictions = get_video_predictions(video_folders, get_action)

    mAP = calculate_map(ground_truth, predictions)
    print(f"mAP: {mAP:.4f}")
