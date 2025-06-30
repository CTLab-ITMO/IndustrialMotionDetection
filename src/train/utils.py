import json
from collections import Counter, defaultdict
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt, patches

from src.metrics_impl.BoundingBox import BoundingBox
from src.metrics_impl.BoundingBoxes import BoundingBoxes
from src.metrics_impl.utils import BBType, MethodAveragePrecision


def summarize_frame_labels(json_paths: List[str]):
    total_frames = 0
    frame_counter = Counter()

    for path in json_paths:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:

            labels_per_frame = defaultdict(set)
            box = item.get("box", [])
            for person in box:
                n_frames = int(person.get("framesCount", 0))
                total_frames += n_frames
                labels = person.get("labels", [])
                for seq in person.get("sequence", []):
                    frame_no = seq.get("frame")

                    if frame_no is not None and 1 <= frame_no <= n_frames:
                        for lbl in labels:
                            labels_per_frame[frame_no].add(lbl)

            for lbls in labels_per_frame.values():
                for lbl in lbls:
                    frame_counter[lbl] += 1

    labeled_frames = sum(frame_counter.values())
    unlabeled = total_frames - labeled_frames

    print(f"total frames: {total_frames}")
    print(f"frames no label:      {unlabeled}\n")

    for lbl, cnt in frame_counter.most_common():
        print(f"  {lbl:20s}: {cnt}")
    print(f"\nunlabeled frames: {unlabeled}")
    weights = dict(frame_counter.most_common())
    weights["background"] = unlabeled + weights.pop("person", None)
    return weights


def summarize_csv_labels(csv_path: str):
    import pandas as pd

    data = pd.read_csv(csv_path, low_memory=False)

    frame_counter = Counter(data['action_category'])

    person_count = frame_counter.pop('person', 0)

    total_frames = sum(frame_counter.values()) + person_count

    print(f"total frames: {total_frames}")
    print(f"frames with person label: {person_count}\n")

    for lbl, cnt in frame_counter.most_common():
        print(f"  {lbl:30s}: {cnt}")

    weights = dict(frame_counter.most_common())

    weights["background"] = 1

    return weights


def split_annotations_csv(csv_path: str, train_ratio: float = 0.8, random_seed: int = 42, output_dir: str = None):
    import pandas as pd
    import os
    import random

    random.seed(random_seed)

    data = pd.read_csv(csv_path, low_memory=False)

    unique_videos = data['video_path'].unique()

    random.shuffle(unique_videos)

    split_idx = int(len(unique_videos) * train_ratio)
    train_videos = unique_videos[:split_idx]
    test_videos = unique_videos[split_idx:]

    train_df = data[data['video_path'].isin(train_videos)]
    test_df = data[data['video_path'].isin(test_videos)]

    print(f"Total videos {len(unique_videos)}")
    print(f"Train videos {len(train_videos)} ({len(train_df)}")
    print(f"Test videos {len(test_videos)} ({len(test_df)}")

    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    train_csv_path = os.path.join(output_dir, 'train_annotations.csv')
    test_csv_path = os.path.join(output_dir, 'test_annotations.csv')

    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    return train_csv_path, test_csv_path


def add_pred_bboxes(bboxes,
                    scores,
                    labels,
                    nameOfImage,
                    bbFormat,
                    coordType,
                    allBoundingBoxes,
                    imgSize=(0, 0)):
    for box, score, label in list(zip(bboxes, scores, labels)):
        x_min, y_min, x_max, y_max = box
        w = x_max - x_min
        h = y_max - y_min
        bb = BoundingBox(nameOfImage,
                         label,
                         x_min, y_min, w, h,
                         coordType,
                         imgSize,
                         BBType.Detected,
                         score,
                         format=bbFormat)
        allBoundingBoxes.addBoundingBox(bb)
    return allBoundingBoxes


def add_gt_bboxes(bboxes,
                  labels,
                  nameOfImage,
                  bbFormat,
                  coordType,
                  allBoundingBoxes,
                  imgSize=(0, 0)):
    for j in range(len(bboxes)):
        x_min, y_min, x_max, y_max = bboxes[j]
        label = np.argmax(labels[j])
        w = x_max - x_min
        h = y_max - y_min
        bb = BoundingBox(nameOfImage,
                         label,
                         x_min, y_min, w, h,
                         coordType,
                         imgSize,
                         BBType.GroundTruth,
                         format=bbFormat)
        allBoundingBoxes.addBoundingBox(bb)
    return allBoundingBoxes


def compute_metrics_batch_detections(
        detection_results,
        x,
        bbFormat,
        coordType,
        evaluator,
        iouThreshold,
        savePath,
        showPlot,
):
    allBoundingBoxes = BoundingBoxes()
    allClasses = []

    for j in range(len(detection_results)):
        boxes = detection_results[j]['boxes'].cpu().detach().numpy()
        scores = detection_results[j]['scores'].cpu().detach().numpy()
        labels = detection_results[j]['labels'].cpu().detach().numpy()

        nameOfImage = "/".join(x['path'][j].split('/')[-2:])
        imgSize = tuple(x['video'].shape[-2:])

        add_pred_bboxes(boxes, scores, labels,
                        nameOfImage, bbFormat, coordType, allBoundingBoxes, imgSize)

        add_gt_bboxes(x['bbox'][j].bbox.cpu().detach().numpy(),
                      x['target'][j].cpu().detach().numpy(),
                      nameOfImage, bbFormat, coordType, allBoundingBoxes, imgSize)

    print(allBoundingBoxes)
    detections = evaluator.PlotPrecisionRecallCurve(
        allBoundingBoxes,
        IOUThreshold=iouThreshold,
        method=MethodAveragePrecision.EveryPointInterpolation,
        showAP=True,
        showInterpolatedPrecision=False,
        savePath=savePath,
        showGraphic=showPlot)

    return detections


def draw_bboxes(video, boxes, scores, labels, bboxes_path, top_k=3):
    fig, ax = plt.subplots(1)
    middle_frame_index = video.shape[1] // 2

    image = video[:, middle_frame_index].permute(1, 2, 0)
    image = (image - image.min()) / (image.max() - image.min())

    ax.imshow(image)

    if len(boxes) > 0:

        for box, score, label in list(zip(boxes, scores, labels))[:top_k]:
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min

            rect = patches.Rectangle((x_min, y_min), width, height,
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            label_text = f"Label: {label}, Score: {score:.2f}"
            ax.text(x_min, y_min - 7, label_text,
                    color='red', fontsize=8, backgroundcolor='white')
    else:

        ax.text(10, 30, "No detections", color='red', fontsize=12, backgroundcolor='white')

    plt.savefig(bboxes_path)
    plt.close(fig)


def draw_bboxes_with_gt(video, boxes, scores, labels, bboxes_path, boxes_gt, labels_gt, idx_to_class, top_k=3):
    fig, ax = plt.subplots(1)
    middle_frame_index = video.shape[1] // 2

    image = video[:, middle_frame_index].permute(1, 2, 0)
    image = (image - image.min()) / (image.max() - image.min())

    ax.imshow(image)

    if len(boxes) > 0:

        for box, score, label in list(zip(boxes, scores, labels))[:top_k]:
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min

            rect = patches.Rectangle((x_min, y_min), width, height,
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            label_text = f"{label} {score:.2f}"
            ax.text(x_min, y_min - 7, label_text,
                    color='red', fontsize=4, backgroundcolor='white')
    else:

        ax.text(10, 30, "No detections", color='red', fontsize=12, backgroundcolor='white')

    if len(boxes_gt) > 0:
        for box, label in list(zip(boxes_gt, labels_gt)):
            print(box)
            x_min, y_min, x_max, y_max = box.bbox[0]
            width = x_max - x_min
            height = y_max - y_min

            rect = patches.Rectangle((x_min, y_min), width, height,
                                     linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            label = idx_to_class[int(torch.argmax(label))]
            label_text = f"{label}"
            ax.text(x_min, y_max + 4, label_text,
                    color='green', fontsize=4, backgroundcolor='white')
    else:
        ax.text(10, 30, "No detections", color='red', fontsize=12, backgroundcolor='white')
    plt.savefig(bboxes_path)
    plt.close(fig)
