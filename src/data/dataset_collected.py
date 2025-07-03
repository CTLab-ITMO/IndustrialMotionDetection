import json
import os
from collections import defaultdict, Counter

from tqdm import tqdm

from src.models.box_list import BoxList


def collate_fn(batch):
    video_list = []
    target_list = []
    bbox_list = []
    path_list = []

    for sample in batch:
        video_list.append(sample["video"])
        target_list.append(sample["target"])
        bbox_list.append(sample["bbox"])
        path_list.append(sample["path"])

    videos = torch.stack(video_list)

    return {
        "video": videos,
        "target": target_list,
        "bbox": bbox_list,
        "path": path_list
    }


import random
import pandas as pd
import torch
from torch.utils.data import IterableDataset

import decord

decord.bridge.set_bridge('torch')


def sample_frame_indices_centered(clip_len, sample_rate, center, total_frames):
    half_span = (clip_len // 2) * sample_rate
    start = max(0, center - half_span)
    inds = list(range(start, min(start + clip_len * sample_rate, total_frames), sample_rate))
    if len(inds) < clip_len:
        extra = list(range(max(0, total_frames - clip_len * sample_rate), total_frames, sample_rate))
        inds = (inds + extra)[:clip_len]
    return inds


class ActionDetectionDataset(IterableDataset):

    def _get_video_lengths(self):
        if os.path.exists(self.lengths_file):
            with open(self.lengths_file, 'r') as f:
                return json.load(f)
        video_lengths = {}

        for vp in tqdm(self.video_paths, desc="Processing videos"):
            full_path = os.path.join(self.curr_path, vp)
            try:
                vr = decord.VideoReader(full_path, num_threads=1)
                video_lengths[vp] = len(vr)
            except Exception as e:
                print(f"Error processing {vp}: {str(e)}")
                video_lengths[vp] = 0

        with open(self.lengths_file, 'w') as f:
            json.dump(video_lengths, f)

        return video_lengths

    def _get_background_frames(self):
        self.background_frames = []

        for vp in self.video_paths:
            T = self.video_lengths[vp]
            annotated_frames = set(self.annots.get(vp, {}).keys())

            for frame_id in range(T):
                if T - self.clip_len > frame_id > self.clip_len + 1 and frame_id not in annotated_frames:
                    self.background_frames.append((vp, frame_id))

    def __init__(self,
                 current_dir,
                 csv_file,
                 background_label='background',
                 epoch_size_ratio=1.0,
                 epoch_size=None,
                 frame_sample_rate=1,
                 clip_len=16,
                 processor=None,
                 class_to_idx=None,
                 oversample=False,
                 oversample_ratio=2.0,
                 augmentations=None,
                 background_prob=0.1,
                 length_file=None):
        super().__init__()
        self.augmentations = augmentations
        self.background_prob = background_prob
        self.data = pd.read_csv(csv_file, low_memory=False)
        self.data = self.data[self.data["action_category"].isin(class_to_idx.keys())]
        self.curr_path = current_dir
        self.annots = {}
        class_frequencies = Counter()
        self.lengths_file = length_file

        for _, r in self.data.iterrows():
            (self.annots.setdefault(r.video_path, {}).
             setdefault(int(r.keyframe_id), []).append(r))
            if r.action_category != "person"  and r.action_category in class_to_idx.keys():
                class_frequencies[r.action_category] += 1

        self.video_paths = list(self.annots.keys())
        self.video_paths_len = len(self.video_paths)
        if epoch_size is None:
            epoch_size = int(self.video_paths_len * epoch_size_ratio)
        self.epoch_size = epoch_size
        self.frame_sample_rate = frame_sample_rate
        self.clip_len = clip_len
        self.processor = processor
        self.class_to_idx = class_to_idx or {}

        self.bg_idx = self.class_to_idx.get(background_label, len(self.class_to_idx))
        if background_label not in self.class_to_idx:
            self.class_to_idx[background_label] = self.bg_idx
        self.num_classes = len(self.class_to_idx)

        self.oversample = oversample
        self.oversample_ratio = oversample_ratio
        self.video_lengths = self._get_video_lengths()
        self._get_background_frames()
        if self.oversample:
            self._setup_oversampling(class_frequencies)
            print(self.class_weights)

    def _setup_oversampling(self, class_frequencies):
        max_freq = max(class_frequencies.values())
        self.class_weights = {}
        for cls, freq in class_frequencies.items():
            if freq > 0:
                self.class_weights[cls] = min(self.oversample_ratio, max_freq / freq)
            else:
                self.class_weights[cls] = self.oversample_ratio

        self.class_samples = defaultdict(list)
        for vp, annots_dict in self.annots.items():
            for frame_id, annots in annots_dict.items():
                if self.clip_len < frame_id < self.video_lengths[vp] - self.clip_len:
                    for row in annots:
                        if row.action_category != "person":
                            self.class_samples[row.action_category].append((vp, frame_id))

    def _sample_with_oversampling(self):
        if random.random() < self.background_prob:
            vp, frame_id = random.choice(self.background_frames)
            return vp, frame_id
        if not self.oversample:
            vp = random.choice(self.video_paths)
            keyframe = random.randint(self.clip_len + 1, self.video_lengths[vp] - self.clip_len - 1)
            return vp, keyframe

        classes = list(self.class_weights.keys())
        weights = [self.class_weights[cls] for cls in classes]
        selected_class = random.choices(classes, weights=weights, k=1)[0]

        if self.class_samples[selected_class]:
            vp, frame_id = random.choice(self.class_samples[selected_class])
            return vp, frame_id

        vp = random.choice(self.video_paths)
        keyframe = random.randint(self.clip_len + 1, self.video_lengths[vp] - self.clip_len - 1)
        return vp, keyframe

    def __len__(self):
        return self.epoch_size

    def __iter__(self):
        for _ in range(self.epoch_size):
            if self.oversample:
                vp, keyframe = self._sample_with_oversampling()
            else:
                vp, keyframe = self._sample_with_oversampling()
            full_path = os.path.join(self.curr_path, vp)
            vr = decord.VideoReader(full_path)
            T = len(vr)

            inds = sample_frame_indices_centered(self.clip_len,
                                                 self.frame_sample_rate,
                                                 keyframe, T)
            frames = vr.get_batch(inds)
            rows = self.annots[vp].get(keyframe, None)
            boxes = []
            labels = []
            frame_w = frames.shape[2]
            frame_h = frames.shape[1]
            if rows is not None:
                for row in rows:
                    if row.action_category != "person":
                        label_idx = self.class_to_idx[row.action_category]
                        x_perc = float(row.xmin)
                        y_perc = float(row.ymin)
                        w_perc = float(row.xmax)
                        h_perc = float(row.ymax)

                        xmin = x_perc * frame_w / 100.0
                        ymin = y_perc * frame_h / 100.0
                        xmax = w_perc * frame_w / 100.0
                        ymax = h_perc * frame_h / 100.0
                        # xmin = float(row.xmin)
                        # ymin = float(row.ymin)
                        # xmax = float(row.xmax)
                        # ymax = float(row.ymax)
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(label_idx)

            if not boxes:
                bbox = BoxList(torch.zeros((0, 4)), (frame_w, frame_h))
                one_hot_target = torch.zeros((1, self.num_classes), dtype=torch.long)
                one_hot_target[0, self.bg_idx] = 1
            else:
                bbox = BoxList(torch.tensor(boxes, dtype=torch.float32), (frame_w, frame_h))
                labels = torch.tensor(labels, dtype=torch.long)
                one_hot_target = torch.nn.functional.one_hot(labels, num_classes=len(self.class_to_idx))
            boxes = bbox.bbox

            frames = (frames.float() / 255.0).clip(0, 1)
            if self.augmentations is not None:
                frames = frames.permute(0, 3, 1, 2)
                boxes = torch.repeat_interleave(boxes[None, ...], frames.shape[0], dim=0)
                frames, boxes = self.augmentations(frames, boxes)
                boxes = boxes[frames.shape[0] // 2]
                frames = frames.permute(0, 2, 3, 1).clip(0, 1)

            frame_w = frames.shape[2]
            frame_h = frames.shape[1]
            if self.processor is not None:

                pv = self.processor(list(frames),
                                    return_tensors="pt", do_rescale=False)["pixel_values"]

                video_tensor = pv.permute(1, 0, 2, 3)

                video_w_after, video_h_after = video_tensor.shape[-2:]

                scale = 256 / min(frame_h, frame_w)
                boxes *= scale
                resized_h, resized_w = frame_h * scale, frame_w * scale

                offset_x = (resized_w - 224) / 2
                offset_y = (resized_h - 224) / 2
                boxes[:, [0, 2]] -= offset_x
                boxes[:, [1, 3]] -= offset_y
                if boxes.shape[0] > 0:
                    boxes_new = []
                    for ind in range(boxes.shape[0]):
                        xmin, ymin, xmax, ymax = boxes[ind]
                        outside = (
                                (xmax < 0)
                                | (xmin > video_w_after)
                                | (ymax < 0)
                                | (ymin > video_h_after)
                        )
                        if not outside:
                            boxes_new.append([xmin, ymin, xmax, ymax])
                    if not boxes_new:
                        bbox = BoxList(torch.zeros((0, 4)), (frame_w, frame_h))
                    else:
                        bbox = BoxList(torch.tensor(boxes_new, dtype=torch.float32), (frame_w, frame_h))
                else:
                    bbox = BoxList(boxes, (video_w_after, video_h_after))
            else:
                video_tensor = frames.permute(3, 0, 1, 2)
            yield {
                'path': f"{vp}-{keyframe}",
                'video': video_tensor,
                'target': one_hot_target,
                'bbox': bbox
            }
