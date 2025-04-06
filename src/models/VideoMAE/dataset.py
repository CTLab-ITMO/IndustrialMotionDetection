import random
import pandas as pd
import numpy as np
import torch
import decord
from collections import defaultdict
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import IterableDataset
from box_list import BoxList
from transformers import AutoImageProcessor


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    low_bound = min(converted_len, seg_len - 1) # <= seg_len
    end_idx = np.random.randint(low=low_bound, high=seg_len) # end_idx = [low, high)
    start_idx = max(end_idx - converted_len, 0) # start_idx = [0, high - low)
    # print(f"{start_idx=} {end_idx=}")
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    # NOTE: may reduce frame_sample_rate
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

# TODO: move to config or external variable 
class_to_idx = {'person_talks_on_phone': 1,
                'person_texts_on_phone': 2,
                'person_interacts_with_laptop': 3,
                'person_reads_document': 4,
                'person_picks_up_object_from_table': 5,
                'background': 0}
                

class RandomDatasetDecord(IterableDataset):
    def __init__(self,
                 csv_file,
                 epoch_size_ratio=1.0,
                 epoch_size=None,
                 frame_transform=None,
                 video_transform=None,
                 frame_sample_rate=1,
                 clip_len=16,
                 num_classes=5):
        """
        Args:
            frame_sample_rate (int): Sample every n-th frame.
            clip_len (int): Total number of frames to sample.
        """
        super().__init__()

        self.data = pd.read_csv(csv_file, low_memory=False)

        self.video_paths = self.data.video_path.unique()
        self.video_paths_len = len(self.video_paths)

        if epoch_size is None:
            epoch_size = int(self.video_paths_len * epoch_size_ratio)

        self.epoch_size = epoch_size
        self.clip_len = clip_len
        self.num_classes = num_classes
        self.frame_sample_rate = frame_sample_rate
        self.frame_transform = frame_transform
        self.video_transform = video_transform

    def __len__(self):
        return self.epoch_size

    def __iter__(self):
        for _ in range(self.epoch_size):
            random_index = random.randint(0, self.video_paths_len - 1)

            video_path = self.video_paths[random_index]

            try:
                vr = decord.VideoReader(video_path)
            except RuntimeError:
                print(f'Can not read {video_path}')
                continue

            total_frames = len(vr)

            indices = sample_frame_indices(self.clip_len, self.frame_sample_rate, total_frames)

            keyframe_id = int(indices[len(indices) // 2])

            video = vr.get_batch(indices)

            row = self.data.loc[(self.data["video_path"] == video_path) & (self.data["keyframe_id"] == keyframe_id)]

            action_category = row["action_category"].item()
            xmin, ymin, xmax, ymax = map(float, [row["xmin"].item(), row["ymin"].item(), row["xmax"].item(), row["ymax"].item()])

            video_w, video_h = video.shape[1], video.shape[2]

            if self.video_transform:
                video = self.video_transform(list(video), return_tensors="pt").pixel_values[0]

            video_w_after, video_h_after = video.shape[-2:]

            one_hot_target = torch.zeros((1, self.num_classes), dtype=torch.bool)
            one_hot_target[0][class_to_idx[action_category]] = True
            one_hot_target = one_hot_target.to(torch.long)

            yield {
                "path": f"{video_path}-{keyframe_id}",
                "video": video.permute(1, 0, 2, 3), # [TxCxHxW] -> [CxTxHxW]
                "target": one_hot_target,
                "bbox": BoxList([[xmin, ymin, xmax, ymax]], (video_w, video_h)).resize((video_w_after, video_h_after))
            }

def collate_fn(batch):
    video_list = []
    target_list = []
    bbox_list = []
    path_list = []

    for sample in batch:
        video_list.append(sample["video"])
        target_list.append(sample["target"]) # [1, 5]
        bbox_list.append(sample["bbox"])
        path_list.append(sample["path"])

    videos = torch.stack(video_list)
    targets = torch.stack(target_list)

    return {
        "video": videos, # [B, C, T, H, W]
        "target": targets, # [B, N, num_cls]
        "bbox": bbox_list, # [B, BoxList[N]]
        "path": path_list
    }


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    

def get_dataloaders(train_csv_file_path: str, 
                    val_csv_file_path: str,
                    num_classes: str,
                    batch_size, 
                    seed, 
                    frame_sample_rate=1, 
                    clip_len=16,
                    val_epoch_size_ratio: float = 1.0,
                    train_epoch_size_ratio: float = 1.0):
    
    image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")

    image_datasets = dict()
    image_datasets['train'] = RandomDatasetDecord(
        train_csv_file_path,
        epoch_size_ratio=train_epoch_size_ratio,
        video_transform=image_processor,
        frame_sample_rate=frame_sample_rate,
        clip_len=clip_len,
        num_classes=num_classes)

    image_datasets['test'] = RandomDatasetDecord(
        val_csv_file_path,
        epoch_size_ratio=val_epoch_size_ratio,
        video_transform=image_processor,
        frame_sample_rate=frame_sample_rate,
        clip_len=clip_len,
        num_classes=num_classes)
    
    g = torch.Generator()
    g.manual_seed(seed)
    
    dataloaders = dict()
    dataloaders['train'] = DataLoader(
        image_datasets['train'],
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=g)
    
    dataloaders['test'] = DataLoader(
        image_datasets['test'],
        batch_size=batch_size,
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=g)

    return dataloaders
