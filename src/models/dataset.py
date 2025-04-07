import math
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from pytorchvideo.data.encoded_video import EncodedVideo
from models.box_list import BoxList


class RandomDatasetDecord(IterableDataset):
    def __init__(self,
                 csv_file,
                 class2idx: dict,
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
        
        self.class2idx = class2idx

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

            frames_per_second = 30
            clip_duration = (self.clip_len * self.frame_sample_rate) / frames_per_second

            video = EncodedVideo.from_path(video_path)

            # start_sec == 1, since 30 padding frames
            offset_sec = 0.4
            start_sec = random.uniform(offset_sec, float(video.duration) - clip_duration - offset_sec)
            end_sec = start_sec + clip_duration
            # Load the desired clip [C, T, H, W]
            video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)['video']

            video.close()

            start_frame = math.floor(start_sec * frames_per_second)
            end_frame = math.floor(end_sec * frames_per_second)

            keyframe_id = (start_frame + end_frame) // 2

            row = self.data.loc[((self.data["video_path"] == video_path)
                                 & (self.data["keyframe_id"] == keyframe_id))]

            if len(row) == 0:
                print("WARNING: zero matches found")
                print(f"{video_path=} {keyframe_id=}")
                print(f"{type(video_path)=} {type(keyframe_id)=}")
                print(f'{self.data["video_path"].dtype=} {self.data["keyframe_id"].dtype=}')
                print()

            # Convert to a PyTorch tensor of shape Nx4
            boxes = torch.tensor(row[["xmin", "ymin", "xmax", "ymax"]].values.astype(float),
                                 dtype=torch.float32)

            action_categories = row[["action_category"]].values.reshape(-1).astype(str)

            one_hot_target = torch.zeros((len(action_categories), self.num_classes),
                                         dtype=torch.bool)

            for i, c in enumerate(action_categories):
                one_hot_target[i][self.class2idx[str(c)]] = True
            one_hot_target = one_hot_target.to(torch.long)

            if len(row) > 1:
                print(f"INFO: more then one matches")
                print(f"paths={row['video_path'].unique().tolist()} {keyframe_id=}")
                print(f"{boxes=} {one_hot_target=}")
                print()

            video_h, video_w = video_data.shape[-2:]

            if self.video_transform:
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]

                transform = Compose([
                    UniformTemporalSubsample(self.clip_len),
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(mean, std),
                ])

                video_data = transform(video_data)

                video_data, boxes = short_side_scale_with_boxes(
                    video_data, size=224, backend="pytorch", boxes=boxes
                )
                video_data, boxes = uniform_crop_with_boxes(
                    video_data, size=224, spatial_idx=1, boxes=boxes
                )

            video_h, video_w = video_data.shape[-2:]

            yield {
                "path": f"{video_path}-{keyframe_id}",
                "video": video_data, # [CxTxHxW]
                "target": one_hot_target, # [N, cls]
                "bbox": BoxList(boxes, (video_w, video_h))
            }


def collate_fn(batch):
    video_list = []
    target_list = []
    bbox_list = []
    path_list = []

    for sample in batch:
        video_list.append(sample["video"])
        target_list.append(sample["target"]) # [N, cls]
        bbox_list.append(sample["bbox"])
        path_list.append(sample["path"])

    videos = torch.stack(video_list)

    return {
        "video": videos, # [B, C, T, H, W]
        "target": target_list, # [B, N, num_cls]
        "bbox": bbox_list, # [B, BoxList[N]]
        "path": path_list
    }
    

def get_dataloaders(train_csv_file_path: str, 
                    val_csv_file_path: str,
                    num_classes: str,
                    batch_size, 
                    seed, 
                    frame_sample_rate=1, 
                    clip_len=16,
                    val_epoch_size_ratio: float = 1.0,
                    train_epoch_size_ratio: float = 1.0):
    
    image_datasets = dict()
    image_datasets['train'] = RandomDatasetDecord(
        train_csv_file_path,
        epoch_size_ratio=train_epoch_size_ratio,
        video_transform=None,
        frame_sample_rate=frame_sample_rate,
        clip_len=clip_len,
        num_classes=num_classes)

    image_datasets['test'] = RandomDatasetDecord(
        val_csv_file_path,
        epoch_size_ratio=val_epoch_size_ratio,
        video_transform=None,
        frame_sample_rate=frame_sample_rate,
        clip_len=clip_len,
        num_classes=num_classes)
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
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
