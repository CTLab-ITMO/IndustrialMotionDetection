import math
import random
from collections import Counter
import pandas as pd
import numpy as np
import torch
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from video.encoded_video import EncodedVideo
from logger import Logger
from models.box_list import BoxList


class RandomDatasetDecord(IterableDataset):
    def __init__(
        self,
        data_folder_path: str,
        csv_file_path: str,
        class2idx: dict,
        video_transform=None,
        epoch_size_ratio=1.0,
        frame_sample_rate=1,
        clip_len=16,
        num_classes=5
    ):
        super().__init__()
        
        SHOW_LOG = True
        self.logger = Logger(SHOW_LOG).get_logger(__name__)

        self.class2idx = class2idx
        
        self.data_folder_path = data_folder_path

        self.data = pd.read_csv(csv_file_path, low_memory=False)

        self.offset_frames = 15
        self.offset_sec = 0.4
        self.frames_per_second = 30
        
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.clip_duration = (self.clip_len * self.frame_sample_rate) / self.frames_per_second
        
        self.video_paths = self.data.video_path.unique()
        self.video_paths_len = len(self.video_paths)
        self.epoch_size = int(self.video_paths_len * epoch_size_ratio)
        
        self.num_classes = num_classes
        self.video_transform = video_transform
        
        unique_tracks_per_category = (
            self.data.groupby(['video_path', 'track_id', 'action_category'])
            .size()
            .groupby('action_category')
            .size()
            .reset_index(name='unique_tracks')
        )
        self.category_to_tracks = dict(zip(
            unique_tracks_per_category['action_category'], 
            unique_tracks_per_category['unique_tracks']))
        
        unique_frames_per_category = (
            self.data.groupby(['video_path', 'keyframe_id', 'action_category'])
            .size()
            .groupby('action_category')
            .size()
            .reset_index(name='unique_frames')
        )
        self.category_to_frames = dict(zip(
            unique_frames_per_category['action_category'], 
            unique_frames_per_category['unique_frames']))
        
        self.action_weights = self._calculate_action_weights()
        print(self.action_weights)
        
        self.grouped_data = self.data.groupby(['video_path', 'track_id'])
        self.groups = list(self.grouped_data.groups.keys())
        
        self.group_weights = self._calculate_group_weights()

    def _calculate_action_weights(self, alpha=0.5):
        """
        Calculate class weights based on inverse frequency with smoothing factor alpha
        """
        total_frames = sum(self.category_to_frames.values())
        total_tracks = sum(self.category_to_tracks.values())
        action_weights = {}
        for k in self.category_to_tracks.keys():
            frames_ratio = total_frames / self.category_to_frames[k]
            tracks_ratio = total_tracks / self.category_to_tracks[k]
            action_weights[k] = frames_ratio * tracks_ratio
        return action_weights
    
    def _calculate_group_weights(self):
        """
        Calculate sampling weights for each (video, track) group based on action frequency
        """
        group_weights = []
        for group_key in self.groups:
            group = self.grouped_data.get_group(group_key)
            main_action = str(group['action_category'].value_counts().idxmax())
            group_weights.append(self.action_weights[main_action])
        total = sum(group_weights)
        return [w/total for w in group_weights]

    def __len__(self):
        return self.epoch_size
    
    def _sample_consecutive_frames(self, group_annots, video_annots):
        group_frames = group_annots['keyframe_id'].unique()
        video_frames = video_annots['keyframe_id'].unique()
        
        max_available_frame_id = max(video_frames) 
        
        def clip(frame_id):
            return max(0, min(max_available_frame_id, frame_id))

        missing_count = self.offset_frames
        count_frames = max(group_frames) - min(group_frames) + 1 
        if count_frames < self.clip_len:
            missing_count = max(missing_count, math.ceil((self.clip_len - count_frames)/2))
        
        act_start_frame_id = random.randint(
            min(group_frames), 
            max(min(group_frames) + 1, max(group_frames) - (self.clip_len - 1))
        )
        
        start_frame_id = clip(act_start_frame_id - missing_count)
        end_frame_id = clip(act_start_frame_id + (self.clip_len - 1) + missing_count)
        return start_frame_id, end_frame_id 

    def __iter__(self):
        for _ in range(self.epoch_size):
            # Sample (video_path, track_id) pair
            group_key = random.choices(self.groups, weights=self.group_weights, k=1)[0]
            video_path, track_id = group_key
            
            group_annots = self.grouped_data.get_group(group_key)
            video_annots = self.data.loc[self.data["video_path"] == video_path]
            
            start_frame_id, end_frame_id = self._sample_consecutive_frames(group_annots, video_annots) 
        
            video_src = EncodedVideo.from_path(f"{self.data_folder_path}/{video_path}")
            # start_sec from offset_sec, since 30 padding frames
            start_sec = start_frame_id / self.frames_per_second
            end_sec = min(video_src.duration, math.ceil(end_frame_id / self.frames_per_second)) 
            # Load the desired clip [C, T, H, W]
            video_data = video_src.get_clip(start_sec=start_sec, end_sec=end_sec)['video']
            video_src.close()
            del video_src
            
            keyframe_id = (start_frame_id + end_frame_id) // 2
            frame_video_annots = video_annots[video_annots["keyframe_id"] == keyframe_id]

            if len(frame_video_annots) == 0:
                self.logger.warning(f"Zero matches found for {video_path} at {keyframe_id}")
                
            # Convert to a PyTorch tensor of shape Nx4
            boxes = torch.tensor(
                frame_video_annots[["xmin", "ymin", "xmax", "ymax"]].values.astype(float),
                dtype=torch.float32)

            action_categories = frame_video_annots[["action_category"]].values.reshape(-1)

            one_hot_target = torch.zeros(
                (len(action_categories), self.num_classes),
                dtype=torch.bool)

            for i, c in enumerate(action_categories):
                one_hot_target[i][self.class2idx[c]] = True

            one_hot_target = one_hot_target.to(torch.long)

            if len(frame_video_annots) > 1:
                self.logger.info(f"{len(frame_video_annots)} matches for {video_path} at {keyframe_id}")
                
            data = {
                "path": f"{video_path}-{keyframe_id}",
                "video": video_data, # [CxTxHxW]
                "target": one_hot_target, # [N, cls]
                "bbox": boxes, # [N, 4]
            }

            if self.video_transform:
                data = self.video_transform(data)
                
            video_h, video_w = data['video'].shape[-2:]
            data['bbox'] = BoxList(data['bbox'], (video_w, video_h))

            yield data


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
    

def get_dataloaders(
    train_csv_file_path: str, 
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
