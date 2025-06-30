import os
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from torch.utils.data import Dataset


class MultiModalActionDataset(Dataset):
    def __init__(self,
                 labels_path: str,
                 split: str = 'train',
                 frame_step: int = 1,
                 sequence_length: int = 30,
                 transform=None):
        """Dataset class for ActionFormer.

        The dataset expects the following directory structure:
        project_root/
            dataset/
                videos/
                        ...video files...
                action_dataset/
                    labels.csv
                    keypoints/
                        ...keypoint .npz files...
                    masks/
                        ...mask .npy files...

        :param labels_path: Path to CSV file containing dataset annotations.
        :param split: Dataset split to use ('train', 'val', 'test').
        :param frame_step: Step between frames when creating sequences.
        :param sequence_length: Number of frames in each sequence.
        :param transform: Optional transform to be applied on masks.
        """
        self.root_dir = Path(labels_path).resolve().parent.parent.parent
        self.split = split
        self.frame_step = frame_step
        self.sequence_length = sequence_length
        self.transform = transform
        self.labels_df = pd.read_csv(os.path.join(self.root_dir, labels_path))
        self.labels_df = self.labels_df[self.labels_df['split'] == split]
        self.samples = self._prepare_samples()
        self.action_to_idx = {action: idx for idx, action in
                              enumerate(sorted(self.labels_df['target'].unique()))}

    def _prepare_samples(self) -> List[Dict]:
        """Prepare list of samples by scanning the dataset directories.

        :return: List of prepared samples, each containing sequence information.
        """
        samples = []

        for _, row in self.labels_df.iterrows():
            action = row['class']
            target = row['target']
            video_path = row['video_path']
            keypoints_path = row['keypoints_path']
            masks_path = row['masks_path']

            keys_dir = os.path.join(self.root_dir, keypoints_path)
            frame_files = sorted([f for f in os.listdir(keys_dir) if f.endswith('.npz')])

            for i in range(0, len(frame_files) - self.sequence_length * self.frame_step + 1, self.frame_step):
                sequence = []
                for j in range(self.sequence_length):
                    frame_idx = i + j * self.frame_step
                    frame_id = frame_files[frame_idx][:-4]
                    sequence.append({
                        'frame_id': frame_id,
                        'keypoints_path': os.path.join(keypoints_path, f'frame_{frame_id.split("_")[-1]}.npz'),
                        'mask_path': os.path.join(masks_path, f'frame_{frame_id.split("_")[-1]}.npy')
                    })

                sample = {
                    'action': action,
                    'video': video_path,
                    'target': target,
                    'sequence': sequence
                }
                samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        """Get sample by index with optional augmentations.

        :param idx: Index.
        :return: Tuple of (features, target).
        """
        sample = self.samples[idx]
        sequence_data = []
        mask_sequence = []

        for frame in sample['sequence']:
            kp_path = os.path.join(self.root_dir, frame['keypoints_path'])
            kp_data = np.load(kp_path, allow_pickle=True)
            points_dict = kp_data['points'].item()

            points = np.zeros((17, 3), dtype=np.float32)
            for i in range(17):
                if i in points_dict:
                    points[i] = points_dict[i]
                else:
                    points[i] = [0, 0, 0]

            sequence_data.append(points)

            mask_path = os.path.join(self.root_dir, frame['mask_path'])
            mask = np.load(mask_path)
            mask = cv2.resize(mask, (512, 512))
            mask_sequence.append(mask)

        sequence_np = np.array(sequence_data)
        keypoints_tensor = torch.from_numpy(sequence_np).float().permute(2, 0, 1)  # [C, T, V] -- [3, 30, 17]

        masks_np = np.stack(mask_sequence)  # [T, H, W]
        masks_tensor = torch.FloatTensor(masks_np).unsqueeze(1)  # [T, 1, H, W]

        if self.transform:
            masks_tensor = self.transform(masks_tensor)

        label = self.action_to_idx[sample['target']]

        return {
            'keypoints': keypoints_tensor,
            'masks': masks_tensor
        }, label

    def get_num_classes(self) -> int:
        return len(self.action_to_idx)

    def get_action_names(self) -> List[str]:
        return list(self.action_to_idx.keys())

    @staticmethod
    def get_adj_matrix() -> torch.Tensor:
        """Get the adjacency matrix for skeleton graph like in COCO."""
        num_nodes = 17
        adj_matrix = torch.zeros(num_nodes, num_nodes)

        edges = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (0, 5), (0, 6), (5, 7), (7, 9),
            (6, 8), (8, 10), (5, 6), (5, 11),
            (6, 12), (11, 13), (13, 15), (12, 14),
            (14, 16), (11, 12)
        ]

        for i, j in edges:
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1

        adj_matrix += torch.eye(num_nodes)

        degree = adj_matrix.sum(dim=1)
        degree_sqrt_inv = torch.diag(1.0 / torch.sqrt(degree))
        adj_matrix_normalized = degree_sqrt_inv @ adj_matrix @ degree_sqrt_inv
        adj_matrix_normalized = adj_matrix_normalized.unsqueeze(0)
        return adj_matrix_normalized
