import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ActionDataset(Dataset):
    """
    data_dir: Папка с данными в формате:
        data_dir/
            masks/         # .npy файлы масок
            optical_flow/  # .npy файлы оптического потока
            keypoints/     # .npz файлы с ключевыми точками и ребрами
            labels.npy     # Метки классов
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.labels = np.load(os.path.join(data_dir, 'labels.npy'))
        self.actions = os.listdir(os.path.join(data_dir, 'keypoints'))
        self.video_paths = []

        for action in self.actions:
            action_dir = os.path.join(data_dir, 'keypoints', action)
            if os.path.isdir(action_dir):
                videos = os.listdir(action_dir)
                for video in videos:
                    self.video_paths.append((action, video))

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        action, video = self.video_paths[idx]

        mask_path = os.path.join(self.data_dir, 'masks', action, video)
        mask = np.load(mask_path)
        mask = torch.FloatTensor(mask).unsqueeze(0)

        flow_path = os.path.join(self.data_dir, 'optical_flow', action, video)
        flow = np.load(flow_path)
        flow = torch.FloatTensor(flow)

        skeleton_path = os.path.join(self.data_dir, 'keypoints', action, video)
        skeleton_data = np.load(skeleton_path)
        points = torch.FloatTensor(skeleton_data['points'])
        edges = torch.LongTensor(skeleton_data['edges'])

        label = torch.LongTensor([self.labels[action]])

        return mask, flow, points, edges.T, label
