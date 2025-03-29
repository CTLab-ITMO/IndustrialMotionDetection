import cv2
import yaml
import torch
import functools
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
from typing import List, Optional, Dict
from pose_estimation.pose_estimator import PoseEstimator
from depth_estimation.depth_estimator import DepthEstimator
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation


class DatasetCreator:
    """
    Class for creating dataset for AdaptedTwoStreamGCN model
        data_dir: directory with files in format:
            data_dir/
                masks/         # .npy files of masks
                    video1
                    ...
                    video_n
                optical_flow/  # .npy files of optical flow
                    video1
                    ...
                    video_n
                skeletons/     # .npz files of skeletons
                    video1
                    ...
                    video_n
                labels.npy     # labels
    """

    def __init__(
            self,
            dataset_input_path: str,
            output_base_path: str,
            model_path_depth: str,
            config_path_depth: str,
            config_path_seg: str,
            model_path_yolo: str,
            device: str = "cpu"
    ):
        self.dataset_input_path = Path(dataset_input_path)
        self.output_base_path = Path(output_base_path)

        # Output directories
        self.output_paths = {
            'images': self.output_base_path / "action_dataset" / "images",
            'masks': self.output_base_path / "action_dataset" / "masks",
            'keypoints': self.output_base_path / "action_dataset" / "keypoints",
            'flow': self.output_base_path / "action_dataset" / "flow",
            'depth': self.output_base_path / "action_dataset" / "depth"
        }

        # Object prompts for different actions
        self.action_prompts = {
            "drink": ["bottle", "cup"],
            "eat": ["food", "plate"],
            "smoke": ["cigarette"],
            "talks_on_phone": ["phone"],
            "text_on_phone": ["phone"],
            "interacts_with_laptop": ["laptop"]
        }

        # Initialize models
        self.clip_processor = CLIPSegProcessor.from_pretrained(config_path_seg)
        self.seg_model = CLIPSegForImageSegmentation.from_pretrained(config_path_seg)
        self.yolo_model = PoseEstimator(model_path_yolo)
        self.depth_model = DepthEstimator(
            model_config=config_path_depth,
            model_path=model_path_depth,
            device=device
        )

        # Create output directories
        for path in self.output_paths.values():
            path.mkdir(parents=True, exist_ok=True)

    def get_object_mask(
            self,
            image: Image.Image,
            prompts: List[str],
            threshold: float = 0.5
    ) -> np.ndarray:
        """Generate binary mask for specified objects"""
        inputs = self.clip_processor(
            text=prompts,
            images=[image] * len(prompts),
            padding="max_length",
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.seg_model(**inputs)

        # Combine masks from all prompts
        combined_mask = torch.zeros_like(outputs.logits)
        for i in range(len(prompts)):
            combined_mask += torch.sigmoid(outputs.logits[i])

        return (combined_mask > threshold).float().squeeze().cpu().numpy()

    @staticmethod
    def compute_optical_flow(
            frame_paths: List[Path],
            output_dir: Path
    ) -> None:
        """Calculate optical flow between consecutive frames"""
        prev_frame = cv2.imread(str(frame_paths[0]), cv2.IMREAD_GRAYSCALE)

        for i, frame_path in enumerate(frame_paths[1:]):
            next_frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)

            flow = cv2.calcOpticalFlowFarneback(
                prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            # Normalize and save flow
            np.save(
                output_dir / f"flow_{i:04d}.npy",
                flow / 20.0  # Empirical normalization
            )
            prev_frame = next_frame

    @staticmethod
    def get_keypoints(pose_result):
        # Get keypoints and boxes for persons
        keypoints = pose_result.keypoints.xy.cpu().numpy()[0]
        conf = pose_result.keypoints.conf.cpu().numpy()[0].squeeze()

        # Process keypoints
        points = {}
        for point_id, (x, y) in enumerate(keypoints, start=0):
            points[point_id] = (float(x), float(y), float(conf[point_id]))  # (x, y, conf)
        return points

    def process_frame(
            self,
            frame_path: Path,
            output_dirs: Dict[str, Path],
            action_name: Optional[str] = None
    ) -> None:
        """Process single frame and save all features"""
        try:
            image = Image.open(frame_path)
            frame_id = frame_path.stem
        except Exception as e:
            print(f"Error processing {frame_path}: {e}")
            return

        keypoints, _ = self.yolo_model.predict(image, verbose=False)
        if keypoints:
            points = self.get_keypoints(keypoints[0])
            edges = self._get_skeleton_edges(len(points))

            np.savez(
                output_dirs['keypoints'] / f"{frame_id}.npz",
                points=points,
                edges=edges
            )

        if action_name and action_name in self.action_prompts:
            mask = self.get_object_mask(
                image=image,
                prompts=self.action_prompts[action_name]
            )
            np.save(output_dirs['masks'] / f"{frame_id}.npy", mask)

        depth_map = self.depth_model.predict(np.array(image))
        np.save(output_dirs['depth'] / f"{frame_id}.npy", depth_map)

    @staticmethod
    def _get_skeleton_edges(num_points: int) -> np.ndarray:
        """Define skeleton connections between keypoints"""
        # TODO:: проверить, что тут правильно соединяются скелетные точки в ребра
        edges = [
            [0, 1], [0, 2], [1, 3], [2, 4],  # Head
            [5, 7], [7, 9], [6, 8], [8, 10],  # Arms
            [11, 13], [13, 15], [12, 14], [14, 16],  # Legs
            [5, 6], [11, 12], [5, 11], [6, 12]  # Body
        ]
        return np.array(edges).T if num_points == 17 else np.zeros((2, 0))

    def process_video_sequence(
            self,
            sequence_path: Path,
            action_name: str
    ) -> None:
        """Process all frames in a video sequence"""
        output_dirs = {
            k: v / action_name / sequence_path.name
            for k, v in self.output_paths.items()
        }

        for dir_path in output_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        frame_paths = sorted(sequence_path.glob("*.jpg"))
        if not frame_paths:
            print(f"No frames found in {sequence_path}")
            return

        self.compute_optical_flow(frame_paths, output_dirs['flow'])

        for frame_path in tqdm(frame_paths, desc=f"Processing {sequence_path.name}"):
            self.process_frame(frame_path, output_dirs, action_name)

        num_frames = len(frame_paths)
        np.save(
            output_dirs['keypoints'].parent.parent / "labels.npy",
            np.array([action_name] * num_frames)
        )

    def run_parallel(self) -> None:
        """Main processing pipeline in parallel"""
        action_dirs = [d for d in self.dataset_input_path.iterdir() if d.is_dir()]

        with Pool(processes=4) as pool:
            for action_dir in tqdm(action_dirs, desc="Processing actions"):
                action_name = action_dir.name
                sequence_dirs = [d for d in action_dir.iterdir() if d.is_dir()]

                process_func = functools.partial(
                    self.process_video_sequence,
                    action_name=action_name
                )

                pool.map(process_func, sequence_dirs)

    def run(self) -> None:
        """Main processing pipeline"""
        action_dirs = [d for d in self.dataset_input_path.iterdir() if d.is_dir()]

        for action_dir in tqdm(action_dirs, desc="Processing actions"):
            action_name = action_dir.name
            sequence_dirs = [d for d in action_dir.iterdir() if d.is_dir()]

            for seq_dir in sequence_dirs:
                self.process_video_sequence(seq_dir, action_name)


if __name__ == "__main__":
    with open("../configs/models.yaml", "r") as f:
        config = yaml.safe_load(f)

    creator = DatasetCreator(
        dataset_input_path=config["dataset"]["frames_path"],
        output_base_path=config["dataset"]["action_dataset"],
        model_path_depth=config["depth_estimation"]["model_path"],
        config_path_depth=config["depth_estimation"]["config_path"],
        config_path_seg=config["segmentation"]["config_path"],
        model_path_yolo=config["pose_estimation"]["model_path"],
        device=config["settings"]["device"]
    )

    creator.run()
