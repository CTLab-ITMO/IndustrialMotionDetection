import os
import cv2
import yaml
from tqdm import tqdm
from pathlib import Path


class VideoFrameExtractor:
    def __init__(self, videos_path: str, output_dir: str):
        """
        Initializes the frame extractor from the video.

        :param videos_path: Path to the directory with video files
        :param output_dir: Path for saving extracted frames
        """
        self.videos_path = Path(videos_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def extract_frames_from_video(video_path: Path, output_path: Path) -> None:
        """
        Extracts frames from a single video file and saves them to the specified directory.

        :param video_path: Path to the video file
        :param output_path: Directory for saving frames
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"Ошибка: Не удалось открыть видео {video_path}")
                return

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_filename = output_path / f"frame_{frame_count:04d}.jpg"
                cv2.imwrite(str(frame_filename), frame)
                frame_count += 1

            cap.release()
        except Exception as e:
            print(f"Ошибка при обработке видео {video_path}: {e}")

    def process_action(self, action_name: str) -> None:
        """
        Processes all videos for the specified action.

        :param action_name: Action name (directory name)
        """
        action_path = self.videos_path / action_name
        if not action_path.is_dir():
            return

        action_output_path = self.output_dir / action_name
        action_output_path.mkdir(exist_ok=True)

        video_files = [f for f in action_path.iterdir() if f.is_file()]
        for video_file in tqdm(video_files, desc=f"Processing {action_name}"):
            video_stem = video_file.stem  # Имя файла без расширения
            frames_output_path = action_output_path / video_stem
            frames_output_path.mkdir(exist_ok=True)

            self.extract_frames_from_video(video_file, frames_output_path)

    def run(self, target_action: str = "person_picks_up_object_from_table") -> None:
        """
        Starts the process of extracting frames for the specified action.

        :param target_action: Name of the action to process (default is 'person_picks_up_object_from_table')
        """
        print(f"Обработка действия: {target_action}")
        self.process_action(target_action)

    def run_all(self):
        """
        Starts the process of extracting frames for the all actions.
        """
        for action_name in os.listdir(VIDEOS_PATH):
            self.run(target_action=action_name)


if __name__ == "__main__":
    with open("configs/models.yaml", "r") as f:
        config = yaml.safe_load(f)

    VIDEOS_PATH = config["dataset"]["videos"]
    OUTPUT_DIR = config["dataset"]["frames"]

    extractor = VideoFrameExtractor(VIDEOS_PATH, OUTPUT_DIR)
    extractor.run_all()
