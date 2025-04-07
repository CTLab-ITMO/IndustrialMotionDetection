import time
import tqdm
import cv2
import torch
import numpy as np
from box_list import BoxList
from pytorchvideo.data.encoded_video import EncodedVideo
from fractions import Fraction
from transforms import get_val_transform
from logger import Logger


class Predictor:
    def __init__(self, model):
        SHOW_LOG = True
        self.logger = Logger(SHOW_LOG).get_logger(__name__)
        
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = model
        self.idx2class = None
        
        self.fps = 30
        self.frames_count = 16
        
    def get_frames(
        self,
        video_src: EncodedVideo,
        start_frame: int,
    ):
        return video_src.get_clip(
            start_sec=Fraction(start_frame, self.fps),
            end_sec=Fraction(start_frame + self.frames_count, self.fps))['video']

    def predict(
        self, 
        input_video_path: str, 
        output_video_path: str = '/content/output_video.avi',
        len_seconds: float = 10.0
    ):
        T = self.fps * len_seconds
        num_batches = T - self.frames_count + 1
        
        video = EncodedVideo.from_path(input_video_path)
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, self.fps, (224, 224))
        
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(num_batches)):
                start = time.time()
                
                video_data = self.get_frames(
                    video, i, frames_count=self.frames_count)

                transform = get_val_transform()
                
                video_data = transform(video_data).unsqueeze(dim=0).to(self.device)

                B, C, T, H, W = video_data.shape

                bboxes = [BoxList([[0, 0, 5, 5]], (H, W)).to(self.device)]
                labels = [torch.Tensor([[0, 0, 0, 1, 0]]).to(self.device)]

                detection_results, _ = self.model(video_data, bboxes, labels)

                elapsed_time = time.time() - start

                frame = video_data[0][:, T // 2].permute(1, 2, 0).cpu().numpy() # [C, H, W]
                frame = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)
                frame = np.ascontiguousarray(frame)

                # print(frame.shape, type(frame), frame.dtype, frame.min(), frame.max())

                # print(detection_results)

                cv2.putText(img=frame, 
                            text=f"{elapsed_time:.2f}s", 
                            org=(25, 25), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale=0.3, 
                            color=(0, 255, 0), 
                            thickness=1, 
                            lineType=cv2.LINE_AA)

                preds_count = len(detection_results[0]['boxes'])

                if preds_count == 0:
                    self.logger.warning(f"Zero prediction for frames {i}-{i+self.frames_count}")
                    continue

                scores = detection_results[0]['scores'].cpu()
                max_score_index = scores.argmax().item()

                x1, y1, x2, y2 = detection_results[0]['boxes'][max_score_index].cpu().numpy().astype(int)
                cv2.rectangle(img=frame, 
                              pt1=(x1, y1),
                              pt2=(x2, y2),
                              color=(0, 255, 0),
                              thickness=1)

                label = detection_results[0]['labels'].cpu().numpy()[max_score_index]
                confidence = detection_results[0]['scores'].cpu().numpy()[max_score_index]
                text = f"{self.idx_to_class[label]}: {confidence:.2f}"

                cv2.putText(img=frame, 
                            text=text, 
                            org=(x1, y1 - 10), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale=0.3, 
                            color=(255, 0, 0), 
                            thickness=1, 
                            lineType=cv2.LINE_AA)

                out.write(frame)

        out.release()
        self.logger.info(f"Output video saved to {output_video_path}")

        video.close()
