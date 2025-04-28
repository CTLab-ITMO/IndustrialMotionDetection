import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from models.box_list import BoxList


class YOLO_VideoMAE(nn.Module):
    def __init__(self, video_model, yolo_model_name="yolo11n.pt"):
        self.yolo_imgsz = 2048
        self.video_model_img_size = 224
        self.yolo_model = YOLO(yolo_model_name)
        self.video_model = video_model
    
    def forward(self, x, gt_boxes, gt_labels):
        b, c, t, h, w = x.shape # [B, C, T, H, W]
        
        targets = []
        for i in range(b):
            d = {}
            d['boxes'] = gt_boxes[i].bbox  # [N, 4] float tensor
            d['labels'] = gt_labels[i].argmax(dim=1) # Tensor[N]
            targets.append(d)

        image = x[:, t//2].numpy().astype(np.uint8).transpose((1, 2, 0))
        results = self.yolo_model(image, imgsz=self.yolo_imgsz)
        proposals = []
        batch_scores = []
        for result in results:
            xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
            names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
            confs = result.boxes.conf  # confidence score of each box
            
            bboxes = []
            scores = []
            for box, cls_name, conf in zip(xyxy, names, confs):
                if conf < 0.5 or cls_name != 'person':
                    continue
                bboxes.append(box)
                scores.append(conf)
            bboxes = torch.stack(bboxes, dim=0)
            
            proposals.append(
                BoxList(bboxes, image_size=(w, h)) \
                    .resize((self.video_model_img_size, self.video_model_img_size)) \
                    .bbox
            )
            batch_scores.append(scores)
        
        video_results = self.video_model(x, proposals) # Tensor[n, num_cls]
        scores = torch.softmax(video_results, dim=1) # Tensor[n, num_cls]
    
        losses = None
        # proposals (List[Tensor[N, 4]]) 
        # targets (List[Dict])
        n = video_results.shape[0]
        detection_results = []
        for i in range(n):
            d['boxes'] = proposals[i].bbox
            d['scores'] = scores[i].max(dim=1)
            d['labels'] = scores[i].argmax(dim=1)
            detection_results.append(d)

        return detection_results, losses
