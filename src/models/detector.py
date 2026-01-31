import random
import cv2
import scipy
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from models.box_list import BoxList
from typing import Tuple
from transforms_impl.transforms import (
    Compose,
    ShortSideScaleWithBoxes,
)


def match_bboxes(bbox_gt, bbox_pred, IOU_THRESH=0.5):
    '''
    Given sets of true and predicted bounding-boxes,
    determine the best possible match.
    Parameters
    ----------
    bbox_gt, bbox_pred : N1x4 and N2x4 np array of bboxes [x1,y1,x2,y2]. 
      The number of bboxes, N1 and N2, need not be the same.
    
    Returns
    -------
    (idxs_true, idxs_pred, ious, labels)
        idxs_true, idxs_pred : indices into gt and pred for matches
        ious : corresponding IOU value of each match
        labels: vector of 0/1 values for the list of detections
    '''
    n_true = bbox_gt.shape[0]
    n_pred = bbox_pred.shape[0]
    MAX_DIST = 1.0
    MIN_IOU = 0.0

    # NUM_GT x NUM_PRED
    iou_matrix = np.zeros((n_true, n_pred))
    for i in range(n_true):
        for j in range(n_pred):
            iou_matrix[i, j] = bbox_iou(bbox_gt[i,:], bbox_pred[j,:])
            
    if n_pred > n_true:
      # there are more predictions than ground-truth - add dummy rows
      diff = n_pred - n_true
      iou_matrix = np.concatenate( 
        (iou_matrix, np.full((diff, n_pred), MIN_IOU)), axis=0)

    if n_true > n_pred:
      # more ground-truth than predictions - add dummy columns
      diff = n_true - n_pred
      iou_matrix = np.concatenate( 
        (iou_matrix, np.full((n_true, diff), MIN_IOU)), axis=1)

    # call the Hungarian matching
    idxs_true, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)

    # remove dummy assignments
    sel_pred = idxs_pred<n_pred
    idx_pred_actual = idxs_pred[sel_pred] 
    idx_gt_actual = idxs_true[sel_pred]
    ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
    sel_valid = (ious_actual > IOU_THRESH)
    label = sel_valid.astype(int)

    return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], label 

    
def bbox_iou(boxA, boxB):
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = xB - xA + 1
    interH = yB - yA + 1

    # Correction: reject non-overlapping boxes
    if interW <=0 or interH <=0 :
        return -1.0

    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def rescale_bboxes(
    boxes: torch.Tensor, 
    box_size: Tuple[int, int], 
    target_size: Tuple[int, int]
) -> torch.Tensor:
    """
    Rescale bounding boxes from original image size to target size.
    
    Args:
        boxes: Tensor of shape [N, 4] where each row is (x1, y1, x2, y2)
        box_size: Original image size as (width, height)
        target_size: Target image size as (width, height)
    
    Returns:
        Tensor of shape [N, 4] with rescaled boxes
    """
    orig_width, orig_height = box_size
    target_width, target_height = target_size

    scales = torch.tensor(
        [target_width / orig_width, target_height / orig_height,
        target_width / orig_width, target_height / orig_height],
        device=boxes.device, dtype=boxes.dtype)
    return boxes * scales


class YOLO_VideoMAE:
    def __init__(
        self, video_model, criterion,          
        yolo_model_name="yolo11n.pt",
        yolo_imgsz=640,
        yolo_lower_conf=0.25,
        yolo_upper_conf=0.5,
        yolo_iou=0.7,
        enable_logging=False,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.device = device
        self.phase = 'test'
        
        self.yolo_imgsz = yolo_imgsz
        self.yolo_lower_conf = yolo_lower_conf
        self.yolo_upper_conf = yolo_upper_conf
        self.yolo_iou = yolo_iou
        self.yolo_classes = [0]
        self.yolo_agnostic_nms = True
        self.yolo_verbose_inference = False
        self.yolo_model = YOLO(
            yolo_model_name
        ).to(self.device)
        
        self.video_model = video_model
        self.criterion = criterion
        
        self.unknown_class_limit = 3
        
        self.video_model_img_size = 224
        self.max_pixel_value = 255
        self.clip_len = 16
        
        self.video_transform = Compose([
            ShortSideScaleWithBoxes(size=self.video_model_img_size),
        ])
        
        self.total_targets = 0
        self.matched_targets = 0
        
        self.enable_logging = enable_logging
        
        self.limit = 1
        self.count_equals_limit = 0
        self.count_classes_ = {i: 0 for i in range(self.video_model.num_classes)}
        
    def get_match_rate(self):
        return 0 if self.total_targets == 0 else self.matched_targets / self.total_targets
    
    def train(self):
        self.phase = 'train'
    
    def test(self):
        self.phase = 'test'
        
    def draw_rect_image(
        self,
        image, 
        box, 
        label: str = None,
        color = (0, 255, 0)
    ):
        font_scale = 0.5
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(
            image, 
            (x1, y1), (x2, y2), color, 2)
        
        if label is not None:
            (text_width, text_height), _ = cv2.getTextSize(
                label, 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            
            cv2.rectangle(
                image, 
                (x1, y1 - text_height - 10), (x1 + text_width, y1), 
                color, -1)
            
            cv2.putText(
                image, label, (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)
    
    def __call__(self, x, gt_boxes, gt_labels):
        if self.enable_logging:
            print(f"{x.shape=}")
        
        batch_count, c, t, h, w = x.shape # [B, C, T, H, W]
        target_size = (w, h)
        
        if self.enable_logging:
            print(f"target_size: {target_size}")
        
        # [B, C, T, H, W] -> [B, C, H, W]
        image = x[:, :, t//2]
        
        results = self.yolo_model(
            image/self.max_pixel_value, 
            conf=self.yolo_lower_conf,
            iou=self.yolo_iou,
            imgsz=self.yolo_imgsz, 
            agnostic_nms=self.yolo_agnostic_nms,
            classes=self.yolo_classes,
            verbose=self.yolo_verbose_inference,
        )
        
        if self.enable_logging:
            print(f"YOLO model output: {len(results)=}")
        
        # For training
        gt_targets = [] # targets (List[Dict])
        yolo_targets = []
        
        # For testing and training
        yolo_proposals = [] # proposals (List[Tensor[N, 4]]) 
        batch_scores = []
        
        for i, result in enumerate(results):
            # img_with_boxes = image.numpy()[i].astype(np.uint8).transpose((1, 2, 0)).copy()
            
            xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
            names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
            confs = result.boxes.conf  # confidence score of each box
            
            bboxes = []
            scores = []
            
            for box, cls_name, conf in zip(xyxy, names, confs):
                if conf < self.yolo_upper_conf:
                    continue
                
                # label = f"{cls_name} {conf:.2f}"
                # self.draw_rect_image(img_with_boxes, box, label)
     
                bboxes.append(box)
                scores.append(conf)
                
            if len(bboxes) == 0:
                bboxes.append(torch.Tensor([0, 0, 1, 1]))
                scores.append(0.0)
            
            if self.enable_logging:           
                print(f"\nGathered {len(bboxes)} proposals for image #{i}")
            
            bboxes = BoxList(torch.stack(bboxes, dim=0).to(self.device), image_size=(w, h)) \
                .resize((self.video_model_img_size, self.video_model_img_size))  
            scores = torch.tensor(scores).to(self.device)
            
            gt_target_dict = {}
            gt_target_dict['video'] = x[i].detach().clone()
            gt_target_dict['bbox'] = gt_boxes[i].bbox.detach().clone()  # [N, 4] float tensor
            gt_target_dict['target'] = gt_labels[i].argmax(dim=1).detach().clone() # Tensor[N]
        
            # for box in target_dict['bbox']:
            #     self.draw_rect_image(img_with_boxes, box.tolist(), color=(255, 0, 0))
            # plt.imshow(img_with_boxes)
            # plt.show()
            
            gt_target_dict = self.video_transform(gt_target_dict)
            gt_targets.append(gt_target_dict)
            
            if self.enable_logging:
                print(f"{gt_target_dict['bbox']=}")
                print(f"{gt_target_dict['target']=}")
                print(f"{bboxes.bbox=}")
                
            idxs_true, idxs_pred, ious, labels_matched = match_bboxes(gt_target_dict['bbox'], bboxes.bbox)
            idxs_pred = torch.tensor(idxs_pred, device=self.device)
            
            assert len(idxs_true) == len(idxs_pred), "true and pred idxs len should be equal"
            
            self.total_targets += len(gt_target_dict['bbox'])
            self.matched_targets += len(idxs_pred) 
            
            if self.enable_logging:
                print(f"\nMatching info:\n{idxs_true=}\n{idxs_pred.tolist()=}\n{ious=}\n{labels_matched=}")
            
            if self.phase == 'train':
                # # Create mask for matched boxes
                # matched_mask = torch.zeros(len(bboxes), dtype=torch.bool)
                # matched_mask[idxs_pred] = True
                # # Get indices of non-matched boxes
                # non_matched_indices = torch.where(~matched_mask)[0].to(device)
                # # Select up to 3 non-matched boxes
                # num_non_matched = min(self.unknown_class_limit, len(non_matched_indices))
                # # Combine matched and selected non-matched indices
                # keep_indices = torch.cat([idxs_pred, non_matched_indices[:num_non_matched]]).unique()
                # # Filter boxes and scores
                # filtered_bboxes = bboxes.keep_boxes(keep_indices)
                # filtered_scores = scores_tensor[keep_indices]

                # Map original matched indices to new indices
                # new_indices = torch.where(torch.isin(keep_indices, idxs_pred))[0]
                
                UKNOWN_CLS = 0
                yolo_target = torch.full(size=(len(bboxes),), fill_value=UKNOWN_CLS).to(self.device)
                yolo_target[idxs_pred] = gt_target_dict['target'][idxs_true]
            
                # permute bboxes to match yolo_target
                # bboxes = BoxList(bboxes.bbox[idxs_pred], bboxes.size, bboxes.mode)
                
                remove_label_idx = []
                for i, l in enumerate(yolo_target.cpu().detach().numpy()):
                    cls_count = self.count_classes_[l] + 1
                    if cls_count > self.limit:
                        remove_label_idx.append(i)
                    elif cls_count == self.limit:
                        self.count_equals_limit += 1
                        self.count_classes_[l] += 1
                    else:
                        self.count_classes_[l] += 1
                        
                assert all([v <= self.limit for v in self.count_classes_.values()])
            
                if self.count_equals_limit == len(self.count_classes_):
                    self.limit += 1
                    self.count_equals_limit = 0
                    
                if len(remove_label_idx) == len(yolo_target): 
                    random_index = random.randint(0, len(remove_label_idx) - 1)
                    removed_element = remove_label_idx.pop(random_index)
                    self.count_classes_[yolo_target.cpu().detach().numpy()[removed_element]] += 1
                    self.count_equals_limit = 0
                    self.limit += 1
                
                # Create a mask to keep indices that are not in remove_label_idx
                keep_mask = torch.ones(len(yolo_target), dtype=torch.bool)
                keep_mask[remove_label_idx] = False
                
                print(f"\nInfo about rejecting samples:")
                print('\n'.join(f"{k}: {v}" for k, v in self.count_classes_.items()))
                print(f"{self.count_equals_limit=}")
                print(f"{remove_label_idx=}")
                print(f"{keep_mask=}")
                print(f"Before {yolo_target=}")

                yolo_target = yolo_target[keep_mask]
                
                bboxes = bboxes.keep_boxes(keep_mask)
                scores = scores[keep_mask]
                
                if self.enable_logging:
                    print("\nFinal yolo_target:")
                    print(f"{yolo_target=}")
            
                # Change target dict to match yolo bboxes
                yolo_targets.append(yolo_target) # Tensor[N]
                
            bboxes = bboxes.enlarge(10.0)
            yolo_proposals.append(bboxes)
            batch_scores.append(scores)
                
        x = torch.stack([t['video'] for t in gt_targets], dim=0)
        
        if self.enable_logging:
            print(f"\nInput video shape: {x.shape=}")
            print("Overall proposals:")
            print(f"{yolo_proposals=}")
        
        logits = self.video_model(x, yolo_proposals) # Tensor[n, num_cls]
        
        if self.enable_logging:
            print(f"\nVideoMAE results shape: {logits.shape=}")
        
        scores = torch.softmax(logits, dim=1) # Tensor[n, num_cls]
        
        # For training
        losses = None
        if self.phase == 'train':
            labels = torch.cat(yolo_targets, dim=0)
            
            if self.enable_logging:
                print(f"\nTarget shape: {labels.shape=}")
                print("Overall labels:")
                print(f"{labels=}")
            
            print(f"\nDistribution stat:")
            print('\n'.join(f"{k}: {v}" for k, v in self.count_classes_.items()))
            print(f"{remove_label_idx=}") 
            print(f"{self.limit=}")
            
            counts = torch.tensor(list(self.count_classes_.values()),  dtype=torch.float32)
            weights = 1.0 / counts
            weights = weights / weights.max()
        
            losses = {'loss': self.criterion(logits, labels, weights)}
        
        detection_results = []
        
        prev = 0
        for i in range(batch_count):
            batch_len = len(yolo_proposals[i])
            
            detection_result_dict = {}
            detection_result_dict['boxes'] = rescale_bboxes(
                yolo_proposals[i].bbox, 
                yolo_proposals[i].size, 
                target_size
            )
            max_values, max_indices = scores[prev:prev+batch_len].max(dim=1)
            detection_result_dict['scores'] = max_values
            detection_result_dict['labels'] = max_indices 
    
            prev += batch_len
            
            detection_results.append(detection_result_dict)

        return detection_results, losses
