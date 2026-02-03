import numpy as np
from src.metrics_impl.BoundingBox import BoundingBox
from src.metrics_impl.BoundingBoxes import BoundingBoxes
from src.metrics_impl.utils import MethodAveragePrecision, BBType, BBFormat, CoordinatesType
from src.metrics_impl.Evaluator import Evaluator


def compute_metrics_batch_detections(
    detection_results,
    batch,
    savePath: str,
    iouThreshold: float = .5,
    bbFormat: BBFormat = BBFormat.XYWH,
    coordType: CoordinatesType = CoordinatesType.Absolute,
    showPlot: bool = False
):
    allBoundingBoxes = BoundingBoxes()

    for j in range(len(detection_results)):
        boxes = detection_results[j]['boxes'].cpu().detach().numpy()
        scores = detection_results[j]['scores'].cpu().detach().numpy()
        labels = detection_results[j]['labels'].cpu().detach().numpy()
        
        gt_boxes = batch['bbox'][j].bbox.cpu().detach().numpy()
        gt_targets = batch['target'][j].cpu().detach().numpy()

        nameOfImage = "/".join(batch['path'][j].split('/')[-2:]) + f'/{j}'
        imgSize = tuple(batch['video'].shape[-2:])
        # print(f"{nameOfImage=} {imgSize}")

        for box, score, label in list(zip(boxes, scores, labels)):
            x_min, y_min, x_max, y_max = box
            w = x_max - x_min
            h = y_max - y_min
            bb = BoundingBox(
                nameOfImage,
                label,
                x_min, y_min, w, h,
                coordType,
                imgSize,
                BBType.Detected,
                score,
                format=bbFormat)
            allBoundingBoxes.addBoundingBox(bb)
    
        for k in range(len(gt_boxes)):
            x_min, y_min, x_max, y_max = gt_boxes[k]
            label = np.argmax(gt_targets[k])
            w = x_max - x_min
            h = y_max - y_min
            bb = BoundingBox(
                nameOfImage,
                label,
                x_min, y_min, w, h,
                coordType,
                imgSize,
                BBType.GroundTruth,
                format=bbFormat)
            allBoundingBoxes.addBoundingBox(bb)
    
    evaluator = Evaluator()

    # Plot Precision x Recall curve
    detections = evaluator.PlotPrecisionRecallCurve(
        allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
        IOUThreshold=iouThreshold,  # IOU threshold
        method=MethodAveragePrecision.EveryPointInterpolation,
        showAP=True,  # Show Average Precision in the title of the plot
        showInterpolatedPrecision=False,  # Don't plot the interpolated precision curve
        savePath=savePath,
        showGraphic=showPlot)

    return detections