from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
import sys
import os


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


def add_pred_bboxes(bboxes,
                    scores,
                    labels,
                    nameOfImage,
                    bbFormat,
                    coordType,
                    allBoundingBoxes,
                    imgSize=(0, 0)):
    for box, score, label in list(zip(bboxes, scores, labels)):
        x_min, y_min, x_max, y_max = box
        w = x_max - x_min
        h = y_max - y_min
        bb = BoundingBox(nameOfImage,
                         label,
                         x_min, y_min, w, h,
                         coordType,
                         imgSize,
                         BBType.Detected,
                         score,
                         format=bbFormat)
        allBoundingBoxes.addBoundingBox(bb)
    return allBoundingBoxes

def add_gt_bboxes(bboxes,
                  labels,
                  nameOfImage,
                  bbFormat,
                  coordType,
                  allBoundingBoxes,
                  imgSize=(0, 0)):
    for j in range(len(bboxes)):
        x_min, y_min, x_max, y_max = bboxes[j]
        label = np.argmax(labels[j])
        w = x_max - x_min
        h = y_max - y_min
        bb = BoundingBox(nameOfImage,
                         label,
                         x_min, y_min, w, h,
                         coordType,
                         imgSize,
                         BBType.GroundTruth,
                         format=bbFormat)
        allBoundingBoxes.addBoundingBox(bb)
    return allBoundingBoxes


def compute_metrics_batch_detections(
    detection_results,
    x
):
      allBoundingBoxes = BoundingBoxes()
      allClasses = []

      for j in range(len(detection_results)):
          boxes = detection_results[j]['boxes'].cpu().detach().numpy()
          scores = detection_results[j]['scores'].cpu().detach().numpy()
          labels = detection_results[j]['labels'].cpu().detach().numpy()

          nameOfImage = "/".join(x['path'][j].split('/')[-2:])
          imgSize = tuple(x['video'].shape[-2:])
          # print(f"{nameOfImage=} {imgSize}")

          add_pred_bboxes(boxes, scores, labels,
                          nameOfImage, bbFormat, coordType, allBoundingBoxes, imgSize)

          add_gt_bboxes(x['bbox'][j].bbox.cpu().detach().numpy(),
                        x['target'][j].cpu().detach().numpy(),
                        nameOfImage, bbFormat, coordType, allBoundingBoxes, imgSize)


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