# Модули обработки обработки видео для распознавания действий

## 1. Оценка глубины (`depth_estimation/depth_estimator.py`)

**Назначение**: Построение карты глубины для анализа пространственных отношений  
**Реализация**:

```python
class DepthEstimator:
    def predict(frame: np.ndarray, optimization_type: str) -> np.ndarray:
        """Возвращает карту глубины в формате HxW"""
```

## 2. Оценка позы (`pose_estimation/pose_estimator.py`)

**Назначение**: Детекция скелетных точек человека (17 keypoints COCO-формат)

**Поддерживаемые модели**:

```python
SUPPORTED_MODELS = [
    "yolo11n-pose", "yolo11s-pose", "yolo11m-pose", "yolo11l-pose", "yolo11x-pose",
    "yolov8n-pose", "yolov8s-pose", "yolov8m-pose", "yolov8l-pose", "yolov8x-pose",
]
```

**Реализация**

```python
class PoseEstimator:
    def predict(frame: np.ndarray, optimization_type: str) -> np.ndarray:
        """Возвращает 17 скелетных точек человека."""
```

## 3. Сегментация объектов

### ClipSeg (`segmentation/segmenter_clip_seg.py`)

**Назначение**: Выделение масок для объектов взаимодействия на основе текстового промпта

**Поддерживаемые классы**:

```python
CLASSES = [
    'food', 'laptop', 'phone',
    'cigarette', 'bottle'
]
```

**Реализация**

```python
class ClipSegmentation:
    def predict(frame: np.ndarray, optimization_type: str) -> np.ndarray:
        """Возвращает маску объекта на основе текстового промпта."""
```

### Sam (`segmentation/segmenter_sam.py`)

**Назначение**: Выделение масок для объектов взаимодействия на основе заданных точек

**Реализация**

```python
class SamSegmentor:
    def predict(frame: np.ndarray, optimization_type: str) -> np.ndarray:
        """Возвращает маску объекта на основе заданных точек."""
```