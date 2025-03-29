import cv2
import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean


class PoseMaskAnalyzer:
    def __init__(self, threshold_distance: int = 10, base_threshold=0.15, size_factor=0.1):
        """Initialize the analyzer with depth and YOLO models.

        Args:
            threshold_distance: threshold distance
        """
        self.threshold_distance = threshold_distance
        self.last_processed_image = None
        self.base_threshold = base_threshold
        self.size_factor = size_factor

    @staticmethod
    def _get_object_size(segmentation_mask):
        """Calculates the characteristic size of an object based on the segmentation mask."""
        contours, _ = cv2.findContours(segmentation_mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        # Возвращаем длину большей стороны ограничивающего прямоугольника
        return max(rect[1])

    def check_hands_in_mask(self,
                            points_3d: Dict[int, Dict[int, Tuple[float, float]]],
                            masks: np.ndarray) -> Dict[int, Dict[str, bool]]:
        """Check if hand points are in or near the mask for each person.

        Args:
            points_3d: Dictionary of person IDs to their keypoints
            masks: Binary mask array

        Returns:
            Dictionary with results for each person: {
                person_id: {
                    'left_hand': bool,
                    'right_hand': bool,
                    'any_hand': bool
                }
            }
        """
        results = {}

        for person_id, skeleton_points in points_3d.items():
            point_9 = skeleton_points.get(9)  # Left hand
            point_10 = skeleton_points.get(10)  # Right hand

            if point_9 is None or point_10 is None:
                print(f"Person {person_id}: Hand points are missing")
                continue

            object_size = self._get_object_size(masks)
            in_left = euclidean(point_9, masks)
            in_right = euclidean(point_10, masks)

            dynamic_threshold = self.base_threshold + object_size * self.size_factor

            is_interacting_right = in_left < dynamic_threshold
            is_interacting_left = in_right < dynamic_threshold

            results[person_id] = {
                'left_hand': is_interacting_left,
                'right_hand': is_interacting_right,
                'any_hand': is_interacting_left or is_interacting_right
            }

        return results

    def process_image(self,
                      image: np.ndarray,
                      keypoints: np.ndarray,
                      depth_map: np.ndarray,
                      visualize: bool = False) -> Dict[int, Dict[int, Tuple[float, float]]]:
        """Process an image to detect persons and extract their 3D keypoints.

        Args:
            image: Input image
            keypoints: 3D keypoints
            depth_map: depth map
            visualize: Whether to draw visualization (bounding boxes)

        Returns:
            Dictionary mapping person IDs to their keypoints
        """
        if image is None:
            raise FileNotFoundError(f"Could not load image")

        self.last_processed_image = image.copy()

        points_3d = {}
        for person_id, result in enumerate(keypoints, start=1):
            person_indices = [i for i, cls in enumerate(result.boxes.cls.cpu().numpy())
                              if result.names[int(cls)] == 'person']

            if not person_indices:
                continue

            keypoints = result.keypoints.xy.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()

            for idx in person_indices:
                if visualize:
                    x1, y1, x2, y2 = map(int, boxes[idx][:4])
                    cv2.rectangle(self.last_processed_image,
                                  (x1, y1), (x2, y2),
                                  (0, 255, 0), 2)

                person_keypoints = keypoints[idx]
                points = {}

                for point_id, (x, y) in enumerate(person_keypoints, start=1):
                    if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                        depth_value = round(depth_map[int(y), int(x)], 2)
                        points[point_id] = (float(x), float(y), depth_value)  # (x, y, depth)

                points_3d[person_id] = points

        return points_3d

    def predict(self,
                image: np.ndarray,
                mask: np.ndarray,
                keypoints: np.ndarray,
                depth_map: np.ndarray,
                visualize: bool = False) -> Dict[int, Dict[str, bool]]:
        """Complete analysis pipeline for an image.

        Args:
            image: Input image
            mask: Binary mask to check against
            keypoints: 3D keypoints
            depth_map: depth map
            visualize: Whether to draw visualization

        Returns:
            Dictionary with hand-in-mask results for each person
        """
        points_3d = self.process_image(image, keypoints, depth_map, visualize)
        return self.check_hands_in_mask(points_3d, mask)

    def show_processed_image(self):
        """Display the last processed image with visualizations."""
        if self.last_processed_image is not None:
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(self.last_processed_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        else:
            print("No processed image available. Call process_image() first.")
