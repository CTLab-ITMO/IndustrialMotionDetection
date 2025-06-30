import cv2
import numpy as np
from typing import Dict, Tuple
from scipy.spatial.distance import euclidean


class PoseMaskAnalyzer:
    def __init__(self, base_threshold=0.15, size_factor=0.1, use_depth=False):
        """Initialize the analyzer with depth and YOLO models.

        :param base_threshold: threshold distance
        :param size_factor: size factor
        :param use_depth: whether to use depth or not
        """
        self.base_threshold = base_threshold
        self.size_factor = size_factor
        self.use_depth = use_depth

    @staticmethod
    def _get_object_size(mask: np.ndarray) -> float:
        """Calculates the characteristic size of an object based on the segmentation mask.

        :param mask: binary mask
        :return: size of object
        """
        contours, _ = cv2.findContours(mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        return max(rect[1])

    def _get_object_centroid(self, mask: np.ndarray,
                             depth_map: np.ndarray) -> np.ndarray:
        """Calculates the centroid coordinates of an object based on the segmentation mask.

        :param mask: binary mask
        :param depth_map: depth map
        :return: centroid coordinates of object
        """
        y_coords, x_coords = np.where(mask > 0)
        if self.use_depth:
            z_values = depth_map[y_coords, x_coords].astype(np.float32)
            x_3d = x_coords * z_values  # условные единицы
            y_3d = y_coords * z_values  # условные единицы
            z_3d = z_values
            x_3d, y_3d, z_3d = np.mean(x_3d), np.mean(y_3d), np.mean(z_3d)
            coords = [x_3d, y_3d, z_3d]
        else:
            x, y = np.mean(x_coords), np.mean(y_coords)
            coords = [x, y]
        return coords

    def check_hands_in_mask(self,
                            person_points: Dict[int, Dict[int, Tuple[float, float]]],
                            object_centroid: np.ndarray,
                            object_size: int) -> bool:
        """Check if hand points are in or near the mask for each person.

        :param person_points: Dictionary of person IDs to their keypoints
        :param object_centroid: centroid of the object
        :param object_size: size of the object

        :return: Dictionary with results for each person: {
                person_id: {
                    'left_hand': bool,
                    'right_hand': bool
                }
            }
        """
        results = {}
        is_interacting_left, is_interacting_right = False, False
        for person_id, skeleton_points in person_points.items():
            point_9 = skeleton_points.get(9)
            point_10 = skeleton_points.get(10)

            if point_9 is None or point_10 is None:
                continue

            in_left = euclidean(point_9, object_centroid)
            in_right = euclidean(point_10, object_centroid)

            dynamic_threshold = self.base_threshold + object_size * self.size_factor
            is_interacting_right = in_left < dynamic_threshold
            is_interacting_left = in_right < dynamic_threshold

            results[person_id] = {
                'left_hand': is_interacting_left,
                'right_hand': is_interacting_right,
                'any_hand': is_interacting_left or is_interacting_right
            }

        return is_interacting_left or is_interacting_right

    def get_person_points_dict(self, keypoints: np.ndarray,
                               depth_map=None) -> Dict[int, Dict[int, Tuple[float, float]]]:
        """Process an image to detect persons and extract their 3D keypoints.

        :param keypoints: 3D keypoints
        :param depth_map: depth map

        :return: Dictionary mapping person IDs to their keypoints
        """
        points_3d = {}
        for person_id, result in enumerate(keypoints, start=1):
            person_indices = [i for i, cls in enumerate(result.boxes.cls.cpu().numpy())
                              if result.names[int(cls)] == 'person']
            if not person_indices:
                continue
            keypoints = result.keypoints.xy.cpu().numpy()
            for idx in person_indices:

                person_keypoints = keypoints[idx]
                points = {}

                for point_id, (x, y) in enumerate(person_keypoints, start=1):
                    if self.use_depth:
                        height, width = depth_map.shape
                        assert 0 <= x < width and 0 <= y < height, "Coordinates outside the image!"
                        z = depth_map[int(y), int(x)].astype(float)
                        x_3d, y_3d, z_3d = x * z, y * z, z
                        points[point_id] = [float(x_3d), float(y_3d), z_3d]
                    else:
                        points[point_id] = [x, y]

                points_3d[person_id] = points

        return points_3d

    def predict(self,
                image: np.ndarray,
                mask: np.ndarray,
                keypoints: np.ndarray,
                depth_map: np.ndarray) -> Dict[int, Dict[str, bool]]:
        """Complete analysis pipeline for an image.

        :param image: Input image
        :param mask: Binary mask to check against
        :param keypoints: 3D keypoints
        :param depth_map: depth map

        :return: Dictionary with hand-in-mask results for each person
        """
        if image is None:
            raise FileNotFoundError(f"Could not load image")

        persons_coords = self.get_person_points_dict(keypoints=keypoints, depth_map=depth_map)
        object_size = self._get_object_size(mask)
        object_centroid = self._get_object_centroid(mask, depth_map)
        if object_centroid is None or object_size is None:
            return False
        return self.check_hands_in_mask(persons_coords, object_centroid, object_size)
