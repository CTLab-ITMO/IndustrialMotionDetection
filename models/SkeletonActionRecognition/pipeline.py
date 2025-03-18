import cv2
import matplotlib.pyplot as plt


def is_point_near_mask(point, mask, threshold_distance=10):
    y, x = int(point[1]), int(point[0])
    h, w = mask.shape

    # Проверяем, что точка внутри изображения
    if 0 <= y < h and 0 <= x < w:
        if mask[y, x]:
            return True

    # Если точка не в маске, проверяем ближайшие точки
    for dy in range(-threshold_distance, threshold_distance + 1):
        for dx in range(-threshold_distance, threshold_distance + 1):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                if mask[ny, nx]:
                    return True
    return False


def check_points_in_mask(points_3d, masks, threshold_distance=10):
    for person_id, skeleton_points in points_3d.items():
        point_9 = skeleton_points.get(9)  # Получаем 9-ю точку (левая кисть)
        point_10 = skeleton_points.get(10)  # Получаем 10-ю точку (правая кисть)

        if point_9 is not None and point_10 is not None:
            point_9_2d = (int(point_9[0]), int(point_9[1]))
            point_10_2d = (int(point_10[0]), int(point_10[1]))

            # Проверяем, находятся ли точки в маске или рядом с ней
            in_mask_9 = is_point_near_mask(point_9_2d, masks, threshold_distance)
            in_mask_10 = is_point_near_mask(point_10_2d, masks, threshold_distance)

            print(f"Person {person_id}: Point 9 in/near mask: {in_mask_9}, Point 10 in/near mask: {in_mask_10}")
            return in_mask_9 or in_mask_10
        else:
            print(f"Person {person_id}: Points 9 or 10 are missing")
            return False


def get_yolo_depth_result(depth_model, yolo_model, image_path: str):
    plt.figure(figsize=(16, 12))
    inputs = cv2.imread(image_path)
    results = yolo_model(inputs)
    person_keypoints = []
    person_boxes = []
    for result in results:
        keypoints = result.keypoints.xy.cpu().numpy()
        person_keypoints.append(keypoints)
        classes = result.boxes.cls.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        for box, cls in zip(boxes, classes):
            if result.names[int(cls)] == 'person':
                x1, y1, x2, y2 = map(int, box[:4])
                person_boxes.append((x1, y1, x2, y2))
                cv2.rectangle(inputs, (x1, y1), (x2, y2), (0, 225, 0), 2)

    depth = depth_model.infer_image(inputs)
    points_3d, points = {}, {}
    id_person, id_keypont = 0, 0
    for keypoints_person in person_keypoints:  # тут итерация по всем людям
        id_person += 1
        for keypoints in keypoints_person:  # тут итерация по одному человеку
            points = {}
            id_keypont = 0
            for x, y in keypoints:  # тут итерация по ключевым точкам человека
                depth_value = round(depth[int(y), int(x)], 2)
                id_keypont += 1
                points[id_keypont] = (x, y)
        points_3d[id_person] = points
    return points_3d
