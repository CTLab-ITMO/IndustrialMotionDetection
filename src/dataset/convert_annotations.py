import json
from pathlib import Path


def update_label_studio_annotations(
        original_annotation_path,
        processed_dir,
        output_path,
        video_name
):
    with open(original_annotation_path, 'r') as f:
        label_studio_data = json.load(f)

    person_annotations = []
    for person_dir in Path(processed_dir).glob('person_*'):
        annotation_file = list(person_dir.glob('*_annotations.json'))
        if annotation_file:
            person_annotations.append(str(annotation_file[0]))

    if not person_annotations:
        print(f"no annotations {processed_dir}")
        return
    print(f"{len(person_annotations)} persons annotations")

    transformations = {}
    for annotation_path in person_annotations:
        with open(annotation_path, 'r') as f:
            person_data = json.load(f)

        person_idx = Path(annotation_path).parent.name.split('_')[1]
        transformations[person_idx] = person_data

    video_item = None
    for item in label_studio_data:
        if  video_name in item["video"]:
            video_item = item
    person_to_box_mapping = {}
    if isinstance(label_studio_data, list):
        if video_item and 'box' in video_item:
            for box_idx, box_item in enumerate(video_item['box']):
                if 'sequence' in box_item:
                    person_idx = update_sequence(box_item['sequence'], transformations)
                    # print(person_idx)
                    if person_idx:
                        person_to_box_mapping[f"person_{person_idx}"] = {
                            "box_index": box_idx,
                            "frame_count": len(box_item['sequence'])
                        }
    with open(output_path, 'w') as f:
        json.dump([video_item], f, indent=2)

    mapping_path = Path(output_path).parent / "person_box_mapping.json"
    with open(mapping_path, 'w') as f:
        json.dump(person_to_box_mapping, f, indent=2)


def update_sequence(sequence, transformations):
    if not sequence:
        return None
    person_idx = find_matching_person_for_sequence(sequence, transformations)
    if not person_idx:
        return None
    transform_data = transformations[person_idx]
    for box in sequence:
        frame_num = box.get('frame', 0)
        if str(frame_num) in transform_data['frame_data']:
            transform = transform_data['frame_data'][str(frame_num)]

            new_x = transform['relative']['x'] / transform_data['max_dimensions']['width'] * 100
            new_y = transform['relative']['y'] / transform_data['max_dimensions']['height'] * 100

            new_width = transform['original']['width'] / transform_data['max_dimensions']['width'] * 100
            new_height = transform['original']['height'] / transform_data['max_dimensions']['height'] * 100

            box['x'] = new_x
            box['y'] = new_y
            box['width'] = new_width
            box['height'] = new_height

    return person_idx


def find_matching_person_for_sequence(sequence, transformations):
    if not sequence:
        return None
    sequence_frames = set(box.get('frame', 0) for box in sequence)
    match_scores = {}

    for person_idx, transform_data in transformations.items():
        person_frames = set(int(f) for f in transform_data['frame_data'].keys())
        common_frames = sequence_frames.intersection(person_frames)
        if not common_frames:
            continue

        overlap = len(common_frames) / len(sequence_frames)
        similarity = 0
        video_width = transform_data['original_dimensions']['width']
        video_height = transform_data['original_dimensions']['height']

        for frame in common_frames:
            seq_box = next((box for box in sequence if box.get('frame') == frame), None)
            if not seq_box:
                continue

            seq_x = seq_box.get('x', 0) * video_width / 100
            seq_y = seq_box.get('y', 0) * video_height / 100
            seq_w = seq_box.get('width', 0) * video_width / 100
            seq_h = seq_box.get('height', 0) * video_height / 100

            person_box = transform_data['frame_data'][str(frame)]['original']
            seq_center_x, seq_center_y = seq_x + seq_w / 2, seq_y + seq_h / 2
            person_center_x = person_box['x'] + person_box['width'] / 2
            person_center_y = person_box['y'] + person_box['height'] / 2

            max_dist = (video_width ** 2 + video_height ** 2) ** 0.5
            distance = ((seq_center_x - person_center_x) ** 2 +
                        (seq_center_y - person_center_y) ** 2) ** 0.5

            similarity = 1 - min(1.0, distance / (max_dist * 0.2))
            similarity += similarity

        if common_frames:
            avg_spatial = similarity / len(common_frames)
            match_scores[person_idx] = 0.3 * overlap + 0.7 * avg_spatial

    if not match_scores:
        return None

    best_match = max(match_scores.items(), key=lambda x: x[1])
    if best_match[1] > 0.2:
        print(f"{best_match[1]} for person_{best_match[0]}")
        return best_match[0]

    print(f"{best_match[1] if match_scores else 0}")
    return None


def calculate_iou(box1, box2):
    box1_x1, box1_y1 = box1['x'], box1['y']
    box1_x2, box1_y2 = box1_x1 + box1['width'], box1_y1 + box1['height']

    box2_x1, box2_y1 = box2['x'], box2['y']
    box2_x2, box2_y2 = box2_x1 + box2['width'], box2_y1 + box2['height']

    x_left = max(box1_x1, box2_x1)
    y_top = max(box1_y1, box2_y1)
    x_right = min(box1_x2, box2_x2)
    y_bottom = min(box1_y2, box2_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    union_area = box1_area + box2_area - intersection

    if union_area == 0:
        return 0.0

    return intersection / union_area


if __name__ == "__main__":
    video_name = "20250309_134203"
    # original_annotation_path = "input/annotations/project-2-at-2025-04-18-16-16-d5a9de12.json"
    original_annotation_path = "input/annotations/project-1-at-2025-04-16-13-42-d41c1b2b.json"
    processed_dir = f"output_matted/{video_name}"
    output_path = f"output_matted/{video_name}/updated_annotations.json"

    update_label_studio_annotations(
        original_annotation_path,
        processed_dir,
        output_path,
        video_name
    )