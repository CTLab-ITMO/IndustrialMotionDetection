import json
import os
import cv2
import torch
from tqdm import tqdm
from inference import Converter


def process_annotations(video_path, annotation_path, output_annot, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(output_annot, 'r') as f:
        output_annot = json.load(f)
    for item in output_annot:
        if video_path.split("/")[-1] in item["video"]:
            output_annot = item

    person_sequences = []
    if 'box' in output_annot:
        for box_data in output_annot['box']:
            if 'sequence' in box_data:
                person_sequences.append(box_data['sequence'])

    with open(annotation_path, 'r') as f:
        data = json.load(f)

    if 'predictions' in data:
        annotations = []
        for prediction in data['predictions']:
            if 'result' in prediction:
                annotations.extend(prediction['result'])
    else:
        annotations = data

    person_sequences = []
    for item in annotations:
        if isinstance(item, dict) and 'value' in item:
            if 'sequence' in item['value']:
                person_sequences.append(item['value']['sequence'])
        elif isinstance(item, dict) and 'box' in item:
            for box in item['box']:
                if 'sequence' in box:
                    person_sequences.append(box['sequence'])

    print(f"found {len(person_sequences)} people in annotations")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"error can't open video {video_path}")
        return

    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    converter = Converter(
        variant='mobilenetv3',
        checkpoint='rvm_mobilenetv3.pth',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    for person_idx, sequence in enumerate(person_sequences):
        print(f"processing person {person_idx + 1} from {len(person_sequences)}")

        person_dir = os.path.join(output_dir, f'person_{person_idx + 1}')
        cropped_dir = os.path.join(person_dir, 'cropped')
        matted_dir = os.path.join(person_dir, 'matted')
        os.makedirs(cropped_dir, exist_ok=True)
        os.makedirs(matted_dir, exist_ok=True)

        frame_to_box = {}
        max_width = 0
        max_height = 0
        transformed_boxes = {}

        for box in sequence:
            if box.get('enabled', True):
                frame_num = box.get('frame')
                x_percent = box.get('x', 0)
                y_percent = box.get('y', 0)
                width_percent = box.get('width', 0)
                height_percent = box.get('height', 0)

                x = int(x_percent * video_width / 100)
                y = int(y_percent * video_height / 100)
                width = int(width_percent * video_width / 100)
                height = int(height_percent * video_height / 100)

                frame_to_box[frame_num] = {
                    'x': x, 'y': y, 'width': width, 'height': height
                }

                max_width = max(max_width, width)
                max_height = max(max_height, height)

        if not frame_to_box:
            print(f"no valid boxes for person {person_idx + 1} skipping")
            continue

        max_width = int(max_width)
        max_height = int(max_height)

        cropped_video_path = os.path.join(cropped_dir, f"person_{person_idx + 1}_cropped.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(cropped_video_path, fourcc, fps, (max_width, max_height))

        cap = cv2.VideoCapture(video_path)
        start_frame = min(frame_to_box.keys())
        end_frame = max(frame_to_box.keys())
        frame_idx = 0
        last_valid_box = None

        with tqdm(total=end_frame, desc=f"cropping person {person_idx + 1}") as pbar:
            while frame_idx <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                if frame_idx < start_frame:
                    pbar.update(1)
                    continue

                box = frame_to_box.get(frame_idx, None)

                if box:
                    last_valid_box = box
                elif last_valid_box:
                    box = last_valid_box
                else:
                    pbar.update(1)
                    continue

                x, y, w, h = box['x'], box['y'], box['width'], box['height']

                center_x = x + w // 2
                center_y = y + h // 2

                crop_x = max(0, center_x - max_width // 2)
                crop_y = max(0, center_y - max_height // 2)

                if crop_x + max_width > video_width:
                    crop_x = max(0, video_width - max_width)
                if crop_y + max_height > video_height:
                    crop_y = max(0, video_height - max_height)

                try:
                    transformed_boxes[frame_idx] = {
                        'original': {
                            'x': x, 'y': y, 'width': w, 'height': h,
                        },
                        'crop': {
                            'x': crop_x, 'y': crop_y, 'width': max_width, 'height': max_height
                        },
                        'relative': {

                            'x': x - crop_x,
                            'y': y - crop_y
                        }
                    }
                    cropped = frame[crop_y:crop_y + max_height, crop_x:crop_x + max_width]

                    if cropped.size == 0 or cropped.shape[0] != max_height or cropped.shape[1] != max_width:

                        center_crop_x = max(0, video_width // 2 - max_width // 2)
                        center_crop_y = max(0, video_height // 2 - max_height // 2)
                        cropped = frame[center_crop_y:center_crop_y + max_height,
                                  center_crop_x:center_crop_x + max_width]
                        transformed_boxes[frame_idx]['crop'] = {
                            'x': center_crop_x, 'y': center_crop_y,
                            'width': max_width, 'height': max_height
                        }
                        transformed_boxes[frame_idx]['relative'] = {
                            'x': x - center_crop_x,
                            'y': y - center_crop_y
                        }

                        if cropped.shape[0] != max_height or cropped.shape[1] != max_width:
                            cropped = cv2.resize(cropped, (max_width, max_height))

                    out.write(cropped)
                except Exception as e:
                    print(f"error cropping frame {frame_idx}: {e}")

                    cropped = cv2.resize(frame, (max_width, max_height))
                    out.write(cropped)

                pbar.update(1)

        cap.release()
        out.release()

        matted_video_path = os.path.join(matted_dir, f"person_{person_idx + 1}_matted.mp4")
        matted_alpha_path = os.path.join(matted_dir, f"person_{person_idx + 1}_alpha.mp4")

        print(f"matting person {person_idx + 1}...")
        converter.convert(
            input_source=cropped_video_path,
            output_type='video',
            output_composition=matted_video_path,
            output_alpha=matted_alpha_path,
            downsample_ratio=1,
        )
        annotation_path = os.path.join(person_dir, f"person_{person_idx + 1}_annotations.json")
        with open(annotation_path, 'w') as f:
            json.dump({
                'original_video': video_path,
                'cropped_video': cropped_video_path,
                'matted_video': matted_video_path,
                'alpha_video': matted_alpha_path,
                'frame_data': transformed_boxes,
                'max_dimensions': {
                    'width': max_width,
                    'height': max_height
                },
                'original_dimensions': {
                    'width': video_width,
                    'height': video_height
                }
            }, f, indent=2)

        print(f"processing person {person_idx + 1}")


if __name__ == "__main__":
    video_name = "20250309_134203"
    video_path = f"input/foregrounds/not_one/{video_name}.mp4"
    annotation_path = f"../local_st/tasks/task{video_name}.json"
    output_dir = f"output_matted/{video_name}"
    output_annotation = "input/annotations/project-1-at-2025-04-16-13-42-d41c1b2b.json"
    # output_annotation = "input/annotations/project-2-at-2025-04-18-16-16-d5a9de12.json"
    process_annotations(video_path, annotation_path, output_annotation, output_dir)
