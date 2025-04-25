import json
import os

import cv2
import numpy as np
import tqdm
from PIL import Image
from libcom.fopa_heat_map import FOPAHeatMapModel
from libcom.fopa_heat_map.source.data.OBdataset import make_composite_PIL
from libcom.image_harmonization import ImageHarmonizationModel


def place_person_on_background_video(
        matted_video_path,
        background_video_path,
        annotations_path,
        mapping_path,
        person_id="1",
        output_dir="fopa_results",
        harmonize=False,
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'cache'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'heatmap'), exist_ok=True)

    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    with open(mapping_path, 'r') as f:
        person_mapping = json.load(f)

    person_key = f"person_{person_id}"
    if person_key not in person_mapping:
        print("error person not in mapping")
        return

    box_idx = person_mapping[person_key]["box_index"]

    video_item = annotations[0]
    box_data = video_item["box"][box_idx]
    sequence = box_data["sequence"]

    fg_cap = cv2.VideoCapture(matted_video_path)
    bg_cap = cv2.VideoCapture(background_video_path)

    if not fg_cap.isOpened() or not bg_cap.isOpened():
        raise Exception("can't open video files")

    fg_fps = fg_cap.get(cv2.CAP_PROP_FPS)
    bg_fps = bg_cap.get(cv2.CAP_PROP_FPS)
    fps = min(fg_fps, bg_fps)
    width = int(bg_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(bg_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fg_width = int(fg_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fg_height = int(fg_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fg_frame_count = int(fg_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video_path = os.path.join(output_dir, "composite_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    net = FOPAHeatMapModel(device=0)

    ret_fg, fg_frame = fg_cap.read()
    ret_bg, bg_frame = bg_cap.read()

    if not (ret_fg and ret_bg):
        raise Exception("can't read first frames from videos")

    first_fg_path = os.path.join(output_dir, 'first_fg.png')
    first_bg_path = os.path.join(output_dir, 'first_bg.png')
    cv2.imwrite(first_fg_path, fg_frame)
    cv2.imwrite(first_bg_path, bg_frame)

    hsv_fg = cv2.cvtColor(fg_frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 50, 50])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv_fg, lower_green, upper_green)
    mask = cv2.bitwise_not(mask)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    first_mask_path = os.path.join(output_dir, 'first_mask.png')
    cv2.imwrite(first_mask_path, mask)

    bboxes, _ = net(
        foreground_image=first_fg_path,
        foreground_mask=first_mask_path,
        background_image=first_bg_path,
        cache_dir=os.path.join(output_dir, 'cache'),
        heatmap_dir=os.path.join(output_dir, 'heatmap'),
        fg_scale_num=24,
        composite_num_choose=1,
        composite_num=25
    )

    if not bboxes:
        print("best frame not found")
        w, h = width // 3, height // 3
        x, y = (width - w) // 2, (height - h) // 2
        best_bbox = [x, y, w, h]
    else:
        best_bbox = bboxes[0]
        print("found best placement:", best_bbox)

    fg_cap.release()
    bg_cap.release()
    fg_cap = cv2.VideoCapture(matted_video_path)
    bg_cap = cv2.VideoCapture(background_video_path)

    new_annotations = []
    frame_count = 0

    if harmonize:
        CDTNet = ImageHarmonizationModel(device=0, model_type='PCTNet')

    with tqdm.tqdm(total=fg_frame_count, desc="Processing video") as pbar:
        while True:
            ret_fg, fg_frame = fg_cap.read()
            ret_bg, bg_frame = bg_cap.read()

            if not (ret_fg and ret_bg):
                break

            frame_count += 1

            hsv_fg = cv2.cvtColor(fg_frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_fg, lower_green, upper_green)
            mask = cv2.bitwise_not(mask)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            x, y, w, h = best_bbox

            background = Image.fromarray(cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB))
            foreground = Image.fromarray(cv2.cvtColor(fg_frame, cv2.COLOR_BGR2RGB))
            mask_pil = Image.fromarray(mask).convert("RGB")

            composite, comp_mask = make_composite_PIL(foreground, mask_pil, background, [x, y, w, h], return_mask=True)
            composite_np = np.array(composite)
            comp_mask_np = np.array(comp_mask)
            comp_mask_gray = cv2.cvtColor(comp_mask_np, cv2.COLOR_RGB2GRAY)

            if harmonize:
                try:
                    harmonized = CDTNet(cv2.cvtColor(composite_np, cv2.COLOR_RGB2BGR), comp_mask_gray)
                    output_frame = harmonized.astype(np.uint8)
                except Exception as e:
                    print(f"problem with harmonization {e}")
                    output_frame = cv2.cvtColor(composite_np, cv2.COLOR_RGB2BGR)
            else:
                output_frame = cv2.cvtColor(composite_np, cv2.COLOR_RGB2BGR)

            if output_frame.dtype != np.uint8:
                output_frame = output_frame.astype(np.uint8)

            out.write(output_frame)

            for box in sequence:
                if box["frame"] == frame_count:
                    box_x, box_y, box_width, box_height = box["x"], box["y"], box["width"], box["height"]
                    bg_width, bg_height = background.size

                    scale_x = w / fg_width
                    scale_y = h / fg_height

                    old_x = box_x * fg_width / 100
                    old_y = box_y * fg_height / 100
                    old_width = box_width * fg_width / 100
                    old_height = box_height * fg_height / 100

                    new_x = (x + old_x * scale_x) / bg_width * 100
                    new_y = (y + old_y * scale_y) / bg_height * 100
                    new_width = (w / fg_width) * old_width / bg_width * 100
                    new_height = (h / fg_height) * old_height / bg_height * 100

                    new_annotations.append({
                        "frame": frame_count,
                        "time": (frame_count - 1) / fps,
                        "x": new_x,
                        "y": new_y,
                        "width": new_width,
                        "height": new_height,
                        "enabled": True,
                        "rotation": 0
                    })
                    break

            pbar.update(1)

    fg_cap.release()
    bg_cap.release()
    out.release()

    labels = box_data.get("labels", ["person"])
    new_annotation = {
        "video": f"/data/local-files/?d=media/{os.path.basename(output_video_path)}",
        "fps": fps,
        "id": 1,
        "box": [
            {
                "framesCount": frame_count,
                "duration": frame_count / fps,
                "sequence": new_annotations,
                "labels": labels
            }
        ],
        "annotator": 1,
        "annotation_id": 1,
        "created_at": "2025-04-03T10:28:12.892153Z",
        "updated_at": "2025-04-03T10:28:12.892196Z",
        "lead_time": 1.0
    }

    new_annotation_path = os.path.join(output_dir, 'label_studio_annotation.json')
    with open(new_annotation_path, 'w') as f:
        json.dump([new_annotation], f, indent=2)

    print(f"results saved to {output_dir}")
    print(f"video saved to {output_video_path}")
    print(f"annotations saved to {new_annotation_path}")

    return output_video_path, new_annotation_path


if __name__ == "__main__":
    video_name = "VID_20250309_114742"
    bg_name = "2025-04-17 18-54-18"
    person_id = "2"
    matted_video_path = f"output_matted/{video_name}/person_{person_id}/matted/person_{person_id}_matted.mp4"
    background_video_path = f"input/backgrounds/{bg_name}.mkv"
    annotations_path = f"output_matted/{video_name}/updated_annotations.json"
    mapping_path = f"output_matted/{video_name}/person_box_mapping.json"
    output_dir = f"output_fopa/{bg_name}/{video_name}"

    place_person_on_background_video(
        matted_video_path=matted_video_path,
        background_video_path=background_video_path,
        annotations_path=annotations_path,
        mapping_path=mapping_path,
        person_id=person_id,
        output_dir=output_dir
    )
