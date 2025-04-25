import os
import sys
import argparse
from pathlib import Path
import json
from tqdm import tqdm

from add_green_screen import process_annotations
from convert_annotations import update_label_studio_annotations
from use_libcom import place_person_on_background_video


def process_backgrounds(
        foregrounds_dir="input/foregrounds",
        backgrounds_dir="input/backgrounds",
        output_base="output_fopa",
        person_filter=None,
        video_filter=None
):
    foregrounds = []
    for file in os.listdir(foregrounds_dir):
        if file.endswith((".mp4")):
            video_name = os.path.splitext(file)[0]
            if video_filter is None or video_name in video_filter:
                foregrounds.append(video_name)

    backgrounds = []
    for file in os.listdir(backgrounds_dir):
        if file.endswith((".mp4", ".mkv")):
            bg_name = os.path.splitext(file)[0]
            backgrounds.append(bg_name)

    print(f"found {len(foregrounds)} foreground videos and {len(backgrounds)} background videos")

    for video_name in tqdm(foregrounds, desc="foreground videos"):
        video_path = os.path.join(foregrounds_dir, f"{video_name}.mp4")
        annotation_path = f"../local_st/tasks/task{video_name}.json"
        output_matted_dir = f"output_matted/{video_name}"
        original_ls_annotations = "input/annotations/project-1-at-2025-04-16-13-42-d41c1b2b.json"

        if not os.path.exists(output_matted_dir):
            print(f"\n{video_name} for matting")
            try:
                process_annotations(
                    video_path=video_path,
                    annotation_path=annotation_path,
                    output_annot=original_ls_annotations,
                    output_dir=output_matted_dir
                )

                update_label_studio_annotations(
                    original_annotation_path=original_ls_annotations,
                    processed_dir=output_matted_dir,
                    output_path=f"{output_matted_dir}/updated_annotations.json",
                    video_name=video_name
                )

            except Exception as e:
                print(f"error processing {video_name}: {e}")
                continue
        else:
            print(f"using existing matted videos for {video_name}")

        available_persons = []
        mapping_path = os.path.join(output_matted_dir, "person_box_mapping.json")

        if os.path.exists(mapping_path):
            with open(mapping_path, 'r') as f:
                mapping = json.load(f)
                for person_key in mapping:
                    person_id = person_key.split('_')[1]
                    if person_filter is None or person_id in person_filter:
                        available_persons.append(person_id)
        else:
            for dir_name in os.listdir(output_matted_dir):
                if dir_name.startswith("person_"):
                    person_id = dir_name.split('_')[1]
                    if person_filter is None or person_id in person_filter:
                        available_persons.append(person_id)

        print(f"found {len(available_persons)} persons in {video_name}: {available_persons}")

        for person_id in available_persons:
            for bg_name in tqdm(backgrounds, desc=f"backgrounds for {video_name} person {person_id}"):
                matted_video_path = f"{output_matted_dir}/person_{person_id}/matted/person_{person_id}_matted.mp4"
                background_video_path = f"{backgrounds_dir}/{bg_name}.mkv"

                if not os.path.exists(background_video_path):
                    for ext in [".mp4", ".mkv"]:
                        alt_path = f"{backgrounds_dir}/{bg_name}{ext}"
                        if os.path.exists(alt_path):
                            background_video_path = alt_path
                            break

                if not os.path.exists(matted_video_path) or not os.path.exists(background_video_path):
                    print(f"{video_name} person {person_id} + {bg_name} missing files")
                    continue

                output_dir = f"{output_base}/{bg_name}/{video_name}/person_{person_id}"

                if os.path.exists(os.path.join(output_dir, "composite_video.mp4")):
                    print(f"skipping {video_name} person {person_id} + {bg_name}")
                    continue

                try:

                    place_person_on_background_video(
                        matted_video_path=matted_video_path,
                        background_video_path=background_video_path,
                        annotations_path=f"{output_matted_dir}/updated_annotations.json",
                        mapping_path=mapping_path,
                        person_id=person_id,
                        output_dir=output_dir
                    )
                    print(f"placed {video_name} person {person_id} on {bg_name}")
                except Exception as e:
                    print(f"error: {video_name} person {person_id} on {bg_name}: {e}")


if __name__ == "__main__":
    process_backgrounds(
        foregrounds_dir="input/foregrounds",
        backgrounds_dir="input/backgrounds",
        output_base="output_last",
        person_filter="1 2 3",
    )
