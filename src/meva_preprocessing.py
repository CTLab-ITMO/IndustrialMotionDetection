import os
import time
import argparse
import yaml
import cv2
from typing import Tuple, Dict
import shutil
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from collections import defaultdict
from config import YamlConfigReader
from utils import (AverageMeter, 
                   get_size, 
                   is_awscli_installed, 
                   get_last_n_path_elements,
                   get_leaf_dirs)
from logger import Logger
import math


class MEVAProcessor:
    def __init__(self, args: argparse.Namespace):
        SHOW_LOG = True
        self.logger = Logger(SHOW_LOG).get_logger(__name__)
        
        self.config_source = YamlConfigReader(args.config) 
        self.params = self.config_source.get_all()
        
        self.videos_root = self.params['videos_root']
        self.annotations_folder = self.params['annotations_folder']
        if not os.path.isdir(self.annotations_folder):
            message = f'Annotations folder not exist {self.annotations_folder}'
            self.logger.error(message)
            raise Exception(message)
        
        self.result_folder = self.params['result_folder']
        os.makedirs(self.result_folder, exist_ok=True)
        
        self.target_activities = self.params['target_activities']
        self.padding_frames = self.params['padding_frames']
        self.bbox_area_limit = self.params['bbox_area_limit']
        self.display_annotations = self.params['display_annotations']
        
        self.annot_df_path = os.path.join(self.result_folder, 'annotations.csv')
        self.train_df_path = os.path.join(self.result_folder, 'train.csv')
        self.test_df_path =  os.path.join(self.result_folder, 'test.csv')
        self.test_size = self.params['test_size']
        self.split_seed = self.params['split_seed']
        
        self.logger.info("MEVAProcessor is ready")

    def parse_geometries(self, annotation_root: str, annot_filename: str) -> Tuple[Dict, Dict]:
        """
        Parse geometry annotations for a given filename.
        Schema Structure:
        - { geom: { id0: detection-id,
                    id1: track-id,
                    ts0: frame-id,
                    g0: geom-str,
                    src: source
                    [ occlusion: (medium|heavy) ]
                    [ cset3: {object: likelihood, … } ]
                    [ evalN: eval-tag... ]
                }
        }
        """
        geo_path = os.path.join(annotation_root, f"{annot_filename}.geom.yml")
        
        track_bbox_area = AverageMeter()
        geometries = defaultdict(dict)
        
        with open(geo_path) as f:
            entries = yaml.safe_load(f) or []

            for entry in entries:
                geom = entry.get('geom', {})
                track_id = geom.get('id1')
                frame_num = geom.get('ts0')
                bbox_str = geom.get('g0', '')

                if track_id is None or frame_num is None:
                    continue
                
                geometries[(track_id, frame_num)] = bbox_str
                
                coords = bbox_str.split()
                message = f'geom {track_id=} {frame_num=} isnt bbox {coords=}'
                assert len(coords) == 4, message
                xmin, ymin, xmax, ymax = map(int, coords)

                bbox_area = (xmax - xmin) * (ymax - ymin)
                track_bbox_area.update({track_id: bbox_area})
                
        return geometries, track_bbox_area.get_average()

    def find_activities(self, annotation_root: str, annot_filename: str) -> Tuple[Dict, Dict]:
        """
        Find all activities with their metadata.
        - { act { actN: {activity_name: likelihood, …},
                id_packet,
                timespan: [{tsr_packet} (... tsr_packet)],
                src: source,
                actors: [ {id_packet,
                            timespan: [{tsr_packet} (... tsr_packet)]}
                            (, next actor identification... )
                        ]
                }
        }
        { act: { act2: {Talking: 1.0},
                id2: 3,
                timespan: [{tsr0: [3293, 3314]}],
                src: truth,
                actors: [{id1: 9, timespan: [{tsr0: [3293, 3314]}]} ,
                            {id1: 12, timespan: [{tsr0: [3293, 3314]}]} ,
                        ]
                }
        }
        """
        activities = {}
        activity_id_counter = 0 # Auto-incremented ID for each activity
        frame_to_activity_ids = defaultdict(list)  # Maps frame numbers to activity IDs

        yml_path = os.path.join(annotation_root, f"{annot_filename}.activities.yml")

        with open(yml_path) as f:
            entries = yaml.safe_load(f) or []
            
            for entry in entries:
                act = entry.get('act', {})

                # skip empty act field
                if not act: 
                    continue

                action_name = list(list(act.values())[0].keys())[0]

                # skip irrelevant target_activities
                if action_name not in self.target_activities:
                    continue

                timespan = act.get('timespan', [{}])[0].get('tsr0', [])

                assert len(timespan) == 2, f'timespan length != 2: {len(timespan)}'

                actors = act.get('actors', [])

                actors = [a.get('id1') for a in actors
                          if a.get('id1') is not None]

                activities[activity_id_counter] = {
                    'action_category': action_name,
                    'start_frame': timespan[0],
                    'end_frame': timespan[1],
                    'actors': actors,
                }

                # Map frames to activity IDs
                for frame_num in range(timespan[0], timespan[1] + 1):
                    frame_to_activity_ids[frame_num].append(activity_id_counter)
                
                activity_id_counter += 1

        return activities, frame_to_activity_ids

    def combine_ranges(self, activities: dict, length: int) -> list:
        """Combine overlapping frame ranges for activities in the same video"""
        combined_ranges = []

        for activity in activities.values():
            start_frame = max(0, activity['start_frame'] - self.padding_frames)
            end_frame = min(length, activity['end_frame'] + 1 + self.padding_frames)
            combined_ranges.append((start_frame, end_frame))

        merged_ranges = []

        combined_ranges.sort()
        
        current_start, current_end = combined_ranges[0]
        for start, end in combined_ranges[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                merged_ranges.append((current_start, current_end))
                current_start, current_end = start, end

        merged_ranges.append((current_start, current_end))
        return merged_ranges

    def process_annotations(self,
                            video_root: str,
                            annotation_root: str,
                            result_folder: str) -> pd.DataFrame:
        """Main function to display annotations on videos and save them"""
        rows = []

        for fname in tqdm(os.listdir(annotation_root),
                          desc='Iterating dir with annotations...',
                          leave=False):
            # fname should end with .activities.yml
            if not fname.endswith('.activities.yml'):
                continue

            annotation_filename = fname.replace('.activities.yml', '')
            
            activities, frame_to_activity_ids = self.find_activities(
                annotation_root, annotation_filename)
            
            if len(activities) == 0:
                self.logger.info(f'{annotation_filename} empty')
                continue

            geometries, track_bbox_area = self.parse_geometries(
                annotation_root, annotation_filename)

            # filter tracks with box area >= 10000
            filtered_tracks = set([id for id, area in track_bbox_area.items() 
                                   if area >= self.bbox_area_limit])
            # filter activities before creating video: reduce memory usage
            activities = {id: activity for id, activity in activities.items()
                          if filtered_tracks.intersection(activity['actors'])}
            
            if len(activities) == 0:
                self.logger.info(f'After filtering bboxes {annotation_filename} empty')
                continue
            
            video_path = os.path.join(video_root, f"{annotation_filename}.r13.avi")
            if not os.path.exists(video_path):
                self.logger.warning(f"Video not found: {video_path}")
                continue
            
            # read video from MEVA dataset
            cap = cv2.VideoCapture(video_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            ranges = self.combine_ranges(activities, length)
            count_ranges = len(ranges)
            self.logger.info(f"Found {count_ranges} ranges at {annotation_filename}")
            
            for start_frame, end_frame in tqdm(ranges, 
                                               total=count_ranges, 
                                               desc='Ranges...', 
                                               leave=False):
                output_video_path = os.path.join(
                    result_folder, f"{annotation_filename}_frange{start_frame}-{end_frame}.avi")

                if os.path.exists(output_video_path):
                    self.logger.info(f"Already processed: {output_video_path}")
                    continue
                
                # initialize output video file
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                for frame_num in tqdm(range(start_frame, end_frame),
                                    desc='Iterating frames...', leave=False):
                    ret, frame = cap.read()
                    if not ret:
                        self.logger.warning(f"Failed to read frame {frame_num} from {video_path}")
                        break

                    # average complexity: O(1) (max activities per frame 2-3)
                    for activity_id in frame_to_activity_ids[frame_num]:
                        activity = activities.get(activity_id)
                                                
                        if activity is None:
                            continue
                        
                        temporal_rows = {}
                        
                        max_bbox_area_track = None
                        max_bbox_area = -math.inf

                        for actor_id in activity['actors']:
                            # skip tracks that less than limit in average across frames
                            # if track_bbox_area[actor_id] < self.bbox_area_limit:
                            #     continue
                            
                            coords = geometries.get((actor_id, frame_num), '').split()
                            if not len(coords) == 4:
                                message = f'geom {actor_id=} {frame_num=} isnt bbox {coords=} for {activity=}'
                                self.logger.warning(message)
                                continue 
                            xmin, ymin, xmax, ymax = map(int, coords)
                            bbox_area = (xmax - xmin) * (ymax - ymin)
                            
                            # skip frames with tracks less than limit
                            if bbox_area < self.bbox_area_limit:
                                continue
                            
                            if max_bbox_area < bbox_area:
                                max_bbox_area_track = actor_id
                                max_bbox_area = bbox_area

                            temporal_rows[actor_id] = {
                                'video_path': output_video_path,
                                'keyframe_id': frame_num - start_frame,
                                'track_id': actor_id,
                                'action_category': activity['action_category'],
                                'xmin': xmin,
                                'ymin': ymin,
                                'xmax': xmax,
                                'ymax': ymax
                            }
                            
                        # filter object bboxes for specific classes
                        if activity['action_category'] == 'person_picks_up_object':
                            assert max_bbox_area_track is not None
                            temporal_rows = {
                                max_bbox_area_track: temporal_rows[max_bbox_area_track]
                            }
                                
                        # draw bboxes on video
                        if self.display_annotations:
                            for r in temporal_rows.values():
                                cv2.putText(img=frame, 
                                            text=r['action_category'], 
                                            org=(20, 50), 
                                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                            fontScale=1, 
                                            color=(0, 255, 0), 
                                            thickness=2)
                                cv2.rectangle(img=frame, 
                                              pt1=(r['xmin'], r['ymin']), 
                                              pt2=(r['xmax'], r['ymax']), 
                                              color=(0, 255, 0), 
                                              thickness=2)
                            
                        rows.extend(list(temporal_rows.values()))

                    out.write(frame)

                out.release()
            
            cap.release()

            self.logger.info(f"Saved annotated video: {output_video_path}")

        return pd.DataFrame(rows)

    def download_meva_data_folder_for_date(self, date: str, dest: str) -> float:
        if not is_awscli_installed():
            error_message = 'awscli not installed'
            self.logger.error(error_message)
            raise Exception(error_message)
         
        self.logger.info(f'Downloading {date} from s3://mevadata-public-01/drops-123-r13 ...')
        start = time.time()
        os.system(f'aws s3 sync --no-sign-request s3://mevadata-public-01/drops-123-r13/{date} {dest}')
        return time.time() - start

    def split_train_test(self) -> bool:
        if not os.path.exists(self.annot_df_path):
            self.logger.error('Annotations file not exist, split omitted')
            return
        
        annotations = pd.read_csv(self.annot_df_path)
        
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        
        categories = annotations['action_category'].unique()
        
        for category in categories:
            category_df = annotations[annotations['action_category'] == category]
            
            unique_videos = category_df['video_path'].unique()
            
            train_videos, test_videos = train_test_split(
                unique_videos, test_size=self.test_size, random_state=self.split_seed)
            
            train_category_df = category_df[category_df['video_path'].isin(train_videos)]
            test_category_df = category_df[category_df['video_path'].isin(test_videos)]
            
            train_df = pd.concat([train_df, train_category_df])
            test_df = pd.concat([test_df, test_category_df])
        
        train_df.to_csv(self.train_df_path, index=False)
        test_df.to_csv(self.test_df_path, index=False)
        
        splits_created = (os.path.isfile(self.train_df_path) 
                          and os.path.isfile(self.test_df_path)) 
        
        if splits_created:
            self.logger.info("train and test data is ready")
            
            self.config_source.update_from_dict({
                'train': self.train_df_path,
                'test': self.test_df_path
            })
            
            return splits_created
        else:
            self.logger.error("train and test is not ready")
            return False
        
    def run(self) -> None:
        self.logger.info(f"Current working directory: {os.getcwd()}")
        
        already_processed = set(
            list(map(lambda x: get_last_n_path_elements(x, 2), 
                     get_leaf_dirs(self.result_folder))))
        self.logger.info(f'{already_processed=}')

        # Initialize an empty dataframe to store all annotations
        all_annotations_df = pd.DataFrame(columns=[
            'video_path', 'keyframe_id', 'track_id', 'action_category', 'xmin', 'ymin', 'xmax', 'ymax'])
        
        self.config_source.update_from_dict({'annotations_csv': self.annot_df_path})
        
        if os.path.exists(self.annot_df_path):
            all_annotations_df = pd.read_csv(self.annot_df_path)
            print("Annotations loaded from existing file.")

        for i, curr_annotation_root in enumerate(get_leaf_dirs(self.annotations_folder)):
            # if i == 1: break

            date = get_last_n_path_elements(curr_annotation_root, 2)
            
            # skip already processed folders
            if date in already_processed: 
                continue
            
            if date != '2018-03-11/12': # and date != '2018-03-15/16' and date != '2018-03-15/14':
                continue

            self.logger.info(f"Processing {curr_annotation_root} ...")
            curr_video_dir = os.path.join(self.videos_root, date)

            elapsed = self.download_meva_data_folder_for_date(date, curr_video_dir)
            self.logger.info(
                "Download completed:\n"
                f"\t {self.videos_root} size: {get_size(self.videos_root):.2f} GB\n"
                f"\t time elapsed: {elapsed} sec"
            )

            video_result_folder = os.path.join(self.result_folder, date)
            os.makedirs(video_result_folder, exist_ok=True)

            annotations_df = self.process_annotations(
                curr_video_dir, curr_annotation_root,
                video_result_folder)

            all_annotations_df = pd.concat(
                [all_annotations_df, annotations_df],
                ignore_index=True)
            
            # save annotations csv file
            all_annotations_df.to_csv(self.annot_df_path, index=False)
            self.logger.info(f"Saved annotations to CSV: {self.annot_df_path}")

            # log resulting data size
            self.logger.info(f"{self.result_folder=} size: {get_size(self.result_folder):.2f} GB")

            # delete initial video folder
            self.logger.info(f"Deleting folder {curr_video_dir} ...")
            shutil.rmtree(curr_video_dir)
            
            self.logger.info(
                "Deletion completed:\n"
                f"\t {self.videos_root} size: {get_size(self.videos_root):.2f} GB"
            )
            
        # save configuration updates
        self.config_source.save()
            
        # split annotations into train and test subsets
        self.split_train_test()


def main(args):
    processor = MEVAProcessor(args)
    processor.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="MEVA preprocessing script")
    parser.add_argument("-c", "--config",
                        type=str,
                        help="Sepcify config file path",
                        required=True,
                        nargs="?")
    args = parser.parse_args()
    main(args)
