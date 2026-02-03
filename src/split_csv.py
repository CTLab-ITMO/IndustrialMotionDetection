import os
import pandas as pd
from tqdm import tqdm


def split_csv(dataset_root, annot_df_path) -> str:
    test_size = 0.2
    train_df_path = f"{dataset_root}/train.csv"
    test_df_path = f"{dataset_root}/test.csv"

    if not os.path.exists(annot_df_path):
        print("Annotations file not exist, split omitted")

    annotations = pd.read_csv(annot_df_path)

    assigned_videos = set()
    train, test = [], []

    # Get sorted actions by rarity (ascending video count)
    actions = sorted(
        annotations.groupby("action_category")["video_path"].nunique().items(),
        key=lambda x: x[1],
    )

    print(actions)

    for action, _ in tqdm(actions):
        # Get videos with this action, sorted by frame count (descending)
        videos = (
            annotations[annotations["action_category"] == action]
            .groupby("video_path")
            .size()
            .sort_values(ascending=False)
            .reset_index(name="frame_count")
        )

        videos = videos[~videos["video_path"].isin(assigned_videos)]
        if videos.empty:
            continue

        n_test = max(1, round(len(videos) * test_size))
        n_train = len(videos) - n_test

        train_videos = videos.iloc[:n_train]["video_path"]
        test_videos = videos.iloc[n_train : n_train + n_test]["video_path"]

        assigned_videos.update(train_videos)
        assigned_videos.update(test_videos)

        train.append(annotations[annotations["video_path"].isin(train_videos)])
        test.append(annotations[annotations["video_path"].isin(test_videos)])

    train_df = pd.concat(train)
    test_df = pd.concat(test)

    train_df.to_csv(train_df_path, index=False)
    test_df.to_csv(test_df_path, index=False)

    return train_df_path, test_df_path
