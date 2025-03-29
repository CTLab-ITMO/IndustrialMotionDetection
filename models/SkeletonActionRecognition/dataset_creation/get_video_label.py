import pandas as pd


def get_sample_action(path, action, desc, num_samples):
    df = pd.read_csv(path, low_memory=False)
    df = df[df['video_path'].str.contains(action, na=False)]
    df = df[['video_path', 'action_category']]
    df_sample = df.sample(n=num_samples, random_state=42)
    df_sample.to_csv(f'/Users/anastasiaspileva/Desktop/actions/{desc}.csv', index=False)


get_sample_action(
    '/Users/anastasiaspileva/Desktop/pip_175k/video_annotations_train.csv',
    'person_talks_on_phone',
    'train',
    100
)

get_sample_action(
    '/Users/anastasiaspileva/Desktop/pip_175k/video_annotations_val.csv',
    'person_talks_on_phone',
    'valid',
    20
)
