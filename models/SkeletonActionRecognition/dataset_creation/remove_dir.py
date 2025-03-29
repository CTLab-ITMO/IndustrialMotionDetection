import os

import pandas as pd


def remove_directory(directory_name, df_train: pd.DataFrame, df_valid: pd.DataFrame):
    dir_exists_train = df_train['video_path'].str.contains(directory_name).any()
    dir_exists_valid = df_valid['video_path'].str.contains(directory_name).any()
    if dir_exists_train or dir_exists_valid:
        return
    else:
        os.remove('/Users/anastasiaspileva/Desktop/actions/person_talks_on_phone/' + directory_name)


df_train = pd.read_csv('/Users/anastasiaspileva/Desktop/actions/train_phone.csv')
df_valid = pd.read_csv('/Users/anastasiaspileva/Desktop/actions/valid_phone.csv')

global_dir = '/Users/anastasiaspileva/Desktop/actions/person_talks_on_phone'
for dir_name in os.listdir(global_dir):
    remove_directory(dir_name, df_train, df_valid)

train_phone = pd.read_csv('/Users/anastasiaspileva/Desktop/actions/train_phone.csv')
valid_phone = pd.read_csv('/Users/anastasiaspileva/Desktop/actions/valid_phone.csv')

train_table = pd.read_csv('/Users/anastasiaspileva/Desktop/actions/train_table.csv')
valid_table = pd.read_csv('/Users/anastasiaspileva/Desktop/actions/valid_table.csv')

train = pd.concat([train_phone, train_table]).sample(frac=1).reset_index(drop=True)
valid = pd.concat([valid_phone, valid_table]).sample(frac=1).reset_index(drop=True)

train.to_csv('/Users/anastasiaspileva/Desktop/actions/train.csv')
valid.to_csv('/Users/anastasiaspileva/Desktop/actions/valid.csv')

df_train = pd.read_csv('/Users/anastasiaspileva/Desktop/actions/train.csv')
df_valid = pd.read_csv('/Users/anastasiaspileva/Desktop/actions/valid.csv')
df_train['video_path'] = df_train['video_path'].str.replace('/content/pip_175k/', '', regex=False).reset_index(
    drop=True)
df_valid['video_path'] = df_valid['video_path'].str.replace('/content/pip_175k/', '', regex=False).reset_index(
    drop=True)

df_train.to_csv('/Users/anastasiaspileva/Desktop/actions/train.csv')
df_valid.to_csv('/Users/anastasiaspileva/Desktop/actions/valid.csv')
