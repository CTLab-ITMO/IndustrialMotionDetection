# Datasets

## MEVA

- create data folder

```bash
mkdir -p data/MEVA
```

- clone MEVA annotations repository (data/MEVA)

```bash
# clones annotations folder
git clone https://gitlab.kitware.com/meva/meva-data-repo.git data/MEVA/meva-data-repo
```

- change config file parameters at `conf/meva_preproc.yaml`

```yaml
annotations_csv: data/MEVA/meva_processed/annotations.csv
annotations_folder: data/MEVA/meva-data-repo/annotation/DIVA-phase-2/MEVA/kitware-meva-training
bbox_area_limit: 10000
display_annotations: false
padding_frames: 30
result_folder: data/MEVA/meva_processed
split_seed: 42
target_activities:
- person_talks_on_phone
- person_texts_on_phone
- person_picks_up_object
- person_reads_document
- person_interacts_with_laptop
test_size: 0.2
videos_root: data/MEVA/mevadata-public-01/drops-123-r13
```

- run preprocessing

```bash
python src/meva_preprocessing.py --config conf/meva_preproc.yaml
```

### Save processed dataset into kaggle dataset collection

- create `.env` file and fill missing variables

```text
KAGGLE_USERNAME=
KAGGLE_KEY=
```

- run command below to upload processed dataset to your remote repo (specify dataset name for remote repo)

```bash
python scripts/upload.py --dataset-name meva-processed-test
```

- run command below to download dataset from your remote repo

```bash
python scripts/download.py \
  --dataset $KAGGLE_USERNAME/meva-processed-test \
  --output data/MEVA/meva-processed-test.zip
```

- unzip donwloaded dataset

```bash
unzip -o -q data/MEVA/meva-processed-test.zip \
  -d data/MEVA/meva-processed-test
```
