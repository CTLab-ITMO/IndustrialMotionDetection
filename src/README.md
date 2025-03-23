# How to use?

- clone repository

```bash
git clone -b ruslan-dev https://github.com/CTLab-ITMO/IndustrialMotionDetection.git
cd IndustrialMotionDetection
```

## Datasets

### Meva preprocessing

- create data folder

```bash
mkdir -p data/MEVA
```

- clone MEVA annotations repository (data/MEVA)

```bash
# clones annotations folder
git clone https://gitlab.kitware.com/meva/meva-data-repo.git data/MEVA
```

- change config file parameters at `conf/meva_preproc.yaml`

```yaml
result_folder: data/MEVA/meva_processed
videos_root: data/MEVA/mevadata-public-01/drops-123-r13
annotations_folder: data/MEVA/meva-data-repo/annotation/DIVA-phase-2/MEVA/kitware-meva-training
target_activities: [
  person_talks_on_phone, 
  person_texts_on_phone,
  person_picks_up_object,
  person_reads_document,
  person_interacts_with_laptop
]
padding_frames: 30
bbox_area_limit: 10000
display_annotations: True
```

- run preprocessing

```bash
python src/meva_preprocessing.py --config conf/meva_preproc.yaml
```
