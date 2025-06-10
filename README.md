# IndustrialMotionDetection. How to start?

- clone repository

```bash
git clone -b ruslan-dev https://github.com/CTLab-ITMO/IndustrialMotionDetection.git
cd IndustrialMotionDetection
```

- setup local venv using poetry.lock and pyproject.toml

## Datasets

- Refer to [DATASETS.md](src/data/DATASETS.md) in order to use avaliable datasets 

## Project Structure

```
.
├── LICENSE
├── Makefile
├── README.md
├── conf
│   └── meva_preproc.yaml
├── data
│   └── MEVA
│       └── meva_processed
├── logfile.log
├── models
├── notebooks
│   ├── meva-processed-eda.ipynb
│   └── videomae-action-recognition.ipynb
├── poetry.lock
├── pyproject.toml
├── references
├── reports
│   └── figures
├── requirements.txt
├── scripts
│   ├── download.py
│   └── upload.py
├── setup.cfg
└── src
    ├── README.md
    ├── __init__.py
    ├── config.py
    ├── logger.py
    ├── meva_preprocessing.py
    ├── models
    │   └── VideoMAE
    │       ├── __init__.py
    │       ├── box_list.py
    │       ├── dataset.py
    │       ├── image_list.py
    │       ├── metrics.py
    │       ├── model.py
    │       ├── predict.py
    │       ├── preprocess.py
    │       └── train.py
    └── utils.py
```
