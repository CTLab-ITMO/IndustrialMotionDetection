import os
import random
from datetime import datetime

import pandas as pd
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from hiera_test.train.run_epoch import train_model
import sys


import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoImageProcessor

from src.metrics_impl.Evaluator import Evaluator
from src.metrics_impl.utils import BBFormat, CoordinatesType
from hiera_test.train.dataset import ActionDetectionDataset, collate_fn
from hiera_test.train.model import HieraActionDetector
from hiera_test.train.utils import summarize_csv_labels
import kornia.augmentation as K

import decord
decord.bridge.set_bridge('torch')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train():
    seed = 2025
    torch_deterministic = True

    lr = 3 * 1e-4

    total_epochs_overall = 20
    num_head_only_epochs = 20
    num_warmup_epochs = 0
    num_finetune_epochs = total_epochs_overall - num_head_only_epochs - num_warmup_epochs
    save_frequency = 1

    batch_size = 6
    clip_len = 16
    frame_sample_rate = 2

    train_epoch_size_ratio = 10
    val_epoch_size_ratio = 10

    exp_name = f"hiera_mydataset_{total_epochs_overall}ep_{batch_size}bs_{clip_len}cliplen-{frame_sample_rate}fsr-2"

    run_name = f"{exp_name}__{seed}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = f"runs/{run_name}"
    writer = SummaryWriter(log_dir=log_dir)

    checkpoint_dir = os.path.join("runs", run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # cls = [
    #     "hand_interacts_with_person",
    #     "person_carries_heavy_object",
    #     "person_interacts_with_laptop",
    #     "person_reads_document",
    #     "person_talks_on_phone",
    #     "person_texts_on_phone"
    # ]

    cls = [
        # "bottle",
        # "eating",
        "smoking",
        "talking phone"
    ]

    train_csv_file_path = "../dataset_final/train_dataset.csv"
    val_csv_file_path = "../dataset_final/part_2/project-2_annotations.csv"

    df1 = pd.read_csv("../dataset_final/part_1/project-1_annotations.csv")
    df2 = pd.read_csv("../dataset_final/synthetic/synthetic_annotations.csv")
    df2["video_path"] = "dataset_final/" + df2["video_path"]
    df3 = pd.concat([df1, df2], axis=0, ignore_index=True)
    df3.to_csv(train_csv_file_path, index=False)

    weights = summarize_csv_labels(train_csv_file_path)

    class_to_idx = {cl: i + 1 for i, cl in enumerate(cls)}
    class_to_idx['background'] = 0
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    inv_freq = {cls: 1 / max(1, cnt) for cls, cnt in weights.items()}
    sum_weights = sum(inv_freq.values())
    class_weights = torch.zeros(len(class_to_idx))
    for name, idx in class_to_idx.items():
        class_weights[idx] = inv_freq[name] / sum_weights
    num_classes = len(class_to_idx)
    print(class_to_idx, idx_to_class)
    print(class_weights)
    print(inv_freq)
    image_processor = AutoImageProcessor.from_pretrained("facebook/hiera-base-224-hf")

    augmentations = K.AugmentationSequential(
        K.Resize(size=256),
        K.RandomHorizontalFlip(p=0.5),
        K.RandomGaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0)),
        K.RandomSharpness(sharpness=2, p=0.5),
        K.ColorJiggle(brightness=0.2, contrast=0.3, saturation=0.2, p=0.6),
        K.RandomPosterize(bits=3, p=0.4),

        same_on_batch=True,
        data_keys=["input", "bbox_xyxy"],
    )

    image_datasets = dict()
    image_datasets['train'] = ActionDetectionDataset("..", train_csv_file_path,
                                                     oversample=True,
                                                     oversample_ratio=3,
                                                     epoch_size_ratio=train_epoch_size_ratio,
                                                     processor=image_processor,
                                                     frame_sample_rate=frame_sample_rate,
                                                     augmentations = augmentations,
                                                     clip_len=clip_len,
                                                     length_file=f"./{train_csv_file_path.split("/")[-1]}.json",
                                                     class_to_idx=class_to_idx)

    image_datasets['test'] = ActionDetectionDataset("..", val_csv_file_path,
                                                    oversample=False,
                                                    epoch_size_ratio=val_epoch_size_ratio,
                                                    processor=image_processor,
                                                    frame_sample_rate=frame_sample_rate,
                                                    clip_len=clip_len,
                                                    length_file=f"./{val_csv_file_path.split("/")[-1]}.json",
                                                    class_to_idx=class_to_idx)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    dataloaders = dict()
    dataloaders['train'] = DataLoader(image_datasets['train'],
                                      batch_size=batch_size,
                                      pin_memory=True,
                                      drop_last=True,
                                      collate_fn=collate_fn,
                                      worker_init_fn=seed_worker,
                                      generator=g)
    dataloaders['test'] = DataLoader(image_datasets['test'],
                                     batch_size=batch_size,
                                     pin_memory=True,
                                     collate_fn=collate_fn,
                                     worker_init_fn=seed_worker,
                                     generator=g)

    print(len(dataloaders['train']), len(dataloaders['test']))

    random.seed(seed)
    np.random.seed(seed)
    bi = 0

    iouThreshold = 0.5
    showPlot = False
    evaluator = Evaluator()

    bbFormat = BBFormat.XYWH
    coordType = CoordinatesType.Absolute

    savePath = os.path.join(os.getcwd(), 'results')
    os.makedirs(savePath, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HieraActionDetector(num_classes, use_focal_loss=True).to(device)
    print(model)
    for name, param in model.named_parameters():
        if "backbone" not in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    print(f"Trainable parameters: {count_parameters(model)}")

    start_epoch = 0
    # load_checkpoint = "runs/hiera_mydataset_20ep_6bs_16cliplen-2fsr-2__2025__20250620_091928/hiera_mydataset_20ep_6bs_16cliplen-2fsr-2_head_epoch9.pth"
    load_checkpoint = None
    if load_checkpoint is not None:
        checkpoint = torch.load(load_checkpoint, weights_only=True, map_location=device)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
    print("start_epoch", start_epoch)
    start_epoch = 0

    if num_head_only_epochs > 0:
        for name, param in model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        optimizer_phase1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-2)
        scheduler_phase1 = CosineAnnealingLR(optimizer_phase1, T_max=num_head_only_epochs * len(dataloaders["train"]),
                                             eta_min=lr * 0.01)
        # if load_checkpoint is not None and "optimizer" in checkpoint:
        #     print('Load optimizer')
        #     optimizer_phase1.load_state_dict(checkpoint["optimizer"])
        #
        # if load_checkpoint is not None and "scheduler" in checkpoint:
        #     print('Load scheduler')
        #     scheduler_phase1.load_state_dict(checkpoint["scheduler"])

        train_model(
            dataloaders,
            idx_to_class,
            model=model,
            seed=seed,
            writer=writer,
            optimizer=optimizer_phase1,
            scheduler=scheduler_phase1,
            start_epoch=start_epoch,
            savePath=checkpoint_dir,
            save_frequency=save_frequency,
            run_name=f"{run_name}",
            exp_name=f"{exp_name}_head",
            num_epochs=num_head_only_epochs,
            device=device,
            num_classes=num_classes,
            evaluator=evaluator,
            showPlot=showPlot,
            bi=bi,
            bbFormat=bbFormat,
            coordType=coordType,
            iouThreshold=iouThreshold,
        )

    if num_warmup_epochs > 0:
        blocks_unfreeze = ["blocks.23"]
        for name, param in model.named_parameters():
            if any(blocks in name for blocks in blocks_unfreeze) or "backbone" not in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        print(f"params {count_parameters(model)}")

        head_params = []
        unfrozen_backbone_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if "backbone" not in name:
                    head_params.append(param)
                elif any(blocks in name for blocks in blocks_unfreeze):
                    unfrozen_backbone_params.append(param)

        optimizer_params_phase2 = []
        if head_params:
            optimizer_params_phase2.append({"params": head_params, "lr": 3 * 1e-4})
        if unfrozen_backbone_params:
            optimizer_params_phase2.append({"params": unfrozen_backbone_params, "lr": 1e-4})

        optimizer_phase2 = optim.AdamW(optimizer_params_phase2, weight_decay=1e-2)
        total_warmup_steps = num_warmup_epochs * len(dataloaders["train"])

        scheduler_phase2 = torch.optim.lr_scheduler.LinearLR(
            optimizer_phase2,
            start_factor=0.001,
            end_factor=1.0,
            total_iters=total_warmup_steps
        )

        train_model(
            dataloaders,
            idx_to_class,
            model=model,
            seed=seed,
            writer=writer,
            optimizer=optimizer_phase2,
            scheduler=scheduler_phase2,
            start_epoch=start_epoch + num_head_only_epochs,
            savePath=checkpoint_dir,
            save_frequency=save_frequency,
            run_name=f"{run_name}",
            exp_name=f"{exp_name}_warmup",
            num_epochs=num_warmup_epochs,
            device=device,
            num_classes=num_classes,
            evaluator=evaluator,
            showPlot=showPlot,
            bi=bi,
            bbFormat=bbFormat,
            coordType=coordType,
            iouThreshold=iouThreshold
        )

    if num_finetune_epochs > 0:
        print("phase 3 finetuning")
        finetune_lr = 1e-4
        optimizer_finetune = optim.AdamW(
            model.parameters(),
            lr=finetune_lr,
            weight_decay=1e-2
        )
        scheduler_finetune = CosineAnnealingLR(
            optimizer_finetune,
            T_max=num_finetune_epochs * len(dataloaders["train"]),
            eta_min=finetune_lr * 0.01
        )

        train_model(
            dataloaders,
            idx_to_class,
            model=model,
            seed=seed,
            writer=writer,
            optimizer=optimizer_finetune,
            scheduler=scheduler_finetune,
            start_epoch=start_epoch + num_head_only_epochs + num_warmup_epochs,
            savePath=checkpoint_dir,
            save_frequency=save_frequency,
            run_name=f"{run_name}",
            exp_name=f"{exp_name}_finetune",
            num_epochs=num_finetune_epochs,
            device=device,
            num_classes=num_classes,
            evaluator=evaluator,
            showPlot=showPlot,
            bi=bi,
            bbFormat=bbFormat,
            coordType=coordType,
            iouThreshold=iouThreshold
        )

    writer.close()


train()
