import os
import random
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import tqdm.auto as tqdm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Union
from collections import OrderedDict
from timm import create_model
from dataset import get_dataloaders
from logger import Logger
from metrics import compute_metrics_batch_detections


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def count_parameters(model: nn.Module):
        return sum(
            p.numel() for p in model.parameters() if p.requires_grad)


class Trainer:
    def __init__(
        self,
        seed: int = 2025,
        eval_frequency: int = 2,
        save_frequency: int = 1,
        num_classes: int = 5,
        lr: float = 1e-4,
        batch_size: int = 2,
        num_epochs: int = 5,
        clip_len: int = 16,
        frame_sample_rate: float = 1.0,
        finetune: str = "/content/checkpoint.pth",
        load_checkpoint: Union[str, None] = None):
        
        SHOW_LOG = True
        self.logger = Logger(SHOW_LOG).get_logger(__name__)
        
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        
        self.seed = seed
        self.save_frequency = save_frequency
        self.num_classes = num_classes
        self.exp_name = f"AL_VideoMAE_MEVA_talksOnPhone_{num_epochs}ep_{batch_size}bs_{clip_len}cliplen-{frame_sample_rate}fsr-test"

        self.save_path = os.path.join(os.getcwd(), 'results') 
        os.makedirs(self.save_path, exist_ok=True)
        
        self.batch_idx = 0
        self.bboxes_path = os.path.join(self.save_path, 'bboxes.jpg')
        
        seed_everything(self.seed)

        self.model = self._init_model(num_classes=self.num_classes)
        
        self._load_pretrained_weights(finetune)
        
        self.logger.info(f"Trainable params: {count_parameters(self.model)}")
        
        start_epoch = 0
        if load_checkpoint is not None:
            checkpoint = torch.load(
                load_checkpoint, weights_only=True, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            start_epoch = checkpoint['epoch'] + 1
            
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        dataloaders = get_dataloaders()
        
        self.run_name = f"{self.exp_name}__{self.seed}__{int(time.time())}"
        writer = SummaryWriter(log_dir=f"runs/{self.run_name}")
        
        self.train_model(start_epoch, dataloaders, writer, num_epochs)
    
        
    def _init_model(self, num_classes):
        return create_model(
            "vit_small_patch16_224",
            pretrained=False,
            num_classes=num_classes,
            all_frames=16,
            tubelet_size=2,
            drop_rate=0.,
            drop_path_rate=0.2,
            attn_drop_rate=0.,
            drop_block_rate=None,
            use_checkpoint=False,
            use_mean_pooling=True,
            init_scale=0.001,
        )

    def _load_pretrained_weights(self, weights_path):
        checkpoint = torch.load(weights_path, map_location=self.device)
        checkpoint_model = checkpoint['model']

        for name, param in self.model.named_parameters():
            if (name.startswith('rpn_network')
                or name.startswith('bbox_head')
                or name.startswith('head')
                or name.startswith('roi_heads')):
                continue
            param.requires_grad = False

        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.'):
                new_dict[key[8:]] = checkpoint_model[key]
            elif (key.startswith('decoder.')
                or key == 'mask_token'
                or key.startswith('encoder_to_decoder')):
                continue
            else:
                new_dict[key] = checkpoint_model[key]

        checkpoint_model = new_dict
        self.model.load_state_dict(checkpoint_model, strict=False)
        
    def train_model(
        self, start_epoch, dataloaders, writer=None, num_epochs=5, 
    ):
        self.logger.info(f"Training model with params:")
        self.logger.info(f"Optim: {self.optimizer}")
        self.logger.info(f"Criterion: {self.criterion}")

        phases = ['train', 'test']
        for phase in dataloaders:
            if phase not in phases:
                phases.append(phase)

        for epoch in tqdm(range(start_epoch, num_epochs + 1)):
            for phase in phases:

                epoch_loss, epoch_mAP, epoch_AP = (
                    self.train_epoch(dataloaders[phase]) 
                    if phase == 'train'
                    else self.test_epoch(dataloaders[phase]))

                if writer:
                    if phase == 'train':
                        writer.add_scalar(f'loss/{phase}', epoch_loss, epoch)

                    elif phase == 'test':
                        writer.add_scalar(f'mAP/{phase}', epoch_mAP, epoch)

                        precision_recall_curves = {
                            f'images/{i}-curve': os.path.join(self.save_path, f'{i}.png')
                            for i in range(self.num_classes)
                        }

                        image_paths = {
                            'images/bboxes': self.bboxes_path,
                            **precision_recall_curves
                        }

                        for k, path in image_paths.items():
                            if os.path.exists(path):
                                writer.add_image(k, plt.imread(path)[:, :, :3], epoch, dataformats='HWC')

                        for k, v in epoch_AP.items():
                            writer.add_scalar(f'{k}/{phase}', v, epoch)

            if epoch % self.save_frequency == 0:
                model_path = f"runs/{self.run_name}/{self.exp_name}.pth"
                self.logger.info(f"Saving model at {model_path}")
                torch.save({'epoch': epoch,
                            'model': self.model.state_dict()}, model_path)

                repo_name = f"{self.exp_name}-{self.seed}"
                os.system(f"zip -r {repo_name}.zip runs")
                os.system(f"cp {repo_name}.zip /content/drive/MyDrive/data")
                
    def test_epoch(self, dataloader):
        with torch.inference_mode():
            return self.run_epoch('test', dataloader)

    def train_epoch(self, dataloader):
        return self.run_epoch('train', dataloader)

    def run_epoch(self, phase, dataloader):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        running_mAP = 0.0
        running_AP = {f"AP-{i}": 0.0 for i in range(self.num_classes)}

        all_elems_count = 0
        batch_count = len(dataloader)

        cur_tqdm = tqdm(dataloader, total=len(dataloader), leave=False)
        for i, batch in enumerate(cur_tqdm):
            acc_AP = 0
            validClasses = 0

            inputs = batch['video'].to(self.device) # [B, C, T, H, W]
            labels = batch['target'].to(self.device) # [B, N, cls]
            boxes = [b.to(self.device) for b in batch['bbox']]

            detection_results, losses = self.model(inputs, boxes, labels)

            bz = inputs.shape[0]
            all_elems_count += bz

            show_dict = {}

            if phase == 'train':
                loss = sum(losses.values())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * bz

                show_dict['Loss'] = f'{loss.item():.6f}'
                for k, v in losses.items(): show_dict[k] = v.item()
                cur_tqdm.set_postfix(show_dict)

                # if i % eval_frequency == 0 and i != 0:
                #     test_epoch(dataloaders['test']) # note: after return model in eval mode not train

            elif phase == 'test':
                self.draw_bboxes(
                    batch['video'][self.batch_idx],
                    detection_results[self.batch_idx]['boxes'].cpu().detach().numpy(),
                    detection_results[self.batch_idx]['scores'].cpu().detach().numpy(),
                    detection_results[self.batch_idx]['labels'].cpu().detach().numpy(),
                    self.bboxes_path)

                detection_metrics = compute_metrics_batch_detections(
                    detection_results, batch
                )

                show_dict['bbox_num_per_video'] = sum(r['boxes'].shape[0] for r in detection_results) / bz
                show_dict['labels_in_batch'] = set(labels.cpu().detach().argmax(dim=2).squeeze().numpy())
                show_dict['pred_labels'] = set(np.concatenate([r['labels'].cpu().detach().numpy() for r in detection_results]))

                for metricsPerClass in detection_metrics:
                    cl = metricsPerClass['class']
                    ap = metricsPerClass['AP']
                    totalPositives = metricsPerClass['total positives']
                    total_TP = metricsPerClass['total TP']
                    total_FP = metricsPerClass['total FP']

                    if totalPositives > 0:
                        validClasses = validClasses + 1
                        acc_AP = acc_AP + ap

                        show_dict[f'AP-{cl}'] = "{0:.2f}%".format(ap * 100)
                        running_AP[f'AP-{cl}'] += ap

                        show_dict[f'total_TP-{cl}'] = total_TP
                        show_dict[f'total_FP-{cl}'] = total_FP
                        show_dict[f'total_Positives-{cl}'] = totalPositives

                mAP = acc_AP / validClasses
                show_dict['mAP'] = "{0:.2f}%".format(mAP * 100)
                running_mAP += mAP

                cur_tqdm.set_postfix(show_dict)

        epoch_loss = running_loss / batch_count
        epoch_mAP = running_mAP / batch_count
        epoch_AP = {k: v / batch_count for k, v in running_AP.items()}

        return epoch_loss, epoch_mAP, epoch_AP
    
    def draw_bboxes(self, video, boxes, scores, labels, bboxes_path, top_k=3):
        # video shape - [c, t, h, w]
        # boxes shape - [n, 4]
        # labels and scores shape - [n]

        fig, ax = plt.subplots(1)
        middle_frame_index = video.shape[1] // 2
        
        # [c, h, w] -> [c, w, h]
        image = video[:, middle_frame_index].permute(1, 2, 0)
        
        image = (image - image.min()) / (image.max() - image.min())

        ax.imshow(image)

        self.logger.info(f"top_k boxes stat:\n{list(zip(boxes, scores, labels))[:top_k]}")
        for box, score, label in list(zip(boxes, scores, labels))[:top_k]:
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min

            rect = patches.Rectangle((x_min, y_min), width, height,
                                    linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            label_text = f"Label: {label}, Score: {score:.2f}"
            ax.text(x_min, y_min - 5, label_text,
                    color='red', fontsize=8, backgroundcolor='white')

        plt.savefig(bboxes_path)


def main():
    t = Trainer()
    
    
if __name__ == "__main__":
    main()
