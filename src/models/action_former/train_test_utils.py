import time
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Any
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score

from interaction_analysis.action_recognizer.model.action_dataset import MultiModalActionDataset
from interaction_analysis.action_recognizer.model.action_former import ActionFormer


def compute_metrics(outputs: torch.tensor, labels: torch.tensor):
    """Calculates precision and recall."""
    _, predicted = torch.max(outputs, 1)
    predicted = predicted.cpu().numpy()
    labels = labels.cpu().numpy()

    precision = precision_score(labels, predicted, average='macro', zero_division=0)
    recall = recall_score(labels, predicted, average='macro', zero_division=0)
    return precision, recall


def train_epoch(model, dataloader, criterion, optimizer, device, writer, epoch):
    model.train()
    total_loss = 0
    all_labels, all_outputs = [], []

    for batch, label in tqdm(dataloader, desc="Training"):
        gcn_input = batch['keypoints'].to(device)
        mask_input = batch['masks'].to(device)
        labels = label.to(device)

        optimizer.zero_grad()

        outputs = model(gcn_input, mask_input)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        all_labels.extend(labels.to(device).numpy())
        all_outputs.extend(outputs.detach().to(device).numpy())

    all_outputs = torch.tensor(np.array(all_outputs))
    all_labels = torch.tensor(np.array(all_labels))
    precision, recall = compute_metrics(all_outputs, all_labels)
    avg_loss = total_loss / len(dataloader)

    writer.add_scalar('train/loss', avg_loss, epoch)
    writer.add_scalar('train/precision', precision, epoch)
    writer.add_scalar('train/recall', recall, epoch)
    return avg_loss, recall


def validate_epoch(model, dataloader, criterion, device, writer, epoch):
    model.eval()
    total_loss = 0
    correct = 0
    all_labels, all_outputs = [], []

    with torch.no_grad():
        for batch, label in tqdm(dataloader, desc="Validation"):
            gcn_input = batch['keypoints'].to(device)
            mask_input = batch['masks'].to(device)
            labels = label.to(device)

            outputs = model(gcn_input, mask_input)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

    all_outputs = torch.tensor(np.array(all_outputs))
    all_labels = torch.tensor(np.array(all_labels))
    precision, recall = compute_metrics(all_outputs, all_labels)
    avg_loss = total_loss / len(dataloader)

    writer.add_scalar('val/loss', avg_loss, epoch)
    writer.add_scalar('val/precision', precision, epoch)
    writer.add_scalar('val/recall', recall, epoch)

    return avg_loss, recall


def train_model(config: Dict[str, Any]) -> ActionFormer:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = MultiModalActionDataset(
        labels_path='/content/dataset/action_dataset/labels.csv',
        split='train',
        frame_step=config['T'],
        sequence_length=config['sequence_length']
    )

    val_dataset = MultiModalActionDataset(
        labels_path='/content/dataset/action_dataset/labels.csv',
        split='test',
        frame_step=config['T'],
        sequence_length=config['sequence_length']
    )

    adj_matrix = MultiModalActionDataset.get_adj_matrix()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )
    model = ActionFormer(
        num_classes=5,
        adj_matrix=adj_matrix,
        temporal_dim=config['T']
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config['lr'])
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)

    best_val_recall = 0.0
    time_experiment = time.time()
    writer = SummaryWriter(f'interaction_analysis/action_recognizer/runs/{time_experiment}')

    for epoch in range(config['epochs']):
        train_loss, train_recall = train_epoch(model, train_loader, criterion, optimizer, device, writer, epoch)
        val_loss, val_recall = validate_epoch(model, val_loader, criterion, device, writer, epoch)

        print(f'Epoch {epoch + 1}:')
        print(f'Train Loss: {train_loss:.4f} | Precision: {train_recall:.2%}')
        print(f'Val Loss: {val_loss:.4f} | Precision: {val_recall:.2%}')
        print("\n")

        scheduler.step(val_recall)

        if val_recall > best_val_recall:
            best_val_recall = val_recall
            torch.save(model.state_dict(), 'interaction_analysis/action_recognizer/saved_models/best_model.pth')
            print("Saved best model!")

    print(f"\nBest Validation Recall: {best_val_recall:.4f}")
    return model


if __name__ == '__main__':
    config = {
        'lr': 0.001,
        'epochs': 20,
        'num_classes': 5,
        'T': 5,
        'sequence_length': 5,
        'batch_size': 8
    }
    model = train_model(config)
