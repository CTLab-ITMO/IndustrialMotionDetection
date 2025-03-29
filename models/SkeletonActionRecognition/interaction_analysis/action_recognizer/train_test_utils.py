import torch
import torch.nn as nn
import torch_geometric
import torch.optim as optim
from typing import Tuple, List
from torch_geometric.data import Batch
from action_dataset import ActionDataset
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, random_split
from interaction_analysis.action_recognizer.recognizer import SuperFormer


def train_model(data_dir: str, num_classes: int, epochs: int = 20, batch_size: int = 8) -> None:
    """Train the SuperFormer action recognition model.

    Args:
        data_dir: Path to the dataset directory
        num_classes: Number of action classes
        epochs: Number of training epochs
        batch_size: Size of training batches
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SuperFormer(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    full_dataset = ActionDataset(data_dir)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    def collate_fn(batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
        """Custom collate function to handle graph data.

        Args:
            batch: List of dataset items

        Returns:
            Tuple containing:
            - masks: Stacked mask tensors
            - flows: Stacked optical flow tensors
            - graph_batch: Batched graph data
            - labels: Stacked label tensors
        """
        masks = torch.stack([item[0] for item in batch])
        flows = torch.stack([item[1] for item in batch])
        labels = torch.tensor([item[4] for item in batch])

        skeletons = [item[2] for item in batch]
        edges = [item[3] for item in batch]

        graph_batch = Batch.from_data_list([
            torch_geometric.data.Data(x=skeletons[i], edge_index=edges[i])
            for i in range(len(batch))
        ])

        return masks, flows, graph_batch, labels

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, collate_fn=collate_fn)

    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for masks, flows, graph_batch, labels in train_loader:
            masks = masks.to(device)
            flows = flows.to(device)
            labels = labels.to(device)
            graph_batch = graph_batch.to(device)

            optimizer.zero_grad()

            outputs = model(masks, flows, graph_batch.x, graph_batch.edge_index)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        avg_train_loss = train_loss / len(train_loader)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f} | Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print('Model saved!')

    print('\nTesting best model...')
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, test_mode=True)
    print(f'Test Loss: {test_loss:.4f} | Acc: {test_acc:.2f}%')


def evaluate(model: nn.Module,
             data_loader: DataLoader,
             criterion: nn.Module,
             device: torch.device,
             test_mode: bool = False) -> Tuple[float, float]:
    """Evaluate the model on validation or test data.

    Args:
        model: The model to evaluate
        data_loader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to run evaluation on
        test_mode: Whether to generate classification report

    Returns:
        Tuple containing average loss and accuracy
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for masks, flows, graph_batch, labels in data_loader:
            masks = masks.to(device)
            flows = flows.to(device)
            labels = labels.to(device)
            graph_batch = graph_batch.to(device)

            outputs = model(masks, flows, graph_batch.x, graph_batch.edge_index)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if test_mode:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = 100. * correct / total

    if test_mode:
        print('\nClassification Report:')
        print(classification_report(all_labels, all_preds))

    return avg_loss, accuracy
