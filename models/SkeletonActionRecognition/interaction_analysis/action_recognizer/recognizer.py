import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """Graph Convolutional Network for processing skeletal points data.

    Args:
        input_dim (int): Dimension of input node features. Default: 3 (x,y,confidence).
        hidden_dim (int): Dimension of hidden layer. Default: 64.
        output_dim (int): Dimension of output features. Default: 32.
    """

    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, output_dim: int = 32) -> None:
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Forward pass of the GCN.

        Args:
            x (Tensor): Node feature matrix [num_nodes, input_dim]
            edge_index (Tensor): Graph connectivity in COO format [2, num_edges]

        Returns:
            Tensor: Output node features [num_nodes, output_dim]
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class SuperFormer(nn.Module):
    """Multimodal action recognition model combining mask, optical flow and skeleton data.

    Args:
        num_classes (int): Number of output action classes
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        # Mask processing stream
        self.mask_stream = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))

        # Optical flow processing stream
        self.temporal_stream = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))

        # Skeletal points processing
        self.gcn = GCN(input_dim=3, hidden_dim=64, output_dim=256)

        # Feature fusion attention
        self.attention = nn.Sequential(
            nn.Linear(256 * 2 + 256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=1))

        # Final classifier
        self.fc = nn.Linear(256 * 2 + 256, num_classes)

    def forward(self,
                mask: Tensor,
                optical_flow: Tensor,
                skeleton_points: Tensor,
                edge_index: Tensor) -> Tensor:
        """Forward pass of the SuperFormer model.

        Args:
            mask (Tensor): Object mask tensor [batch_size, 1, H, W]
            optical_flow (Tensor): Optical flow tensor [batch_size, 2, H, W]
            skeleton_points (Tensor): Skeletal points coordinates [num_nodes, 3] (x,y,confidence)
            edge_index (Tensor): Skeleton graph edges [2, num_edges]

        Returns:
            Tensor: Classification logits [batch_size, num_classes]
        """
        # Process mask stream
        mask_features = self.mask_stream(mask).flatten(1)  # [B, 256]

        # Process optical flow stream
        flow_features = self.temporal_stream(optical_flow).flatten(1)  # [B, 256]

        # Process skeleton points
        skeleton_features = self.gcn(skeleton_points, edge_index)  # [N, 256]
        skeleton_features = torch.max(skeleton_features, dim=0)[0].unsqueeze(0)  # Global max pooling [1, 256]

        # Attention-weighted feature fusion
        combined = torch.cat([mask_features, flow_features, skeleton_features], dim=1)
        weights = self.attention(combined)
        combined = (weights[:, 0:1] * mask_features +
                    weights[:, 1:2] * flow_features +
                    weights[:, 2:3] * skeleton_features)

        return self.fc(combined)
