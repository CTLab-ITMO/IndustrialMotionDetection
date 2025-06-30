import torch
import torch.nn as nn

from interaction_analysis.action_recognizer.model.mask_stream import MaskStream
from interaction_analysis.action_recognizer.model.skeleton_stream import ST_GCN_Net


class ActionFormer(nn.Module):
    def __init__(self, num_classes, adj_matrix, temporal_dim=16):
        """Two-stream action recognition model combining skeleton and mask information.

        :param num_classes: Number of action classes to predict
        :param adj_matrix: Adjacency matrix for the graph convolution
        :param temporal_dim: Temporal dimension for mask processing.
        """
        super().__init__()
        self.st_gcn_net = ST_GCN_Net(num_classes, adj_matrix)
        self.mask_stream = MaskStream(in_channels=1, base_channels=16, temporal_dim=temporal_dim)
        self.st_gcn_net.classifier = nn.Identity()
        self.gcn_feat_dim = 128
        self.mask_feat_dim = 160
        self.attention = nn.MultiheadAttention(
            embed_dim=self.gcn_feat_dim + self.mask_feat_dim,
            num_heads=8,
            dropout=0.1
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.gcn_feat_dim + self.mask_feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        self.norm = nn.LayerNorm(self.gcn_feat_dim + self.mask_feat_dim)

    def forward(self, gcn_input, mask_input):
        """Forward pass.

        :param gcn_input: Input tensor for skeleton stream of shape: (batch_size, num_channels, num_frames, num_joints)
        :param mask_input: Input tensor for mask stream of shape: (batch_size, 1, temporal_dim, height, width)

        :return: Classification logits for each action class of shape: (batch_size, num_classes)
        """
        gcn_feat = self.st_gcn_net(gcn_input)
        mask_feat = self.mask_stream(mask_input)
        combined = torch.cat([gcn_feat, mask_feat], dim=1)  #
        combined = combined.unsqueeze(0)
        attn_output, _ = self.attention(combined, combined, combined)
        attn_output = attn_output.squeeze(0)
        attn_output = self.norm(attn_output)
        out = self.classifier(attn_output)
        return out
