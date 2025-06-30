import torch.nn as nn


class MaskStream(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, temporal_dim=16):
        """Initialize the mask stream.

        :param in_channels: Number of input channels (1 for binary masks)
        :param base_channels: Base number of channels after first conv
        :param temporal_dim: Number of frames (T)
        """
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3),
                      stride=(1, 2, 2), padding=(1, 1, 1), groups=in_channels),
            nn.Conv3d(in_channels, base_channels, kernel_size=1),
            nn.GroupNorm(4, base_channels),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(4, base_channels * 2),
            nn.ReLU()
        )

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool3d((temporal_dim, 1, 1)),
            nn.Dropout(0.2))

    def forward(self, x):
        """Forward pass of the mask stream."""
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        return x.flatten(1)
