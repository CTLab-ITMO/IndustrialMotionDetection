import torch
import torch.nn as nn


class ConvTemporalGraph(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 adj_matrix,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        """A module that performs temporal convolution followed by graph convolution.

        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Size of the graph convolution kernel
        :param adj_matrix: Adjacency matrix/matrices for graph convolution
        :param t_kernel_size: Size of the temporal convolution kernel
        :param t_stride: Stride of the temporal convolution
        :param t_padding: Padding for the temporal convolution
        :param t_dilation: Dilation for the temporal convolution
        :param bias: Whether to add bias
        """
        super().__init__()

        self.adj_matrix = adj_matrix
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x):
        assert self.adj_matrix.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, self.adj_matrix))

        return x.contiguous()


class ST_GCN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 adj_matrix,
                 stride=1,
                 residual=True):
        """A Spatio-Temporal Graph Convolutional Network block combining temporal and graph convolutions with residual connection.

        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Tuple of (temporal_kernel_size, graph_kernel_size)
        :param adj_matrix: Adjacency matrix/matrices for graph convolution
        :param stride: Temporal stride (default: 1)
        :param residual: Whether to use residual connection (default: True)
        """
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        self.adj_matrix = adj_matrix
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraph(in_channels, out_channels,
                                     kernel_size[1], adj_matrix=adj_matrix)

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass."""
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x) + res
        return self.relu(x)


class ST_GCN_Net(nn.Module):
    def __init__(self, num_classes, adj_matrix):
        """

        :param num_classes: Number of classes
        :param adj_matrix: Adjacency matrix for graph convolution
        """
        super().__init__()
        self.adj_matrix = adj_matrix
        self.st_gcn_blocks = nn.Sequential(
            ST_GCN(3, 64, (3, 1), adj_matrix),
            ST_GCN(64, 128, (3, 1), adj_matrix)
        )

        self.pooling = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        """Initializes network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass."""
        # (B, C, T, V) -- (B, 3, T, 17)
        x = self.st_gcn_blocks(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
