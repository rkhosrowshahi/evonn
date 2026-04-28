import torch
from torch import nn

from .separable_conv import SeparableConv2d


class MNIST30K(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # The first convolution uses a 5x5 kernel and has 16 filters
        self.conv1 = SeparableConv2d(1, 16, kernel_size=5, stride=1, padding=2)
        # Then max pooling is applied with a kernel size of 2
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # The second convolution uses a 5x5 kernel and has 32 filters
        self.conv2 = SeparableConv2d(16, 32, kernel_size=5, stride=1, padding=2)
        # Another max pooling is applied with a kernel size of 2
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Apply layer normalization after the second pool
        self.norm = nn.BatchNorm1d(1568, affine=False)

        # A final linear layer maps outputs to the 10 target classes
        self.out = nn.Linear(1568, 10)

        # All activations are ReLU
        self.act = nn.ReLU()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # Apply the first conv + pool
        data = self.pool1(self.act(self.conv1(data)))
        # Apply the second conv + pool
        data = self.pool2(self.act(self.conv2(data)))

        # Apply layer norm
        data = self.norm(data.flatten(start_dim=1))

        # Flatten and apply the output linear layer
        data = self.out(data)

        return data


class MNIST500K(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # The first convolution uses a 5x5 kernel and has 16 filters
        self.conv1 = SeparableConv2d(1, 32, kernel_size=5, stride=1, padding=2)
        # Then max pooling is applied with a kernel size of 2
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # The second convolution uses a 5x5 kernel and has 32 filters
        self.conv2 = SeparableConv2d(32, 64, kernel_size=5, stride=1, padding=2)
        # Another max pooling is applied with a kernel size of 2
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Apply layer normalization after the second pool
        self.norm = nn.BatchNorm1d(3136, affine=False)

        # A final linear layer maps outputs to the 10 target classes
        self.out1 = nn.Linear(3136, 128)
        self.out2 = nn.Linear(128, 10)

        # All activations are ReLU
        self.act = nn.ReLU()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # Apply the first conv + pool
        data = self.pool1(self.act(self.conv1(data)))
        # Apply the second conv + pool
        data = self.pool2(self.act(self.conv2(data)))

        # Apply layer norm
        data = self.norm(data.flatten(start_dim=1))

        # Flatten and apply the output linear layer
        data = self.act(self.out1(data))
        data = self.out2(data)

        return data


class MNIST3M(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # The first convolution uses a 5x5 kernel and has 16 filters
        self.conv1 = SeparableConv2d(1, 32, kernel_size=5, stride=1, padding=2)
        # self.bn1 = nn.BatchNorm2d(32)  # Batch Norm for conv1
        # Then max pooling is applied with a kernel size of 2
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # The second convolution uses a 5x5 kernel and has 32 filters
        self.conv2 = SeparableConv2d(32, 64, kernel_size=5, stride=1, padding=2)
        # self.bn2 = nn.BatchNorm2d(64)  # Batch Norm for conv2
        # Another max pooling is applied with a kernel size of 2
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Apply layer normalization after the second pool
        self.norm = nn.BatchNorm1d(3136, affine=False)

        # A final linear layer maps outputs to the 10 target classes
        self.out1 = nn.Linear(3136, 1024)
        self.out2 = nn.Linear(1024, 10)

        # All activations are ReLU
        self.act = nn.ReLU()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # Apply the first conv + pool
        data = self.pool1(self.act(self.conv1(data)))
        # Apply the second conv + pool
        data = self.pool2(self.act(self.conv2(data)))

        # Apply layer norm
        data = self.norm(data.flatten(start_dim=1))

        # Flatten and apply the output linear layer
        data = self.act(self.out1(data))
        data = self.out2(data)

        return data

