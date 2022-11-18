from .task_nn import TaskNN
import torch
from torch import nn
from torch.nn import functional as F


@TaskNN.register("weizmann-horse-seg-s")
class WeizmannHorseSegTaskNN(TaskNN):
    def __init__(self):
        super().__init__()  # type: ignore
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv2d(3, 64, 5, 1, padding=2)
        self.conv1_ = nn.Conv2d(64, 1, 1, 1)
        self.conv2 = nn.Conv2d(64, 128, 5, 2, padding=2)
        self.conv2_ = nn.Conv2d(128, 1, 1, 1)
        self.conv3 = nn.Conv2d(128, 128, 5, 2, padding=2)
        self.conv3_ = nn.Conv2d(128, 1, 1, 1)
        # nn.ConvTranspose2d(
        #   in_channels, out_channels, kernel_size, stride=1, padding=0,
        #   output_padding=0, groups=1, bias=True, dilation=1
        # )
        self.deconv2 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv1 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x) -> torch.Tensor:
        x1 = F.relu(self.conv1(x))
        # print(x1.size()) # torch.Size([batch, 64, 24, 24])
        x2 = F.relu(self.conv2(x1))
        # print(x2.size()) # torch.Size([batch, 128, 12, 12])
        x3 = F.relu(self.conv3(x2))
        # print(x3.size()) # torch.Size([batch, 128, 6, 6])

        x4 = self.deconv2(self.conv3_(x3)) + self.conv2_(x2)
        # print(x4.size()) # torch.Size([batch, 1, 12, 12])
        return self.deconv1(x4) + self.conv1_(x1) # (batch, 1, 24, 24) of unnormalized logits


@TaskNN.register("weizmann-horse-seg")
class WeizmannHorseSegTaskNN(TaskNN):
    def __init__(self):
        super().__init__()  # type: ignore
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv2d(3, 64, 5, 1, padding=2)
        self.conv1_ = nn.Conv2d(64, 2, 1, 1)
        self.conv2 = nn.Conv2d(64, 128, 5, 2, padding=2)
        self.conv2_ = nn.Conv2d(128, 2, 1, 1)
        self.conv3 = nn.Conv2d(128, 128, 5, 2, padding=2)
        self.conv3_ = nn.Conv2d(128, 2, 1, 1)
        # nn.ConvTranspose2d(
        #   in_channels, out_channels, kernel_size, stride=1, padding=0,
        #   output_padding=0, groups=1, bias=True, dilation=1
        # )
        self.deconv2 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv1 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x) -> torch.Tensor:
        x1 = F.relu(self.conv1(x))
        # print(x1.size()) # torch.Size([batch, 64, 24, 24])
        x2 = F.relu(self.conv2(x1))
        # print(x2.size()) # torch.Size([batch, 128, 12, 12])
        x3 = F.relu(self.conv3(x2))
        # print(x3.size()) # torch.Size([batch, 128, 6, 6])

        x4 = self.deconv2(self.conv3_(x3)) + self.conv2_(x2)
        # print(x4.size()) # torch.Size([batch, 2, 12, 12])
        return self.deconv1(x4) + self.conv1_(x1) # (batch, 2, 24, 24) of unnormalized logits