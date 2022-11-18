from typing import List, Tuple, Union, Dict, Any, Optional
from .task_nn import TaskNN
from .score_nn import ScoreNN
import torch
from torch import nn
from torch.nn import functional as F


@ScoreNN.register("weizmann-horse-seg")
class WeizmannHorseSegScoreNN(ScoreNN):
    def __init__(self, task_nn: TaskNN, dropout: float = 0.25):
        super().__init__(task_nn)  # type:ignore

        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv2d(4, 64, 5, 1, padding=2)
        self.conv2 = nn.Conv2d(64, 128, 5, 2, padding=2)
        self.conv3 = nn.Conv2d(128, 128, 5, 2, padding=2)

        # Linear(in_features, out_features, bias=True)
        self.fc1 = nn.Linear(128 * 6 * 6, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 1)

        # apply dropout on the first FC layer as paper mentioned
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x: torch.Tensor, y: torch.Tensor, buffer: Any = None) -> torch.Tensor:
        """
        :param x: (b, 3, 24, 24)
        :param y: (b, n or 1+n, 1, 24, 24) for scorenn training; (b, 1, 24, 24) during tasknn training or DVN inference
        :return: (b, n or 1+n) for scorenn training and (b,) during tasknn training
        """
        size_prefix = y.size()[:-3]
        if len(y.size()) > 4: # scorenn training
            x = x.unsqueeze(-4) # (b, 1, 3, 24, 24)
            x = x.repeat(*((1,)*len(x.size()[:-4])), y.size()[-4], *((1,)*len(x.size()[-3:]))) # (b, n or 1+n, 3, 24, 24)
            x = x.view(-1, *x.size()[-3:])
            y = y.view(-1, *y.size()[-3:])

        z = torch.cat((x, y), -3) # concatenate image and mask along the channel dimension
        z = F.relu(self.conv1(z))
        z = F.relu(self.conv2(z))
        z = F.relu(self.conv3(z))

        # flatten before FC layers
        z = z.view((*z.size()[:-3], 128 * 6 * 6))
        z = F.relu(self.fc1(z))
        z = self.dropout(z)
        z = F.relu(self.fc2(z))
        z = self.fc3(z)
        return z.view(*size_prefix) # (b, n or 1+n) for scorenn training and (b,) during tasknn training