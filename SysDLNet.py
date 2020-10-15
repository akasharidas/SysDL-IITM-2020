import torch
import torch.nn as nn
from modules import Fuse, Hswish, Hsigmoid


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            Hswish(),
        )

        self.fuse1 = Fuse(16, 3, 16, 16, True, "relu", 2)
        self.fuse2 = Fuse(16, 3, 72, 24, False, "relu", 2)
        self.fuse3 = Fuse(24, 3, 88, 24, False, "relu", 1)
        self.fuse4 = Fuse(24, 5, 96, 40, True, "hswish", 2)
        self.fuse5 = Fuse(40, 5, 240, 40, True, "hswish", 1)
        self.fuse6 = Fuse(40, 5, 240, 40, True, "hswish", 1)
        self.fuse7 = Fuse(40, 5, 120, 48, True, "hswish", 1)
        self.fuse8 = Fuse(48, 5, 144, 48, True, "hswish", 1)
        self.fuse9 = Fuse(48, 5, 288, 96, True, "hswish", 2)
        self.fuse10 = Fuse(96, 5, 576, 96, True, "hswish", 1)
        self.fuse11 = Fuse(96, 5, 576, 96, True, "hswish", 1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 576, 1, stride=1, bias=False), nn.BatchNorm2d(576), Hswish()
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(576, 1024, 1, bias=False),
            Hswish(),
            nn.Conv2d(1024, 100, 1, bias=False),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)

        x = self.fuse1(x)
        x = self.fuse2(x)
        x = self.fuse3(x)
        x = self.fuse4(x)
        x = self.fuse5(x)
        x = self.fuse6(x)
        x = self.fuse7(x)
        x = self.fuse8(x)
        x = self.fuse9(x)
        x = self.fuse10(x)
        x = self.fuse11(x)

        x = self.avg_pool(self.conv2(x))
        x = self.classifier(x)

        return x.view(-1, 1)
