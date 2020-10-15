import torch
import torch.nn.functional as F
import torch.nn as nn


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6


class Fuse(nn.Module):
    def __init__(self, inC, K, exp, oup, is_SE, NL, stride):
        super().__init__()
        self.is_SE = is_SE
        self.NL = nn.ReLU() if NL == "relu" else Hswish()

        self.conv1 = nn.Sequential(
            nn.Conv2d(inC, exp, kernel_size=1, bias=False), nn.BatchNorm2d(exp)
        )

        self.dw_conv1 = nn.Sequential(
            nn.Conv2d(
                exp,
                exp,
                kernel_size=(1, K),
                stride=stride,
                padding=(0, (K - 1) // 2),
                groups=exp,
                bias=False,
            ),
            nn.BatchNorm2d(exp),
        )

        self.dw_conv2 = nn.Sequential(
            nn.Conv2d(
                exp,
                exp,
                kernel_size=(K, 1),
                stride=stride,
                padding=((K - 1) // 2, 0),
                groups=exp,
                bias=False,
            ),
            nn.BatchNorm2d(exp),
        )

        self.SE = None
        if is_SE:
            self.SE = nn.Sequential(SEModule(2 * exp), Hsigmoid())

        self.conv2 = nn.Sequential(
            nn.Conv2d(2 * exp, oup, kernel_size=1, bias=False), nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        x = self.conv1(x)

        x = self.NL(x)

        out1 = self.dw_conv1(x)
        out2 = self.dw_conv2(x)

        x = torch.cat([out1, out2], 1)

        if self.is_SE:
            x = self.SE(x)

        x = self.NL(x)

        out = self.conv2(x)

        return out
