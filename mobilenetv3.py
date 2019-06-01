'''
MobileNetV3 in PyTorch.

Searching for MobileNetV3
https://arxiv.org/abs/1905.02244
'''

import torch.nn as nn
import torch.nn.functional as F


class HSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3, inplace=self.inplace) / 6


class HSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3, inplace=self.inplace) / 6


class SEModule(nn.Module):
    def __init__(self, inp, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(inp, inp // reduction, bias=False), nn.ReLU(inplace=True),
            nn.Linear(inp // reduction, inp, bias=False), HSigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Block(nn.Module):
    def __init__(self, kernel_size, inp, exp, oup, semodule, nolinear, stride):
        super(Block, self).__init__()
        self.use_res_connection = stride == 1 and inp == oup

        self.semodule = semodule(exp) if semodule != None else None
        self.nolinear = nolinear

        self.conv1 = nn.Conv2d(
            inp, exp, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(exp)

        self.conv2 = nn.Conv2d(
            exp,
            exp,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=exp,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(exp)

        self.conv3 = nn.Conv2d(
            exp, oup, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn3 = nn.BatchNorm2d(oup)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.nolinear(out)

        out = self.bn2(self.conv2(out))
        if self.semodule != None:
            out = self.semodule(out)
        out = self.nolinear(out)

        out = self.bn3(self.conv3(out))

        if self.use_res_connection:
            return x + out
        else:
            return out


class MobileNetV3(nn.Module):
    def __init__(self, n_class=1000, mode='small'):
        super(MobileNetV3, self).__init__()

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = HSwish()

        if mode == 'small':
            self.bnecks = nn.Sequential(
                Block(3, 16, 16, 16, SEModule, nn.ReLU(inplace=True), 2),
                Block(3, 16, 72, 24, None, nn.ReLU(inplace=True), 2),
                Block(3, 24, 88, 24, None, nn.ReLU(inplace=True), 1),
                Block(5, 24, 96, 40, SEModule, HSwish(inplace=True), 2),
                Block(5, 40, 240, 40, SEModule, HSwish(inplace=True), 1),
                Block(5, 40, 240, 40, SEModule, HSwish(inplace=True), 1),
                Block(5, 40, 120, 48, SEModule, HSwish(inplace=True), 1),
                Block(5, 48, 144, 48, SEModule, HSwish(inplace=True), 1),
                Block(5, 48, 288, 96, SEModule, HSwish(inplace=True), 2),
                Block(5, 96, 576, 96, SEModule, HSwish(inplace=True), 1),
                Block(5, 96, 576, 96, SEModule, HSwish(inplace=True), 1),
            )
            self.last = nn.Sequential(
                nn.Conv2d(96, 576, kernel_size=1, stride=1),
                nn.BatchNorm2d(576),
                SEModule(576),
                HSwish(inplace=True),
                nn.AvgPool2d(7),
                nn.Conv2d(576, 1280, kernel_size=1, stride=1),
                nn.BatchNorm2d(1280),
                HSwish(inplace=True),
                nn.Conv2d(1280, n_class, kernel_size=1, stride=1),
                nn.BatchNorm2d(n_class),
                HSwish(inplace=True),
            )
        elif mode == 'large':
            self.bnecks = nn.Sequential(
                Block(3, 16, 16, 16, None, nn.ReLU(inplace=True), 1),
                Block(3, 16, 64, 24, None, nn.ReLU(inplace=True), 2),
                Block(3, 24, 72, 24, None, nn.ReLU(inplace=True), 1),
                Block(5, 24, 72, 40, SEModule, nn.ReLU(inplace=True), 2),
                Block(5, 40, 120, 40, SEModule, nn.ReLU(inplace=True), 1),
                Block(5, 40, 240, 40, SEModule, nn.ReLU(inplace=True), 1),
                Block(3, 40, 240, 80, None, HSwish(inplace=True), 2),
                Block(3, 80, 200, 80, None, HSwish(inplace=True), 1),
                Block(3, 80, 184, 80, None, HSwish(inplace=True), 1),
                Block(3, 80, 184, 80, None, HSwish(inplace=True), 1),
                Block(3, 80, 480, 112, SEModule, HSwish(inplace=True), 1),
                Block(3, 112, 672, 112, SEModule, HSwish(inplace=True), 1),
                Block(5, 112, 672, 112, SEModule, HSwish(inplace=True), 1),
                Block(5, 112, 672, 160, SEModule, HSwish(inplace=True), 2),
                Block(5, 160, 960, 160, SEModule, HSwish(inplace=True), 1),
            )
            self.last = nn.Sequential(
                nn.Conv2d(160, 960, kernel_size=1, stride=1),
                nn.BatchNorm2d(960),
                HSwish(inplace=True),
                nn.AvgPool2d(7),
                nn.Conv2d(960, 1280, kernel_size=1, stride=1),
                HSwish(inplace=True),
                nn.Conv2d(1280, n_class, kernel_size=1, stride=1),
            )
          
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
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
        x = self.hs1(self.bn1(self.conv1(x)))
        x = self.bnecks(x)
        x = self.last(x)

        return x.view(x.size(0), -1)
