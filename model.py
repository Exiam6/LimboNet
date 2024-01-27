import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1).long(), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class CobbBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, cardinality, base_width, widen_factor):
        super(CobbBottleneck, self).__init__()
        D = out_channels // 4

        # 1x1 followed by 7x7 convolution branch
        self.conv1x1_7x7 = nn.Sequential(
            nn.Conv2d(in_channels, D, kernel_size=1, bias=False),
            nn.BatchNorm2d(D),
            nn.ReLU(inplace=True),
            nn.Conv2d(D, D, kernel_size=7, stride=stride, padding=3, bias=False),
            nn.BatchNorm2d(D),
            nn.ReLU(inplace=True)
        )
        # 1x1 followed by 3x3 convolution branch
        self.conv1x1_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, D, kernel_size=1, bias=False),
            nn.BatchNorm2d(D),
            nn.ReLU(inplace=True),
            nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(D),
            nn.ReLU(inplace=True)
        )

        # 1x1 followed by 5x5 convolution branch
        self.conv1x1_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, D, kernel_size=1, bias=False),
            nn.BatchNorm2d(D),
            nn.ReLU(inplace=True),
            nn.Conv2d(D, D, kernel_size=5, stride=stride, padding=2, bias=False),
            nn.BatchNorm2d(D),
            nn.ReLU(inplace=True)
        )

        # 3x3 maxpool followed by 1x1 convolution branch
        self.maxpool_conv1x1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
            nn.Conv2d(in_channels, D, kernel_size=1, bias=False),
            nn.BatchNorm2d(D),
            nn.ReLU(inplace=True)
        )

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out1x1_7x7 = self.conv1x1_7x7(x) 
        out1x1_3x3 = self.conv1x1_3x3(x)
        out1x1_5x5 = self.conv1x1_5x5(x)
        out_maxpool_conv1x1 = self.maxpool_conv1x1(x)

        # Concatenate along the channel dimension
        out = torch.cat([out1x1_7x7, out1x1_3x3, out1x1_5x5, out_maxpool_conv1x1], dim=1)

        residual = self.shortcut(x)
        out += residual
        out = F.relu(out)

        return out

class LimboNet(nn.Module):
    def __init__(self, num_classes=20, cardinality=4, base_width=4, widen_factor=4):
        super(LimboNet, self).__init__()
        self.cardinality = cardinality
        self.base_width = base_width
        self.widen_factor = widen_factor

        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
    
        self.stage_1 = self._make_stage(64, 128, 1)
        self.stage_2 = self._make_stage(128, 256, 2)  
        self.stage_3 = self._make_stage(256, 512, 2)  
        self.stage_4 = self._make_stage(512, 1024, 2)
        self.dropout = nn.Dropout(p=0.2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1024, num_classes)  
        init.kaiming_normal_(self.classifier.weight)

    def _make_stage(self, in_channels, out_channels, stride):
        return CobbBottleneck(in_channels, out_channels, stride, self.cardinality, self.base_width, self.widen_factor)

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x