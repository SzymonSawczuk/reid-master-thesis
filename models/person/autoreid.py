"""
Auto-ReID+ - Automated Neural Architecture Search for Person Re-ID

Reference:
    Gu H, Fu G, Li J, Zhu J.
    Auto-reid+: Searching for a multi-branch convnet for person re-identification.
    Neurocomputing. 2021;435:53-66.
    DOI: 10.1016/j.neucom.2021.01.015
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class AutoReIDBlock(nn.Module):
    """Auto-ReID building block with SE and depthwise separable convolutions."""

    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super(AutoReIDBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = DepthwiseSeparableConv(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.se = SEBlock(out_channels) if use_se else None

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AutoReIDPlus(nn.Module):
    """
    Auto-ReID+ model with part-aware features.

    Features:
    - Efficient building blocks from NAS
    - Part-based feature extraction
    - SE attention modules
    - Depthwise separable convolutions
    """

    def __init__(
        self,
        num_classes=751,
        loss='softmax',
        num_parts=6,
        feature_dim=2048,
        **kwargs
    ):
        """
        Args:
            num_classes: Number of identities
            loss: Loss type ('softmax', 'triplet', 'softmax+triplet')
            num_parts: Number of horizontal parts for part-based features
            feature_dim: Feature dimension
        """
        super(AutoReIDPlus, self).__init__()

        self.num_classes = num_classes
        self.loss = loss
        self.num_parts = num_parts

        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stages
        self.stage1 = self._make_stage(64, 128, num_blocks=2)
        self.stage2 = self._make_stage(128, 256, num_blocks=3, stride=2)
        self.stage3 = self._make_stage(256, 512, num_blocks=4, stride=2)
        self.stage4 = self._make_stage(512, feature_dim, num_blocks=3, stride=1)

        # Global pooling
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        # Part pooling
        self.part_avgpool = nn.AdaptiveAvgPool2d((num_parts, 1))

        # Bottleneck
        self.bottleneck = nn.BatchNorm1d(feature_dim)
        self.bottleneck.bias.requires_grad_(False)

        # Part bottlenecks
        self.part_bottlenecks = nn.ModuleList([
            nn.BatchNorm1d(feature_dim) for _ in range(num_parts)
        ])

        # Global classifier
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)

        # Part classifiers
        self.part_classifiers = nn.ModuleList([
            nn.Linear(feature_dim, num_classes, bias=False) for _ in range(num_parts)
        ])

        self._init_params()

    def _make_stage(self, in_channels, out_channels, num_blocks, stride=1):
        """Build a stage with multiple blocks."""
        layers = []
        layers.append(AutoReIDBlock(in_channels, out_channels, stride=stride, use_se=True))

        for _ in range(1, num_blocks):
            layers.append(AutoReIDBlock(out_channels, out_channels, stride=1, use_se=True))

        return nn.Sequential(*layers)

    def _init_params(self):
        """Initialize parameters."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: Input images (B, 3, H, W)

        Returns:
            Depending on training mode and loss type
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        global_feat = self.global_avgpool(x)
        global_feat = global_feat.view(global_feat.size(0), -1)
        global_feat = self.bottleneck(global_feat)

        part_feat = self.part_avgpool(x)
        part_feats = []
        for i in range(self.num_parts):
            part = part_feat[:, :, i, :]
            part = part.view(part.size(0), -1)
            part = self.part_bottlenecks[i](part)
            part_feats.append(part)

        if not self.training:
            feat = torch.cat([global_feat] + part_feats, dim=1)
            return feat

        global_logits = self.classifier(global_feat)
        part_logits = [self.part_classifiers[i](part_feats[i]) for i in range(self.num_parts)]

        if self.loss == 'softmax':
            return [global_logits] + part_logits
        elif self.loss == 'triplet':
            return [global_feat] + part_feats
        elif self.loss == 'softmax+triplet':
            return [global_logits] + part_logits, [global_feat] + part_feats
        else:
            raise KeyError(f"Unsupported loss: {self.loss}")


def autoreid_plus(num_classes=751, loss='softmax', num_parts=6, **kwargs):
    """
    Build Auto-ReID+ model.

    Args:
        num_classes: Number of identities
        loss: Loss type
        num_parts: Number of parts

    Returns:
        Auto-ReID+ model
    """
    model = AutoReIDPlus(
        num_classes=num_classes,
        loss=loss,
        num_parts=num_parts,
        **kwargs
    )
    return model
