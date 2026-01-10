"""
AAVER - Attention-Aware Vehicle Re-identification

Reference:
    Khorramshahi P, Kumar A, Peri N, Rambhatla SS, Chen JC, Chellappa R.
    A dual-path model with adaptive attention for vehicle re-identification.
    In: Proceedings of the IEEE International Conference on Computer Vision, 2019; p. 6132-6141.
    DOI: 10.1109/ICCV.2019.00622
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SpatialAttention(nn.Module):
    """Spatial attention module."""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return x * self.sigmoid(attention)


class ChannelAttention(nn.Module):
    """Channel attention module."""

    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class DualPathBlock(nn.Module):
    """Dual-path block with appearance and attention features."""

    def __init__(self, in_channels):
        super(DualPathBlock, self).__init__()

        # Appearance path
        self.appearance_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Attention path
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        # Appearance features
        appearance = self.appearance_conv(x)

        # Attention features
        attention = self.channel_attention(x)
        attention = self.spatial_attention(attention)

        # Combine
        out = appearance + attention
        return out


class AAVER(nn.Module):
    """
    AAVER - Attention-Aware Vehicle Re-identification model.

    Features:
    - ResNet-50 backbone
    - Dual-path architecture (appearance + attention)
    - Channel and spatial attention
    - Multi-branch outputs for robust matching
    """

    def __init__(
        self,
        num_classes=576,
        loss='softmax',
        pretrained=True,
        **kwargs
    ):
        """
        Args:
            num_classes: Number of vehicle identities
            loss: Loss type ('softmax', 'triplet', 'softmax+triplet')
            pretrained: Use ImageNet pretrained weights
        """
        super(AAVER, self).__init__()

        self.num_classes = num_classes
        self.loss = loss

        resnet50 = models.resnet50(pretrained=pretrained)

        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4

        self.dual_path3 = DualPathBlock(1024)
        self.dual_path4 = DualPathBlock(2048)

        self.feature_dim = 2048

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self.bottleneck = nn.BatchNorm1d(self.feature_dim)
        self.bottleneck.bias.requires_grad_(False)

        self.classifier = nn.Linear(self.feature_dim, num_classes, bias=False)

        self.aux_bottleneck = nn.BatchNorm1d(1024)
        self.aux_bottleneck.bias.requires_grad_(False)
        self.aux_classifier = nn.Linear(1024, num_classes, bias=False)

        self._init_params()

    def _init_params(self):
        """Initialize parameters."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
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

        x = self.layer1(x)
        x = self.layer2(x)

        x3 = self.layer3(x)
        x3 = self.dual_path3(x3)

        aux_feat = self.global_avgpool(x3)
        aux_feat = aux_feat.view(aux_feat.size(0), -1)
        aux_feat_bn = self.aux_bottleneck(aux_feat)

        x4 = self.layer4(x3)
        x4 = self.dual_path4(x4)

        main_feat = self.global_avgpool(x4)
        main_feat = main_feat.view(main_feat.size(0), -1)
        main_feat_bn = self.bottleneck(main_feat)

        if not self.training:
            feat = torch.cat([main_feat_bn, aux_feat_bn], dim=1)
            return feat

        main_logits = self.classifier(main_feat_bn)
        aux_logits = self.aux_classifier(aux_feat_bn)

        if self.loss == 'softmax':
            return main_logits, aux_logits
        elif self.loss == 'triplet':
            return main_feat_bn, aux_feat_bn
        elif self.loss == 'softmax+triplet':
            return (main_logits, aux_logits), (main_feat_bn, aux_feat_bn)
        else:
            raise KeyError(f"Unsupported loss: {self.loss}")


def aaver(num_classes=576, loss='softmax', pretrained=True, **kwargs):
    """
    Build AAVER model.

    Args:
        num_classes: Number of vehicle identities
        loss: Loss type
        pretrained: Use pretrained weights

    Returns:
        AAVER model
    """
    model = AAVER(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        **kwargs
    )
    return model
