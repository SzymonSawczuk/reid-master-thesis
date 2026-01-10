"""
RPTM - Relation Preserving Triplet Mining for Vehicle Re-identification

Reference:
    Ghosh A, Shanmugalingam K, Lin WY.
    Relation preserving triplet mining for stabilising the triplet loss in re-identification systems.
    In: Proceedings of the IEEE/CVF Winter Conference on applications of computer vision, 2023; p. 4840-4849.
    DOI: 10.1109/WACV56688.2023.00483

Note:
    The "relation preserving triplet mining" is a training strategy (hard negative selection),
    not part of the model architecture. Use hard mining triplet loss during training:

        criterion_triplet = torchreid.losses.TripletLoss(
            margin=0.3,
            distance_metric='euclidean'
        )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class KeypointAttention(nn.Module):
    """Keypoint-aware attention module for vehicle parts."""

    def __init__(self, in_channels, num_keypoints=20):
        super(KeypointAttention, self).__init__()

        self.num_keypoints = num_keypoints

        self.keypoint_conv = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_keypoints, 1)
        )

        self.refine_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        heatmaps = self.keypoint_conv(x)
        heatmaps = torch.sigmoid(heatmaps)

        refined = self.refine_conv(x)

        keypoint_feats = []
        for i in range(self.num_keypoints):
            heatmap = heatmaps[:, i:i+1, :, :]
            feat = refined * heatmap
            keypoint_feats.append(feat)

        attended = torch.stack(keypoint_feats, dim=1)
        attended = torch.sum(attended, dim=1)

        return attended, heatmaps


class RegionBranch(nn.Module):
    """Region-specific branch for extracting part features."""

    def __init__(self, in_channels, feature_dim=512):
        super(RegionBranch, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x


class RPTM(nn.Module):
    """
    RPTM - Region-based Part-based Triplet Mining for Vehicle Re-ID.

    Features:
    - ResNet-50 backbone
    - Keypoint-aware attention
    - Multi-region feature extraction
    - Part-based triplet mining
    """

    def __init__(
        self,
        num_classes=576,
        loss='softmax',
        pretrained=True,
        num_keypoints=20,
        num_regions=4,
        **kwargs
    ):
        """
        Args:
            num_classes: Number of vehicle identities
            loss: Loss type ('softmax', 'triplet', 'softmax+triplet')
            pretrained: Use ImageNet pretrained weights
            num_keypoints: Number of vehicle keypoints
            num_regions: Number of regions for part-based features
        """
        super(RPTM, self).__init__()

        self.num_classes = num_classes
        self.loss = loss
        self.num_regions = num_regions

        resnet50 = models.resnet50(pretrained=pretrained)

        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4

        self.keypoint_attention = KeypointAttention(2048, num_keypoints)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = 2048

        self.region_pool = nn.AdaptiveAvgPool2d((num_regions, 1))
        self.region_branches = nn.ModuleList([
            RegionBranch(2048, 512) for _ in range(num_regions)
        ])

        self.global_bottleneck = nn.BatchNorm1d(self.feature_dim)
        self.global_bottleneck.bias.requires_grad_(False)
        self.global_classifier = nn.Linear(self.feature_dim, num_classes, bias=False)

        self.region_bottlenecks = nn.ModuleList([
            nn.BatchNorm1d(512) for _ in range(num_regions)
        ])
        self.region_classifiers = nn.ModuleList([
            nn.Linear(512, num_classes, bias=False) for _ in range(num_regions)
        ])

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
        x = self.layer3(x)
        x = self.layer4(x)

        x_att, heatmaps = self.keypoint_attention(x)

        global_feat = self.global_pool(x_att)
        global_feat = global_feat.view(global_feat.size(0), -1)
        global_feat_bn = self.global_bottleneck(global_feat)

        region_feats = []
        region_feats_bn = []

        region_maps = self.region_pool(x_att)
        for i in range(self.num_regions):
            region = region_maps[:, :, i:i+1, :]
            region_feat = self.region_branches[i](region)
            region_feat_bn = self.region_bottlenecks[i](region_feat)
            region_feats.append(region_feat)
            region_feats_bn.append(region_feat_bn)

        if not self.training:
            feat = torch.cat([global_feat_bn] + region_feats_bn, dim=1)
            return feat

        global_logits = self.global_classifier(global_feat_bn)
        region_logits = [self.region_classifiers[i](region_feats_bn[i])
                        for i in range(self.num_regions)]

        if self.loss == 'softmax':
            return [global_logits] + region_logits
        elif self.loss == 'triplet':
            return [global_feat_bn] + region_feats_bn
        elif self.loss == 'softmax+triplet':
            return [global_logits] + region_logits, [global_feat_bn] + region_feats_bn
        else:
            raise KeyError(f"Unsupported loss: {self.loss}")


def rptm(num_classes=576, loss='softmax', pretrained=True, num_regions=4, **kwargs):
    """
    Build RPTM model.

    Args:
        num_classes: Number of vehicle identities
        loss: Loss type
        pretrained: Use pretrained weights
        num_regions: Number of regions

    Returns:
        RPTM model
    """
    model = RPTM(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        num_regions=num_regions,
        **kwargs
    )
    return model
