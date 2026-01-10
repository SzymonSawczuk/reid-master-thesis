"""
VAT - Vehicle Attribute Transformer for Vehicle Re-identification

Reference:
    Yu Z, Pei J, Zhu M, Zhang J, Li J.
    Multi-attribute adaptive aggregation transformer for vehicle re-identification.
    Inform Process Manag. 2022;59(2): 102868.
    DOI: 10.1016/j.ipm.2021.102868
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from timm.models.vision_transformer import Block


class VehicleAttributeHead(nn.Module):
    """
    Vehicle attribute prediction head.
    Predicts vehicle attributes like color, type, make, etc.
    """

    def __init__(self, in_dim, num_colors=10, num_types=10):
        super(VehicleAttributeHead, self).__init__()

        self.color_head = nn.Linear(in_dim, num_colors)
        self.type_head = nn.Linear(in_dim, num_types)

    def forward(self, x):
        color_logits = self.color_head(x)
        type_logits = self.type_head(x)
        return color_logits, type_logits


class TransformerEncoder(nn.Module):
    """Transformer encoder for feature refinement."""

    def __init__(self, dim=2048, depth=3, num_heads=8, mlp_ratio=4.0, drop_rate=0.1):
        super(TransformerEncoder, self).__init__()

        self.blocks = nn.ModuleList([
            Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=True, proj_drop=drop_rate, attn_drop=drop_rate)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: (B, N, D) where N is number of tokens
        """
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x


class VAT(nn.Module):
    """
    VAT - Vehicle Attribute Transformer.

    Features:
    - ResNet-50 backbone for initial feature extraction
    - Transformer encoder for global context modeling
    - Vehicle attribute prediction
    - Multi-scale feature fusion
    """

    def __init__(
        self,
        num_classes=576,
        loss='softmax',
        pretrained=True,
        num_colors=10,
        num_types=10,
        transformer_depth=3,
        num_heads=8,
        **kwargs
    ):
        """
        Args:
            num_classes: Number of vehicle identities
            loss: Loss type ('softmax', 'triplet', 'softmax+triplet')
            pretrained: Use ImageNet pretrained weights
            num_colors: Number of color classes for attribute prediction
            num_types: Number of vehicle type classes
            transformer_depth: Depth of transformer encoder
            num_heads: Number of attention heads
        """
        super(VAT, self).__init__()

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

        self.feature_dim = 2048

        self.spatial_to_seq = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 4)), 
        )
        self.num_patches = 8 * 4  

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.feature_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.feature_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.transformer = TransformerEncoder(
            dim=self.feature_dim,
            depth=transformer_depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            drop_rate=0.1
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.bottleneck = nn.BatchNorm1d(self.feature_dim)
        self.bottleneck.bias.requires_grad_(False)

        self.classifier = nn.Linear(self.feature_dim, num_classes, bias=False)

        self.attribute_head = VehicleAttributeHead(
            self.feature_dim,
            num_colors=num_colors,
            num_types=num_types
        )

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

    def forward(self, x, get_attributes=False):
        """
        Args:
            x: Input images (B, 3, H, W)
            get_attributes: Whether to return attribute predictions

        Returns:
            Depending on training mode and loss type
        """
        B = x.size(0)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) 

        x_seq = self.spatial_to_seq(x) 
        B, C, H, W = x_seq.shape
        x_seq = x_seq.flatten(2).transpose(1, 2) 

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_seq = torch.cat([cls_tokens, x_seq], dim=1)  

        x_seq = x_seq + self.pos_embed

        x_trans = self.transformer(x_seq)  

        global_feat = x_trans[:, 0] 

        feat_bn = self.bottleneck(global_feat)

        if get_attributes or self.training:
            color_logits, type_logits = self.attribute_head(global_feat)

        if not self.training:
            if get_attributes:
                return feat_bn, color_logits, type_logits
            return feat_bn

        id_logits = self.classifier(feat_bn)

        if self.loss == 'softmax':
            return id_logits, color_logits, type_logits
        elif self.loss == 'triplet':
            return feat_bn
        elif self.loss == 'softmax+triplet':
            return (id_logits, color_logits, type_logits), feat_bn
        else:
            raise KeyError(f"Unsupported loss: {self.loss}")


def vat(num_classes=576, loss='softmax', pretrained=True, **kwargs):
    """
    Build VAT model.

    Args:
        num_classes: Number of vehicle identities
        loss: Loss type
        pretrained: Use pretrained weights

    Returns:
        VAT model
    """
    model = VAT(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        **kwargs
    )
    return model
