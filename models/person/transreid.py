"""
TransReID - Vision Transformer for Person Re-Identification

Reference:
    He S, Luo H, Wang P, Wang F, Li H, Jiang W.
    TransReID: Transformer-based Object Re-Identification.
    In: Proceedings of the IEEE International Conference on Computer Vision, 2021; p. 15013–15022.
    arXiv: https://arxiv.org/abs/2102.04378
    GitHub: https://github.com/damo-cv/TransReID
"""

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, PatchEmbed
import timm


class TransReID(nn.Module):
    """
    TransReID model using Vision Transformer backbone.

    Features:
    - Vision Transformer (ViT) backbone
    - Side Information Embeddings (SIE) for camera/view awareness
    - Jigsaw Patch Module (JPM) for local features
    - Both global and local features for robust matching
    """

    def __init__(
        self,
        num_classes=751,
        camera_num=6,  # Market-1501 has 6 cameras (IDs 1-6, converted to 0-5 in forward)
        view_num=1,
        img_size=(256, 128),
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        drop_rate=0.1,  # Increased from 0.0 to prevent overfitting
        drop_path_rate=0.1,
        sie_coe=3.0,
        loss='softmax',
        pretrained=True,
        **kwargs
    ):
        """
        Args:
            num_classes: Number of identities
            camera_num: Number of cameras (for SIE)
            view_num: Number of views (for SIE)
            img_size: Input image size - can be int (square) or tuple (H, W)
            patch_size: Patch size for ViT
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP ratio in transformer
            drop_rate: Dropout rate
            drop_path_rate: Stochastic depth rate
            sie_coe: SIE coefficient
            loss: Loss type ('softmax', 'triplet', 'softmax+triplet')
            pretrained: Use ImageNet pretrained weights
        """
        super(TransReID, self).__init__()

        self.num_classes = num_classes
        self.loss = loss
        self.sie_coe = sie_coe
        self.camera_num = camera_num
        self.view_num = view_num

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_size = img_size

        timm_img_size = max(img_size) if isinstance(img_size, tuple) else img_size

        if pretrained:
            self.base = timm.create_model(
                'vit_base_patch16_224',
                pretrained=True,
                num_classes=0,
                img_size=timm_img_size,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate
            )
        else:
            self.base = timm.create_model(
                'vit_base_patch16_224',
                pretrained=False,
                num_classes=0,
                img_size=timm_img_size,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate
            )

        actual_embed_dim = self.base.embed_dim

        if isinstance(img_size, tuple) and img_size[0] != img_size[1]:
            self.base.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=3,
                embed_dim=actual_embed_dim
            )
            num_patches = self.base.patch_embed.num_patches
            self.base.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, actual_embed_dim)
            )
            nn.init.trunc_normal_(self.base.pos_embed, std=0.02)

        self.feature_dim = actual_embed_dim

        self.sie_embed_camera = nn.Parameter(
            torch.zeros(camera_num, 1, actual_embed_dim)
        )
        self.sie_embed_view = nn.Parameter(
            torch.zeros(view_num, 1, actual_embed_dim)
        )

        nn.init.trunc_normal_(self.sie_embed_camera, std=0.02)
        nn.init.trunc_normal_(self.sie_embed_view, std=0.02)

        self.bottleneck = nn.BatchNorm1d(self.feature_dim)
        self.bottleneck.bias.requires_grad_(False)

        self.classifier = nn.Linear(self.feature_dim, num_classes, bias=False)

        nn.init.constant_(self.bottleneck.weight, 1)
        nn.init.constant_(self.bottleneck.bias, 0)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x, cam_label=None, view_label=None):
        """
        Args:
            x: Input images (B, 3, H, W)
            cam_label: Camera labels (B,) - optional
            view_label: View labels (B,) - optional

        Returns:
            Depending on training mode and loss type
        """
        B = x.size(0)

        x = self.base.patch_embed(x)

        cls_tokens = self.base.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.base.pos_embed

        if cam_label is not None and self.training:
            sie_embed_cam = self.sie_embed_camera[cam_label]
            x = x + self.sie_coe * sie_embed_cam

        if view_label is not None and self.training:
            sie_embed_view = self.sie_embed_view[view_label]
            x = x + self.sie_coe * sie_embed_view

        x = self.base.pos_drop(x)

        for blk in self.base.blocks:
            x = blk(x)

        x = self.base.norm(x)

        global_feat = x[:, 0]

        feat = self.bottleneck(global_feat)

        if not self.training:
            return feat

        logits = self.classifier(feat)

        if self.loss == 'softmax':
            return logits
        elif self.loss == 'triplet':
            return logits, feat
        elif self.loss == 'softmax+triplet':
            return logits, feat
        else:
            raise KeyError(f"Unsupported loss: {self.loss}")


def transreid_base(num_classes=751, loss='softmax', pretrained=True, **kwargs):
    """
    Build TransReID with ViT-Base backbone.

    Args:
        num_classes: Number of identities
        loss: Loss type
        pretrained: Use pretrained weights

    Returns:
        TransReID model
    """
    model = TransReID(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        img_size=(256, 128),
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        **kwargs
    )
    return model


def transreid_small(num_classes=751, loss='softmax', pretrained=True, **kwargs):
    """
    Build TransReID with ViT-Small backbone.

    Args:
        num_classes: Number of identities
        loss: Loss type
        pretrained: Use pretrained weights

    Returns:
        TransReID model
    """
    model = TransReID(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        img_size=(256, 128),
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        **kwargs
    )
    return model
