"""
OSNet model wrapper for torchreid.
OSNet is available in torchreid library - this is a convenience wrapper.

OSNet variants:
- osnet_x1_0: Standard OSNet

Reference:
    Zhou K, Yang Y, Cavallaro A, Xiang T.
    Omni-Scale Feature Learning for Person Re-Identification.
    In: Proceedings of the IEEE International Conference on Computer Vision, 2019; p. 3702-3712.
    arXiv: https://arxiv.org/abs/1905.00953
"""

import torchreid


def build_osnet(num_classes, variant='osnet_x1_0', loss='softmax', pretrained=True):
    """
    Build OSNet model using torchreid.

    Args:
        num_classes: Number of identities for classification
        variant: OSNet variant ('osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'osnet_x0_25', 'osnet_ibn_x1_0')
        loss: Loss type ('softmax', 'triplet', or 'softmax+triplet')
        pretrained: Whether to use pretrained weights

    Returns:
        OSNet model instance
    """
    model = torchreid.models.build_model(
        name=variant,
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained
    )
    return model


def osnet_x1_0(num_classes, loss='softmax', pretrained=True):
    """OSNet x1.0 variant."""
    return build_osnet(num_classes, variant='osnet_x1_0', loss=loss, pretrained=pretrained)

