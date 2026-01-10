"""
HACNN (Harmonious Attention CNN) model wrapper for torchreid.
HACNN is available in torchreid library - this is a convenience wrapper.

Reference:
    Li W, Zhu X, Gong S.
    Harmonious Attention CNN for Person Re-Identification.
    In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018; p. 2285-2294.
    arXiv: https://arxiv.org/abs/1802.08122
"""

import torchreid


def build_hacnn(num_classes, loss='softmax', pretrained=True):
    """
    Build HACNN model using torchreid.

    Args:
        num_classes: Number of identities for classification
        loss: Loss type ('softmax', 'triplet', or 'softmax+triplet')
        pretrained: Whether to use pretrained weights

    Returns:
        HACNN model instance
    """
    model = torchreid.models.build_model(
        name='hacnn',
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained
    )
    return model


def hacnn(num_classes, loss='softmax', pretrained=True):
    """Build HACNN model."""
    return build_hacnn(num_classes, loss=loss, pretrained=pretrained)
