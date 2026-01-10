"""
PCB (Part-based Convolutional Baseline) model wrapper for torchreid.
PCB is available in torchreid library - this is a convenience wrapper.

Reference:
    Sun Y, Zheng L, Yang Y, Tian Q, Wang S.
    Beyond Part Models: Person Retrieval with Refined Part Pooling (and A Strong Convolutional Baseline).
    In: European Conference on Computer Vision (ECCV), 2018; p. 480-496.
    arXiv: https://arxiv.org/abs/1711.09349
"""

import torchreid


def build_pcb(num_classes, loss='softmax', pretrained=True, num_stripes=6, last_stride=1):
    """
    Build PCB model using torchreid.

    Args:
        num_classes: Number of identities for classification
        loss: Loss type ('softmax', 'triplet', or 'softmax+triplet')
        pretrained: Whether to use pretrained weights
        num_stripes: Number of horizontal stripes (default: 6)
        last_stride: Stride of last convolutional layer (1 or 2)

    Returns:
        PCB model instance
    """
    model = torchreid.models.build_model(
        name='pcb_p6',  # PCB with 6 parts
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained
    )
    return model


def pcb_p6(num_classes, loss='softmax', pretrained=True):
    """PCB with 6 parts."""
    return build_pcb(num_classes, loss=loss, pretrained=pretrained, num_stripes=6)


def pcb_p4(num_classes, loss='softmax', pretrained=True):
    """
    PCB with 4 parts.
    """
    return build_pcb(num_classes, loss=loss, pretrained=pretrained, num_stripes=4)
