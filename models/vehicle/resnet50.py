"""
ResNet-50 for Vehicle Re-Identification

This is a wrapper/adapter that uses torchreid's ResNet-50
"""

import torchreid


def build_resnet50_vehicle(num_classes, loss='softmax', pretrained=True):
    """
    Build ResNet-50 for vehicle re-identification using torchreid.

    Args:
        num_classes: Number of vehicle identities
        loss: Loss type ('softmax', 'triplet', or 'softmax+triplet')
        pretrained: Whether to use ImageNet pretrained weights

    Returns:
        ResNet-50 model instance configured for vehicle ReID
    """
    model = torchreid.models.build_model(
        name='resnet50',
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained
    )
    return model


def resnet50_vehicle(num_classes=576, loss='softmax', pretrained=True):
    """
    ResNet-50 for vehicle re-identification.

    Args:
        num_classes: Number of vehicle identities (default: 576 for VeRi-776)
        loss: Loss type
        pretrained: Use ImageNet pretrained weights

    Returns:
        ResNet-50 model
    """
    return build_resnet50_vehicle(num_classes, loss=loss, pretrained=pretrained)
