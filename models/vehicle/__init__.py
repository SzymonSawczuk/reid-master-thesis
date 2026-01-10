"""
Vehicle Re-Identification Models

Available models:
- ResNet-50 (adapted for vehicles)
- AAVER (Attention-Aware Vehicle Re-ID)
- RPTM (Region-based Part-based Triplet Mining)
- VAT (Vehicle Attribute Transformer)
"""

from .resnet50 import build_resnet50_vehicle, resnet50_vehicle
from .aaver import aaver
from .rptm import rptm
from .vat import vat

__all__ = [
    # ResNet-50
    'build_resnet50_vehicle', 'resnet50_vehicle',
    # AAVER
    'aaver',
    # RPTM
    'rptm',
    # VAT
    'vat',
]
