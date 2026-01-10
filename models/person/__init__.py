"""
Person Re-Identification Models

Available models:
- ResNet-50 (via torchreid)
- PCB (Part-based Convolutional Baseline via torchreid)
- OSNet (Omni-Scale Network via torchreid)
- HACNN (Harmonious Attention CNN via torchreid)
- TransReID (Transformer-based ReID)
- Auto-ReID+ (Neural Architecture Search)
"""

from .osnet import build_osnet, osnet_x1_0
from .pcb import build_pcb, pcb_p6, pcb_p4
from .ha_cnn import build_hacnn, hacnn
from .transreid import transreid_base, transreid_small
from .autoreid import autoreid_plus

__all__ = [
    # OSNet
    'build_osnet', 'osnet_x1_0',
    # PCB
    'build_pcb', 'pcb_p6', 'pcb_p4',
    # HACNN
    'build_hacnn', 'hacnn',
    # TransReID
    'transreid_base', 'transreid_small',
    # Auto-ReID+
    'autoreid_plus',
]
