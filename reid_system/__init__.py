"""
Re-Identification System

A modular person and vehicle re-identification system with:
- Detection module (YOLO)
- Feature extraction module (multiple ReID models)
- Matching and ranking module (with reranking)
- Database management for identity storage
- Visualization module
"""

__version__ = "1.0.0"
__author__ = "Master Thesis Project"

from .detection.detector import DetectionModule
from .extraction.extractor import ExtractionModule
from .matching.matcher import MatchingModule
from .database.db_manager import DatabaseManager
from .visualization.visualizer import Visualizer

__all__ = [
    'DetectionModule',
    'ExtractionModule',
    'MatchingModule',
    'DatabaseManager',
    'Visualizer'
]
