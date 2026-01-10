"""
Feature Extraction Module

Handles feature extraction using pre-trained ReID models.
Supports both person and vehicle ReID models.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Union
from pathlib import Path
import cv2
from torchvision import transforms
import sys


project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.person import *
from models.vehicle import *


class ExtractionModule:
    """
    Feature extraction module for person and vehicle ReID.
    """

    def __init__(
        self,
        person_model_path: Optional[str] = None,
        vehicle_model_path: Optional[str] = None,
        person_model_name: str = 'osnet_x1_0',
        vehicle_model_name: str = 'resnet50',
        person_loss: str = 'softmax',
        vehicle_loss: str = 'softmax',
        device: str = 'cuda',
        img_height: int = 256,
        img_width: int = 128
    ):
        """
        Initialize extraction module.

        Args:
            person_model_path: Path to person ReID model (.pth file)
            vehicle_model_path: Path to vehicle ReID model (.pth file)
            person_model_name: Name of person model architecture
            vehicle_model_name: Name of vehicle model architecture
            person_loss: Person model loss function
            vehicle_loss: Vehicle model loss function
            device: Device to run inference on
            img_height: Input image height
            img_width: Input image width
        """
        self.device = device
        self.img_height = img_height
        self.img_width = img_width
        self.person_loss = person_loss
        self.vehicle_loss = vehicle_loss

        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        
        self.person_model = None
        if person_model_path:
            print(f"Loading person ReID model: {person_model_name} (loss: {person_loss})")
            self.person_model = self._load_model(
                person_model_path,
                person_model_name,
                person_loss,
                model_type='person'
            )

        
        self.vehicle_model = None
        if vehicle_model_path:
            print(f"Loading vehicle ReID model: {vehicle_model_name} (loss: {vehicle_loss})")
            self.vehicle_model = self._load_model(
                vehicle_model_path,
                vehicle_model_name,
                vehicle_loss,
                model_type='vehicle'
            )

        print(f"Extraction module initialized (device: {device})")

    def _load_model(
        self,
        model_path: str,
        model_name: str,
        loss: str,
        model_type: str
    ) -> torch.nn.Module:
        """
        Load a ReID model from checkpoint.

        Args:
            model_path: Path to model checkpoint
            model_name: Model architecture name
            loss: Loss function name
            model_type: 'person' or 'vehicle'

        Returns:
            Loaded model
        """
        
        checkpoint = torch.load(model_path, map_location='cpu')

        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        
        num_classes = state_dict['classifier.weight'].shape[0]

        
        if model_type == 'person':
            model = self._build_person_model(model_name, num_classes, loss)
        else:
            model = self._build_vehicle_model(model_name, num_classes, loss)

        
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()

        print(f"  Loaded from: {model_path}")
        print(f"  Num classes: {num_classes}")

        return model

    def _build_person_model(self, model_name: str, num_classes: int, loss: str):
        """Build person ReID model."""
        if model_name == 'osnet_x1_0':
            return osnet_x1_0(num_classes=num_classes, loss=loss, pretrained=False)
        elif model_name == 'pcb_p6':
            return pcb_p6(num_classes=num_classes, loss=loss, pretrained=False)
        elif model_name == 'hacnn':
            return hacnn(num_classes=num_classes, loss=loss, pretrained=False)
        elif model_name == 'transreid_base':
            return transreid_base(num_classes=num_classes, loss=loss, pretrained=False)
        else:
            raise ValueError(f"Unknown person model: {model_name}")

    def _build_vehicle_model(self, model_name: str, num_classes: int, loss: str):
        """Build vehicle ReID model."""
        if model_name == 'resnet50':
            return resnet50_vehicle(num_classes=num_classes, loss=loss, pretrained=False)
        elif model_name == 'aaver':
            return aaver(num_classes=num_classes, loss=loss, pretrained=False)
        elif model_name == 'rptm':
            return rptm(num_classes=num_classes, loss=loss, pretrained=False)
        elif model_name == 'vat':
            return vat(num_classes=num_classes, loss=loss, pretrained=False)
        else:
            raise ValueError(f"Unknown vehicle model: {model_name}")

    def extract_features(
        self,
        images: List[np.ndarray],
        model_type: str = 'person'
    ) -> np.ndarray:
        """
        Extract features from images.

        Args:
            images: List of images (BGR format)
            model_type: 'person' or 'vehicle'

        Returns:
            Feature vectors (shape: [N, feature_dim])
        """
        if len(images) == 0:
            return np.array([])

        
        if model_type == 'person':
            model = self.person_model
            if model is None:
                raise ValueError("Person model not loaded")
        else:
            model = self.vehicle_model
            if model is None:
                raise ValueError("Vehicle model not loaded")

        
        batch = []
        for img in images:
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img_tensor = self.transform(img_rgb)
            batch.append(img_tensor)

        
        batch = torch.stack(batch).to(self.device)

        
        with torch.no_grad():
            outputs = model(batch)

            
            if isinstance(outputs, list):
                features = torch.cat(outputs, dim=1)
            elif isinstance(outputs, tuple):
                if len(outputs) == 2 and isinstance(outputs[1], torch.Tensor):
                    features = outputs[1]
                else:
                    features = outputs[0]
            else:
                features = outputs

            
            features = F.normalize(features, p=2, dim=1)

        return features.cpu().numpy()

    def extract_single(
        self,
        image: np.ndarray,
        model_type: str = 'person'
    ) -> np.ndarray:
        """
        Extract features from a single image.

        Args:
            image: Image (BGR format)
            model_type: 'person' or 'vehicle'

        Returns:
            Feature vector (shape: [feature_dim])
        """
        features = self.extract_features([image], model_type)
        return features[0] if len(features) > 0 else np.array([])
