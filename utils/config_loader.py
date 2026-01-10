"""
Configuration loader utilities for ReID project.
Loads and manages YAML configuration files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """Load and manage configuration from YAML files."""

    def __init__(self, config_dir: str = "config"):
        """
        Initialize config loader.

        Args:
            config_dir: Directory containing config files
        """
        self.config_dir = Path(config_dir)
        self._datasets_config = None
        self._training_config = None
        self._train_experiments_config = None

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load a YAML file."""
        config_path = self.config_dir / filename
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    @property
    def datasets(self) -> Dict[str, Any]:
        """Load datasets configuration."""
        if self._datasets_config is None:
            self._datasets_config = self._load_yaml('datasets.yaml')
        return self._datasets_config

    @property
    def training(self) -> Dict[str, Any]:
        """Load training configuration."""
        if self._training_config is None:
            self._training_config = self._load_yaml('training.yaml')
        return self._training_config

    @property
    def train_experiments(self) -> Dict[str, Any]:
        """Load train experiments configuration."""
        if self._train_experiments_config is None:
            self._train_experiments_config = self._load_yaml('train_experiments.yaml')
        return self._train_experiments_config

    def get_augmentation_config(self, data_type: str = 'preprocessed') -> Dict[str, bool]:
        """
        Get OpenCV augmentation configuration from training config.

        Args:
            data_type: Data type ('preprocessed', 'augmented', 'original')

        Returns:
            Augmentation configuration dictionary
        """
        # If data_type is 'original', no augmentation
        if data_type == 'original':
            return {
                'rotation': False,
                'flip': False,
                'brightness': False,
                'contrast': False,
                'saturation': False,
                'hue': False,
                'noise': False,
                'blur': False,
                'crop': False,
            }

        # If data_type is 'augmented', use full augmentation
        if data_type == 'augmented':
            try:
                return self.training['training']['augmentation']['opencv']
            except KeyError:
                pass

        # For 'preprocessed' or fallback, use default config
        try:
            return self.training['training']['augmentation']['opencv']
        except KeyError:
            # Return default config if not found
            return {
                'rotation': True,
                'flip': True,
                'brightness': True,
                'contrast': True,
                'saturation': True,
                'hue': True,
                'noise': False,
                'blur': True,
                'crop': True,
            }

    def get_data_loading_config(self) -> Dict[str, Any]:
        """Get data loading configuration."""
        try:
            return self.datasets['data_loading']
        except KeyError:
            # Return defaults
            return {
                'height': 256,
                'width': 128,
                'batch_size_train': 32,
                'batch_size_test': 100,
                'num_workers': 4,
                'use_opencv_augmentation': True
            }

    def get_few_shot_config(self) -> Dict[str, Any]:
        """Get few-shot configuration."""
        try:
            return self.datasets['few_shot']
        except KeyError:
            # Return defaults
            return {
                'k_shots': [1, 2, 4, 8],
                'default_k': 4,
                'seed': 42,
                'num_instances': 4
            }

    def get_batch_size(self, mode: str = 'train') -> int:
        """
        Get batch size for training or testing.

        Args:
            mode: 'train' or 'test'

        Returns:
            Batch size
        """
        if mode == 'train':
            try:
                return self.training['training']['batch_size']
            except KeyError:
                return self.get_data_loading_config()['batch_size_train']
        else:
            return self.get_data_loading_config()['batch_size_test']

    def get_image_size(self, model_name: Optional[str] = None, model_type: Optional[str] = None) -> tuple:
        """
        Get image size (height, width).

        Args:
            model_name: Model name to get specific image size from train_experiments.yaml
            model_type: Model type ('person' or 'vehicle')

        Returns:
            Tuple of (height, width)
        """
        # Try to get from train_experiments config if model specified
        if model_name and model_type:
            try:
                experiments = self.train_experiments
                if model_type == 'person':
                    model_cfg = experiments['person_reid_experiments'].get(model_name)
                else:
                    model_cfg = experiments['vehicle_reid_experiments'].get(model_name)

                if model_cfg:
                    height = model_cfg['training'].get('img_height', 256)
                    width = model_cfg['training'].get('img_width', 128)
                    return (height, width)
            except (KeyError, TypeError):
                pass

        # Fallback to data_loading config
        config = self.get_data_loading_config()
        return (config['height'], config['width'])

    def get_model_config(self, model_name: str, model_type: str = 'person') -> Dict[str, Any]:
        """
        Get model configuration from train_experiments.yaml.

        Args:
            model_name: Name of the model
            model_type: Type of model ('person' or 'vehicle')

        Returns:
            Model configuration dictionary
        """
        try:
            experiments = self.train_experiments
            if model_type == 'person':
                return experiments['person_reid_experiments'][model_name]
            else:
                return experiments['vehicle_reid_experiments'][model_name]
        except KeyError:
            raise ValueError(f"Model {model_name} not found in {model_type}_reid_experiments")



def get_dataloader_params_from_config(
    config_dir: str = "config",
    k_shot: Optional[int] = None,
    model_name: Optional[str] = None,
    model_type: Optional[str] = None,
    data_type: str = 'preprocessed'
) -> Dict[str, Any]:
    """
    Get data loader parameters from config files.

    Args:
        config_dir: Directory containing config files
        k_shot: Override k_shot value (if None, uses default from config)
        model_name: Model name to get specific image size from train_experiments.yaml
        model_type: Model type ('person' or 'vehicle')
        data_type: Data type ('preprocessed', 'augmented', 'original')

    Returns:
        Dictionary with all data loader parameters
    """
    loader = ConfigLoader(config_dir)

    data_config = loader.get_data_loading_config()
    few_shot_config = loader.get_few_shot_config()
    augment_config = loader.get_augmentation_config(data_type)

    # Get image size (model-specific or default)
    height, width = loader.get_image_size(model_name, model_type)

    # Determine if augmentation should be used based on data_type
    use_augmentation = data_type == 'augmented'

    params = {
        'height': height,
        'width': width,
        'batch_size_train': loader.get_batch_size('train'),
        'batch_size_test': loader.get_batch_size('test'),
        'num_workers': data_config['num_workers'],
        'k_shot': k_shot or few_shot_config['default_k'],
        'num_instances': few_shot_config['num_instances'],
        'seed': few_shot_config['seed'],
        'opencv_augment': use_augmentation,
        'augment_config': augment_config
    }

    return params


