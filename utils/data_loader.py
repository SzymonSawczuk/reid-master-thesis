"""
Custom data loader for object ReID with few-shot support and OpenCV augmentations.
Compatible with TorchReID.
"""

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from pathlib import Path
import random
from collections import defaultdict
from typing import Optional, Tuple, Dict, Callable
import torchvision.transforms as T

try:
    from utils.config_loader import get_dataloader_params_from_config
except ImportError:
    get_dataloader_params_from_config = None


class OpenCVAugmentation:
    """OpenCV-based augmentation for object ReID."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
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

    def random_rotation(self, image: np.ndarray, angle_range: Tuple[int, int] = (-10, 10)) -> np.ndarray:
        """Rotate image by random angle within range."""
        angle = random.uniform(angle_range[0], angle_range[1])
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return rotated

    def random_horizontal_flip(self, image: np.ndarray, p: float = 0.5) -> np.ndarray:
        """Randomly flip image horizontally."""
        if random.random() < p:
            return cv2.flip(image, 1)
        return image

    def random_brightness(self, image: np.ndarray, delta_range: Tuple[int, int] = (-30, 30)) -> np.ndarray:
        """Adjust brightness randomly."""
        delta = random.uniform(delta_range[0], delta_range[1])
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + delta, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def random_contrast(self, image: np.ndarray, alpha_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """Adjust contrast randomly."""
        alpha = random.uniform(alpha_range[0], alpha_range[1])
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        return adjusted

    def random_saturation(self, image: np.ndarray, saturation_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """Adjust saturation randomly."""
        factor = random.uniform(saturation_range[0], saturation_range[1])
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def random_hue(self, image: np.ndarray, delta_range: Tuple[int, int] = (-10, 10)) -> np.ndarray:
        """Adjust hue randomly."""
        delta = random.uniform(delta_range[0], delta_range[1])
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + delta) % 180
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def random_gaussian_noise(self, image: np.ndarray, sigma_range: Tuple[int, int] = (0, 5)) -> np.ndarray:
        """Add random Gaussian noise."""
        sigma = random.uniform(sigma_range[0], sigma_range[1])
        noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy

    def random_gaussian_blur(self, image: np.ndarray, kernel_size_range: Tuple[int, int] = (3, 7), p: float = 0.3) -> np.ndarray:
        """Apply random Gaussian blur."""
        if random.random() < p:
            kernel_size = random.choice(range(kernel_size_range[0], kernel_size_range[1] + 1, 2))
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return image

    def random_crop_and_resize(self, image: np.ndarray, crop_ratio_range: Tuple[float, float] = (0.8, 1.0)) -> np.ndarray:
        """Random crop and resize to original size."""
        h, w = image.shape[:2]
        ratio = random.uniform(crop_ratio_range[0], crop_ratio_range[1])
        crop_h, crop_w = int(h * ratio), int(w * ratio)

        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        cropped = image[top:top+crop_h, left:left+crop_w]
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        return resized

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply a series of augmentations based on config."""
        aug_image = image.copy()

        if self.config.get('flip', False):
            aug_image = self.random_horizontal_flip(aug_image)

        if self.config.get('rotation', False):
            aug_image = self.random_rotation(aug_image)

        if self.config.get('crop', False):
            aug_image = self.random_crop_and_resize(aug_image)

        if self.config.get('brightness', False):
            aug_image = self.random_brightness(aug_image)

        if self.config.get('contrast', False):
            aug_image = self.random_contrast(aug_image)

        if self.config.get('saturation', False):
            aug_image = self.random_saturation(aug_image)

        if self.config.get('hue', False):
            aug_image = self.random_hue(aug_image)

        if self.config.get('noise', False):
            aug_image = self.random_gaussian_noise(aug_image)

        if self.config.get('blur', False):
            aug_image = self.random_gaussian_blur(aug_image)

        return aug_image


class ReIDDataset(Dataset):
    """
    Universal Dataset class for ReID datasets with OpenCV augmentations.
    Supports Market-1501, DukeMTMC-reID, VeRi-776, CityFlow formats.
    Compatible with TorchReID.

    All datasets output the same structure: (img, label, camid, img_path)
    """

    def __init__(
        self,
        data_dir: str,
        dataset_type: str = 'market1501',
        height: int = 256,
        width: int = 128,
        transform: Optional[Callable] = None,
        opencv_augment: bool = True,
        augment_config: Optional[Dict] = None
    ):
        """
        Args:
            data_dir: Path to dataset directory containing images
            dataset_type: Type of dataset ('market1501', 'dukemtmc', 'veri776', 'cityflow')
            height: Target image height
            width: Target image width
            transform: PyTorch transforms (applied after OpenCV augmentations)
            opencv_augment: Whether to apply OpenCV augmentations
            augment_config: Configuration dict for augmentations
        """
        self.data_dir = Path(data_dir)
        self.dataset_type = dataset_type.lower()
        self.height = height
        self.width = width
        self.transform = transform
        self.opencv_augment = opencv_augment

        if opencv_augment:
            self.opencv_aug = OpenCVAugmentation(augment_config)

        # For CityFlow: load camera ID and vehicle ID mappings
        self._cityflow_camid_map = {}
        self._cityflow_pid_map = {}  # Maps filename -> vehicleID
        if self.dataset_type == 'cityflow':
            self._load_cityflow_xml()
            self._load_cityflow_tracks()

        self.img_paths = []
        self.pids = []
        self.camids = []

        self._parse_data()

        self.pid_set = sorted(list(set(self.pids)))
        self.pid2label = {pid: label for label, pid in enumerate(self.pid_set)}

        self.num_pids = len(self.pid_set)
        self.num_imgs = len(self.img_paths)

    def _load_cityflow_xml(self):
        """Load CityFlow camera IDs from XML files."""
        import xml.etree.ElementTree as ET

        parent_dir = self.data_dir.parent
        xml_files = list(parent_dir.glob('*_label.xml'))

        xml_files.extend(list(self.data_dir.glob('*_label.xml')))

        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                for item in root.findall('.//Item'):
                    image_name = item.get('imageName')
                    camera_id_str = item.get('cameraID')

                    if image_name and camera_id_str:
                        try:
                            camid = int(camera_id_str[1:])
                            self._cityflow_camid_map[image_name] = camid
                        except (ValueError, IndexError):
                            pass

            except Exception as e:
                pass

    def _load_cityflow_tracks(self):
        """
        Load CityFlow vehicle ID mapping from track files.

        Parses train_track.txt and test_track.txt where each line contains
        filenames for images of the same vehicle. The line number becomes the vehicleID.
        """
        parent_dir = self.data_dir.parent

        # Look for track files in parent directory and current directory
        track_files = list(parent_dir.glob('*_track.txt'))
        track_files.extend(list(self.data_dir.glob('*_track.txt')))

        if not track_files:
            # Fallback: if no track files found, warn but continue
            # (will use filename-based PID parsing as fallback)
            print(f"Warning: No track files found for CityFlow in {parent_dir} or {self.data_dir}")
            print(f"  Falling back to filename-based vehicle ID parsing")
            return

        for track_file in track_files:
            try:
                with open(track_file, 'r') as f:
                    for vehicle_id, line in enumerate(f, start=1):
                        # Each line contains space-separated filenames for one vehicle
                        filenames = line.strip().split()

                        for filename in filenames:
                            # Map each filename to its vehicle ID
                            self._cityflow_pid_map[filename] = vehicle_id

                print(f"Loaded {len(self._cityflow_pid_map)} image->vehicleID mappings from {track_file.name}")

            except Exception as e:
                print(f"Warning: Failed to load track file {track_file}: {e}")
                continue

    def _parse_market1501(self, filename: str) -> Optional[Tuple[int, int]]:
        """
        Parse Market-1501
        Returns: (pid, camid) or None
        """
        if filename.startswith('-1') or filename.startswith('0000'):
            return None

        parts = filename.split('_')
        if len(parts) < 2:
            return None

        try:
            pid = int(parts[0])
            camid = int(parts[1][1])  
            return (pid, camid)
        except (ValueError, IndexError):
            return None

    def _parse_dukemtmc(self, filename: str) -> Optional[Tuple[int, int]]:
        """
        Parse DukeMTMC-reID
        Returns: (pid, camid) or None
        """
        if filename.startswith('-1') or filename.startswith('0000'):
            return None

        parts = filename.split('_')
        if len(parts) < 2:
            return None

        try:
            pid = int(parts[0])
            camid = int(parts[1][1])  
            return (pid, camid)
        except (ValueError, IndexError):
            return None

    def _parse_veri(self, filename: str) -> Optional[Tuple[int, int]]:
        """
        Parse VeRi-776
        Returns: (pid, camid) or None
        """
        parts = filename.split('_')
        if len(parts) < 2:
            return None

        try:
            pid = int(parts[0])
            camid = int(parts[1][1:]) 
            return (pid, camid)
        except (ValueError, IndexError):
            return None

    def _parse_cityflow(self, filename: str) -> Optional[Tuple[int, int]]:
        """
        Parse CityFlow format using track files.

        Vehicle ID comes from track files (train_track.txt, test_track.txt).
        Camera ID comes from XML files (*_label.xml).

        Fallback: If track files are not available, use filename number as vehicleID.

        Returns: (pid, camid) or None
        """
        # Try to get vehicleID from track mapping first
        pid = self._cityflow_pid_map.get(filename)

        if pid is None:
            # Fallback: parse filename as vehicleID (e.g., 000001.jpg -> vehicleID=1)
            # This is less accurate but works if track files are missing
            name_without_ext = filename.split('.')[0]
            try:
                pid = int(name_without_ext)
            except (ValueError, IndexError):
                return None

        # Get cameraID from XML mapping
        camid = self._cityflow_camid_map.get(filename, 0)

        return (pid, camid)

    def _parse_data(self):
        """Parse dataset based on dataset_type."""
        img_paths = list(self.data_dir.glob('*.jpg')) + list(self.data_dir.glob('*.png'))

        # Debug for CityFlow
        if self.dataset_type == 'cityflow':
            print(f"[CityFlow] Found {len(img_paths)} image files in {self.data_dir}")
            if len(img_paths) > 0:
                print(f"[CityFlow] Sample filenames: {[p.name for p in img_paths[:5]]}")

        parser_map = {
            'market1501': self._parse_market1501,
            'dukemtmc': self._parse_dukemtmc,
            'dukemtmc-reid': self._parse_dukemtmc,
            'veri776': self._parse_veri,
            'veri-776': self._parse_veri,
            'cityflow': self._parse_cityflow,
        }

        parser = parser_map.get(self.dataset_type, self._parse_market1501)

        parsed_count = 0
        for img_path in img_paths:
            filename = img_path.name
            result = parser(filename)

            if result is not None:
                pid, camid = result
                self.img_paths.append(str(img_path))
                self.pids.append(pid)
                self.camids.append(camid)
                parsed_count += 1

        # Debug for CityFlow
        if self.dataset_type == 'cityflow':
            print(f"[CityFlow] Successfully parsed {parsed_count}/{len(img_paths)} images")
            if parsed_count < len(img_paths):
                print(f"[CityFlow] WARNING: {len(img_paths) - parsed_count} images were skipped!")

    def __len__(self) -> int:
        return self.num_imgs

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int, str]:
        """
        Returns:
            img: Image tensor
            pid: Person ID (original PID, not mapped label)
            camid: Camera ID
            img_path: Image path
        """
        img_path = self.img_paths[index]
        pid = self.pids[index]
        camid = self.camids[index]

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        if self.opencv_augment:
            img = self.opencv_aug(img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path


class FewShotSampler(Sampler):
    """
    Few-shot sampler that samples K images per identity.
    """

    def __init__(
        self,
        data_source: Dataset,
        k_shot: int = 4,
        num_instances: int = 4,
        seed: Optional[int] = None
    ):
        """
        Args:
            data_source: Dataset instance
            k_shot: Number of images per identity to keep
            num_instances: Number of instances per identity in each batch
            seed: Random seed for reproducibility
        """
        self.data_source = data_source
        self.k_shot = k_shot
        self.num_instances = num_instances

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.pid_index_map = defaultdict(list)
        for idx, pid in enumerate(data_source.pids):
            label = data_source.pid2label[pid]
            self.pid_index_map[label].append(idx)

        self.sampled_indices = []
        self.pid_to_sampled_indices = defaultdict(list)
        for label, indices in self.pid_index_map.items():
            if len(indices) <= k_shot:
                sampled = indices
            else:
                sampled = random.sample(indices, k_shot)

            self.sampled_indices.extend(sampled)
            self.pid_to_sampled_indices[label] = sampled

        self.pids = list(self.pid_index_map.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        """Sample all k-shot instances for training."""
        indices = list(self.sampled_indices)
        random.shuffle(indices)
        for idx in indices:
            yield idx

    def __len__(self) -> int:
        return len(self.sampled_indices)


def create_fewshot_dataset(
    data_dir: str,
    k_shot: int = 4,
    batch_size: int = 32,
    num_instances: int = 4,
    height: int = 256,
    width: int = 128,
    num_workers: int = 4,
    seed: Optional[int] = None,
    opencv_augment: bool = True,
    augment_config: Optional[Dict] = None,
    dataset_type: str = 'market1501'
) -> DataLoader:
    """
    Create a few-shot dataloader for person ReID training.

    Args:
        data_dir: Path to training data directory
        k_shot: Number of images per identity
        batch_size: Batch size for training
        num_instances: Number of instances per identity in each batch
        height: Image height
        width: Image width
        num_workers: Number of data loading workers
        seed: Random seed
        opencv_augment: Whether to use OpenCV augmentations
        augment_config: Augmentation configuration

    Returns:
        DataLoader with few-shot sampling
    """
    dataset = ReIDDataset(
        data_dir=data_dir,
        dataset_type=dataset_type,
        height=height,
        width=width,
        opencv_augment=opencv_augment,
        augment_config=augment_config
    )

    sampler = FewShotSampler(
        data_source=dataset,
        k_shot=k_shot,
        num_instances=num_instances,
        seed=seed
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    return dataloader


def create_test_dataloader(
    data_dir: str,
    batch_size: int = 100,
    height: int = 256,
    width: int = 128,
    num_workers: int = 4,
    dataset_type: str = 'market1501'
) -> DataLoader:
    """
    Create a test/query dataloader (no augmentation).

    Args:
        data_dir: Path to test/query data directory
        batch_size: Batch size
        height: Image height
        width: Image width
        num_workers: Number of workers

    Returns:
        DataLoader for testing
    """
    dataset = ReIDDataset(
        data_dir=data_dir,
        dataset_type=dataset_type,
        height=height,
        width=width,
        opencv_augment=False
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def get_cityflow_dataloaders(
    dataset_path: Path,
    k_shot: int = 4,
    batch_size_train: int = 32,
    batch_size_test: int = 100,
    height: int = 256,
    width: int = 128,
    num_workers: int = 4,
    seed: Optional[int] = None,
    opencv_augment: bool = True,
    augment_config: Optional[Dict] = None
) -> Dict[str, DataLoader]:
    """
    Create CityFlow dataloaders with proper query/gallery splitting.

    For CityFlow, the test set is split as follows:
    - Query: First image from each vehicle track
    - Gallery: Remaining images from each vehicle track

    This ensures query and gallery don't overlap while maintaining vehicleID info.

    Args:
        dataset_path: Path to CityFlow dataset directory
        k_shot: Number of shots for few-shot learning
        batch_size_train: Training batch size
        batch_size_test: Test batch size
        height: Image height
        width: Image width
        num_workers: Number of workers
        seed: Random seed
        opencv_augment: Whether to use OpenCV augmentations
        augment_config: Augmentation configuration

    Returns:
        Dictionary with 'train', 'query', and 'gallery' dataloaders
    """
    print(f"[CityFlow] Loading dataset from {dataset_path}")

    # Create train loader (standard approach)
    train_dir = dataset_path / 'image_train'
    if not train_dir.exists():
        print(f"[CityFlow] WARNING: Training directory not found: {train_dir}")

    train_loader = create_fewshot_dataset(
        data_dir=str(train_dir),
        k_shot=k_shot,
        batch_size=batch_size_train,
        height=height,
        width=width,
        num_workers=num_workers,
        seed=seed,
        opencv_augment=opencv_augment,
        augment_config=augment_config,
        dataset_type='cityflow'
    )

    # Load test set and split into query and gallery
    test_dir = dataset_path / 'image_test'
    if not test_dir.exists():
        print(f"[CityFlow] ERROR: Test directory not found: {test_dir}")
        raise FileNotFoundError(f"CityFlow test directory not found: {test_dir}")

    # Load track file to determine query and gallery splits
    test_track_file = dataset_path / 'test_track.txt'
    if not test_track_file.exists():
        # Try parent directory
        test_track_file = dataset_path.parent / 'test_track.txt'

    if not test_track_file.exists():
        print(f"[CityFlow] ERROR: test_track.txt not found!")
        raise FileNotFoundError(f"test_track.txt not found in {dataset_path} or parent directory")

    # Parse track file to determine query and gallery splits
    print(f"[CityFlow] Loading track file: {test_track_file}")
    query_filenames = set()
    gallery_filenames = set()
    single_image_tracks = 0

    with open(test_track_file, 'r') as f:
        for line in f:
            filenames = line.strip().split()
            if len(filenames) > 0:
                # First image of each track = query
                query_filenames.add(filenames[0])

                # Remaining images = gallery
                if len(filenames) > 1:
                    gallery_filenames.update(filenames[1:])
                else:
                    # Single-image track: add to both query and gallery
                    # This is standard practice in ReID - self-matches will be filtered during evaluation
                    gallery_filenames.add(filenames[0])
                    single_image_tracks += 1

    print(f"[CityFlow] Query images: {len(query_filenames)}")
    print(f"[CityFlow] Gallery images: {len(gallery_filenames)}")
    if single_image_tracks > 0:
        print(f"[CityFlow] Single-image tracks: {single_image_tracks} (added to both query and gallery)")

    # Load full test dataset
    full_test_dataset = ReIDDataset(
        data_dir=str(test_dir),
        dataset_type='cityflow',
        height=height,
        width=width,
        opencv_augment=False
    )

    # Filter into query and gallery based on filenames
    query_img_paths = []
    query_pids = []
    query_camids = []
    gallery_img_paths = []
    gallery_pids = []
    gallery_camids = []

    # CityFlow-specific: Assign different camera IDs to query vs gallery
    # This is required because torchreid's evaluation filters out gallery samples
    # with matching (PID, camID) pairs. CityFlow tracks contain images from the
    # same camera, so we assign a special camera ID (999) to all query images.
    QUERY_CAMERA_ID = 999

    for idx, img_path in enumerate(full_test_dataset.img_paths):
        filename = Path(img_path).name

        if filename in query_filenames:
            query_img_paths.append(img_path)
            query_pids.append(full_test_dataset.pids[idx])
            # Use special camera ID for query to ensure it differs from gallery
            query_camids.append(QUERY_CAMERA_ID)

        if filename in gallery_filenames:
            gallery_img_paths.append(img_path)
            gallery_pids.append(full_test_dataset.pids[idx])
            # Keep original camera ID for gallery
            gallery_camids.append(full_test_dataset.camids[idx])

    print(f"[CityFlow] Query indices found: {len(query_img_paths)}")
    print(f"[CityFlow] Gallery indices found: {len(gallery_img_paths)}")

    # Debug: Check PID overlap
    query_pid_set = set(query_pids)
    gallery_pid_set = set(gallery_pids)
    overlap = query_pid_set.intersection(gallery_pid_set)
    missing = query_pid_set - gallery_pid_set

    print(f"[CityFlow] Query unique PIDs: {len(query_pid_set)}")
    print(f"[CityFlow] Gallery unique PIDs: {len(gallery_pid_set)}")
    print(f"[CityFlow] PIDs in both: {len(overlap)}")
    if len(missing) > 0:
        print(f"[CityFlow] ERROR: Query PIDs missing from gallery: {len(missing)}")
        print(f"[CityFlow]   Missing PIDs (first 10): {sorted(list(missing))[:10]}")

    # Debug: Check camera ID distribution
    query_camid_set = set(query_camids)
    gallery_camid_set = set(gallery_camids)
    print(f"[CityFlow] Query unique camera IDs: {sorted(query_camid_set)}")
    print(f"[CityFlow] Gallery unique camera IDs: {sorted(gallery_camid_set)}")

    # Debug: Check for each query PID, how many different camera IDs exist in gallery
    pid_camid_issues = 0
    for q_pid in list(query_pid_set)[:5]:  # Check first 5 PIDs as sample
        q_indices = [i for i, pid in enumerate(query_pids) if pid == q_pid]
        g_indices = [i for i, pid in enumerate(gallery_pids) if pid == q_pid]

        q_camids_for_pid = [query_camids[i] for i in q_indices]
        g_camids_for_pid = [gallery_camids[i] for i in g_indices]

        print(f"[CityFlow]   PID {q_pid}: Query camids={q_camids_for_pid}, Gallery camids={g_camids_for_pid}")

        # Check if query and gallery have different camera IDs for same PID
        if set(q_camids_for_pid) == set(g_camids_for_pid) and len(g_indices) > 0:
            pid_camid_issues += 1

    # Create custom dataset classes with filtered data
    class FilteredReIDDataset(Dataset):
        """Filtered dataset for CityFlow query/gallery split."""
        def __init__(self, img_paths, pids, camids, height, width):
            self.img_paths = img_paths
            self.pids = pids
            self.camids = camids
            self.height = height
            self.width = width

            # Required for torchreid compatibility
            self.pid_set = sorted(list(set(self.pids)))
            self.pid2label = {pid: label for label, pid in enumerate(self.pid_set)}
            self.num_pids = len(self.pid_set)
            self.num_imgs = len(self.img_paths)

        def __len__(self):
            return len(self.img_paths)

        def __getitem__(self, idx):
            img_path = self.img_paths[idx]
            pid = self.pids[idx]
            camid = self.camids[idx]

            # Load and preprocess image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.width, self.height))

            # Convert to tensor and normalize
            img = img.astype(np.float32) / 255.0
            img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            img = torch.from_numpy(img.transpose(2, 0, 1)).float()

            return img, pid, camid, img_path

    query_dataset = FilteredReIDDataset(
        query_img_paths, query_pids, query_camids, height, width
    )
    gallery_dataset = FilteredReIDDataset(
        gallery_img_paths, gallery_pids, gallery_camids, height, width
    )

    # Create dataloaders
    query_loader = DataLoader(
        query_dataset,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"[CityFlow] Dataloaders created successfully!")
    print(f"  Train: {len(train_loader.dataset) if hasattr(train_loader, 'dataset') else 'N/A'} images")
    print(f"  Query: {len(query_dataset)} images")
    print(f"  Gallery: {len(gallery_dataset)} images")

    return {
        'train': train_loader,
        'query': query_loader,
        'gallery': gallery_loader
    }


def get_dataloaders(
    root: str,
    dataset_name: str = 'market1501',
    k_shot: int = 4,
    batch_size_train: int = 32,
    batch_size_test: int = 100,
    height: int = 256,
    width: int = 128,
    num_workers: int = 4,
    seed: Optional[int] = None,
    opencv_augment: bool = True,
    augment_config: Optional[Dict] = None
) -> Dict[str, DataLoader]:
    """
    Create train, query, and gallery dataloaders.

    Args:
        root: Root directory containing the dataset
        dataset_name: Dataset name (e.g., 'market1501', 'dukemtmc', 'veri776', 'cityflow')
        k_shot: Number of shots for few-shot learning
        batch_size_train: Training batch size
        batch_size_test: Test batch size
        height: Image height
        width: Image width
        num_workers: Number of workers
        seed: Random seed
        opencv_augment: Whether to use OpenCV augmentations
        augment_config: Augmentation configuration

    Returns:
        Dictionary with 'train', 'query', and 'gallery' dataloaders
    """
    root_path = Path(root)
    dataset_path = root_path / dataset_name

    # Determine dataset type and folder names
    dataset_type = dataset_name.lower().replace('_preprocessed', '')

    # Special handling for CityFlow - split test set into query and gallery
    if dataset_type == 'cityflow':
        return get_cityflow_dataloaders(
            dataset_path=dataset_path,
            k_shot=k_shot,
            batch_size_train=batch_size_train,
            batch_size_test=batch_size_test,
            height=height,
            width=width,
            num_workers=num_workers,
            seed=seed,
            opencv_augment=opencv_augment,
            augment_config=augment_config
        )

    # Folder mapping for different datasets
    folder_map = {
        'market1501': {'train': 'bounding_box_train', 'test': 'bounding_box_test', 'query': 'query'},
        'dukemtmc': {'train': 'bounding_box_train', 'test': 'bounding_box_test', 'query': 'query'},
        'dukemtmc-reid': {'train': 'bounding_box_train', 'test': 'bounding_box_test', 'query': 'query'},
        'veri776': {'train': 'bounding_box_train', 'test': 'bounding_box_test', 'query': 'query'},
        'veri-776': {'train': 'bounding_box_train', 'test': 'bounding_box_test', 'query': 'query'},
    }

    folders = folder_map.get(dataset_type, folder_map['market1501'])

    train_loader = create_fewshot_dataset(
        data_dir=str(dataset_path / folders['train']),
        k_shot=k_shot,
        batch_size=batch_size_train,
        height=height,
        width=width,
        num_workers=num_workers,
        seed=seed,
        opencv_augment=opencv_augment,
        augment_config=augment_config,
        dataset_type=dataset_type
    )

    query_loader = create_test_dataloader(
        data_dir=str(dataset_path / folders['query']),
        batch_size=batch_size_test,
        height=height,
        width=width,
        num_workers=num_workers,
        dataset_type=dataset_type
    )

    gallery_loader = create_test_dataloader(
        data_dir=str(dataset_path / folders['test']),
        batch_size=batch_size_test,
        height=height,
        width=width,
        num_workers=num_workers,
        dataset_type=dataset_type
    )

    return {
        'train': train_loader,
        'query': query_loader,
        'gallery': gallery_loader
    }


def get_dataloaders_from_config(
    root: str,
    dataset_name: str = 'market1501',
    config_dir: str = "config",
    k_shot: Optional[int] = None,
    model_name: Optional[str] = None,
    model_type: Optional[str] = None,
    data_type: str = 'preprocessed',
    override_params: Optional[Dict] = None
) -> Dict[str, DataLoader]:
    """
    Create train, query, and gallery dataloaders using parameters from config files.

    Args:
        root: Root directory containing the dataset
        dataset_name: Dataset name (default: 'market1501')
        config_dir: Directory containing config files
        k_shot: Override k_shot value (if None, uses default from config)
        model_name: Model name to get specific image size from train_experiments.yaml
        model_type: Model type ('person' or 'vehicle')
        data_type: Data type ('preprocessed', 'augmented', 'original')
        override_params: Optional dict to override specific parameters

    Returns:
        Dictionary with 'train', 'query', and 'gallery' dataloaders
    """
    if get_dataloader_params_from_config is None:
        raise ImportError("config_loader module not available. Use get_dataloaders() instead.")

    params = get_dataloader_params_from_config(
        config_dir=config_dir,
        k_shot=k_shot,
        model_name=model_name,
        model_type=model_type,
        data_type=data_type
    )

    if override_params:
        params.update(override_params)

    return get_dataloaders(
        root=root,
        dataset_name=dataset_name,
        k_shot=params['k_shot'],
        batch_size_train=params['batch_size_train'],
        batch_size_test=params['batch_size_test'],
        height=params['height'],
        width=params['width'],
        num_workers=params['num_workers'],
        seed=params['seed'],
        opencv_augment=params['opencv_augment'],
        augment_config=params['augment_config']
    )
