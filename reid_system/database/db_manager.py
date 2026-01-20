"""
Database Manager

Manages identity database with features and metadata.
Supports adding, updating, and querying identities.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pickle
from datetime import datetime
import shutil


class DatabaseManager:
    """
    Database manager for storing and querying identities.
    """

    def __init__(
        self,
        db_path: str,
        feature_dim: int = 512,
        auto_save: bool = True
    ):
        """
        Initialize database manager.

        Args:
            db_path: Path to database directory
            feature_dim: Dimension of feature vectors
            auto_save: Whether to auto-save after modifications
        """
        self.db_path = Path(db_path)
        self.feature_dim = feature_dim
        self.auto_save = auto_save

        
        self.identities = {}  
        self.features = {}    
        self.metadata = {
            'feature_dim': feature_dim,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'version': '1.0'
        }

        
        self.db_path.mkdir(parents=True, exist_ok=True)

        
        self.load()

        print(f"Database initialized at: {db_path}")
        print(f"  Total identities: {len(self.identities)}")

    def add_identity(
        self,
        identity_id: str,
        feature: np.ndarray,
        label: Optional[str] = None,
        category: str = 'person',
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Add a new identity to the database.

        Args:
            identity_id: Unique identity ID
            feature: Feature vector
            label: Human-readable label (optional)
            category: 'person' or 'vehicle'
            metadata: Additional metadata

        Returns:
            True if added, False if already exists
        """
        if identity_id in self.identities:
            print(f"Identity {identity_id} already exists. Use update_identity() to modify.")
            return False

        
        if feature.shape[0] != self.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.feature_dim}, "
                f"got {feature.shape[0]}"
            )

        
        identity_data = {
            'identity_id': identity_id,
            'label': label or f"{category}_{identity_id}",
            'category': category,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }

        
        self.identities[identity_id] = identity_data
        self.features[identity_id] = feature

        
        self.metadata['updated_at'] = datetime.now().isoformat()

        
        if self.auto_save:
            self.save()

        print(f"Added identity: {identity_id} ({label})")
        return True

    def update_identity(
        self,
        identity_id: str,
        feature: Optional[np.ndarray] = None,
        label: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Update an existing identity.

        Args:
            identity_id: Identity ID to update
            feature: New feature vector (optional)
            label: New label (optional)
            metadata: New metadata (optional)

        Returns:
            True if updated, False if not found
        """
        if identity_id not in self.identities:
            print(f"Identity {identity_id} not found.")
            return False

        
        if feature is not None:
            if feature.shape[0] != self.feature_dim:
                raise ValueError(
                    f"Feature dimension mismatch: expected {self.feature_dim}, "
                    f"got {feature.shape[0]}"
                )
            self.features[identity_id] = feature

        
        if label is not None:
            self.identities[identity_id]['label'] = label

        
        if metadata is not None:
            self.identities[identity_id]['metadata'].update(metadata)

        
        self.identities[identity_id]['updated_at'] = datetime.now().isoformat()
        self.metadata['updated_at'] = datetime.now().isoformat()

        
        if self.auto_save:
            self.save()

        print(f"Updated identity: {identity_id}")
        return True

    def add_feature_to_identity(
        self,
        identity_id: str,
        feature: np.ndarray
    ) -> bool:
        """
        Add an additional feature vector to an existing identity.

        Args:
            identity_id: Identity ID to add feature to
            feature: New feature vector to add

        Returns:
            True if added, False if identity not found
        """
        if identity_id not in self.identities:
            print(f"Identity {identity_id} not found.")
            return False

        if feature.shape[0] != self.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.feature_dim}, "
                f"got {feature.shape[0]}"
            )

        current_features = self.features[identity_id]
        if not isinstance(current_features, list):
            current_features = [current_features]

        current_features.append(feature)
        self.features[identity_id] = current_features

        self.identities[identity_id]['updated_at'] = datetime.now().isoformat()
        self.metadata['updated_at'] = datetime.now().isoformat()

        if self.auto_save:
            self.save()

        print(f"Added feature to identity: {identity_id} (now has {len(current_features)} features)")
        return True

    def remove_identity(self, identity_id: str) -> bool:
        """
        Remove an identity from the database.

        Args:
            identity_id: Identity ID to remove

        Returns:
            True if removed, False if not found
        """
        if identity_id not in self.identities:
            print(f"Identity {identity_id} not found.")
            return False

        del self.identities[identity_id]
        del self.features[identity_id]


        self.metadata['updated_at'] = datetime.now().isoformat()


        if self.auto_save:
            self.save()

        print(f"Removed identity: {identity_id}")
        return True

    def get_identity(self, identity_id: str) -> Optional[Dict]:
        """
        Get identity data.

        Args:
            identity_id: Identity ID

        Returns:
            Identity data or None if not found
        """
        return self.identities.get(identity_id)

    def get_feature(self, identity_id: str) -> Optional[np.ndarray]:
        """
        Get identity feature vector.

        Args:
            identity_id: Identity ID

        Returns:
            Feature vector or None if not found
        """
        return self.features.get(identity_id)

    def get_all_features(
        self,
        category: Optional[str] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Get all feature vectors (expands multiple features per identity).

        If an identity has multiple feature vectors stored as a list,
        each vector is returned as a separate entry with the same identity_id.

        Args:
            category: Filter by category ('person' or 'vehicle')

        Returns:
            features: Feature matrix (shape: [N, D]) where N = total feature vectors
            identity_ids: List of corresponding identity IDs (with duplicates if multiple features)
        """
        if category:

            filtered_ids = [
                iid for iid, data in self.identities.items()
                if data['category'] == category
            ]
        else:
            filtered_ids = list(self.identities.keys())

        if len(filtered_ids) == 0:
            return np.array([]), []

        all_features = []
        all_identity_ids = []

        for iid in filtered_ids:
            feature_data = self.features[iid]

            for feature_vec in feature_data:
                all_features.append(feature_vec)
                all_identity_ids.append(iid)

        if len(all_features) == 0:
            return np.array([]), []

        features = np.stack(all_features)
        return features, all_identity_ids

    def query(
        self,
        feature: np.ndarray,
        category: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Query database for similar identities.

        Args:
            feature: Query feature vector
            category: Filter by category
            top_k: Number of top matches to return

        Returns:
            List of matches with identity data and distances
        """
        
        gallery_features, identity_ids = self.get_all_features(category)

        if len(gallery_features) == 0:
            return []

        
        distances = np.linalg.norm(gallery_features - feature, axis=1)

        
        top_indices = np.argsort(distances)[:top_k]

        results = []
        for idx in top_indices:
            identity_id = identity_ids[idx]
            distance = float(distances[idx])

            results.append({
                'identity_id': identity_id,
                'identity_data': self.identities[identity_id],
                'distance': distance,
                'similarity': float(1.0 / (1.0 + distance))  
            })

        return results

    def save(self):
        """Save database to disk."""
        
        identities_path = self.db_path / 'identities.json'
        with open(identities_path, 'w') as f:
            json.dump(self.identities, f, indent=2)

        
        features_path = self.db_path / 'features.pkl'
        with open(features_path, 'wb') as f:
            pickle.dump(self.features, f)

        
        metadata_path = self.db_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        print(f"Database saved ({len(self.identities)} identities)")

    def load(self):
        """Load database from disk."""
        identities_path = self.db_path / 'identities.json'
        features_path = self.db_path / 'features.pkl'
        metadata_path = self.db_path / 'metadata.json'

        
        if identities_path.exists():
            with open(identities_path, 'r') as f:
                self.identities = json.load(f)

        
        if features_path.exists():
            with open(features_path, 'rb') as f:
                self.features = pickle.load(f)

        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)

        if len(self.identities) > 0:
            print(f"Loaded {len(self.identities)} identities from database")

    def backup(self, backup_path: str):
        """
        Create a backup of the database.

        Args:
            backup_path: Path to backup directory
        """
        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)

        
        for file in self.db_path.glob('*'):
            if file.is_file():
                shutil.copy2(file, backup_path / file.name)

        print(f"Database backed up to: {backup_path}")

    def get_statistics(self) -> Dict:
        """
        Get database statistics.

        Returns:
            Dictionary with statistics
        """
        person_count = sum(
            1 for data in self.identities.values()
            if data['category'] == 'person'
        )
        vehicle_count = sum(
            1 for data in self.identities.values()
            if data['category'] == 'vehicle'
        )

        return {
            'total_identities': len(self.identities),
            'person_count': person_count,
            'vehicle_count': vehicle_count,
            'feature_dim': self.feature_dim,
            'created_at': self.metadata.get('created_at'),
            'updated_at': self.metadata.get('updated_at')
        }
