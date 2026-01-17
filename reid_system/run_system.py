#!/usr/bin/env python3
"""
ReID System CLI

Command-line interface for running the Re-Identification system.
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import time
from collections import defaultdict
import json

from detection.detector import DetectionModule
from extraction.extractor import ExtractionModule
from matching.matcher import MatchingModule
from database.db_manager import DatabaseManager
from visualization.visualizer import Visualizer


class ReIDSystem:
    """
    Complete Re-Identification System.
    """

    def __init__(
        self,
        person_model_path: str,
        vehicle_model_path: str,
        database_person_path: str,
        database_vehicle_path: str,
        person_model_name: str = 'osnet_x1_0',
        vehicle_model_name: str = 'resnet50',
        person_loss: str = 'softmax',
        vehicle_loss: str = 'softmax',
        yolo_model: str = 'yolov8x.pt',
        device: str = 'cuda',
        confidence_threshold: float = 0.7,
        use_reranking: bool = True
    ):
        """
        Initialize ReID system.

        Args:
            person_model_path: Path to person ReID model
            vehicle_model_path: Path to vehicle ReID model
            database_person_path: Path to person identity database
            database_vehicle_path: Path to vehicle identity database
            person_model_name: Person model architecture name
            vehicle_model_name: Vehicle model architecture name
            person_loss: Person model loss function
            vehicle_loss: Vehicle model loss function
            yolo_model: YOLO model name
            device: Device to use
            confidence_threshold: Confidence threshold for identity matching
            use_reranking: Whether to use reranking
        """
        print("=" * 80)
        print("INITIALIZING RE-IDENTIFICATION SYSTEM")
        print("=" * 80)

        self.device = device
        self.confidence_threshold = confidence_threshold

        # Initialize statistics tracking
        self.stats = {
            'total_frames': 0,
            'total_frame_time': 0.0,
            'total_reid_time': 0.0,
            'person_queries': 0,
            'vehicle_queries': 0,
            'person_found': 0,
            'vehicle_found': 0,
            'person_new': 0,
            'vehicle_new': 0,
            'per_frame_stats': []
        }
        
        print("\n[1/5] Initializing detection module...")
        self.detector = DetectionModule(
            model_name=yolo_model,
            device=device
        )

        print("\n[2/5] Initializing feature extraction module...")
        self.extractor = ExtractionModule(
            person_model_path=person_model_path,
            vehicle_model_path=vehicle_model_path,
            person_model_name=person_model_name,
            vehicle_model_name=vehicle_model_name,
            person_loss=person_loss,
            vehicle_loss=vehicle_loss,
            device=device
        )

        print("\n[3/5] Initializing matching module...")
        self.matcher = MatchingModule(use_reranking=use_reranking)

        print("\n[4/5] Initializing databases...")
        print("  Loading person database...")
        self.database_person = DatabaseManager(db_path=database_person_path)
        print("  Loading vehicle database...")
        self.database_vehicle = DatabaseManager(db_path=database_vehicle_path)

        print("\n[5/5] Initializing visualizer...")
        self.visualizer = Visualizer()

        print("\n" + "=" * 80)
        print("SYSTEM READY")
        print("=" * 80)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        collect_stats: bool = True,
        return_detections: bool = False
    ) -> tuple:
        """
        Process a single frame.

        Args:
            frame: Input frame
            frame_id: Frame number
            collect_stats: Whether to collect statistics
            return_detections: Whether to return detections and results for crop saving

        Returns:
            If return_detections=False: (annotated_frame, frame_stats)
            If return_detections=True: (annotated_frame, frame_stats, detections_data)
            where detections_data = {'persons': (detections, results), 'vehicles': (detections, results)}
        """
        frame_start = time.time()
        frame_stats = {
            'frame_id': frame_id,
            'person_queries': 0,
            'vehicle_queries': 0,
            'person_found': 0,
            'vehicle_found': 0,
            'person_new': 0,
            'vehicle_new': 0,
            'reid_time': 0.0,
            'frame_time': 0.0
        }

        
        detections = self.detector.detect(frame, frame_id)

        
        reid_start = time.time()
        person_results = []
        if len(detections['persons']) > 0:
            person_results = self._process_detections(
                detections['persons'],
                category='person'
            )

            if collect_stats:
                frame_stats['person_queries'] = len(detections['persons'])
                for result in person_results:
                    if result.get('is_known', False):
                        frame_stats['person_found'] += 1
                    else:
                        frame_stats['person_new'] += 1

        
        vehicle_results = []
        if len(detections['vehicles']) > 0:
            vehicle_results = self._process_detections(
                detections['vehicles'],
                category='vehicle'
            )

            if collect_stats:
                frame_stats['vehicle_queries'] = len(detections['vehicles'])
                for result in vehicle_results:
                    if result.get('is_known', False):
                        frame_stats['vehicle_found'] += 1
                    else:
                        frame_stats['vehicle_new'] += 1

        reid_end = time.time()
        frame_stats['reid_time'] = reid_end - reid_start

        # Visualize results
        frame_vis = frame.copy()

        if len(detections['persons']) > 0:
            frame_vis = self.visualizer.draw_reid_results(
                frame_vis,
                detections['persons'],
                person_results,
                category='Person',
                confidence_threshold=self.confidence_threshold
            )

        if len(detections['vehicles']) > 0:
            frame_vis = self.visualizer.draw_reid_results(
                frame_vis,
                detections['vehicles'],
                vehicle_results,
                category='Vehicle',
                confidence_threshold=self.confidence_threshold
            )

        frame_end = time.time()
        frame_stats['frame_time'] = frame_end - frame_start

        if return_detections:
            detections_data = {
                'persons': (detections['persons'], person_results),
                'vehicles': (detections['vehicles'], vehicle_results)
            }
            return frame_vis, frame_stats, detections_data
        else:
            return frame_vis, frame_stats

    def _process_detections(
        self,
        detections: list,
        category: str
    ) -> list:
        """
        Process detections: extract features, match, and identify.

        Args:
            detections: List of detections
            category: 'person' or 'vehicle'

        Returns:
            List of ReID results
        """
        results = []

        
        cropped_images = [det['cropped_image'] for det in detections]
        features = self.extractor.extract_features(
            cropped_images,
            model_type=category
        )

        
        database = self.database_person if category == 'person' else self.database_vehicle

        
        gallery_features, identity_ids = database.get_all_features(category=category)

        if len(gallery_features) == 0:
            
            return [{
                'identity_id': None,
                'label': 'Unknown',
                'similarity': 0.0,
                'is_known': False
            }] * len(features)

        
        indices, distances = self.matcher.match_batch(
            query_features=features,
            gallery_features=gallery_features,
            top_k=1
        )

        
        for idx, dist in zip(indices[:, 0], distances[:, 0]):
            identity_id = identity_ids[idx]
            distance_original = float(dist)

            
            
            
            distance = max(distance_original, 0.1)  

            
            
            
            
            similarity = float(1.0 / (1.0 + (distance * 3.0) ** 2))

            if similarity >= self.confidence_threshold:
                
                results.append({
                    'identity_id': identity_id,
                    'label': database.identities[identity_id]['label'],
                    'similarity': similarity,
                    'distance': distance,
                    'is_known': True
                })
            else:
                
                results.append({
                    'identity_id': None,
                    'label': 'Unknown',
                    'similarity': similarity,
                    'is_known': False
                })

        
        results = self._remove_duplicate_identities(results, detections)

        
        assert len(results) == len(detections), \
            f"Results length ({len(results)}) doesn't match detections length ({len(detections)})"

        return results

    def _remove_duplicate_identities(
        self,
        results: list,
        detections: list
    ) -> list:
        """
        Remove duplicate identities from results - keep only highest confidence.

        Args:
            results: List of ReID results
            detections: List of detections

        Returns:
            Filtered results with no duplicate identities
        """
        
        identity_groups = {}
        for i, result in enumerate(results):
            identity_id = result.get('identity_id')
            if identity_id is None or not result.get('is_known', False):
                
                continue

            if identity_id not in identity_groups:
                identity_groups[identity_id] = []
            identity_groups[identity_id].append((i, result['similarity']))

        
        indices_to_remove = set()
        for identity_id, occurrences in identity_groups.items():
            if len(occurrences) > 1:
                
                occurrences.sort(key=lambda x: x[1], reverse=True)

                
                print(f"  [DEBUG] Found duplicate identity: {identity_id}")
                print(f"    Keeping detection #{occurrences[0][0]} (conf: {occurrences[0][1]:.3f})")
                print(f"    Removing: {[(idx, conf) for idx, conf in occurrences[1:]]}")

                
                for idx, _ in occurrences[1:]:
                    indices_to_remove.add(idx)

        
        filtered_results = []
        for i, result in enumerate(results):
            if i in indices_to_remove:
                filtered_results.append({
                    'identity_id': None,
                    'label': 'Unknown',
                    'similarity': result['similarity'],
                    'is_known': False
                })
            else:
                filtered_results.append(result)

        return filtered_results

    def run_on_frames(
        self,
        frames_dir: str,
        output_dir: str,
        save_frames: bool = True,
        save_stats: bool = True,
        save_crops: bool = True
    ):
        """
        Run system on a directory of frames.

        Args:
            frames_dir: Directory containing frames
            output_dir: Output directory for results
            save_frames: Whether to save annotated frames
            save_stats: Whether to save statistics
            save_crops: Whether to save cropped detection images organized by identity
        """
        frames_path = Path(frames_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        
        self.stats = {
            'total_frames': 0,
            'total_frame_time': 0.0,
            'total_reid_time': 0.0,
            'person_queries': 0,
            'vehicle_queries': 0,
            'person_found': 0,
            'vehicle_found': 0,
            'person_new': 0,
            'vehicle_new': 0,
            'per_frame_stats': []
        }

        
        frame_files = sorted(
            list(frames_path.glob('*.jpg')) +
            list(frames_path.glob('*.png')) +
            list(frames_path.glob('*.jpeg'))
        )

        if len(frame_files) == 0:
            print(f"No frames found in {frames_dir}")
            return

        print(f"\nProcessing {len(frame_files)} frames...")
        print(f"Output directory: {output_path}")

        
        if save_crops:
            crops_dir = output_path / 'detection_crops'
            crops_dir.mkdir(parents=True, exist_ok=True)
            print(f"Detection crops will be saved to: {crops_dir}")

        
        for i, frame_file in enumerate(tqdm(frame_files, desc="Processing frames")):
            
            frame = cv2.imread(str(frame_file))
            if frame is None:
                print(f"Failed to read frame: {frame_file}")
                continue

            
            if save_crops:
                frame_vis, frame_stats, detections_data = self.process_frame(
                    frame, frame_id=i, collect_stats=True, return_detections=True
                )
            else:
                frame_vis, frame_stats = self.process_frame(
                    frame, frame_id=i, collect_stats=True, return_detections=False
                )

            # Update global statistics
            self.stats['total_frames'] += 1
            self.stats['total_frame_time'] += frame_stats['frame_time']
            self.stats['total_reid_time'] += frame_stats['reid_time']
            self.stats['person_queries'] += frame_stats['person_queries']
            self.stats['vehicle_queries'] += frame_stats['vehicle_queries']
            self.stats['person_found'] += frame_stats['person_found']
            self.stats['vehicle_found'] += frame_stats['vehicle_found']
            self.stats['person_new'] += frame_stats['person_new']
            self.stats['vehicle_new'] += frame_stats['vehicle_new']
            self.stats['per_frame_stats'].append(frame_stats)

            # Save detection crops
            if save_crops:
                person_detections, person_results = detections_data['persons']
                vehicle_detections, vehicle_results = detections_data['vehicles']

                if len(person_detections) > 0:
                    self._save_detection_crops(
                        person_detections, person_results, 'person', crops_dir, i
                    )

                if len(vehicle_detections) > 0:
                    self._save_detection_crops(
                        vehicle_detections, vehicle_results, 'vehicle', crops_dir, i
                    )

            # Save annotated frame
            if save_frames:
                output_file = output_path / f"result_{frame_file.name}"
                self.visualizer.save_image(frame_vis, str(output_file))

        # Print and save statistics
        self._print_statistics()

        if save_stats:
            stats_file = output_path / 'statistics.json'
            self._save_statistics(stats_file)

        print(f"\nProcessing complete! Results saved to: {output_path}")

    def _print_statistics(self):
        """Print comprehensive statistics."""
        print("\n" + "=" * 80)
        print("REID SYSTEM STATISTICS")
        print("=" * 80)

        # FPS calculations
        if self.stats['total_frames'] > 0:
            avg_frame_time = self.stats['total_frame_time'] / self.stats['total_frames']
            avg_reid_time = self.stats['total_reid_time'] / self.stats['total_frames']
            whole_frame_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            reid_fps = 1.0 / avg_reid_time if avg_reid_time > 0 else 0

            print(f"\n--- Performance Metrics ---")
            print(f"Total frames processed: {self.stats['total_frames']}")
            print(f"Total processing time: {self.stats['total_frame_time']:.2f}s")
            print(f"Average time per frame (full pipeline): {avg_frame_time*1000:.2f}ms")
            print(f"Average time per frame (ReID only): {avg_reid_time*1000:.2f}ms")
            print(f"FPS (full pipeline): {whole_frame_fps:.2f}")
            print(f"FPS (ReID only): {reid_fps:.2f}")

        # Query statistics
        total_queries = self.stats['person_queries'] + self.stats['vehicle_queries']
        total_found = self.stats['person_found'] + self.stats['vehicle_found']
        total_new = self.stats['person_new'] + self.stats['vehicle_new']

        print(f"\n--- Query Statistics ---")
        print(f"Total queries: {total_queries}")
        print(f"  Person queries: {self.stats['person_queries']}")
        print(f"  Vehicle queries: {self.stats['vehicle_queries']}")

        print(f"\n--- Match Statistics ---")
        print(f"Found (existing identities): {total_found} ({100*total_found/total_queries:.1f}%)" if total_queries > 0 else "Found: 0")
        print(f"  Persons found: {self.stats['person_found']} ({100*self.stats['person_found']/self.stats['person_queries']:.1f}%)" if self.stats['person_queries'] > 0 else "  Persons found: 0")
        print(f"  Vehicles found: {self.stats['vehicle_found']} ({100*self.stats['vehicle_found']/self.stats['vehicle_queries']:.1f}%)" if self.stats['vehicle_queries'] > 0 else "  Vehicles found: 0")

        print(f"\nNew (unknown identities): {total_new} ({100*total_new/total_queries:.1f}%)" if total_queries > 0 else "New: 0")
        print(f"  Persons new: {self.stats['person_new']} ({100*self.stats['person_new']/self.stats['person_queries']:.1f}%)" if self.stats['person_queries'] > 0 else "  Persons new: 0")
        print(f"  Vehicles new: {self.stats['vehicle_new']} ({100*self.stats['vehicle_new']/self.stats['vehicle_queries']:.1f}%)" if self.stats['vehicle_queries'] > 0 else "  Vehicles new: 0")

        print("\n" + "=" * 80)

    def _save_statistics(self, output_file: Path):
        """Save statistics to JSON file."""
        # Calculate derived metrics
        stats_output = self.stats.copy()

        if self.stats['total_frames'] > 0:
            avg_frame_time = self.stats['total_frame_time'] / self.stats['total_frames']
            avg_reid_time = self.stats['total_reid_time'] / self.stats['total_frames']

            stats_output['metrics'] = {
                'avg_frame_time_ms': avg_frame_time * 1000,
                'avg_reid_time_ms': avg_reid_time * 1000,
                'fps_full_pipeline': 1.0 / avg_frame_time if avg_frame_time > 0 else 0,
                'fps_reid_only': 1.0 / avg_reid_time if avg_reid_time > 0 else 0
            }

            total_queries = self.stats['person_queries'] + self.stats['vehicle_queries']
            total_found = self.stats['person_found'] + self.stats['vehicle_found']
            total_new = self.stats['person_new'] + self.stats['vehicle_new']

            stats_output['match_rates'] = {
                'total_queries': total_queries,
                'total_found': total_found,
                'total_new': total_new,
                'found_rate': total_found / total_queries if total_queries > 0 else 0,
                'new_rate': total_new / total_queries if total_queries > 0 else 0,
                'person_found_rate': self.stats['person_found'] / self.stats['person_queries'] if self.stats['person_queries'] > 0 else 0,
                'vehicle_found_rate': self.stats['vehicle_found'] / self.stats['vehicle_queries'] if self.stats['vehicle_queries'] > 0 else 0
            }

        # Save to file
        with open(output_file, 'w') as f:
            json.dump(stats_output, f, indent=2)

        print(f"\nStatistics saved to: {output_file}")

    def _save_detection_crops(
        self,
        detections: list,
        results: list,
        category: str,
        crops_dir: Path,
        frame_id: int
    ):
        """
        Save cropped detection images organized by identity ID.

        Args:
            detections: List of detections with cropped_image
            results: List of ReID results
            category: 'person' or 'vehicle'
            crops_dir: Base directory for saving crops
            frame_id: Frame number
        """
        for i, (det, result) in enumerate(zip(detections, results)):
            if result.get('is_known', False):
                # Save found identities
                identity_id = result['identity_id']
                identity_label = result['label']

                # Create folder: crops_dir/found/{category}/{identity_id}_{label}/
                save_dir = crops_dir / 'found' / category / f"{identity_id}_{identity_label}"
                save_dir.mkdir(parents=True, exist_ok=True)

                
                similarity = result.get('similarity', 0.0)
                filename = f"frame{frame_id:06d}_det{i}_conf{similarity:.3f}.jpg"
                save_path = save_dir / filename

                cv2.imwrite(str(save_path), det['cropped_image'])
            else:
                # Save new/unknown identities
                save_dir = crops_dir / 'new' / category
                save_dir.mkdir(parents=True, exist_ok=True)

                
                filename = f"frame{frame_id:06d}_det{i}.jpg"
                save_path = save_dir / filename

                cv2.imwrite(str(save_path), det['cropped_image'])

    def add_identity_interactive(self, category: str = 'person'):
        """
        Interactive mode to add a new identity to the database.

        Args:
            category: 'person' or 'vehicle'
        """
        print(f"\n=== Add New {category.title()} Identity ===")

        
        database = self.database_person if category == 'person' else self.database_vehicle

        
        image_path = input("Enter path to reference image: ").strip()
        if not Path(image_path).exists():
            print("Error: Image not found")
            return

        
        label = input(f"Enter label for this {category}: ").strip()
        if not label:
            print("Error: Label cannot be empty")
            return

        
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Failed to read image")
            return

        
        print("Extracting features...")
        feature = self.extractor.extract_single(image, model_type=category)

        
        identity_id = f"{category}_{len(database.identities) + 1:04d}"

        
        success = database.add_identity(
            identity_id=identity_id,
            feature=feature,
            label=label,
            category=category
        )

        if success:
            print(f"Successfully added {label} to database (ID: {identity_id})")
        else:
            print("Failed to add identity")


def main():
    parser = argparse.ArgumentParser(
        description='Re-Identification System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process frames from a directory with OSNet and AAVeR
  python run_system.py --frames-dir ./camera_frames --output-dir ./results \\
      --person-model /path/to/osnet.pth \\
      --vehicle-model /path/to/aaver.pth \\
      --person-model-name osnet_x1_0 \\
      --vehicle-model-name aaver \\
      --person-loss softmax \\
      --vehicle-loss softmax \\
      --database-person ./reid_database_person \\
      --database-vehicle ./reid_database_vehicle \\
      --confidence 0.8

  # Add a new identity to the database
  python run_system.py --add-identity person \\
      --person-model /path/to/osnet.pth \\
      --person-model-name osnet_x1_0 \\
      --database-person ./reid_database_person \\
      --database-vehicle ./reid_database_vehicle

Note: Model paths should point to Google Drive or local .pth files
Supported person models: osnet_x1_0, transreid, hacnn, pcb, autoreid
Supported vehicle models: resnet50, aaver, rptm, vat
        """
    )

    parser.add_argument(
        '--frames-dir',
        type=str,
        help='Directory containing camera frames'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./reid_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--person-model',
        type=str,
        required=True,
        help='Path to person ReID model (.pth file)'
    )
    parser.add_argument(
        '--vehicle-model',
        type=str,
        help='Path to vehicle ReID model (.pth file)'
    )
    parser.add_argument(
        '--person-model-name',
        type=str,
        default='osnet_x1_0',
        help='Person model architecture (osnet_x1_0, transreid, hacnn, etc.)'
    )
    parser.add_argument(
        '--vehicle-model-name',
        type=str,
        default='resnet50',
        help='Vehicle model architecture (resnet50, aaver, rptm, vat, etc.)'
    )
    parser.add_argument(
        '--person-loss',
        type=str,
        default='softmax',
        help='Person model loss function (softmax, triplet, etc.)'
    )
    parser.add_argument(
        '--vehicle-loss',
        type=str,
        default='softmax',
        help='Vehicle model loss function (softmax, triplet, etc.)'
    )
    parser.add_argument(
        '--database-person',
        type=str,
        default='./identity_database_person',
        help='Path to person identity database directory'
    )
    parser.add_argument(
        '--database-vehicle',
        type=str,
        default='./identity_database_vehicle',
        help='Path to vehicle identity database directory'
    )
    parser.add_argument(
        '--yolo-model',
        type=str,
        default='yolov8x.pt',
        help='YOLO model name'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.7,
        help='Confidence threshold for identity matching'
    )
    parser.add_argument(
        '--no-reranking',
        action='store_true',
        help='Disable reranking'
    )
    parser.add_argument(
        '--add-identity',
        type=str,
        choices=['person', 'vehicle'],
        help='Add a new identity to the database'
    )
    parser.add_argument(
        '--no-save-crops',
        action='store_true',
        help='Disable saving detection crops organized by identity'
    )

    args = parser.parse_args()

    # Initialize system
    system = ReIDSystem(
        person_model_path=args.person_model,
        vehicle_model_path=args.vehicle_model,
        database_person_path=args.database_person,
        database_vehicle_path=args.database_vehicle,
        person_model_name=args.person_model_name,
        vehicle_model_name=args.vehicle_model_name,
        person_loss=args.person_loss,
        vehicle_loss=args.vehicle_loss,
        yolo_model=args.yolo_model,
        device=args.device,
        confidence_threshold=args.confidence,
        use_reranking=not args.no_reranking
    )

    # Run mode
    if args.add_identity:
        # Add identity mode
        system.add_identity_interactive(category=args.add_identity)
    elif args.frames_dir:
        # Process frames mode
        system.run_on_frames(
            frames_dir=args.frames_dir,
            output_dir=args.output_dir,
            save_crops=not args.no_save_crops
        )
    else:
        print("Error: Please specify --frames-dir or --add-identity")
        parser.print_help()


if __name__ == '__main__':
    main()
