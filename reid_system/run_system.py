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

            
            if save_frames:
                output_file = output_path / f"result_{frame_file.name}"
                self.visualizer.save_image(frame_vis, str(output_file))

        
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
                
                identity_id = result['identity_id']
                identity_label = result['label']

                
                save_dir = crops_dir / 'found' / category / f"{identity_id}_{identity_label}"
                save_dir.mkdir(parents=True, exist_ok=True)

                
                similarity = result.get('similarity', 0.0)
                filename = f"frame{frame_id:06d}_det{i}_conf{similarity:.3f}.jpg"
                save_path = save_dir / filename

                cv2.imwrite(str(save_path), det['cropped_image'])
            else:
                
                save_dir = crops_dir / 'new' / category
                save_dir.mkdir(parents=True, exist_ok=True)

                
                filename = f"frame{frame_id:06d}_det{i}.jpg"
                save_path = save_dir / filename

                cv2.imwrite(str(save_path), det['cropped_image'])

    def evaluate_yolo_detection(
        self,
        frames_dir: str,
        gt_file: str,
        output_dir: str,
        iou_threshold: float = 0.5
    ):
        """
        Evaluate YOLO detection quality against MOT17 ground truth.

        Args:
            frames_dir: Directory containing video frames
            gt_file: Path to MOT17 ground truth file (gt.txt)
            output_dir: Output directory for results
            iou_threshold: IoU threshold for matching (default: 0.5)
        """
        from collections import defaultdict

        print("\n" + "=" * 80)
        print("YOLO DETECTION EVALUATION")
        print("=" * 80)

        
        print(f"\nLoading ground truth from: {gt_file}")
        gt_data = self._parse_mot17_gt(gt_file)
        print(f"  Found {len(gt_data)} frames with annotations")
        print(f"  Total GT boxes: {sum(len(boxes) for boxes in gt_data.values())}")

        
        frames_path = Path(frames_dir)
        frame_files = sorted(
            list(frames_path.glob('*.jpg')) +
            list(frames_path.glob('*.png'))
        )

        print(f"\nFound {len(frame_files)} frames in {frames_dir}")

        
        stats = {
            'total_frames': 0,
            'total_gt_boxes': 0,
            'total_yolo_detections': 0,
            'total_matches': 0,
            'total_false_negatives': 0,
            'total_false_positives': 0,
            'iou_scores': [],
            'per_frame_stats': []
        }

        
        print(f"\nEvaluating detection (IoU threshold: {iou_threshold})...")
        for frame_file in tqdm(frame_files, desc="Evaluating YOLO"):
            
            frame_num = int(frame_file.stem)

            
            if frame_num not in gt_data:
                continue

            
            frame = cv2.imread(str(frame_file))
            if frame is None:
                print(f"  Warning: Failed to read {frame_file}")
                continue

            
            gt_boxes = gt_data[frame_num]

            
            detections = self.detector.detect(frame, frame_id=frame_num)
            yolo_persons = detections['persons']

            
            matches, unmatched_gt, unmatched_yolo = self._match_detections(
                gt_boxes, yolo_persons, iou_threshold
            )

            
            stats['total_frames'] += 1
            stats['total_gt_boxes'] += len(gt_boxes)
            stats['total_yolo_detections'] += len(yolo_persons)
            stats['total_matches'] += len(matches)
            stats['total_false_negatives'] += len(unmatched_gt)
            stats['total_false_positives'] += len(unmatched_yolo)

            
            for _, _, iou in matches:
                stats['iou_scores'].append(iou)

            
            frame_recall = len(matches) / len(gt_boxes) if len(gt_boxes) > 0 else 0
            frame_precision = len(matches) / len(yolo_persons) if len(yolo_persons) > 0 else 0

            stats['per_frame_stats'].append({
                'frame_num': frame_num,
                'gt_boxes': len(gt_boxes),
                'yolo_detections': len(yolo_persons),
                'matches': len(matches),
                'false_negatives': len(unmatched_gt),
                'false_positives': len(unmatched_yolo),
                'recall': frame_recall,
                'precision': frame_precision
            })

        
        self._print_yolo_evaluation_results(stats, iou_threshold)

        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        results_file = output_path / 'yolo_evaluation.json'
        self._save_yolo_evaluation_results(stats, iou_threshold, results_file)

    def _parse_mot17_gt(self, gt_file: str):
        """
        Parse MOT17 ground truth file.

        Format: frame,id,x,y,w,h,conf,class,vis

        Returns:
            dict: {frame_id: [bbox1, bbox2, ...]}
        """
        from collections import defaultdict

        gt_data = defaultdict(list)

        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 9:
                    continue

                frame_id = int(parts[0])
                person_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                conf = float(parts[6])
                cls = int(parts[7])
                vis = float(parts[8])

                
                if cls == 1 and vis > 0.0:
                    gt_data[frame_id].append({
                        'bbox': [x, y, w, h],
                        'person_id': person_id,
                        'visibility': vis
                    })

        return gt_data

    def _calculate_iou(self, box1, box2):
        """
        Calculate IoU between two boxes.

        Args:
            box1: [x, y, w, h] format (MOT17 GT)
            box2: [x1, y1, x2, y2] format (YOLO output)

        Returns:
            float: IoU value
        """
        
        x1_box1, y1_box1 = box1[0], box1[1]
        x2_box1, y2_box1 = box1[0] + box1[2], box1[1] + box1[3]

        
        x1_box2, y1_box2, x2_box2, y2_box2 = box2

        
        x1_inter = max(x1_box1, x1_box2)
        y1_inter = max(y1_box1, y1_box2)
        x2_inter = min(x2_box1, x2_box2)
        y2_inter = min(y2_box1, y2_box2)

        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        
        box1_area = box1[2] * box1[3]
        box2_area = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def _match_detections(self, gt_boxes, yolo_detections, iou_threshold):
        """
        Match YOLO detections to GT boxes using greedy matching.

        Returns:
            matches: List of (gt_idx, yolo_idx, iou)
            unmatched_gt: List of unmatched GT indices
            unmatched_yolo: List of unmatched YOLO indices
        """
        if len(gt_boxes) == 0 or len(yolo_detections) == 0:
            return [], list(range(len(gt_boxes))), list(range(len(yolo_detections)))

        
        iou_matrix = np.zeros((len(gt_boxes), len(yolo_detections)))
        for i, gt_box in enumerate(gt_boxes):
            for j, yolo_det in enumerate(yolo_detections):
                iou_matrix[i, j] = self._calculate_iou(gt_box['bbox'], yolo_det['bbox'])

        
        matches = []
        matched_gt = set()
        matched_yolo = set()

        
        candidates = []
        for i in range(len(gt_boxes)):
            for j in range(len(yolo_detections)):
                if iou_matrix[i, j] >= iou_threshold:
                    candidates.append((i, j, iou_matrix[i, j]))

        candidates.sort(key=lambda x: x[2], reverse=True)

        for gt_idx, yolo_idx, iou in candidates:
            if gt_idx not in matched_gt and yolo_idx not in matched_yolo:
                matches.append((gt_idx, yolo_idx, iou))
                matched_gt.add(gt_idx)
                matched_yolo.add(yolo_idx)

        unmatched_gt = [i for i in range(len(gt_boxes)) if i not in matched_gt]
        unmatched_yolo = [i for i in range(len(yolo_detections)) if i not in matched_yolo]

        return matches, unmatched_gt, unmatched_yolo

    def _print_yolo_evaluation_results(self, stats, iou_threshold):
        """Print YOLO evaluation results."""
        print("\n" + "=" * 80)
        print("YOLO DETECTION EVALUATION RESULTS")
        print("=" * 80)

        recall = stats['total_matches'] / stats['total_gt_boxes'] if stats['total_gt_boxes'] > 0 else 0
        precision = stats['total_matches'] / stats['total_yolo_detections'] if stats['total_yolo_detections'] > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        avg_iou = np.mean(stats['iou_scores']) if len(stats['iou_scores']) > 0 else 0

        print(f"\n--- Detection Quality ---")
        print(f"IoU Threshold: {iou_threshold}")
        print(f"Frames evaluated: {stats['total_frames']}")

        print(f"\n--- Counts ---")
        print(f"Ground Truth boxes: {stats['total_gt_boxes']}")
        print(f"YOLO detections: {stats['total_yolo_detections']}")
        print(f"Matched (True Positives): {stats['total_matches']}")
        print(f"False Negatives (missed GT): {stats['total_false_negatives']}")
        print(f"False Positives (extra detections): {stats['total_false_positives']}")

        print(f"\n--- Metrics ---")
        print(f"Recall (Detection Rate): {recall*100:.2f}%")
        print(f"  → {recall*100:.2f}% of GT boxes were detected by YOLO")
        print(f"Precision: {precision*100:.2f}%")
        print(f"  → {precision*100:.2f}% of YOLO detections matched GT boxes")
        print(f"F1-Score: {f1*100:.2f}%")
        print(f"Average IoU (for matches): {avg_iou:.3f}")

        print("\n" + "=" * 80)

    def _save_yolo_evaluation_results(self, stats, iou_threshold, output_file):
        """Save YOLO evaluation results to JSON."""
        recall = stats['total_matches'] / stats['total_gt_boxes'] if stats['total_gt_boxes'] > 0 else 0
        precision = stats['total_matches'] / stats['total_yolo_detections'] if stats['total_yolo_detections'] > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        avg_iou = np.mean(stats['iou_scores']) if len(stats['iou_scores']) > 0 else 0

        results = {
            'config': {
                'iou_threshold': iou_threshold
            },
            'summary': {
                'frames_evaluated': stats['total_frames'],
                'total_gt_boxes': stats['total_gt_boxes'],
                'total_yolo_detections': stats['total_yolo_detections'],
                'true_positives': stats['total_matches'],
                'false_negatives': stats['total_false_negatives'],
                'false_positives': stats['total_false_positives'],
                'recall': recall,
                'precision': precision,
                'f1_score': f1,
                'average_iou': avg_iou
            },
            'per_frame_stats': stats['per_frame_stats']
        }

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nYOLO evaluation results saved to: {output_file}")

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

  # Evaluate YOLO detection quality against MOT17 ground truth
  python run_system.py --frames-dir ./MOT17/train/MOT17-02/img1 \\
      --gt-file ./MOT17/train/MOT17-02/gt/gt.txt \\
      --output-dir ./yolo_eval_results \\
      --evaluate-yolo \\
      --iou-threshold 0.5 \\
      --person-model /path/to/osnet.pth \\
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
    parser.add_argument(
        '--evaluate-yolo',
        action='store_true',
        help='Enable YOLO detection evaluation against MOT17 ground truth'
    )
    parser.add_argument(
        '--gt-file',
        type=str,
        help='Path to MOT17 ground truth file (gt.txt) for YOLO evaluation'
    )
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.5,
        help='IoU threshold for detection matching (default: 0.5)'
    )

    args = parser.parse_args()

    
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

    
    if args.add_identity:
        
        system.add_identity_interactive(category=args.add_identity)
    elif args.frames_dir:
        
        if args.evaluate_yolo:
            if not args.gt_file:
                print("Error: --gt-file is required when --evaluate-yolo is enabled")
                parser.print_help()
                return

            system.evaluate_yolo_detection(
                frames_dir=args.frames_dir,
                gt_file=args.gt_file,
                output_dir=args.output_dir,
                iou_threshold=args.iou_threshold
            )
        else:
            
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
