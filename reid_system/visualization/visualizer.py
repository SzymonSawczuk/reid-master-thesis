"""
Visualization Module

Handles visualization of detection and identification results.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class Visualizer:
    """
    Visualizer for detection and ReID results.
    """

    def __init__(
        self,
        font_scale: float = 0.6,
        thickness: int = 2,
        bbox_thickness: int = 2
    ):
        """
        Initialize visualizer.

        Args:
            font_scale: Font scale for text
            thickness: Text thickness
            bbox_thickness: Bounding box thickness
        """
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.thickness = thickness
        self.bbox_thickness = bbox_thickness

        
        self.colors = self._generate_colors(100)

        print("Visualizer initialized")

    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """
        Generate n distinct colors.

        Args:
            n: Number of colors to generate

        Returns:
            List of BGR color tuples
        """
        colors = []
        for i in range(n):
            hue = int(180 * i / n)
            color = cv2.cvtColor(
                np.uint8([[[hue, 255, 255]]]),
                cv2.COLOR_HSV2BGR
            )[0][0]
            colors.append(tuple(map(int, color)))
        return colors

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        category: str = 'person',
        show_detection_id: bool = True
    ) -> np.ndarray:
        """
        Draw detections on frame.

        Args:
            frame: Input frame
            detections: List of detections with bbox and detection_id
            category: Category label
            show_detection_id: Whether to show detection ID

        Returns:
            Frame with drawn detections
        """
        frame_vis = frame.copy()

        for det in detections:
            bbox = det['bbox']
            
            det_id = det.get('detection_id', )
            confidence = det.get('confidence', 1.0)

            
            color = self.colors[det_id % len(self.colors)] if det_id >= 0 else (0, 255, 0)

            
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, self.bbox_thickness)

            
            if show_detection_id and det_id >= 0:
                label = f"{category} #{det_id}"
            else:
                label = category

            label += f" {confidence:.2f}"

            (text_width, text_height), baseline = cv2.getTextSize(
                label, self.font, self.font_scale, self.thickness
            )
            cv2.rectangle(
                frame_vis,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )

            
            cv2.putText(
                frame_vis,
                label,
                (x1, y1 - baseline - 5),
                self.font,
                self.font_scale,
                (255, 255, 255),
                self.thickness
            )

        return frame_vis

    def draw_reid_results(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        reid_results: List[Optional[Dict]],
        category: str = 'person',
        confidence_threshold: float = 0.5
    ) -> np.ndarray:
        """
        Draw ReID results on frame.

        Args:
            frame: Input frame
            detections: List of detections
            reid_results: List of ReID results (one per detection)
            category: Category label
            confidence_threshold: Confidence threshold for showing identity

        Returns:
            Frame with drawn ReID results
        """
        frame_vis = frame.copy()

        
        assert len(detections) == len(reid_results), \
            f"Detections ({len(detections)}) and reid_results ({len(reid_results)}) length mismatch!"

        for i, (det, reid_result) in enumerate(zip(detections, reid_results)):
            bbox = det['bbox']
            det_id = det.get('detection_id')

            
            if reid_result is not None:
                identity_id = reid_result.get('identity_id', 'Unknown')
                identity_label = reid_result.get('label', identity_id)
                confidence = reid_result.get('similarity', 0.0)
                is_known = confidence >= confidence_threshold
            else:
                identity_id = 'Unknown'
                identity_label = 'Unknown'
                confidence = 0.0
                is_known = False

            
            if is_known:
                
                color_idx = hash(identity_id) % len(self.colors)
                color = self.colors[color_idx]
            else:
                
                color = (128, 128, 128)

            
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, self.bbox_thickness)

            
            if is_known:
                label = f"{identity_label}"
                conf_label = f"Conf: {confidence:.2f}"
            else:
                label = f"New {category}"
                conf_label = ""

            
            (text_width, text_height), baseline = cv2.getTextSize(
                label, self.font, self.font_scale, self.thickness
            )
            cv2.rectangle(
                frame_vis,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )

            
            cv2.putText(
                frame_vis,
                label,
                (x1, y1 - baseline - 5),
                self.font,
                self.font_scale,
                (255, 255, 255),
                self.thickness
            )

            
            if is_known and conf_label:
                cv2.putText(
                    frame_vis,
                    conf_label,
                    (x1, y2 + text_height + 5),
                    self.font,
                    self.font_scale * 0.8,
                    color,
                    self.thickness
                )

        return frame_vis

    def save_image(self, image: np.ndarray, path: str):
        """
        Save image to file.

        Args:
            image: Image to save
            path: Output path
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)

    def create_comparison_grid(
        self,
        query_image: np.ndarray,
        gallery_images: List[np.ndarray],
        labels: List[str],
        scores: List[float],
        max_cols: int = 5
    ) -> np.ndarray:
        """
        Create a comparison grid showing query and top matches.

        Args:
            query_image: Query image
            gallery_images: List of gallery images
            labels: List of labels for gallery images
            scores: List of similarity scores
            max_cols: Maximum number of columns

        Returns:
            Grid image
        """
        n_gallery = len(gallery_images)
        n_cols = min(n_gallery + 1, max_cols)
        n_rows = (n_gallery + n_cols) // n_cols

        
        target_h, target_w = 256, 128
        query_resized = cv2.resize(query_image, (target_w, target_h))

        gallery_resized = [
            cv2.resize(img, (target_w, target_h))
            for img in gallery_images
        ]

        
        grid_h = n_rows * (target_h + 40)
        grid_w = n_cols * (target_w + 10)
        grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

        
        y_offset = 0
        x_offset = 0
        grid[y_offset:y_offset + target_h, x_offset:x_offset + target_w] = query_resized

        
        cv2.putText(
            grid,
            "Query",
            (x_offset, y_offset + target_h + 25),
            self.font,
            self.font_scale,
            (0, 0, 255),
            self.thickness
        )

        
        for i, (img, label, score) in enumerate(zip(gallery_resized, labels, scores)):
            row = (i + 1) // n_cols
            col = (i + 1) % n_cols

            y_offset = row * (target_h + 40)
            x_offset = col * (target_w + 10)

            grid[y_offset:y_offset + target_h, x_offset:x_offset + target_w] = img

            
            text = f"{label[:15]}"
            score_text = f"{score:.2f}"

            cv2.putText(
                grid,
                text,
                (x_offset, y_offset + target_h + 20),
                self.font,
                self.font_scale * 0.7,
                (0, 0, 0),
                self.thickness
            )

            cv2.putText(
                grid,
                score_text,
                (x_offset, y_offset + target_h + 35),
                self.font,
                self.font_scale * 0.6,
                (0, 128, 0),
                self.thickness - 1
            )

        return grid
