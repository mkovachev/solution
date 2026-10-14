"""
Marker Detector — Computer Vision Recruitment Task

Implement the MarkerDetector class and the utility functions below.
See README.md for full task description.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple
from operator import itemgetter


class MarkerDetector:
    """
    Detects colored markers in images using classical computer vision techniques.

    Each detection is a dictionary with the following fields:
        - 'color':  str           — one of 'red', 'green', 'blue', 'yellow'
        - 'bbox':   (x, y, w, h) — bounding rectangle of the detected contour
        - 'center': (cx, cy)     — center coordinates of the bounding box
        - 'area':   float        — area of the detected contour

    Optionally, if you attempt the bonus task:
        - 'shape':  str          — one of 'circle', 'triangle', 'rectangle'
    """

    # Define HSV color ranges for each target color.
    # Each entry maps a color name to a list of (lower_bound, upper_bound) tuples.
    # Use np.array([H, S, V]) for bounds. OpenCV uses H: 0-179, S: 0-255, V: 0-255.
    #
    # Hint: Red wraps around the hue spectrum (both ~0-10 and ~170-179 are red),
    # so you will likely need TWO ranges for red.
    COLOR_RANGES = {
        # Example format:
        # "green": [(np.array([35, 80, 80]), np.array([85, 255, 255]))],
        "green": [(np.array([35, 80, 80]), np.array([85, 255, 255]))],
        "blue": [(np.array([100, 80, 80]), np.array([130, 255, 255]))],
        "yellow": [(np.array([20, 80, 80]), np.array([34, 255, 255]))],
        "red": [
            (np.array([0, 80, 80]), np.array([10, 255, 255])),
            (np.array([170, 80, 80]), np.array([179, 255, 255]))
        ]
    }

    # Minimum contour area to consider (filters noise)
    min_area = 500

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect colored markers in the given BGR image.

        Args:
            image: Input image in BGR format (as loaded by cv2.imread).

        Returns:
            A list of detection dictionaries, each containing:
            'color', 'bbox', 'center', and 'area' keys.
        """
        shape_tolerance = 1
        
        detections=[]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        for color, ranges in self.COLOR_RANGES.items():
            for lower, upper in ranges:
                mask=cv2.inRange(hsv, lower, upper)
                contours=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]

                for i in contours:
                    area=cv2.contourArea(i)
                    if area<self.min_area:
                        continue

                    approximation = cv2.approxPolyDP(i, shape_tolerance, True)
                    shape = None
                    if len(approximation) == 3:
                        shape = "triangle"
                    elif len(approximation) == 4:
                        shape = "rectangle"
                    elif len(approximation) > 4:
                        shape = "circle"
                    x,y,w,h=cv2.boundingRect(i)
                    center =(x+w//2, y+h//2)

                    detections.append({
                        "color":color,
                        "bbox":(x, y, w, h),
                        "center":center,
                        "area":area,
                        "shape":shape,
                    })
        
        return detections


def compute_iou(box_a: Tuple, box_b: Tuple) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Each box is represented as (x, y, w, h) where:
        - (x, y) is the top-left corner
        - (w, h) is the width and height

    Args:
        box_a: First bounding box as (x, y, w, h).
        box_b: Second bounding box as (x, y, w, h).

    Returns:
        IoU value as a float between 0.0 (no overlap) and 1.0 (perfect overlap).
    """
    intersected_y=max(0, min(box_a[1]+box_a[3], box_b[1]+box_b[3])-max(box_a[1],box_b[1]))
    intersected_x=max(0, min(box_a[0]+box_a[2], box_b[0]+box_b[2])-max(box_a[0],box_b[0]))
    intersected_area=intersected_x*intersected_y

    unioned_area=(box_a[2]*box_a[3])+(box_b[2]*box_b[3])-intersected_area

    if unioned_area == 0:
        return 0.0
    else:
        return intersected_area/unioned_area


def filter_detections(
    detections: List[Dict], iou_threshold: float = 0.5
) -> List[Dict]:
    """
    Filter overlapping detections using Non-Maximum Suppression (NMS).

    When two detections overlap (IoU > iou_threshold), keep the one with the
    larger area and discard the other.

    Args:
        detections: List of detection dictionaries (each must have 'bbox' and 'area').
        iou_threshold: IoU threshold above which two detections are considered overlapping.

    Returns:
        Filtered list of detections with overlapping duplicates removed.
    """
    kept_detections = []
    sorted_detections=sorted(detections, key=itemgetter("area"), reverse=True)
    for i in sorted_detections:
        keep = True
        for j in kept_detections:
            if compute_iou(i["bbox"], j["bbox"]) > iou_threshold:
                keep = False
                break
        if keep:
            kept_detections.append(i)

    return kept_detections