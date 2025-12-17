#!/usr/bin/env python3
"""
Embryo region mapping module for stage 27 (and stage 22-23) embryos.

This module provides functions to map spark coordinates to anatomical regions
based on a reference embryo map with bounding boxes for different regions.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List, Dict
from pathlib import Path


# Region definitions based on stage 27 embryo map
# Regions are defined as bounding boxes in normalized coordinates (0-1 range)
# Format: (x_min, y_min, x_max, y_max) where (0,0) is top-left
# These are approximate coordinates based on the image description
# User should provide actual coordinates or we can extract from image

REGION_BOUNDING_BOXES = [
    {
        "name": "Tailbud",
        "bbox": (0.0, 0.5, 0.15, 0.7),  # Left side, lower-middle
    },
    {
        "name": "Tail",
        "bbox": (0.0, 0.3, 0.25, 0.5),  # Left side, upper-middle
    },
    {
        "name": "Gut primordia",
        "bbox": (0.15, 0.5, 0.55, 0.75),  # Large middle region, lower
    },
    {
        "name": "Trunk",
        "bbox": (0.25, 0.3, 0.55, 0.5),  # Middle region, upper
    },
    {
        "name": "Kidney",
        "bbox": (0.55, 0.25, 0.65, 0.4),  # Right side, upper-middle
    },
    {
        "name": "Otic Vesicle",
        "bbox": (0.65, 0.2, 0.75, 0.35),  # Right side, upper (labeled as "Ear")
    },
    {
        "name": "Neural Crest",
        "bbox": (0.55, 0.4, 0.75, 0.6),  # Right side, middle-lower
    },
    {
        "name": "Brain",
        "bbox": (0.7, 0.0, 0.95, 0.25),  # Right side, top
    },
    {
        "name": "Eye",
        "bbox": (0.75, 0.25, 0.95, 0.45),  # Right side, upper-middle
    },
    {
        "name": "Cement Gland",
        "bbox": (0.85, 0.6, 0.95, 0.75),  # Right side, lower
    },
    {
        "name": "Heart",
        "bbox": (0.55, 0.6, 0.75, 0.75),  # Right side, lower-middle
    },
]


def extract_bboxes_from_image(image_path: str) -> List[Dict]:
    """
    Extract bounding box coordinates from a map image with red rectangles.
    
    This function attempts to detect red bounding boxes in the image and
    extract their coordinates. Returns normalized coordinates (0-1 range).
    
    Args:
        image_path: Path to the map image file
        
    Returns:
        List of dicts with 'name' and 'bbox' keys, or None if extraction fails
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Detect red/magenta rectangles (bounding boxes)
    # Red range in HSV
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Find contours (rectangles)
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for contour in contours:
        x, y, w_box, h_box = cv2.boundingRect(contour)
        # Normalize coordinates
        x_min = x / w
        y_min = y / h
        x_max = (x + w_box) / w
        y_max = (y + h_box) / h
        bboxes.append({
            "name": "Unknown",  # Would need OCR or manual labeling
            "bbox": (x_min, y_min, x_max, y_max)
        })
    
    return bboxes if bboxes else None


def load_region_map(map_path: Optional[str] = None) -> List[Dict]:
    """
    Load region bounding boxes from image or use defaults.
    
    Args:
        map_path: Optional path to map image. If None, uses default coordinates.
        
    Returns:
        List of region dicts with 'name' and 'bbox' keys
    """
    if map_path and Path(map_path).exists():
        extracted = extract_bboxes_from_image(map_path)
        if extracted:
            return extracted
    
    # Use default coordinates
    return REGION_BOUNDING_BOXES.copy()


def create_embryo_transform(
    embryo_head: Tuple[float, float],
    embryo_tail: Tuple[float, float],
    map_head: Tuple[float, float] = (0.95, 0.1),  # Right side, top (Brain region)
    map_tail: Tuple[float, float] = (0.1, 0.5),    # Left side, middle (Tail region)
) -> Dict:
    """
    Create transformation parameters to map embryo coordinates to map coordinates.
    
    Args:
        embryo_head: (x, y) coordinates of embryo head
        embryo_tail: (x, y) coordinates of embryo tail
        map_head: (x, y) normalized coordinates of head in map (default: right side)
        map_tail: (x, y) normalized coordinates of tail in map (default: left side)
        
    Returns:
        Dict with transformation parameters:
        - 'scale': scaling factor
        - 'rotation': rotation angle in radians
        - 'translation': (dx, dy) translation vector
        - 'map_head': map head position
        - 'map_tail': map tail position
    """
    # Calculate embryo axis
    emb_axis = np.array(embryo_tail) - np.array(embryo_head)
    emb_length = np.linalg.norm(emb_axis)
    
    if emb_length < 1e-6:
        # Invalid axis, return identity transform
        return {
            'scale': 1.0,
            'rotation': 0.0,
            'translation': (0.0, 0.0),
            'map_head': map_head,
            'map_tail': map_tail,
        }
    
    emb_axis_norm = emb_axis / emb_length
    
    # Calculate map axis
    map_axis = np.array(map_tail) - np.array(map_head)
    map_length = np.linalg.norm(map_axis)
    
    if map_length < 1e-6:
        map_axis_norm = np.array([-1.0, 0.0])  # Default: left-to-right
    else:
        map_axis_norm = map_axis / map_length
    
    # Calculate scale
    scale = map_length / emb_length if emb_length > 0 else 1.0
    
    # Calculate rotation angle
    # Angle from embryo axis to map axis
    emb_angle = np.arctan2(emb_axis_norm[1], emb_axis_norm[0])
    map_angle = np.arctan2(map_axis_norm[1], map_axis_norm[0])
    rotation = map_angle - emb_angle
    
    # Translation: align heads after rotation and scaling
    # First rotate embryo head, then scale, then translate to map head
    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)
    rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    
    emb_head_rotated = rot_matrix @ np.array(embryo_head)
    emb_head_scaled = emb_head_rotated * scale
    translation = np.array(map_head) - emb_head_scaled
    
    return {
        'scale': scale,
        'rotation': rotation,
        'translation': tuple(translation),
        'map_head': map_head,
        'map_tail': map_tail,
    }


def transform_point_to_map(
    x: float,
    y: float,
    transform: Dict
) -> Tuple[float, float]:
    """
    Transform a point from embryo coordinates to map coordinates.
    
    Args:
        x, y: Point coordinates in embryo space
        transform: Transformation dict from create_embryo_transform()
        
    Returns:
        (x_map, y_map) coordinates in normalized map space (0-1 range)
    """
    point = np.array([x, y])
    
    # Apply rotation
    cos_r = np.cos(transform['rotation'])
    sin_r = np.sin(transform['rotation'])
    rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    point_rotated = rot_matrix @ point
    
    # Apply scaling
    point_scaled = point_rotated * transform['scale']
    
    # Apply translation
    point_translated = point_scaled + np.array(transform['translation'])
    
    return tuple(point_translated)


def get_region_for_point(
    x: float,
    y: float,
    transform: Dict,
    regions: Optional[List[Dict]] = None
) -> str:
    """
    Determine which anatomical region a point belongs to.
    
    Args:
        x, y: Point coordinates in embryo space
        transform: Transformation dict from create_embryo_transform()
        regions: Optional list of region dicts. If None, uses default regions.
        
    Returns:
        Region name string, or "unknown" if point doesn't match any region
    """
    if regions is None:
        regions = REGION_BOUNDING_BOXES
    
    # Transform point to map coordinates
    x_map, y_map = transform_point_to_map(x, y, transform)
    
    # Check which bounding box contains the point
    for region in regions:
        x_min, y_min, x_max, y_max = region['bbox']
        if x_min <= x_map <= x_max and y_min <= y_map <= y_max:
            return region['name']
    
    return "unknown"


def get_region_for_points_batch(
    points: np.ndarray,
    transform: Dict,
    regions: Optional[List[Dict]] = None
) -> np.ndarray:
    """
    Determine regions for multiple points efficiently.
    
    Args:
        points: Nx2 array of (x, y) coordinates in embryo space
        transform: Transformation dict from create_embryo_transform()
        regions: Optional list of region dicts. If None, uses default regions.
        
    Returns:
        Array of region name strings (or "unknown")
    """
    if regions is None:
        regions = REGION_BOUNDING_BOXES
    
    # Transform all points to map coordinates
    n_points = len(points)
    regions_out = np.full(n_points, "unknown", dtype=object)
    
    # Apply transformation
    cos_r = np.cos(transform['rotation'])
    sin_r = np.sin(transform['rotation'])
    rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    translation = np.array(transform['translation'])
    scale = transform['scale']
    
    # Transform all points
    points_rotated = (rot_matrix @ points.T).T
    points_scaled = points_rotated * scale
    points_transformed = points_scaled + translation
    
    # Check each point against each region
    for i, (x_map, y_map) in enumerate(points_transformed):
        for region in regions:
            x_min, y_min, x_max, y_max = region['bbox']
            if x_min <= x_map <= x_max and y_min <= y_map <= y_max:
                regions_out[i] = region['name']
                break
    
    return regions_out
