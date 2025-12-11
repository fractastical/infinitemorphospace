#!/usr/bin/env python3
"""
Generate visualization and summary table of embryo detection results.

This script:
1. Creates visualization images showing embryo outlines, head/tail labels, and poke locations
2. Generates a markdown table summarizing all detections for manual fact-checking
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Polygon
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from pathlib import Path
import argparse
import re
from collections import defaultdict
import cv2
import tifffile as tiff
import os

try:
    from scipy.spatial import ConvexHull
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, will use bounding boxes instead of convex hulls")


def extract_folder_and_video(filename):
    """
    Extract folder name and video identifier from filename.
    Example: "1/B1 - Substack (1-301).tif" -> ("1", "B1 - Substack (1-301).tif")
    Example: "2/Video1.tif" -> ("2", "Video1.tif")
    """
    if pd.isna(filename):
        return None, None
    
    # Remove page numbers first
    filename_clean = re.sub(r' \(page \d+\)', '', str(filename))
    
    # Split by path separator
    parts = filename_clean.split('/')
    if len(parts) > 1:
        folder = parts[0]
        video_name = '/'.join(parts[1:])  # Keep full path including extension
    else:
        # No folder, just filename
        folder = "unknown"
        video_name = filename_clean
    
    return folder, video_name


def detect_embryo_from_tiff(tiff_path, embryo_id=None):
    """
    Detect embryo boundaries directly from TIFF file using the same method as the parser.
    Returns dict with 'contour', 'mask', 'head', 'tail', 'centroid' for each embryo.
    
    Args:
        tiff_path: Path to TIFF file
        embryo_id: Optional, 'A' or 'B' to get specific embryo
    
    Returns:
        Dict mapping embryo_id to detection data, or single dict if embryo_id specified
    """
    try:
        # Read first page of TIFF
        with tiff.TiffFile(tiff_path) as tif:
            if len(tif.pages) == 0:
                return None
            
            # Try to read as 16-bit first
            try:
                raw_16bit = tif.pages[0].asarray()
                if raw_16bit.ndim == 2:
                    gray = raw_16bit.astype(np.float32)
                else:
                    # Multi-channel, convert to grayscale
                    gray = cv2.cvtColor(raw_16bit, cv2.COLOR_RGB2GRAY).astype(np.float32)
            except:
                # Fallback to BGR
                img = tif.pages[0].asarray()
                if img.ndim == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
                else:
                    gray = img.astype(np.float32)
    except Exception as e:
        print(f"    ⚠ Could not read TIFF {tiff_path}: {e}")
        return None
    
    h, w = gray.shape
    
    # Apply same detection method as parser
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    flat_intensities = blur.flatten()
    
    # Calculate three threshold levels: old (restrictive), intermediate (better head capture), new (inclusive)
    background_percentile_old = 10
    background_percentile_intermediate = 7
    background_percentile_new = 5
    background_threshold_old = np.percentile(flat_intensities, background_percentile_old)
    background_threshold_intermediate = np.percentile(flat_intensities, background_percentile_intermediate)
    background_threshold_new = np.percentile(flat_intensities, background_percentile_new)
    
    median_intensity = np.median(flat_intensities)
    p25 = np.percentile(flat_intensities, 25)
    p20 = np.percentile(flat_intensities, 20)
    p15 = np.percentile(flat_intensities, 15)
    mean_intensity = flat_intensities.mean()
    
    # Old (restrictive) threshold - for inner outline
    embryo_threshold_old = max(median_intensity * 0.7, p25, mean_intensity * 0.8, background_threshold_old * 1.2)
    
    # Intermediate threshold - for middle outline (better head capture)
    # Targets "background of the embryos" - tissue that's clearly part of embryo but lighter than core
    embryo_threshold_intermediate = max(median_intensity * 0.65, p20, mean_intensity * 0.75, background_threshold_intermediate * 1.15)
    
    # New (inclusive) threshold - for outer outline
    embryo_threshold_new = max(median_intensity * 0.6, p25 * 0.9, p15, mean_intensity * 0.7, background_threshold_new * 1.1)
    
    # Create three masks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Inner mask (old, restrictive)
    embryo_mask_old = (blur >= embryo_threshold_old).astype(np.uint8) * 255
    embryo_mask_old = cv2.morphologyEx(embryo_mask_old, cv2.MORPH_CLOSE, kernel, iterations=2)
    embryo_mask_old = cv2.morphologyEx(embryo_mask_old, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Intermediate mask (better head capture)
    embryo_mask_intermediate = (blur >= embryo_threshold_intermediate).astype(np.uint8) * 255
    embryo_mask_intermediate = cv2.morphologyEx(embryo_mask_intermediate, cv2.MORPH_CLOSE, kernel, iterations=2)  # 2-3 iterations
    embryo_mask_intermediate = cv2.morphologyEx(embryo_mask_intermediate, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Outer mask (new, inclusive)
    embryo_mask_new = (blur >= embryo_threshold_new).astype(np.uint8) * 255
    embryo_mask_new = cv2.morphologyEx(embryo_mask_new, cv2.MORPH_CLOSE, kernel, iterations=3)
    embryo_mask_new = cv2.morphologyEx(embryo_mask_new, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Use the new mask for detection, but we'll draw all three
    embryo_mask = embryo_mask_new
    
    # Find contours for all three masks
    contours_new, _ = cv2.findContours(embryo_mask_new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_intermediate, _ = cv2.findContours(embryo_mask_intermediate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_old, _ = cv2.findContours(embryo_mask_old, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by minimum area
    min_area = 0.003 * h * w
    emb_contours_new = [c for c in contours_new if cv2.contourArea(c) >= min_area]
    emb_contours_intermediate = [c for c in contours_intermediate if cv2.contourArea(c) >= min_area]
    emb_contours_old = [c for c in contours_old if cv2.contourArea(c) >= min_area]
    
    if not emb_contours_new:
        return None
    
    # Score contours based on "good" criteria: ~80% horizontal space, clear head taper
    def score_contour(contour):
        """Score contour based on how well it matches ideal embryo shape."""
        contour_points = contour.reshape(-1, 2).astype(np.float32)
        x_coords = contour_points[:, 0]
        y_coords = contour_points[:, 1]
        
        # Horizontal extent (should be ~80% of image width)
        x_range = x_coords.max() - x_coords.min()
        x_extent_ratio = x_range / w
        ideal_extent = 0.8
        extent_score = max(0, 1.0 - abs(x_extent_ratio - ideal_extent) / ideal_extent)
        
        # Check for clear head taper (wider at one end)
        try:
            mean, eigenvectors, eigenvalues = cv2.PCACompute2(contour_points, mean=None)
            v = eigenvectors[0]
            v_norm = v / (np.linalg.norm(v) + 1e-9)
            u_perp = np.array([-v_norm[1], v_norm[0]])
            
            proj = np.dot(contour_points - mean.reshape(1, 2), v_norm.reshape(2, 1)).ravel()
            proj_normalized = (proj - proj.min()) / (proj.max() - proj.min() + 1e-9)
            
            # Calculate width at ends
            end1_mask = proj_normalized < 0.2
            end2_mask = proj_normalized > 0.8
            if end1_mask.sum() > 0 and end2_mask.sum() > 0:
                end1_points = contour_points[end1_mask]
                end2_points = contour_points[end2_mask]
                deltas1 = end1_points - mean.reshape(1, 2)
                deltas2 = end2_points - mean.reshape(1, 2)
                widths1 = np.abs(deltas1 @ u_perp.reshape(2, 1)).ravel()
                widths2 = np.abs(deltas2 @ u_perp.reshape(2, 1)).ravel()
                width1 = np.percentile(widths1, 75) * 2 if len(widths1) > 0 else 0
                width2 = np.percentile(widths2, 75) * 2 if len(widths2) > 0 else 0
                
                # Head should be wider (taper score)
                if width1 > 0 and width2 > 0:
                    width_ratio = max(width1, width2) / min(width1, width2)
                    taper_score = min(width_ratio / 1.5, 1.0)  # Prefer ratio > 1.5
                else:
                    taper_score = 0.5
            else:
                taper_score = 0.5
        except:
            taper_score = 0.5
        
        # Combined score (weight extent more)
        total_score = extent_score * 0.6 + taper_score * 0.4
        return total_score
    
    # Score and sort contours - prefer "good" sized ones
    emb_contours_with_scores = [(c, score_contour(c)) for c in emb_contours_new]
    emb_contours_with_scores.sort(key=lambda x: x[1], reverse=True)
    emb_contours = [c for c, _ in emb_contours_with_scores]
    
    # Store old and intermediate contours for triple outline drawing (match by centroid proximity)
    emb_contours_old_sorted = sorted(emb_contours_old, key=lambda c: cv2.moments(c)["m10"] / (cv2.moments(c)["m00"] + 1e-9))
    emb_contours_intermediate_sorted = sorted(emb_contours_intermediate, key=lambda c: cv2.moments(c)["m10"] / (cv2.moments(c)["m00"] + 1e-9))
    
    # Detect and split connected embryos (figure-eight shapes)
    def split_connected_embryos(contour):
        """Detect if contour represents two connected embryos and split them."""
        contour_points = contour.reshape(-1, 2).astype(np.float32)
        area = cv2.contourArea(contour)
        
        # More aggressive thresholds for detecting connected embryos
        # Typical single embryo area is roughly 0.01-0.02 of image
        typical_embryo_area = 0.015 * h * w  # Typical single embryo area
        
        # Check multiple criteria for connected embryos:
        # 1. Area is significantly larger than typical (more aggressive threshold)
        # 2. Very elongated aspect ratio (connected embryos are often dumbbell-shaped)
        # 3. Has a constriction in the middle
        
        # Calculate bounding box and aspect ratio
        x_coords = contour_points[:, 0]
        y_coords = contour_points[:, 1]
        width = x_coords.max() - x_coords.min()
        height = y_coords.max() - y_coords.min()
        aspect_ratio = max(width, height) / (min(width, height) + 1e-9)
        
        # More aggressive: if area > 1.3x typical OR very elongated (>3:1) OR very large (>0.03 of image) OR spans >70% width
        is_large = area > typical_embryo_area * 1.3  # Lowered from 1.5
        is_very_large = area > 0.03 * h * w
        is_very_elongated = aspect_ratio > 3.0
        spans_wide = width > w * 0.7  # Spans >70% of image width
        
        # If none of these criteria, likely a single embryo
        if not (is_large or is_very_large or is_very_elongated or spans_wide):
            return [contour]  # Single embryo, no split needed
        
        # Calculate width profile along the contour
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(contour_points, mean=None)
        v = eigenvectors[0]  # principal axis
        v_norm = v / (np.linalg.norm(v) + 1e-9)
        u_perp = np.array([-v_norm[1], v_norm[0]])
        
        # Project all points onto principal axis
        proj = np.dot(contour_points - mean.reshape(1, 2), v_norm.reshape(2, 1)).ravel()
        proj_normalized = (proj - proj.min()) / (proj.max() - proj.min() + 1e-9)
        
        # Calculate width at each point along the axis
        widths = []
        n_bins = 50
        bin_edges = np.linspace(0, 1, n_bins + 1)
        
        for i in range(n_bins):
            bin_mask = (proj_normalized >= bin_edges[i]) & (proj_normalized < bin_edges[i+1])
            if i == n_bins - 1:  # Include last edge
                bin_mask = (proj_normalized >= bin_edges[i]) & (proj_normalized <= bin_edges[i+1])
            
            if bin_mask.sum() > 0:
                bin_points = contour_points[bin_mask]
                deltas = bin_points - mean.reshape(1, 2)
                dists = np.abs(deltas @ u_perp.reshape(2, 1)).ravel()
                widths.append(np.percentile(dists, 75) * 2)  # Approximate width
            else:
                widths.append(0)
        
        widths = np.array(widths)
        
        if not HAS_SCIPY:
            return [contour]  # Can't split without scipy
        
        # Find two head regions (local maxima in width)
        # Smooth the width profile
        widths_smooth = gaussian_filter1d(widths, sigma=2)
        
        # Find peaks (head regions) - use lower threshold to catch more cases
        median_width = np.median(widths_smooth[widths_smooth > 0])
        peak_height_threshold = max(median_width * 0.7, np.percentile(widths_smooth, 40))
        
        peaks, properties = find_peaks(widths_smooth, height=peak_height_threshold, 
                                       distance=max(n_bins // 6, 5))  # Allow closer peaks
        
        # If we have 2+ peaks, use the two largest
        if len(peaks) >= 2:
            # Sort peaks by height and take top 2
            peak_heights = widths_smooth[peaks]
            top2_idx = np.argsort(peak_heights)[-2:]
            peaks = peaks[top2_idx]
        elif len(peaks) == 1:
            # Only one peak found - might still be connected if very large
            # Try to find a second peak with lower threshold
            peaks2, _ = find_peaks(widths_smooth, height=median_width * 0.5, distance=n_bins // 8)
            if len(peaks2) >= 2:
                peak_heights2 = widths_smooth[peaks2]
                top2_idx = np.argsort(peak_heights2)[-2:]
                peaks = peaks2[top2_idx]
            else:
                # If very large area, force a split at the middle
                if is_very_large or (is_large and aspect_ratio > 2.5):
                    # Force split at the narrowest point
                    min_width_idx = np.argmin(widths_smooth)
                    # Create two peaks around this point
                    peaks = np.array([min_width_idx - n_bins // 4, min_width_idx + n_bins // 4])
                    peaks = np.clip(peaks, 0, n_bins - 1)
                else:
                    return [contour]  # Can't confidently split
        else:
            # No clear peaks - if very large, try to split anyway
            if is_very_large or (is_large and aspect_ratio > 2.0):
                # Split at the narrowest point or middle
                min_width_idx = np.argmin(widths_smooth)
                # Create two peaks around the narrowest point
                peak1 = max(0, min_width_idx - n_bins // 4)
                peak2 = min(n_bins - 1, min_width_idx + n_bins // 4)
                peaks = np.array([peak1, peak2])
            else:
                return [contour]  # Can't confidently split
        
        # Find the constriction (minimum width between the two peaks)
        peak1_idx, peak2_idx = peaks[0], peaks[1]
        if peak1_idx > peak2_idx:
            peak1_idx, peak2_idx = peak2_idx, peak1_idx
        
        # Find minimum width in the region between peaks
        between_region = widths_smooth[peak1_idx:peak2_idx]
        if len(between_region) == 0:
            return [contour]
        
        min_width_idx = np.argmin(between_region) + peak1_idx
        split_proj_value = bin_edges[min_width_idx]
        
        # Find the actual split point on the contour
        # Use the point closest to the split projection value
        split_mask = np.abs(proj_normalized - split_proj_value) < 0.05
        if split_mask.sum() == 0:
            return [contour]
        
        split_points = contour_points[split_mask]
        # Use the point closest to the mean (center of constriction)
        split_dists = np.linalg.norm(split_points - mean.reshape(1, 2), axis=1)
        split_point_idx = np.where(split_mask)[0][np.argmin(split_dists)]
        
        # Split contour into two parts
        # Find the contour point closest to the split point
        contour_array = contour.reshape(-1, 2)
        split_pt = contour_points[split_point_idx]
        dists_to_split = np.linalg.norm(contour_array - split_pt.reshape(1, 2), axis=1)
        actual_split_idx = np.argmin(dists_to_split)
        
        # Split contour into two parts at the constriction
        # Use a more robust splitting method
        contour_array = contour.reshape(-1, 2)
        n_points = len(contour_array)  # Define early for use in all branches
        
        # Find the two points on the contour closest to the split projection value
        # This gives us two points on opposite sides of the constriction
        split_tolerance = 0.08  # Wider tolerance to find points on both sides
        split_mask = np.abs(proj_normalized - split_proj_value) < split_tolerance
        
        if split_mask.sum() < 2:
            # Fallback: use the single closest point and find opposite side
            closest_idx = np.argmin(np.abs(proj_normalized - split_proj_value))
            # Find point on opposite side (180 degrees around contour)
            opposite_idx = (closest_idx + n_points // 2) % n_points
            split_indices = [closest_idx, opposite_idx]
        else:
            # Find the two points that are furthest apart among the candidates
            split_candidates = np.where(split_mask)[0]
            if len(split_candidates) >= 2:
                # Find pair with maximum distance
                max_dist = 0
                split_indices = [split_candidates[0], split_candidates[1]]
                for i in range(len(split_candidates)):
                    for j in range(i+1, len(split_candidates)):
                        idx1, idx2 = split_candidates[i], split_candidates[j]
                        # Distance along contour (accounting for wrap-around)
                        dist1 = abs(idx2 - idx1)
                        dist2 = n_points - dist1
                        dist = min(dist1, dist2)
                        if dist > max_dist:
                            max_dist = dist
                            split_indices = [idx1, idx2]
            else:
                split_indices = [split_candidates[0], (split_candidates[0] + n_points // 2) % n_points]
        
        split_idx1, split_idx2 = sorted(split_indices)
        
        # Create two separate contours
        # Part 1: from split_idx1 to split_idx2
        # Part 2: from split_idx2 back to split_idx1 (wrapping around)
        part1_indices = list(range(split_idx1, split_idx2 + 1))
        part2_indices = list(range(split_idx2, n_points)) + list(range(0, split_idx1 + 1))
        
        part1_points = contour_array[part1_indices]
        part2_points = contour_array[part2_indices]
        
        # Close the contours
        if len(part1_points) > 2:
            if not np.array_equal(part1_points[0], part1_points[-1]):
                part1_points = np.vstack([part1_points, part1_points[0:1]])
        if len(part2_points) > 2:
            if not np.array_equal(part2_points[0], part2_points[-1]):
                part2_points = np.vstack([part2_points, part2_points[0:1]])
        
        # Ensure minimum points for valid contour
        if len(part1_points) < 3 or len(part2_points) < 3:
            # Fallback: simple geometric split at middle of principal axis
            if is_very_large or (is_large and aspect_ratio > 2.0) or spans_wide:
                # Split at the middle point along principal axis
                mid_proj = (proj.min() + proj.max()) / 2
                mid_mask = proj <= mid_proj
                
                if mid_mask.sum() > 3 and (~mid_mask).sum() > 3:
                    part1_simple = contour_points[mid_mask]
                    part2_simple = contour_points[~mid_mask]
                    
                    # Close them
                    if len(part1_simple) > 2:
                        part1_simple = np.vstack([part1_simple, part1_simple[0:1]])
                    if len(part2_simple) > 2:
                        part2_simple = np.vstack([part2_simple, part2_simple[0:1]])
                    
                    if len(part1_simple) >= 3 and len(part2_simple) >= 3:
                        return [part1_simple.reshape(-1, 1, 2).astype(np.int32),
                                part2_simple.reshape(-1, 1, 2).astype(np.int32)]
            
            # Split failed - if very large, log warning
            if is_very_large or spans_wide:
                print(f"    ⚠ Warning: Large contour (area={area:.0f}, width={width:.0f}px) could not be split - may need manual review")
            return [contour]  # Split failed, return original
        
        return [part1_points.reshape(-1, 1, 2).astype(np.int32), 
                part2_points.reshape(-1, 1, 2).astype(np.int32)]
    
    # Split any connected embryos
    split_contours = []
    for contour in emb_contours:
        original_area = cv2.contourArea(contour)
        split_result = split_connected_embryos(contour)
        if len(split_result) > 1:
            # Splitting occurred
            total_split_area = sum(cv2.contourArea(c) for c in split_result)
            print(f"    → Split large contour (area={original_area:.0f}) into {len(split_result)} parts")
        split_contours.extend(split_result)
    
    emb_contours = split_contours
    
    # Sort by centroid x position (leftmost = A, rightmost = B)
    def contour_cx(c):
        M = cv2.moments(c)
        if M["m00"] == 0:
            return 0
        return M["m10"] / M["m00"]
    
    emb_contours.sort(key=contour_cx)
    
    # Single embryo detection: check if we should treat as single embryo
    if len(emb_contours) == 1:
        # Only one contour - single embryo (no A/B assignment needed)
        # But we'll still process it for visualization
        emb_contours = emb_contours[:1]
    elif len(emb_contours) >= 2:
        # Check if one is tiny (likely artifact)
        areas = [cv2.contourArea(c) for c in emb_contours[:2]]
        total_area = sum(areas)
        min_area_ratio = min(areas) / total_area if total_area > 0 else 0
        
        if min_area_ratio < 0.05:  # One is <5% of total - likely artifact
            # Keep only the larger one
            larger_idx = 0 if areas[0] > areas[1] else 1
            emb_contours = [emb_contours[larger_idx]]
            print(f"    → Filtered out tiny contour (artifact) - using single embryo")
        else:
            # Check if centroids are very close (likely same embryo)
            M1 = cv2.moments(emb_contours[0])
            M2 = cv2.moments(emb_contours[1])
            if M1["m00"] > 0 and M2["m00"] > 0:
                cx1 = M1["m10"] / M1["m00"]
                cx2 = M2["m10"] / M2["m00"]
                centroid_sep = abs(cx2 - cx1)
                
                if centroid_sep < w * 0.1:  # Centroids <10% image width apart
                    # Very close - likely same embryo, use the larger one
                    larger_idx = 0 if areas[0] > areas[1] else 1
                    emb_contours = [emb_contours[larger_idx]]
                    print(f"    → Centroids very close ({centroid_sep:.1f}px) - treating as single embryo")
    
    # Process embryos (1 or 2)
    results = {}
    labels = ["A", "B"]
    
    for idx, contour in enumerate(emb_contours[:2]):
        label = labels[idx] if idx < len(labels) else f"E{idx}"
        
        # Get contour points
        contour_points = contour.reshape(-1, 2).astype(np.float32)
        
        # Calculate PCA for head-tail axis
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(contour_points, mean=None)
        v = eigenvectors[0]  # principal axis
        v_norm = v / (np.linalg.norm(v) + 1e-9)
        
        proj = np.dot(contour_points - mean.reshape(1, 2), v_norm.reshape(2, 1)).ravel()
        min_idx = np.argmin(proj)
        max_idx = np.argmax(proj)
        end1 = contour_points[min_idx]
        end2 = contour_points[max_idx]
        
        # Calculate centroid
        M = cv2.moments(contour)
        if M["m00"] == 0:
            cx, cy = float(mean[0, 0]), float(mean[0, 1])
        else:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        
        # Determine head vs tail using width analysis (same as parser)
        proj_normalized = (proj - proj.min()) / (proj.max() - proj.min() + 1e-9)
        end1_region_mask = (proj_normalized < 0.2)
        end2_region_mask = (proj_normalized > 0.8)
        end1_region = contour_points[end1_region_mask]
        end2_region = contour_points[end2_region_mask]
        
        if len(end1_region) > 0 and len(end2_region) > 0:
            u_perp = np.array([-v_norm[1], v_norm[0]])
            # Avoid division by zero or invalid values
            if np.any(np.isnan(u_perp)) or np.any(np.isinf(u_perp)):
                head = end1
                tail = end2
            else:
                delta1 = end1_region - mean.reshape(1, 2)
                delta2 = end2_region - mean.reshape(1, 2)
                try:
                    dist1 = np.abs(delta1 @ u_perp.reshape(2, 1)).ravel()
                    dist2 = np.abs(delta2 @ u_perp.reshape(2, 1)).ravel()
                    # Filter out invalid values
                    dist1 = dist1[np.isfinite(dist1)]
                    dist2 = dist2[np.isfinite(dist2)]
                    
                    avg_width1 = dist1.mean() if len(dist1) > 0 else 0
                    avg_width2 = dist2.mean() if len(dist2) > 0 else 0
                    
                    width_diff_ratio = abs(avg_width1 - avg_width2) / (max(avg_width1, avg_width2) + 1e-9)
                    if width_diff_ratio > 0.1 and np.isfinite(width_diff_ratio):
                        if avg_width1 > avg_width2:
                            head = end1
                            tail = end2
                        else:
                            head = end2
                            tail = end1
                    else:
                        head = end1
                        tail = end2
                except:
                    head = end1
                    tail = end2
        else:
            head = end1
            tail = end2
        
        # Find matching old and intermediate contours (by centroid proximity)
        old_contour = None
        intermediate_contour = None
        
        if len(emb_contours_old_sorted) > idx:
            # Try to match old contour by centroid
            old_candidate = emb_contours_old_sorted[idx]
            M_old = cv2.moments(old_candidate)
            if M_old["m00"] > 0:
                old_cx = M_old["m10"] / M_old["m00"]
                old_cy = M_old["m01"] / M_old["m00"]
                dist = np.sqrt((cx - old_cx)**2 + (cy - old_cy)**2)
                if dist < 100:  # Close enough to be the same embryo
                    old_contour = old_candidate.reshape(-1, 2).astype(np.float32)
        
        if len(emb_contours_intermediate_sorted) > idx:
            # Try to match intermediate contour by centroid
            intermediate_candidate = emb_contours_intermediate_sorted[idx]
            M_int = cv2.moments(intermediate_candidate)
            if M_int["m00"] > 0:
                int_cx = M_int["m10"] / M_int["m00"]
                int_cy = M_int["m01"] / M_int["m00"]
                dist = np.sqrt((cx - int_cx)**2 + (cy - int_cy)**2)
                if dist < 100:  # Close enough to be the same embryo
                    intermediate_contour = intermediate_candidate.reshape(-1, 2).astype(np.float32)
        
        results[label] = {
            'contour': contour_points,  # New (inclusive) contour
            'contour_intermediate': intermediate_contour,  # Intermediate contour for middle outline
            'contour_old': old_contour,  # Old (restrictive) contour for inner outline
            'head': (float(head[0]), float(head[1])),
            'tail': (float(tail[0]), float(tail[1])),
            'centroid': (float(cx), float(cy)),
            'split_info': None  # Will be set if this was split from a connected pair
        }
    
    # If we have 2 embryos, validate and reassign if necessary
    if len(results) == 2:
        emb_A = results.get('A')
        emb_B = results.get('B')
        if emb_A and emb_B:
            contour_A = emb_A['contour']
            contour_B = emb_B['contour']
            
            # Check for overlap in x-coordinates
            x_A_min, x_A_max = contour_A[:, 0].min(), contour_A[:, 0].max()
            x_B_min, x_B_max = contour_B[:, 0].min(), contour_B[:, 0].max()
            
            # Validation: Check if assignment is correct
            # A should be left of B (A's centroid < B's centroid, A's rightmost < B's leftmost ideally)
            centroid_A = emb_A['centroid']
            centroid_B = emb_B['centroid']
            
            needs_reassignment = False
            
            # STRICT CHECK: A's rightmost should be < B's leftmost (or at least A's centroid < B's centroid)
            if x_A_max > x_B_min:
                # They overlap or A extends into B's region
                if centroid_A[0] > centroid_B[0]:
                    # A's centroid is to the right - definitely wrong
                    needs_reassignment = True
                    print(f"    → Reassigning: A centroid ({centroid_A[0]:.1f}) > B centroid ({centroid_B[0]:.1f})")
                elif x_A_max > x_B_max:
                    # A extends further right than B - wrong assignment
                    needs_reassignment = True
                    print(f"    → Reassigning: A extends further right than B")
                elif (x_A_max - x_B_min) > (x_B_max - x_A_min):
                    # A overlaps significantly with B
                    needs_reassignment = True
                    print(f"    → Reassigning: A overlaps B significantly")
            
            # Also check if centroids are wrong even without overlap
            if not needs_reassignment and centroid_A[0] > centroid_B[0]:
                needs_reassignment = True
                print(f"    → Reassigning: A centroid ({centroid_A[0]:.1f}) > B centroid ({centroid_B[0]:.1f})")
            
            # Reassign if needed
            if needs_reassignment:
                # Swap A and B
                results['A'], results['B'] = results['B'], results['A']
                emb_A, emb_B = results['A'], results['B']
                contour_A, contour_B = emb_A['contour'], emb_B['contour']
                x_A_min, x_A_max, x_B_min, x_B_max = x_B_min, x_B_max, x_A_min, x_A_max
                centroid_A, centroid_B = emb_B['centroid'], emb_A['centroid']
                print(f"    ✓ Reassigned: Leftmost → A, Rightmost → B")
            
            # Check if they overlap
            overlap = (x_A_max > x_B_min) and (x_B_max > x_A_min)
            
            # ALWAYS enforce separation when we have 2 embryos
            # Find closest points between contours
            dists = np.linalg.norm(contour_A[:, np.newaxis] - contour_B, axis=2)
            min_idx_A, min_idx_B = np.unravel_index(np.argmin(dists), dists.shape)
            closest_A_pt = contour_A[min_idx_A]
            closest_B_pt = contour_B[min_idx_B]
            
            # Calculate split_x: ALWAYS ensure A is left of B
            # If they overlap, split at the midpoint of overlap region
            # If they don't overlap, split at midpoint between boundaries
            # CRITICAL: After reassignment, x_A_max should be <= x_B_min ideally, but if they overlap, split in the middle
            if x_A_max > x_B_min:
                # They overlap - split at midpoint of overlap region
                overlap_start = max(x_A_min, x_B_min)
                overlap_end = min(x_A_max, x_B_max)
                split_x = (overlap_start + overlap_end) / 2
                print(f"    → Overlap detected: splitting at {split_x:.1f} (overlap region: {overlap_start:.1f} to {overlap_end:.1f})")
            else:
                # They don't overlap - split at midpoint between boundaries
                split_x = (x_A_max + x_B_min) / 2
                print(f"    → No overlap: splitting at {split_x:.1f} (A ends at {x_A_max:.1f}, B starts at {x_B_min:.1f})")
            
            # If B's leftmost point < A's rightmost point (shouldn't happen after validation, but check anyway)
            if x_B_min < x_A_max:
                # Find constriction point (narrowest region between them)
                # Calculate width profile in the region between embryos
                mid_x = (x_A_max + x_B_min) / 2
                search_range = abs(x_B_min - x_A_max) * 0.5
                search_min = max(0, mid_x - search_range)
                search_max = min(w, mid_x + search_range)
                
                # Sample points along y-axis in the middle region to find narrowest point
                y_samples = np.linspace(0, h, 20)
                min_width = float('inf')
                constriction_x = split_x
                
                for y in y_samples:
                    # Find closest points on each contour at this y level
                    A_points_at_y = contour_A[np.abs(contour_A[:, 1] - y) < 10]
                    B_points_at_y = contour_B[np.abs(contour_B[:, 1] - y) < 10]
                    
                    if len(A_points_at_y) > 0 and len(B_points_at_y) > 0:
                        A_rightmost = A_points_at_y[:, 0].max()
                        B_leftmost = B_points_at_y[:, 0].min()
                        width_at_y = B_leftmost - A_rightmost
                        
                        if width_at_y < min_width and width_at_y > 0:
                            min_width = width_at_y
                            constriction_x = (A_rightmost + B_leftmost) / 2
                
                if min_width < float('inf'):
                    split_x = constriction_x
            
            # Validate split_x is reasonable
            if split_x < x_A_min or split_x > x_B_max:
                # Fallback to midpoint
                split_x = (x_A_max + x_B_min) / 2
            
            # ALWAYS enforce separation when 2 embryos detected
            centroid_A = emb_A['centroid']
            centroid_B = emb_B['centroid']
            centroid_sep = abs(centroid_B[0] - centroid_A[0])
            
            # Always clip to ensure strict separation
            if True:  # Always enforce separation
                
                # Clip contour A: keep only points to the left of split_x, add intersection
                def clip_contour_at_x(contour, split_x_val, keep_left=True):
                    """Clip contour at split_x, keeping left or right side."""
                    points_list = []
                    n = len(contour)
                    
                    for i in range(n):
                        p1 = contour[i]
                        p2 = contour[(i + 1) % n]
                        
                        p1_keep = (p1[0] <= split_x_val) if keep_left else (p1[0] >= split_x_val)
                        p2_keep = (p2[0] <= split_x_val) if keep_left else (p2[0] >= split_x_val)
                        
                        if p1_keep:
                            points_list.append(p1)
                        
                        # If edge crosses split line, add intersection
                        if p1_keep != p2_keep:
                            t = (split_x_val - p1[0]) / (p2[0] - p1[0] + 1e-9)
                            y_intersect = p1[1] + t * (p2[1] - p1[1])
                            intersect_pt = np.array([split_x_val, y_intersect], dtype=np.float32)
                            points_list.append(intersect_pt)
                    
                    if len(points_list) < 3:
                        return None
                    
                    clipped = np.array(points_list, dtype=np.float32)
                    # Close the contour
                    if not np.array_equal(clipped[0], clipped[-1]):
                        clipped = np.vstack([clipped, clipped[0:1]])
                    return clipped
                
                # Clip A (keep left side)
                contour_A_clipped = clip_contour_at_x(contour_A, split_x, keep_left=True)
                if contour_A_clipped is not None and len(contour_A_clipped) > 2:
                    contour_A = contour_A_clipped
                    results['A']['contour'] = contour_A
                
                # Clip B (keep right side)
                contour_B_clipped = clip_contour_at_x(contour_B, split_x, keep_left=False)
                if contour_B_clipped is not None and len(contour_B_clipped) > 2:
                    contour_B = contour_B_clipped
                    results['B']['contour'] = contour_B
                
                # ALWAYS recalculate head/tail positions after clipping using SIMPLE position-based logic
                # NO PCA - just use leftmost/rightmost points of the CLIPPED contour
                for label in ['A', 'B']:
                    contour = results[label]['contour']
                    if len(contour) > 2:
                        contour_points = contour.astype(np.float32)
                        x_coords = contour_points[:, 0]
                        
                        # Get the actual min/max x coordinates of the CLIPPED contour
                        leftmost_idx = np.argmin(x_coords)
                        rightmost_idx = np.argmax(x_coords)
                        leftmost_pt = contour_points[leftmost_idx]
                        rightmost_pt = contour_points[rightmost_idx]
                        
                        if label == 'A':
                            # A: head is LEFTMOST point (farthest left), tail is RIGHTMOST point (at split boundary)
                            # Since A is clipped to x <= split_x, leftmost is head, rightmost is tail
                            head = leftmost_pt.copy()
                            tail = rightmost_pt.copy()
                            
                            # Verify: head should be leftmost, tail should be rightmost
                            if head[0] > tail[0]:
                                head, tail = tail.copy(), head.copy()
                            
                            # Final check: head must be the absolute leftmost
                            if head[0] != x_coords.min():
                                head = contour_points[np.argmin(x_coords)].copy()
                            
                        elif label == 'B':
                            # B: head is RIGHTMOST point (farthest right), tail is LEFTMOST point (at split boundary)
                            # Since B is clipped to x >= split_x, rightmost is head, leftmost is tail
                            head = rightmost_pt.copy()
                            tail = leftmost_pt.copy()
                            
                            # Verify: head should be rightmost, tail should be leftmost
                            if head[0] < tail[0]:
                                head, tail = tail.copy(), head.copy()
                            
                            # Final check: head must be the absolute rightmost
                            if head[0] != x_coords.max():
                                head = contour_points[np.argmax(x_coords)].copy()
                        
                        # Store the corrected positions
                        results[label]['head'] = (float(head[0]), float(head[1]))
                        results[label]['tail'] = (float(tail[0]), float(tail[1]))
                        
                        # Log if there's still an issue
                        if label == 'B' and head[0] < split_x:
                            print(f"    ⚠ WARNING: B head ({head[0]:.1f}) is still left of split_x ({split_x:.1f})!")
                        if label == 'A' and head[0] > split_x:
                            print(f"    ⚠ WARNING: A head ({head[0]:.1f}) is still right of split_x ({split_x:.1f})!")
                        
                        # Recalculate centroid
                        M = cv2.moments(contour.astype(np.int32).reshape(-1, 1, 2))
                        if M["m00"] > 0:
                            results[label]['centroid'] = (M["m10"] / M["m00"], M["m01"] / M["m00"])
            
            # Post-processing validation: ensure A is left of B
            contour_A = results['A']['contour']
            contour_B = results['B']['contour']
            x_A_max = contour_A[:, 0].max()
            x_B_min = contour_B[:, 0].min()
            
            if x_A_max > x_B_min:
                # Still overlapping - force split at midpoint and re-clip
                print(f"    → Post-validation: Still overlapping, forcing re-clip")
                split_x = (x_A_max + x_B_min) / 2
                
                # Re-clip both contours (clip_contour_at_x is already defined in scope above)
                contour_A_clipped = clip_contour_at_x(contour_A, split_x, keep_left=True)
                contour_B_clipped = clip_contour_at_x(contour_B, split_x, keep_left=False)
                
                if contour_A_clipped is not None and len(contour_A_clipped) > 2:
                    results['A']['contour'] = contour_A_clipped
                    contour_A = contour_A_clipped
                if contour_B_clipped is not None and len(contour_B_clipped) > 2:
                    results['B']['contour'] = contour_B_clipped
                    contour_B = contour_B_clipped
                
                # Recalculate head/tail positions after re-clipping
                for label in ['A', 'B']:
                    contour = results[label]['contour']
                    if len(contour) > 2:
                        contour_points = contour.astype(np.float32)
                        x_coords = contour_points[:, 0]
                        leftmost_idx = np.argmin(x_coords)
                        rightmost_idx = np.argmax(x_coords)
                        leftmost_pt = contour_points[leftmost_idx]
                        rightmost_pt = contour_points[rightmost_idx]
                        
                        if label == 'A':
                            results[label]['head'] = (float(leftmost_pt[0]), float(leftmost_pt[1]))
                            results[label]['tail'] = (float(rightmost_pt[0]), float(rightmost_pt[1]))
                        elif label == 'B':
                            results[label]['head'] = (float(rightmost_pt[0]), float(rightmost_pt[1]))
                            results[label]['tail'] = (float(leftmost_pt[0]), float(leftmost_pt[1]))
            
            # FINAL VALIDATION: Ensure head positions are absolutely correct
            contour_A_final = results['A']['contour']
            contour_B_final = results['B']['contour']
            x_A_final_max = contour_A_final[:, 0].max()
            x_B_final_min = contour_B_final[:, 0].min()
            
            # A head MUST be the leftmost point of A's contour
            a_head_x = results['A']['head'][0]
            a_leftmost_x = contour_A_final[:, 0].min()
            if abs(a_head_x - a_leftmost_x) > 5:  # Allow small tolerance
                results['A']['head'] = (float(a_leftmost_x), float(contour_A_final[np.argmin(contour_A_final[:, 0]), 1]))
                print(f"    → FORCED A head to leftmost: {a_head_x:.1f} → {a_leftmost_x:.1f}")
            
            # B head MUST be the rightmost point of B's contour
            b_head_x = results['B']['head'][0]
            b_rightmost_x = contour_B_final[:, 0].max()
            if abs(b_head_x - b_rightmost_x) > 5:  # Allow small tolerance
                results['B']['head'] = (float(b_rightmost_x), float(contour_B_final[np.argmax(contour_B_final[:, 0]), 1]))
                print(f"    → FORCED B head to rightmost: {b_head_x:.1f} → {b_rightmost_x:.1f}")
            
            # CRITICAL CHECK: A's rightmost must be <= B's leftmost (or very close)
            if x_A_final_max > x_B_final_min + 10:  # Allow 10px tolerance
                print(f"    ⚠ CRITICAL: A still extends into B! A max: {x_A_final_max:.1f}, B min: {x_B_final_min:.1f}")
                # Force split at midpoint and re-clip
                emergency_split_x = (x_A_final_max + x_B_final_min) / 2
                print(f"    → Emergency re-clipping at {emergency_split_x:.1f}")
                
                # Re-clip both
                contour_A_emergency = clip_contour_at_x(contour_A_final, emergency_split_x, keep_left=True)
                contour_B_emergency = clip_contour_at_x(contour_B_final, emergency_split_x, keep_left=False)
                
                if contour_A_emergency is not None and len(contour_A_emergency) > 2:
                    results['A']['contour'] = contour_A_emergency
                    results['A']['head'] = (float(contour_A_emergency[:, 0].min()), 
                                           float(contour_A_emergency[np.argmin(contour_A_emergency[:, 0]), 1]))
                    results['A']['tail'] = (float(contour_A_emergency[:, 0].max()), 
                                           float(contour_A_emergency[np.argmax(contour_A_emergency[:, 0]), 1]))
                
                if contour_B_emergency is not None and len(contour_B_emergency) > 2:
                    results['B']['contour'] = contour_B_emergency
                    results['B']['head'] = (float(contour_B_emergency[:, 0].max()), 
                                           float(contour_B_emergency[np.argmax(contour_B_emergency[:, 0]), 1]))
                    results['B']['tail'] = (float(contour_B_emergency[:, 0].min()), 
                                           float(contour_B_emergency[np.argmin(contour_B_emergency[:, 0]), 1]))
            
            # Volume assessment: Check that each embryo's area is reasonable (within 30% of typical)
            # Typical single embryo area is roughly 0.01-0.02 of image area
            typical_embryo_area = 0.015 * h * w  # Typical single embryo area
            area_tolerance = 0.30  # 30% tolerance
            
            for label in ['A', 'B']:
                contour = results[label]['contour']
                area = cv2.contourArea(contour)
                area_ratio = area / (h * w)
                
                # Check if area is within 30% of typical
                area_diff_pct = abs(area - typical_embryo_area) / typical_embryo_area
                if area_diff_pct > area_tolerance:
                    print(f"    ⚠ Volume check failed for {label}: area={area:.0f} ({area_ratio*100:.2f}% of image), "
                          f"expected ~{typical_embryo_area:.0f} (±{area_tolerance*100:.0f}%)")
                    # This is a warning, but we'll still use it (might be legitimate variation)
            
            # B head/tail position check: B head should be in the right half of the image
            # If B's head is more than 20% into the left half, it's wrong
            image_center_x = w / 2
            b_head_x = results['B']['head'][0]
            b_tail_x = results['B']['tail'][0]
            
            # B head should be in the right half (x >= center)
            # If it's more than 20% into the left half (x < center - 0.2*w), reject it
            left_threshold = image_center_x - 0.20 * w  # 20% into left half
            
            if b_head_x < left_threshold:
                print(f"    ⚠ B head position check FAILED: head at {b_head_x:.1f} is in left half "
                      f"(threshold: {left_threshold:.1f}, center: {image_center_x:.1f})")
                # B head is way too far left - definitely wrong
                # Force correction: use rightmost point of B's contour
                contour_B = results['B']['contour']
                contour_points = contour_B.astype(np.float32)
                x_coords = contour_points[:, 0]
                
                # Use absolute rightmost point
                rightmost_idx = np.argmax(x_coords)
                new_b_head = contour_points[rightmost_idx]
                results['B']['head'] = (float(new_b_head[0]), float(new_b_head[1]))
                print(f"    → FORCED B head correction: {b_head_x:.1f} → {new_b_head[0]:.1f} (rightmost point)")
                
                # Also update tail to be leftmost
                leftmost_idx = np.argmin(x_coords)
                new_b_tail = contour_points[leftmost_idx]
                results['B']['tail'] = (float(new_b_tail[0]), float(new_b_tail[1]))
            
            # Also check if B head is in the left half at all (shouldn't be)
            elif b_head_x < image_center_x:
                print(f"    ⚠ B head position warning: head at {b_head_x:.1f} is in left half "
                      f"(center: {image_center_x:.1f}) - may need correction")
            
            # Store separation line info (for visualization)
            # Use final contours (after all clipping)
            contour_A = results['A']['contour']
            contour_B = results['B']['contour']
            # Find closest points between the two contours
            dists = np.linalg.norm(contour_A[:, np.newaxis] - contour_B, axis=2)
            min_idx_A, min_idx_B = np.unravel_index(np.argmin(dists), dists.shape)
            closest_A_pt = contour_A[min_idx_A]
            closest_B_pt = contour_B[min_idx_B]
            
            # Calculate split_x if not already defined (for non-overlapping but close cases)
            if 'split_x' not in locals():
                # Use midpoint between rightmost of A and leftmost of B
                split_x = (x_A_max + x_B_min) / 2
            
            results['A']['split_info'] = {
                'line_start': (float(closest_A_pt[0]), float(closest_A_pt[1])),
                'line_end': (float(closest_B_pt[0]), float(closest_B_pt[1])),
                'split_x': split_x if 'split_x' in locals() else None
            }
            results['B']['split_info'] = results['A']['split_info']
    
    if embryo_id:
        return results.get(embryo_id)
    return results


def get_embryo_boundary_from_tiff(tiff_path, embryo_id):
    """
    Get embryo boundary contour from TIFF file detection.
    """
    detection = detect_embryo_from_tiff(tiff_path, embryo_id)
    if detection and 'contour' in detection:
        # Close the contour
        contour = detection['contour']
        return np.vstack([contour, contour[0:1]])  # Add first point at end to close
    return None


def get_embryo_boundary_from_sparks(df_embryo):
    """
    Reconstruct approximate embryo boundary from spark locations (fallback method).
    Uses convex hull or alpha shape approximation.
    """
    if len(df_embryo) == 0:
        return None
    
    # Get all spark locations for this embryo
    points = df_embryo[['x', 'y']].dropna().values
    
    if len(points) < 3:
        return None
    
    # Use convex hull as approximation
    if HAS_SCIPY:
        try:
            hull = ConvexHull(points)
            boundary_points = points[hull.vertices]
            # Close the polygon
            boundary_points = np.vstack([boundary_points, boundary_points[0]])
            return boundary_points
        except:
            pass
    
    # Fallback: use bounding box with padding
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    padding = max((x_max - x_min) * 0.1, (y_max - y_min) * 0.1, 10)
    return np.array([
        [x_min - padding, y_min - padding],
        [x_max + padding, y_min - padding],
        [x_max + padding, y_max + padding],
        [x_min - padding, y_max + padding],
        [x_min - padding, y_min - padding]  # Close the polygon
    ])


def find_extreme_spark_points(df_embryo):
    """
    Find extreme spark points for fact-checking embryo outlines.
    Returns farthest left, right, top-left, bottom-left, top-right, bottom-right points.
    """
    valid_xy = df_embryo[df_embryo['x'].notna() & df_embryo['y'].notna()].copy()
    if len(valid_xy) == 0:
        return None
    
    x_coords = valid_xy['x'].values
    y_coords = valid_xy['y'].values
    
    # Calculate centroid to determine left/right sides
    cx = x_coords.mean()
    cy = y_coords.mean()
    
    # Find farthest points in cardinal directions
    farthest_left_idx = np.argmin(x_coords)
    farthest_right_idx = np.argmax(x_coords)
    
    # For left/right side extremes, split by centroid x
    left_mask = x_coords < cx
    right_mask = x_coords >= cx
    
    extremes = {}
    
    # Farthest left and right (overall)
    extremes['left'] = (x_coords[farthest_left_idx], y_coords[farthest_left_idx])
    extremes['right'] = (x_coords[farthest_right_idx], y_coords[farthest_right_idx])
    
    # Left side extremes
    if left_mask.sum() > 0:
        left_x = x_coords[left_mask]
        left_y = y_coords[left_mask]
        farthest_top_left_idx = np.argmin(left_y)  # Top = minimum y
        farthest_bottom_left_idx = np.argmax(left_y)  # Bottom = maximum y
        extremes['top_left'] = (left_x[farthest_top_left_idx], left_y[farthest_top_left_idx])
        extremes['bottom_left'] = (left_x[farthest_bottom_left_idx], left_y[farthest_bottom_left_idx])
    else:
        extremes['top_left'] = None
        extremes['bottom_left'] = None
    
    # Right side extremes
    if right_mask.sum() > 0:
        right_x = x_coords[right_mask]
        right_y = y_coords[right_mask]
        farthest_top_right_idx = np.argmin(right_y)  # Top = minimum y
        farthest_bottom_right_idx = np.argmax(right_y)  # Bottom = maximum y
        extremes['top_right'] = (right_x[farthest_top_right_idx], right_y[farthest_top_right_idx])
        extremes['bottom_right'] = (right_x[farthest_bottom_right_idx], right_y[farthest_bottom_right_idx])
    else:
        extremes['top_right'] = None
        extremes['bottom_right'] = None
    
    return extremes


def get_head_tail_positions(df_embryo):
    """
    Extract head and tail positions from embryo data.
    Uses ap_norm: 0 = head, 1 = tail
    """
    if 'ap_norm' not in df_embryo.columns:
        return None, None
    
    valid = df_embryo[df_embryo['ap_norm'].notna()]
    if len(valid) == 0:
        return None, None
    
    # Find head (ap_norm closest to 0) and tail (ap_norm closest to 1)
    head_idx = valid['ap_norm'].idxmin()
    tail_idx = valid['ap_norm'].idxmax()
    
    head_data = valid.loc[head_idx]
    tail_data = valid.loc[tail_idx]
    
    head_pos = (head_data['x'], head_data['y']) if pd.notna(head_data['x']) and pd.notna(head_data['y']) else None
    tail_pos = (tail_data['x'], tail_data['y']) if pd.notna(tail_data['x']) and pd.notna(tail_data['y']) else None
    
    return head_pos, tail_pos


def infer_poke_location(df_file):
    """
    Infer poke location from earliest spark clusters or distance data.
    """
    # Method 1: Use dist_from_poke_px = 0 (or minimum)
    if 'dist_from_poke_px' in df_file.columns:
        valid_dist = df_file[df_file['dist_from_poke_px'].notna()]
        if len(valid_dist) > 0:
            min_dist = valid_dist['dist_from_poke_px'].min()
            if min_dist < 50:  # Reasonable threshold
                poke_data = valid_dist[valid_dist['dist_from_poke_px'] == min_dist].iloc[0]
                if pd.notna(poke_data['x']) and pd.notna(poke_data['y']):
                    return (poke_data['x'], poke_data['y'])
    
    # Method 2: Use earliest sparks (first frame/time)
    if 'time_s' in df_file.columns:
        valid_time = df_file[df_file['time_s'].notna()]
        if len(valid_time) > 0:
            earliest = valid_time[valid_time['time_s'] == valid_time['time_s'].min()]
            if len(earliest) > 0:
                # Use centroid of earliest sparks
                valid_xy = earliest[earliest['x'].notna() & earliest['y'].notna()]
                if len(valid_xy) > 0:
                    poke_x = valid_xy['x'].mean()
                    poke_y = valid_xy['y'].mean()
                    if pd.notna(poke_x) and pd.notna(poke_y):
                        return (poke_x, poke_y)
    
    return None


def determine_orientation(head_pos, tail_pos):
    """
    Determine orientation string like "head left, tail right"
    """
    if head_pos is None or tail_pos is None:
        return "not detected"
    
    # Determine relative positions
    dx = head_pos[0] - tail_pos[0]
    dy = head_pos[1] - tail_pos[1]
    
    # Use the more significant direction
    if abs(dx) > abs(dy):
        # Horizontal orientation
        if dx < 0:
            return "head left, tail right"
        else:
            return "head right, tail left"
    else:
        # Vertical orientation
        if dy < 0:
            return "head top, tail bottom"
        else:
            return "head bottom, tail top"


def get_global_bounds(df_tracks):
    """
    Calculate global spatial bounds from all data.
    Returns (x_min, x_max, y_min, y_max) with padding.
    """
    valid_xy = df_tracks[df_tracks['x'].notna() & df_tracks['y'].notna()]
    if len(valid_xy) == 0:
        return None
    
    x_min, x_max = valid_xy['x'].min(), valid_xy['x'].max()
    y_min, y_max = valid_xy['y'].min(), valid_xy['y'].max()
    
    # Add padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    x_min -= x_range * 0.05
    x_max += x_range * 0.05
    y_min -= y_range * 0.05
    y_max += y_range * 0.05
    
    return (x_min, x_max, y_min, y_max)


def find_tiff_file(folder, video, base_path=None):
    """
    Find the actual TIFF file path from folder and video name.
    """
    if base_path is None:
        # Try common locations - check if folder/video exists as-is
        possible_paths = [
            Path(folder) / video,
            Path(folder) / f"{video}.tif",
            Path(folder) / f"{video}.tiff",
        ]
    else:
        possible_paths = [
            Path(base_path) / folder / video,
            Path(base_path) / folder / f"{video}.tif",
            Path(base_path) / folder / f"{video}.tiff",
        ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None


def create_embryo_visualization(df_tracks, output_dir, folder_video_key, global_bounds=None, tiff_base_path=None):
    """
    Create visualization for a specific folder/video combination.
    
    Args:
        df_tracks: DataFrame with track data
        output_dir: Output directory
        folder_video_key: (folder, video) tuple
        global_bounds: Optional (x_min, x_max, y_min, y_max) to use consistent dimensions
        tiff_base_path: Optional base path to search for TIFF files
    """
    folder, video = folder_video_key
    
    # Filter data for this folder/video - match against base_filename
    # base_filename should be like "9/B1 - Substack (1-301).tif"
    base_filename = f"{folder}/{video}"
    df_file = df_tracks[df_tracks['base_filename'] == base_filename].copy()
    if len(df_file) == 0:
        # Try without extension
        base_filename_no_ext = base_filename.replace('.tif', '')
        df_file = df_tracks[df_tracks['base_filename'].str.startswith(base_filename_no_ext, na=False)].copy()
    if len(df_file) == 0:
        return None
    
    # Try to find the actual TIFF file for embryo detection
    tiff_path = find_tiff_file(folder, video, tiff_base_path)
    
    # Use consistent figure size (fixed dimensions for all images)
    # Use fixed size for all images to ensure they're identical in the PDF
    fig_width = 14
    fig_height = 12
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Use global bounds if provided, otherwise calculate from this file's data
    if global_bounds:
        x_min, x_max, y_min, y_max = global_bounds
    else:
        # Get spatial bounds from this file
        valid_xy = df_file[df_file['x'].notna() & df_file['y'].notna()]
        if len(valid_xy) == 0:
            return None
        
        x_min, x_max = valid_xy['x'].min(), valid_xy['x'].max()
        y_min, y_max = valid_xy['y'].min(), valid_xy['y'].max()
        
        # Add padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        # Ensure minimum range to avoid singular transformation
        if x_range < 10:
            x_range = 10
            x_min = x_min - 5
            x_max = x_max + 5
        else:
            x_min -= x_range * 0.1
            x_max += x_range * 0.1
        
        if y_range < 10:
            y_range = 10
            y_min = y_min - 5
            y_max = y_max + 5
        else:
            y_min -= y_range * 0.1
            y_max += y_range * 0.1
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    
    # Detect all embryos from TIFF once (if available)
    tiff_detections = {}
    if tiff_path and tiff_path.exists():
        try:
            all_detections = detect_embryo_from_tiff(tiff_path, embryo_id=None)
            if all_detections:
                tiff_detections = all_detections
        except Exception as e:
            print(f"    ⚠ Error detecting embryos from TIFF: {e}")
    
    # Match TIFF detections to spark data by centroid position
    # First, get spark data centroids for each embryo
    spark_centroids = {}
    for embryo_id in ['A', 'B']:
        df_embryo = df_file[df_file['embryo_id'] == embryo_id].copy()
        if len(df_embryo) > 0:
            valid_xy = df_embryo[df_embryo['x'].notna() & df_embryo['y'].notna()]
            if len(valid_xy) > 0:
                spark_centroids[embryo_id] = (valid_xy['x'].mean(), valid_xy['y'].mean())
    
    # Match TIFF detections to spark labels by closest centroid (one-to-one mapping)
    tiff_to_spark_mapping = {}
    if tiff_detections and spark_centroids:
        # For each spark label, find the closest unmatched TIFF detection
        used_tiff_labels = set()
        for spark_label in spark_centroids.keys():
            spark_centroid = spark_centroids[spark_label]
            min_dist = float('inf')
            best_tiff_label = None
            for tiff_label in tiff_detections.keys():
                if tiff_label in used_tiff_labels:
                    continue  # Already matched
                tiff_centroid = tiff_detections[tiff_label]['centroid']
                dist = np.sqrt((tiff_centroid[0] - spark_centroid[0])**2 + 
                              (tiff_centroid[1] - spark_centroid[1])**2)
                if dist < min_dist and dist < 200:  # Reasonable distance threshold
                    min_dist = dist
                    best_tiff_label = tiff_label
            if best_tiff_label:
                tiff_to_spark_mapping[spark_label] = best_tiff_label
                used_tiff_labels.add(best_tiff_label)
    
    # Color scheme: A = cyan, B = orange
    embryo_colors = {
        'A': {'outline': 'cyan', 'head': 'lime', 'tail': 'red', 'axis': 'yellow'},
        'B': {'outline': 'orange', 'head': 'lime', 'tail': 'red', 'axis': 'yellow'}
    }
    
    # Plot embryo outlines and labels
    for embryo_id in ['A', 'B']:
        df_embryo = df_file[df_file['embryo_id'] == embryo_id].copy()
        if len(df_embryo) == 0:
            continue
        
        colors = embryo_colors.get(embryo_id, embryo_colors['A'])
        
        # Try to get boundary from TIFF detection first (actual grey tissue)
        boundary = None
        tiff_head = None
        tiff_tail = None
        used_tiff_detection = False
        
        # Use matched TIFF detection if available
        boundary_old = None
        boundary_intermediate = None
        if embryo_id in tiff_to_spark_mapping:
            tiff_label = tiff_to_spark_mapping[embryo_id]
            if tiff_label in tiff_detections:
                detection = tiff_detections[tiff_label]
                boundary = detection.get('contour')
                boundary_intermediate = detection.get('contour_intermediate')
                boundary_old = detection.get('contour_old')
                if boundary is not None:
                    # Close the contour
                    boundary = np.vstack([boundary, boundary[0:1]])
                if boundary_intermediate is not None:
                    # Close the intermediate contour
                    boundary_intermediate = np.vstack([boundary_intermediate, boundary_intermediate[0:1]])
                if boundary_old is not None:
                    # Close the old contour
                    boundary_old = np.vstack([boundary_old, boundary_old[0:1]])
                tiff_head = detection.get('head')
                tiff_tail = detection.get('tail')
                used_tiff_detection = True
        
        # Fallback to spark-based boundary if TIFF detection failed
        if boundary is None:
            boundary = get_embryo_boundary_from_sparks(df_embryo)
        
        if boundary is not None:
            # Draw triple outlines: inner (old), middle (intermediate), outer (new)
            if used_tiff_detection:
                # Draw inner outline (old, restrictive) - lighter, thinner
                if boundary_old is not None:
                    poly_inner = Polygon(boundary_old, fill=False, edgecolor=colors['outline'], 
                                        linewidth=1.5, alpha=0.6, linestyle='-')
                    ax.add_patch(poly_inner)
                
                # Draw middle outline (intermediate, better head capture) - medium, visible
                if boundary_intermediate is not None:
                    poly_middle = Polygon(boundary_intermediate, fill=False, edgecolor=colors['outline'], 
                                        linewidth=2.0, alpha=0.8, linestyle='-')
                    ax.add_patch(poly_middle)
                
                # Draw outer outline (new, inclusive) - thicker, more visible
                poly_outer = Polygon(boundary, fill=False, edgecolor=colors['outline'], 
                                    linewidth=2.5, alpha=0.9, linestyle='-')
                ax.add_patch(poly_outer)
            else:
                # Reconstructed from sparks (approximate)
                poly = Polygon(boundary, fill=False, edgecolor=colors['outline'], 
                              linewidth=2, alpha=0.6, linestyle='--')
                ax.add_patch(poly)
        
        # Get head/tail positions - prefer TIFF detection, fallback to spark data
        if tiff_head and tiff_tail:
            head_pos = tiff_head
            tail_pos = tiff_tail
        else:
            head_pos, tail_pos = get_head_tail_positions(df_embryo)
        
        # Draw head
        if head_pos:
            ax.plot(head_pos[0], head_pos[1], 'o', color=colors['head'], markersize=12, 
                   markeredgecolor='white', markeredgewidth=1.5, label=f'Embryo {embryo_id} Head')
            ax.annotate(f'{embryo_id} Head', head_pos, xytext=(10, 10), 
                       textcoords='offset points', color=colors['head'], fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor=colors['head'], linewidth=2))
        
        # Draw tail
        if tail_pos:
            ax.plot(tail_pos[0], tail_pos[1], 'o', color=colors['tail'], markersize=12,
                   markeredgecolor='white', markeredgewidth=1.5, label=f'Embryo {embryo_id} Tail')
            ax.annotate(f'{embryo_id} Tail', tail_pos, xytext=(10, -10), 
                       textcoords='offset points', color=colors['tail'], fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor=colors['tail'], linewidth=2))
        
        # Draw head-tail axis
        if head_pos and tail_pos:
            ax.plot([head_pos[0], tail_pos[0]], [head_pos[1], tail_pos[1]], 
                   color=colors['axis'], linestyle='--', linewidth=1.5, alpha=0.6)
        
        # Plot extreme spark points for fact-checking
        extreme_points = find_extreme_spark_points(df_embryo)
        if extreme_points:
            # Use a distinct color for spark fact-check points (white with colored edge)
            spark_marker_size = 8
            spark_alpha = 0.8
            
            # Farthest left and right
            if extreme_points['left']:
                ax.plot(extreme_points['left'][0], extreme_points['left'][1], 
                       'o', color='white', markersize=spark_marker_size, 
                       markeredgecolor=colors['outline'], markeredgewidth=2,
                       alpha=spark_alpha, zorder=5, label='Extreme Sparks' if embryo_id == 'A' else '')
            if extreme_points['right']:
                ax.plot(extreme_points['right'][0], extreme_points['right'][1], 
                       'o', color='white', markersize=spark_marker_size,
                       markeredgecolor=colors['outline'], markeredgewidth=2,
                       alpha=spark_alpha, zorder=5)
            
            # Left side extremes
            if extreme_points['top_left']:
                ax.plot(extreme_points['top_left'][0], extreme_points['top_left'][1], 
                       '^', color='white', markersize=spark_marker_size,
                       markeredgecolor=colors['outline'], markeredgewidth=2,
                       alpha=spark_alpha, zorder=5)
            if extreme_points['bottom_left']:
                ax.plot(extreme_points['bottom_left'][0], extreme_points['bottom_left'][1], 
                       'v', color='white', markersize=spark_marker_size,
                       markeredgecolor=colors['outline'], markeredgewidth=2,
                       alpha=spark_alpha, zorder=5)
            
            # Right side extremes
            if extreme_points['top_right']:
                ax.plot(extreme_points['top_right'][0], extreme_points['top_right'][1], 
                       '^', color='white', markersize=spark_marker_size,
                       markeredgecolor=colors['outline'], markeredgewidth=2,
                       alpha=spark_alpha, zorder=5)
            if extreme_points['bottom_right']:
                ax.plot(extreme_points['bottom_right'][0], extreme_points['bottom_right'][1], 
                       'v', color='white', markersize=spark_marker_size,
                       markeredgecolor=colors['outline'], markeredgewidth=2,
                       alpha=spark_alpha, zorder=5)
    
    # Draw separation line between connected embryos (if they're close together)
    if len(tiff_detections) == 2:
        emb_A_det = tiff_detections.get('A')
        emb_B_det = tiff_detections.get('B')
        if emb_A_det and emb_B_det:
            # Check if they're close enough to need a separation line
            centroid_A = emb_A_det.get('centroid')
            centroid_B = emb_B_det.get('centroid')
            if centroid_A and centroid_B:
                dist = np.sqrt((centroid_A[0] - centroid_B[0])**2 + 
                              (centroid_A[1] - centroid_B[1])**2)
                # If centroids are within 300 pixels, draw separation line
                if dist < 300:
                    split_info = emb_A_det.get('split_info')
                    if split_info:
                        line_start = split_info['line_start']
                        line_end = split_info['line_end']
                    else:
                        # Calculate separation line from closest points on contours
                        contour_A = emb_A_det.get('contour')
                        contour_B = emb_B_det.get('contour')
                        if contour_A is not None and contour_B is not None:
                            # Find closest points between contours
                            dists = np.linalg.norm(contour_A[:, np.newaxis] - contour_B, axis=2)
                            min_idx_A, min_idx_B = np.unravel_index(np.argmin(dists), dists.shape)
                            line_start = (float(contour_A[min_idx_A, 0]), float(contour_A[min_idx_A, 1]))
                            line_end = (float(contour_B[min_idx_B, 0]), float(contour_B[min_idx_B, 1]))
                        else:
                            line_start = line_end = None
                    
                    if line_start and line_end:
                        # Draw a white dashed line to show the separation
                        ax.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], 
                               'w--', linewidth=2.5, alpha=0.9, label='Embryo Separation', zorder=10)
    
    # Draw poke location
    poke_pos = infer_poke_location(df_file)
    if poke_pos:
        ax.plot(poke_pos[0], poke_pos[1], 'mX', markersize=15, markeredgewidth=2,
               markeredgecolor='white', label='Poke Location')
        ax.annotate('POKE', poke_pos, xytext=(0, 20), 
                   textcoords='offset points', color='magenta', fontsize=12, 
                   fontweight='bold', ha='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('X (pixels)', color='white', fontsize=12)
    ax.set_ylabel('Y (pixels)', color='white', fontsize=12)
    ax.set_title(f'Folder {folder} - {video}\nEmbryo Detection & Poke Location', 
                color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3, color='gray')
    
    # Save with fixed dimensions
    safe_video_name = re.sub(r'[^\w\-_\.]', '_', video)
    output_path = output_dir / f"folder_{folder}_{safe_video_name}_detection.png"
    # Use fixed bbox (not tight) to ensure all images are exactly the same size
    # This ensures consistent dimensions regardless of content
    from matplotlib.transforms import Bbox
    bbox = Bbox([[0, 0], [fig_width, fig_height]])
    plt.savefig(output_path, dpi=150, bbox_inches=bbox, facecolor='black', edgecolor='none')
    plt.close()
    
    return output_path


def create_summary_table_page(summary_data, output_pdf_path):
    """
    Create a summary table page as the first page of the PDF.
    
    Args:
        summary_data: List of dicts with keys: folder, video, emb_a_orient, emb_b_orient, poke_str, healed_wound, img_ref
        output_pdf_path: Path to output PDF (will be opened in append mode)
    """
    from matplotlib.table import Table
    
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    ax.set_title('Embryo Detection Summary', fontsize=16, fontweight='bold', pad=20)
    
    # Prepare table data
    table_data = [['Folder', 'Video', 'Embryo A', 'Embryo B', 'Poke Location', 'Healed Wound']]
    
    for row in summary_data:
        table_data.append([
            str(row.get('folder', '')),
            str(row.get('video', '')),
            str(row.get('emb_a_orient', '')),
            str(row.get('emb_b_orient', '')),
            str(row.get('poke_str', '')),
            str(row.get('healed_wound', ''))
        ])
    
    # Create table
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                    cellLoc='left', loc='center',
                    colWidths=[0.08, 0.25, 0.15, 0.15, 0.20, 0.17])
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows (alternating)
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('white')
    
    plt.tight_layout()
    
    # Save to PDF
    with PdfPages(output_pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_pdf_from_images(image_paths, output_pdf_path, images_per_page=1, summary_data=None):
    """
    Create a PDF from a list of image paths.
    
    Args:
        image_paths: List of (path, title) tuples
        output_pdf_path: Path to output PDF
        images_per_page: Number of images per page (1 or 2)
        summary_data: Optional list of summary data dicts for first page
    """
    with PdfPages(output_pdf_path) as pdf:
        # Add summary table as first page if provided
        if summary_data:
            from matplotlib.table import Table
            
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('off')
            ax.set_title('Embryo Detection Summary', fontsize=16, fontweight='bold', pad=20)
            
            # Prepare table data
            table_data = [['Folder', 'Video', 'Embryo A', 'Embryo B', 'Poke Location', 'Healed Wound']]
            
            for row in summary_data:
                table_data.append([
                    str(row.get('folder', '')),
                    str(row.get('video', '')),
                    str(row.get('emb_a_orient', '')),
                    str(row.get('emb_b_orient', '')),
                    str(row.get('poke_str', '')),
                    str(row.get('healed_wound', ''))
                ])
            
            # Create table
            table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                            cellLoc='left', loc='center',
                            colWidths=[0.08, 0.25, 0.15, 0.15, 0.20, 0.17])
            
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 2)
            
            # Style header row
            for i in range(len(table_data[0])):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Style data rows (alternating)
            for i in range(1, len(table_data)):
                for j in range(len(table_data[0])):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#f0f0f0')
                    else:
                        table[(i, j)].set_facecolor('white')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Add visualization images
        for i, (img_path, title) in enumerate(image_paths):
            if not img_path.exists():
                continue
            
            try:
                # Open image
                img = Image.open(img_path)
                
                # Create figure
                if images_per_page == 1:
                    fig, ax = plt.subplots(figsize=(11, 8.5))
                    ax.imshow(img)
                    ax.axis('off')
                    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
                else:
                    # Two images per page
                    if i % 2 == 0:
                        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
                        fig.suptitle('Embryo Detection Visualizations', fontsize=14, fontweight='bold')
                    
                    ax_idx = i % 2
                    axes[ax_idx].imshow(img)
                    axes[ax_idx].axis('off')
                    axes[ax_idx].set_title(title, fontsize=10, fontweight='bold', pad=5)
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                
                # If two per page and this is the second image, close the figure
                if images_per_page == 2 and i % 2 == 1:
                    plt.close()
                    
            except Exception as e:
                print(f"  ⚠ Warning: Could not add {img_path.name} to PDF: {e}")
                continue
    
    print(f"✓ Generated PDF: {output_pdf_path}")


def generate_summary_table(df_tracks, output_path, output_dir, image_paths_dict=None):
    """
    Generate markdown table summarizing all detections.
    
    Args:
        df_tracks: DataFrame with track data
        output_path: Path to output markdown file
        output_dir: Directory containing images
        image_paths_dict: Dict mapping (folder, video) to image path
    """
    # Group by folder and video
    df_tracks = df_tracks.copy()
    df_tracks['base_filename'] = df_tracks['filename'].str.replace(r' \(page \d+\)', '', regex=True)
    
    # Extract folder and video
    folder_video_data = []
    for base_file in df_tracks['base_filename'].unique():
        folder, video = extract_folder_and_video(base_file)
        if folder and video:
            folder_video_data.append((folder, video, base_file))
    
    # Sort by folder (numeric), then video
    def sort_key(x):
        folder, video, _ = x
        try:
            folder_num = int(folder)
        except:
            folder_num = 999
        return (folder_num, video)
    
    folder_video_data.sort(key=sort_key)
    
    # Build table rows
    table_rows = []
    table_rows.append("| Folder | Video | Embryo A | Embryo B | Poke Location | Healed Wound | Visualization |")
    table_rows.append("|--------|-------|----------|----------|---------------|--------------|---------------|")
    
    # Track folder counts for labeling (1a, 1b, etc.)
    folder_counts = defaultdict(int)
    folder_video_map = {}
    
    for folder, video, base_file in folder_video_data:
        folder_counts[folder] += 1
        folder_video_map[(folder, video)] = folder_counts[folder]
    
    for folder, video, base_file in folder_video_data:
        df_file = df_tracks[df_tracks['base_filename'] == base_file].copy()
        
        # Get embryo A info
        emb_a = df_file[df_file['embryo_id'] == 'A']
        emb_a_orient = "not detected"
        if len(emb_a) > 0:
            head_a, tail_a = get_head_tail_positions(emb_a)
            if head_a and tail_a:
                emb_a_orient = determine_orientation(head_a, tail_a)
        
        # Get embryo B info
        emb_b = df_file[df_file['embryo_id'] == 'B']
        emb_b_orient = "not detected"
        if len(emb_b) > 0:
            head_b, tail_b = get_head_tail_positions(emb_b)
            if head_b and tail_b:
                emb_b_orient = determine_orientation(head_b, tail_b)
        
        # Get poke location
        poke_pos = infer_poke_location(df_file)
        poke_str = f"({poke_pos[0]:.1f}, {poke_pos[1]:.1f})" if poke_pos else "not detected"
        
        # Healed wound (placeholder - need to check if this data exists)
        healed_wound = "not detected"  # TODO: implement if data available
        
        # Determine video label (1a, 1b, etc. if multiple videos per folder)
        videos_in_folder = folder_counts[folder]
        if videos_in_folder > 1:
            video_idx = folder_video_map[(folder, video)] - 1
            video_label = f"{folder}{chr(97 + video_idx)}"  # 1a, 1b, etc.
        else:
            video_label = folder
        
        # Get image reference
        if image_paths_dict and (folder, video) in image_paths_dict:
            img_path = image_paths_dict[(folder, video)]
            img_name = img_path.name
            img_ref = f"[View]({img_name})"
        else:
            img_ref = "-"
        
        table_rows.append(f"| {video_label} | {video} | {emb_a_orient} | {emb_b_orient} | {poke_str} | {healed_wound} | {img_ref} |")
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write("# Embryo Detection Summary\n\n")
        f.write("This table summarizes embryo detection results for manual fact-checking.\n\n")
        f.write("## Legend\n\n")
        f.write("- **Embryo A/B**: Orientation format is 'head [direction], tail [direction]'\n")
        f.write("- **Poke Location**: Coordinates in pixels (x, y)\n")
        f.write("- **Healed Wound**: Location of previously healed wounds (if detected)\n")
        f.write("- **Visualization**: Link to individual image (see also compiled PDF)\n\n")
        f.write("## Compiled PDF\n\n")
        f.write("All visualizations are compiled in: `detection_visualizations.pdf`\n\n")
        f.write("\n".join(table_rows))
        f.write("\n")
    
    print(f"✓ Generated summary table: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualization and summary table of embryo detections"
    )
    parser.add_argument('tracks_csv', help='Path to spark_tracks.csv')
    parser.add_argument('--output-dir', default='analysis_results/detection_summary',
                       help='Output directory for visualizations and table')
    parser.add_argument('--skip-visualizations', action='store_true',
                       help='Skip generating visualization images (only create table)')
    parser.add_argument('--tiff-base-path', default='/Users/jdietz/Documents/Levin/Embryos',
                       help='Base path to search for TIFF files (default: /Users/jdietz/Documents/Levin/Embryos)')
    
    args = parser.parse_args()
    
    # Verify the TIFF base path exists
    if args.tiff_base_path:
        tiff_base = Path(args.tiff_base_path)
        if not tiff_base.exists():
            print(f"  ⚠ Warning: TIFF base path does not exist: {args.tiff_base_path}")
            print(f"    → Will use spark-based embryo reconstruction instead")
            args.tiff_base_path = None
        else:
            print(f"  → Using TIFF base path: {args.tiff_base_path}")
    
    # Load data
    print(f"Loading {args.tracks_csv}...")
    df_tracks = pd.read_csv(args.tracks_csv)
    print(f"  → Loaded {len(df_tracks):,} track states")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    image_paths_dict = {}
    image_paths_for_pdf = []
    
    if not args.skip_visualizations:
        print("\nGenerating visualizations...")
        df_tracks['base_filename'] = df_tracks['filename'].str.replace(r' \(page \d+\)', '', regex=True)
        
        # Calculate global bounds for consistent dimensions
        print("  → Calculating global bounds for consistent image dimensions...")
        global_bounds = get_global_bounds(df_tracks)
        if global_bounds:
            x_min, x_max, y_min, y_max = global_bounds
            print(f"    Global bounds: X=[{x_min:.1f}, {x_max:.1f}], Y=[{y_min:.1f}, {y_max:.1f}]")
        else:
            print("    ⚠ Warning: Could not calculate global bounds, using per-image bounds")
            global_bounds = None
        
        folder_video_keys = set()
        for base_file in df_tracks['base_filename'].unique():
            folder, video = extract_folder_and_video(base_file)
            if folder and video:
                folder_video_keys.add((folder, video))
        
        print(f"  → Found {len(folder_video_keys)} unique folder/video combinations")
        
        # Check if TIFF files can be found
        if args.tiff_base_path:
            print(f"  → Using TIFF base path: {args.tiff_base_path}")
            # Test if we can find any TIFF files
            test_found = 0
            for folder, video in list(folder_video_keys)[:5]:  # Test first 5
                tiff_path = find_tiff_file(folder, video, args.tiff_base_path)
                if tiff_path and tiff_path.exists():
                    test_found += 1
            if test_found > 0:
                print(f"    ✓ Found {test_found}/5 test TIFF files - will use actual embryo detection")
            else:
                print(f"    ⚠ Could not find TIFF files - will use spark-based reconstruction")
        else:
            print(f"  → No TIFF base path provided - using spark-based embryo reconstruction")
            print(f"    (Use --tiff-base-path to enable actual embryo tissue detection from TIFF files)")
        
        for folder_video_key in sorted(folder_video_keys, key=lambda x: (int(x[0]) if x[0].isdigit() else 999, x[1])):
            folder, video = folder_video_key
            print(f"  → Processing folder {folder}, video {video}...")
            try:
                output_path = create_embryo_visualization(df_tracks, output_dir, folder_video_key, global_bounds, args.tiff_base_path)
                if output_path:
                    print(f"    ✓ Saved: {output_path.name}")
                    image_paths_dict[(folder, video)] = output_path
                    # Create title for PDF
                    video_label = folder
                    if len([k for k in folder_video_keys if k[0] == folder]) > 1:
                        folder_list = sorted([k for k in folder_video_keys if k[0] == folder], key=lambda x: x[1])
                        video_idx = folder_list.index((folder, video))
                        video_label = f"{folder}{chr(97 + video_idx)}"
                    title = f"Folder {video_label}: {video}"
                    image_paths_for_pdf.append((output_path, title))
                else:
                    print(f"    ⚠ No data found for this combination")
            except Exception as e:
                print(f"    ✗ Error: {e}")
                import traceback
                traceback.print_exc()
        
        # Create PDF from all images
        if image_paths_for_pdf:
            print(f"\nCompiling {len(image_paths_for_pdf)} images into PDF...")
            pdf_path = output_dir / "detection_visualizations.pdf"
            # Prepare summary data for PDF table
            summary_data = []
            for folder, video in sorted(folder_video_keys, key=lambda x: (int(x[0]) if x[0].isdigit() else 999, x[1])):
                base_file = f"{folder}/{video}"
                df_file = df_tracks[df_tracks['base_filename'] == base_file].copy()
                if len(df_file) == 0:
                    base_file_no_ext = base_file.replace('.tif', '')
                    df_file = df_tracks[df_tracks['base_filename'].str.startswith(base_file_no_ext, na=False)].copy()
                
                # Get embryo A info
                emb_a = df_file[df_file['embryo_id'] == 'A']
                emb_a_orient = "not detected"
                if len(emb_a) > 0:
                    head_a, tail_a = get_head_tail_positions(emb_a)
                    if head_a and tail_a:
                        emb_a_orient = determine_orientation(head_a, tail_a)
                
                # Get embryo B info
                emb_b = df_file[df_file['embryo_id'] == 'B']
                emb_b_orient = "not detected"
                if len(emb_b) > 0:
                    head_b, tail_b = get_head_tail_positions(emb_b)
                    if head_b and tail_b:
                        emb_b_orient = determine_orientation(head_b, tail_b)
                
                # Get poke location
                poke_pos = infer_poke_location(df_file)
                poke_str = f"({poke_pos[0]:.1f}, {poke_pos[1]:.1f})" if poke_pos else "not detected"
                
                # Healed wound (placeholder)
                healed_wound = "not detected"
                
                summary_data.append({
                    'folder': folder,
                    'video': video,
                    'emb_a_orient': emb_a_orient,
                    'emb_b_orient': emb_b_orient,
                    'poke_str': poke_str,
                    'healed_wound': healed_wound
                })
            
            create_pdf_from_images(image_paths_for_pdf, pdf_path, images_per_page=1, summary_data=summary_data)
    
    # Generate summary table (after visualizations so we can include image references)
    print("\nGenerating summary table...")
    generate_summary_table(df_tracks, output_dir / "detection_summary.md", output_dir, image_paths_dict)
    
    print("\n✓ Complete!")


if __name__ == '__main__':
    main()

