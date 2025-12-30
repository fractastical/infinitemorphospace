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
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
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


def _detect_cement_gland_for_visualization(end1, end2, axis_direction, mask, gray_enhanced):
    """
    Detect cement gland location for visualization purposes.
    Uses same logic as parser's _detect_cement_gland function.
    
    Returns: (x, y) location or None if not found
    """
    h, w = mask.shape
    u_perp = np.array([-axis_direction[1], axis_direction[0]])
    search_radius = min(30, np.linalg.norm(end2 - end1) * 0.15)  # 15% of embryo length, max 30px
    ventral_search_dist = search_radius * 0.5  # Look on ventral side
    
    results = {}
    
    for end_name, end_pt in [('end1', end1), ('end2', end2)]:
        center_x, center_y = int(round(end_pt[0])), int(round(end_pt[1]))
        
        # Skip if endpoint is outside image bounds
        if center_x < 0 or center_x >= w or center_y < 0 or center_y >= h:
            continue
        
        # Check ventral region (try both directions of perpendicular)
        for perp_dir in [u_perp, -u_perp]:
            ventral_center = end_pt + perp_dir * ventral_search_dist
            vx, vy = int(round(ventral_center[0])), int(round(ventral_center[1]))
            
            # Skip if ventral center is outside bounds
            if vx < 0 or vx >= w or vy < 0 or vy >= h:
                continue
            
            # Search in a small circular region around ventral center
            dark_pixels = []
            for dy in range(-int(search_radius), int(search_radius) + 1):
                for dx in range(-int(search_radius), int(search_radius) + 1):
                    x, y = vx + dx, vy + dy
                    if 0 <= x < w and 0 <= y < h:
                        dist = np.hypot(dx, dy)
                        if dist <= search_radius and mask[y, x]:
                            # This is an embryo pixel in the search region
                            intensity = gray_enhanced[y, x]
                            dark_pixels.append((intensity, (x, y)))
            
            if dark_pixels:
                # Find the darkest pixels (cement gland should be darker than surrounding tissue)
                dark_pixels.sort(key=lambda p: p[0])  # Sort by intensity (darkest first)
                
                # Take darkest 10% of pixels in this region
                num_dark = max(1, len(dark_pixels) // 10)
                darkest = dark_pixels[:num_dark]
                avg_dark_intensity = np.mean([p[0] for p in darkest])
                
                # Compare to average intensity in the endpoint region
                endpoint_pixels = []
                for dy in range(-int(search_radius), int(search_radius) + 1):
                    for dx in range(-int(search_radius), int(search_radius) + 1):
                        x, y = center_x + dx, center_y + dy
                        if 0 <= x < w and 0 <= y < h and mask[y, x]:
                            dist = np.hypot(dx, dy)
                            if dist <= search_radius:
                                endpoint_pixels.append(gray_enhanced[y, x])
                
                if endpoint_pixels:
                    avg_endpoint_intensity = np.mean(endpoint_pixels)
                    # Cement gland should be darker than surrounding tissue
                    # Use more lenient threshold for visualization (0.9 = 10% darker)
                    # Also accept if darkest pixels are significantly below average
                    darkness_ratio = avg_dark_intensity / (avg_endpoint_intensity + 1e-9)
                    intensity_diff = avg_endpoint_intensity - avg_dark_intensity
                    # Accept if: (1) 10% darker ratio, OR (2) absolute difference > 20 intensity units
                    if darkness_ratio < 0.9 or intensity_diff > 20:
                        if end_name not in results or results[end_name][1] > darkness_ratio:
                            results[end_name] = (darkest[0][1], darkness_ratio)
    
    # Determine which end has the cement gland
    if results:
        # Return location from the darker detection (most confident)
        best = min(results.items(), key=lambda x: x[1][1])
        return best[1][0]  # Return location tuple (x, y)
    return None


def detect_embryo_from_tiff(tiff_path, embryo_id=None, logger=None):
    """
    Detect embryo boundaries directly from TIFF file using the same method as the parser.
    Returns dict with 'contour', 'mask', 'head', 'tail', 'centroid', 'cement_gland' for each embryo.
    
    Args:
        tiff_path: Path to TIFF file
        embryo_id: Optional, 'A' or 'B' to get specific embryo
    
    Returns:
        Dict mapping embryo_id to detection data, or single dict if embryo_id specified
    """
    try:
        # Read first page of TIFF
        with tiff.TiffFile(tiff_path) as tif:
            # Skip single-frame TIFFs - only process multi-frame ones
            if len(tif.pages) <= 1:
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
            
            # CRITICAL CHECK: If both parts are unusually large, it's likely a single embryo that was incorrectly split
            typical_embryo_area = 0.015 * h * w
            area_A_ratio = areas[0] / (h * w)
            area_B_ratio = areas[1] / (h * w)
            
            # If both are >15% of image area, they're both too large to be separate embryos
            # A single large embryo might be 10-20% of image, but two separate embryos should each be ~1-2%
            if area_A_ratio > 0.15 and area_B_ratio > 0.15:
                # Both are very large - likely a single embryo incorrectly split
                # Merge them back into one
                print(f"    → Both contours unusually large (A: {area_A_ratio*100:.1f}%, B: {area_B_ratio*100:.1f}% of image)")
                print(f"    → Likely single embryo incorrectly split - merging back")
                
                # Combine both contours into one
                combined_points = np.vstack([
                    emb_contours[0].reshape(-1, 2),
                    emb_contours[1].reshape(-1, 2)
                ])
                
                # Create a new contour from the combined points (convex hull approximation)
                if HAS_SCIPY:
                    try:
                        from scipy.spatial import ConvexHull
                        hull = ConvexHull(combined_points)
                        combined_contour = combined_points[hull.vertices].reshape(-1, 1, 2).astype(np.int32)
                        emb_contours = [combined_contour]
                    except:
                        # Fallback: use the larger one
                        larger_idx = 0 if areas[0] > areas[1] else 1
                        emb_contours = [emb_contours[larger_idx]]
                else:
                    # Fallback: use the larger one
                    larger_idx = 0 if areas[0] > areas[1] else 1
                    emb_contours = [emb_contours[larger_idx]]
            
            # Additional check: if total area of both is >25% of image, likely single embryo
            elif total_area / (h * w) > 0.25:
                # Total area is too large for two separate embryos
                # Check if they overlap significantly or are connected
                contour1_points = emb_contours[0].reshape(-1, 2)
                contour2_points = emb_contours[1].reshape(-1, 2)
                
                # Check overlap in x-direction
                x1_min, x1_max = contour1_points[:, 0].min(), contour1_points[:, 0].max()
                x2_min, x2_max = contour2_points[:, 0].min(), contour2_points[:, 0].max()
                
                overlap_x = min(x1_max, x2_max) - max(x1_min, x2_min)
                total_span_x = max(x1_max, x2_max) - min(x1_min, x2_min)
                overlap_ratio = overlap_x / total_span_x if total_span_x > 0 else 0
                
                # If they overlap >50% in x-direction, likely same embryo
                if overlap_ratio > 0.5:
                    print(f"    → Contours overlap significantly ({overlap_ratio*100:.1f}%) - treating as single embryo")
                    # Use the larger one or combine
                    larger_idx = 0 if areas[0] > areas[1] else 1
                    emb_contours = [emb_contours[larger_idx]]
    
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
        
        # Detect cement gland location for visualization
        cement_gland_location = None
        try:
            # Create a proper mask from the contour
            mask_image = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask_image, [contour], -1, 255, thickness=-1)
            mask_bool = mask_image == 255
            
            # Create enhanced grayscale for cement gland detection (within embryo region only)
            if mask_bool.sum() > 0:
                gray_min = gray[mask_bool].min()
                gray_max = gray[mask_bool].max()
                if gray_max > gray_min:
                    gray_enhanced = ((gray - gray_min) / (gray_max - gray_min) * 255).astype(np.uint8)
                else:
                    gray_enhanced = gray.astype(np.uint8)
            else:
                # Fallback if mask is empty
                gray_min = gray.min()
                gray_max = gray.max()
                if gray_max > gray_min:
                    gray_enhanced = ((gray - gray_min) / (gray_max - gray_min) * 255).astype(np.uint8)
                else:
                    gray_enhanced = gray.astype(np.uint8)
            
            # Detect cement gland near the HEAD position (where it should be)
            # Use head position instead of both endpoints
            cement_gland_location = _detect_cement_gland_for_visualization(
                head, tail, v_norm, mask_bool, gray_enhanced
            )
            if cement_gland_location:
                if logger:
                    logger(f"    ✓ Cement gland detected at ({cement_gland_location[0]:.1f}, {cement_gland_location[1]:.1f})")
        except Exception as e:
            if logger:
                logger(f"    ⚠ Could not detect cement gland: {e}")
            import traceback
            if logger:
                logger(f"    Traceback: {traceback.format_exc()}")
        
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
        
        # CRITICAL VALIDATION: Enforce spatial constraints before storing results
        image_center_x = w / 2
        head_x = head[0]
        tail_x = tail[0]
        head_tail_dist = np.sqrt((head[0] - tail[0])**2 + (head[1] - tail[1])**2)
        min_head_tail_length = max(w * 0.05, 50)  # 5% of image width or 50 pixels
        
        corrections_made = False
        
        # Validation rule 1: A head/tail must NEVER be on right side
        if label == 'A':
            if head_x >= image_center_x:
                print(f"    [Embryo A] ⚠ CRITICAL: Head at x={head_x:.1f} is on RIGHT side (center={image_center_x:.1f})")
                print(f"    → Correcting: using leftmost point of contour as head")
                leftmost_idx = np.argmin(contour_points[:, 0])
                head = contour_points[leftmost_idx]
                head_x = head[0]
                corrections_made = True
            
            if tail_x >= image_center_x:
                print(f"    [Embryo A] ⚠ CRITICAL: Tail at x={tail_x:.1f} is on RIGHT side (center={image_center_x:.1f})")
                print(f"    → Correcting: using rightmost point on LEFT side as tail")
                left_side_pts = contour_points[contour_points[:, 0] < image_center_x]
                if len(left_side_pts) > 0:
                    rightmost_left_idx = np.argmax(left_side_pts[:, 0])
                    tail = left_side_pts[rightmost_left_idx]
                    tail_x = tail[0]
                    corrections_made = True
                else:
                    print(f"    → No left-side points available - REJECTING this detection")
                    continue  # Skip this embryo
        
        # Validation rule 2: B head/tail must NEVER be on left side
        elif label == 'B':
            if head_x < image_center_x:
                print(f"    [Embryo B] ⚠ CRITICAL: Head at x={head_x:.1f} is on LEFT side (center={image_center_x:.1f})")
                print(f"    → Correcting: using rightmost point of contour as head")
                rightmost_idx = np.argmax(contour_points[:, 0])
                head = contour_points[rightmost_idx]
                head_x = head[0]
                corrections_made = True
            
            if tail_x < image_center_x:
                print(f"    [Embryo B] ⚠ CRITICAL: Tail at x={tail_x:.1f} is on LEFT side (center={image_center_x:.1f})")
                print(f"    → Correcting: using leftmost point on RIGHT side as tail")
                right_side_pts = contour_points[contour_points[:, 0] >= image_center_x]
                if len(right_side_pts) > 0:
                    leftmost_right_idx = np.argmin(right_side_pts[:, 0])
                    tail = right_side_pts[leftmost_right_idx]
                    tail_x = tail[0]
                    corrections_made = True
                else:
                    print(f"    → No right-side points available - REJECTING this detection")
                    continue  # Skip this embryo
        
        # FINAL VALIDATION: After corrections, ensure constraints are still met
        head_x = head[0]
        tail_x = tail[0]
        head_tail_dist = np.sqrt((head[0] - tail[0])**2 + (head[1] - tail[1])**2)
        
        # Re-check spatial constraints after corrections
        if label == 'A':
            if head_x >= image_center_x or tail_x >= image_center_x:
                print(f"    [Embryo A] ⚠ CRITICAL: Still violates spatial constraints after correction")
                print(f"    → Head x={head_x:.1f}, Tail x={tail_x:.1f}, Center={image_center_x:.1f}")
                print(f"    → REJECTING this detection")
                continue  # Skip this embryo
        elif label == 'B':
            if head_x < image_center_x or tail_x < image_center_x:
                print(f"    [Embryo B] ⚠ CRITICAL: Still violates spatial constraints after correction")
                print(f"    → Head x={head_x:.1f}, Tail x={tail_x:.1f}, Center={image_center_x:.1f}")
                print(f"    → REJECTING this detection")
                continue  # Skip this embryo
        
        # Validation rule 3: Head-tail distance must be above minimum threshold
        if head_tail_dist < min_head_tail_length:
            print(f"    [Embryo {label}] ⚠ CRITICAL: Head-tail distance too short ({head_tail_dist:.1f}px < {min_head_tail_length:.1f}px)")
            print(f"    → REJECTING this detection")
            continue  # Skip this embryo
        
        if corrections_made:
            print(f"    → Corrections applied - final: head x={head_x:.1f}, tail x={tail_x:.1f}, distance={head_tail_dist:.1f}px")
        
        results[label] = {
            'contour': contour_points,  # New (inclusive) contour
            'contour_intermediate': intermediate_contour,  # Intermediate contour for middle outline
            'contour_old': old_contour,  # Old (restrictive) contour for inner outline
            'head': (float(head[0]), float(head[1])),
            'tail': (float(tail[0]), float(tail[1])),
            'centroid': (float(cx), float(cy)),
            'cement_gland': cement_gland_location,  # Cement gland location if detected
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
                            print(f"    [Embryo B] ⚠ WARNING: Head ({head[0]:.1f}) is still left of split_x ({split_x:.1f})!")
                        if label == 'A' and head[0] > split_x:
                            print(f"    [Embryo A] ⚠ WARNING: Head ({head[0]:.1f}) is still right of split_x ({split_x:.1f})!")
                        
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
                print(f"    [Embryo A] → FORCED head to leftmost: {a_head_x:.1f} → {a_leftmost_x:.1f}")
            
            # B head MUST be the rightmost point of B's contour
            b_head_x = results['B']['head'][0]
            b_rightmost_x = contour_B_final[:, 0].max()
            if abs(b_head_x - b_rightmost_x) > 5:  # Allow small tolerance
                results['B']['head'] = (float(b_rightmost_x), float(contour_B_final[np.argmax(contour_B_final[:, 0]), 1]))
                print(f"    [Embryo B] → FORCED head to rightmost: {b_head_x:.1f} → {b_rightmost_x:.1f}")
            
            # CRITICAL CHECK: A's rightmost must be <= B's leftmost (or very close)
            if x_A_final_max > x_B_final_min + 10:  # Allow 10px tolerance
                print(f"    [Both Embryos] ⚠ CRITICAL: A still extends into B! A max: {x_A_final_max:.1f}, B min: {x_B_final_min:.1f}")
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
            
            # Volume assessment: Check that each embryo's area is reasonable
            # Compare A and B areas - they should be similar (within 30% of each other)
            # Also check against typical single embryo area
            contour_A = results['A']['contour']
            contour_B = results['B']['contour']
            area_A = cv2.contourArea(contour_A)
            area_B = cv2.contourArea(contour_B)
            
            # Typical single embryo area is roughly 0.01-0.02 of image area
            typical_embryo_area = 0.015 * h * w
            area_tolerance = 0.30  # 30% tolerance
            
            # Check if A and B areas are similar (within 30% of each other)
            avg_area = (area_A + area_B) / 2
            area_diff_pct = abs(area_A - area_B) / avg_area if avg_area > 0 else 0
            
            if area_diff_pct > area_tolerance:
                print(f"    [Both Embryos] ⚠ Volume mismatch: A area={area_A:.0f}, B area={area_B:.0f}, "
                      f"difference={area_diff_pct*100:.1f}% (>30% threshold)")
                # If one is much larger, it might be encompassing both embryos
                if area_A > area_B * 1.5:
                    print(f"    [Both Embryos] ⚠ A is {area_A/area_B:.1f}x larger than B - A might encompass both embryos")
                elif area_B > area_A * 1.5:
                    print(f"    [Both Embryos] ⚠ B is {area_B/area_A:.1f}x larger than A - B might encompass both embryos")
            
            # Check each against typical size
            for label, area in [('A', area_A), ('B', area_B)]:
                area_ratio = area / (h * w)
                area_diff_from_typical_pct = abs(area - typical_embryo_area) / typical_embryo_area
                
                # Flag if significantly different from typical (but allow some variation)
                if area_diff_from_typical_pct > area_tolerance * 2:  # 60% threshold for individual check
                    print(f"    [Embryo {label}] ⚠ Volume unusual: area={area:.0f} ({area_ratio*100:.2f}% of image), "
                          f"typical ~{typical_embryo_area:.0f} (±{area_tolerance*100:.0f}%)")
            
            # Head-tail length validation: filter out detections that are too short
            # Minimum head-tail length should be at least 5% of image width or 50 pixels, whichever is larger
            min_head_tail_length = max(w * 0.05, 50)
            
            for label in ['A', 'B']:
                if label not in results:
                    continue
                head = results[label].get('head')
                tail = results[label].get('tail')
                
                if head and tail:
                    head_tail_dist = np.sqrt((head[0] - tail[0])**2 + (head[1] - tail[1])**2)
                    
                    if head_tail_dist < min_head_tail_length:
                        print(f"    [Embryo {label}] ⚠ CRITICAL: Head-tail distance too short ({head_tail_dist:.1f}px < {min_head_tail_length:.1f}px threshold) - removing detection")
                        # Remove this embryo from results
                        del results[label]
                    elif head_tail_dist < min_head_tail_length * 1.5:
                        print(f"    [Embryo {label}] ⚠ Head-tail distance is short ({head_tail_dist:.1f}px, threshold: {min_head_tail_length:.1f}px)")
            
            # If we removed an embryo, return early (single embryo case)
            if len(results) < 2:
                return results
            
            # B head/tail position check: B head should be in the right half of the image
            # If B's head is more than 20% into the left half, it's wrong
            image_center_x = w / 2
            b_head_x = results['B']['head'][0]
            b_tail_x = results['B']['tail'][0]
            
            # B head should be in the right half (x >= center)
            # If it's more than 20% into the left half (x < center - 0.2*w), reject it
            left_threshold = image_center_x - 0.20 * w  # 20% into left half
            
            if b_head_x < left_threshold:
                print(f"    [Embryo B] ⚠ Head position check FAILED: head at {b_head_x:.1f} is in left half "
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
                print(f"    [Embryo B] ⚠ Head position warning: head at {b_head_x:.1f} is in left half "
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


def detect_embryos_vector_first(df_tracks, tiff_path=None, image_width=None, image_height=None):
    """
    NEW APPROACH: Start with vector data if available, then refine with contours.
    
    Logic:
    1. If vectors exist and cover significant area, use them to identify left/right regions
    2. Find centroids in each region → assign A (left) and B (right)
    3. Use contour detection to refine boundaries
    4. Use vector directions to determine head/tail (waves propagate from head)
    
    Args:
        df_tracks: DataFrame with spark data (x, y, vx, vy, speed, embryo_id)
        tiff_path: Optional path to TIFF file for contour refinement
        image_width: Optional image width (if None, calculated from data)
        image_height: Optional image height (if None, calculated from data)
    
    Returns:
        Dict with 'A' and 'B' keys, each containing:
        - 'contour': boundary contour
        - 'centroid': (x, y) centroid
        - 'head': (x, y) head position
        - 'tail': (x, y) tail position
        - 'vector_region': bounding box of vector data region
    """
    results = {}
    
    # Get image dimensions
    valid_xy = df_tracks[df_tracks['x'].notna() & df_tracks['y'].notna()]
    if len(valid_xy) == 0:
        return results
    
    if image_width is None:
        image_width = valid_xy['x'].max() - valid_xy['x'].min()
    if image_height is None:
        image_height = valid_xy['y'].max() - valid_xy['y'].min()
    
    image_center_x = (valid_xy['x'].min() + valid_xy['x'].max()) / 2
    
    # Step 1: Check if we have vector data
    valid_vectors = df_tracks[(df_tracks['vx'].notna()) & (df_tracks['vy'].notna()) & 
                             (df_tracks['speed'].notna()) & (df_tracks['speed'] > 0)].copy()
    
    has_vectors = len(valid_vectors) > 100  # Need sufficient vector data
    
    # Step 2: Identify left vs right regions using available data
    if has_vectors:
        # FIRST: Check if we have actual embryo_id labels in the data
        # If so, use those to determine left/right instead of just splitting by center
        has_embryo_labels = 'embryo_id' in valid_vectors.columns and valid_vectors['embryo_id'].notna().any()
        
        if has_embryo_labels:
            # Use actual embryo labels to determine left/right
            emb_A_data = valid_vectors[valid_vectors['embryo_id'] == 'A'].copy()
            emb_B_data = valid_vectors[valid_vectors['embryo_id'] == 'B'].copy()
            
            # Only create B if it has substantial data (not just a few outliers)
            if len(emb_A_data) > 50:
                left_vectors = emb_A_data
                # Only use B if it has substantial data
                if len(emb_B_data) >= 100:  # Need at least 100 points for B
                    right_vectors = emb_B_data
                else:
                    b_count = len(emb_B_data)
                    right_vectors = pd.DataFrame()  # Empty - no B
                    if b_count > 0:
                        print(f"    → Embryo B has only {b_count} data points - treating as single embryo (A only)")
            else:
                # Not enough A data, fall back to center split
                left_vectors = valid_vectors[valid_vectors['x'] < image_center_x].copy()
                right_vectors = valid_vectors[valid_vectors['x'] >= image_center_x].copy()
        else:
            # No embryo labels - split by center
            left_vectors = valid_vectors[valid_vectors['x'] < image_center_x].copy()
            right_vectors = valid_vectors[valid_vectors['x'] >= image_center_x].copy()
        
        # Calculate centroids from vector data
        left_centroid = None
        right_centroid = None
        left_bbox = None
        right_bbox = None
        
        if len(left_vectors) > 50:
            left_centroid = (left_vectors['x'].mean(), left_vectors['y'].mean())
            # Calculate bounding box
            left_bbox = {
                'x_min': left_vectors['x'].min(),
                'x_max': left_vectors['x'].max(),
                'y_min': left_vectors['y'].min(),
                'y_max': left_vectors['y'].max()
            }
        
        if len(right_vectors) > 50:
            right_centroid = (right_vectors['x'].mean(), right_vectors['y'].mean())
            # Calculate bounding box
            right_bbox = {
                'x_min': right_vectors['x'].min(),
                'x_max': right_vectors['x'].max(),
                'y_min': right_vectors['y'].min(),
                'y_max': right_vectors['y'].max()
            }
        
        # CRITICAL VALIDATION: Right side must have substantial data to be a real embryo
        # Check if right side has enough data points AND covers a reasonable area
        right_side_valid = False
        if right_centroid and right_bbox:
            right_data_count = len(right_vectors)
            right_x_span = right_bbox['x_max'] - right_bbox['x_min']
            right_y_span = right_bbox['y_max'] - right_bbox['y_min']
            right_area_span = right_x_span * right_y_span
            total_area_span = image_width * image_height
            
            # Right side must have:
            # 1. At least 100 data points (not just a few outliers)
            # 2. Cover at least 5% of the image area (not just a tiny region)
            if right_data_count >= 100 and (right_area_span / total_area_span) > 0.05:
                right_side_valid = True
            else:
                print(f"    → Right side has insufficient data ({right_data_count} points, {right_area_span/total_area_span*100:.1f}% of image) - treating as single embryo")
        
        # Assign A (left) and B (right) based on vector centroids
        # Only create B if right side has substantial data
        if left_centroid and left_bbox:
            if right_side_valid and right_centroid and right_bbox:
                # Two embryos: A (left) and B (right)
                results['A'] = {
                    'centroid': left_centroid,
                    'vector_region': left_bbox,
                    'vector_data': left_vectors,
                    'contour': None,  # Will be filled by contour detection
                    'head': None,
                    'tail': None
                }
                results['B'] = {
                    'centroid': right_centroid,
                    'vector_region': right_bbox,
                    'vector_data': right_vectors,
                    'contour': None,
                    'head': None,
                    'tail': None
                }
            else:
                # Only left side has substantial data - single embryo (A)
                results['A'] = {
                    'centroid': left_centroid,
                    'vector_region': left_bbox,
                    'vector_data': left_vectors,
                    'contour': None,
                    'head': None,
                    'tail': None
                }
    
    # If no vectors or insufficient data, fall back to spark location centroids
    if len(results) == 0:
        # FIRST: Check if we have actual embryo_id labels
        has_embryo_labels = 'embryo_id' in valid_xy.columns and valid_xy['embryo_id'].notna().any()
        
        if has_embryo_labels:
            # Use actual embryo labels
            left_sparks = valid_xy[valid_xy['embryo_id'] == 'A'].copy()
            right_sparks = valid_xy[valid_xy['embryo_id'] == 'B'].copy()
            
            # Only create B if it has substantial data
            b_count = len(right_sparks)
            if b_count < 100:
                right_sparks = pd.DataFrame()  # Empty - no B
                if b_count > 0:
                    print(f"    → Embryo B has only {b_count} spark points - treating as single embryo (A only)")
        else:
            # No labels - split by x-position
            left_sparks = valid_xy[valid_xy['x'] < image_center_x].copy()
            right_sparks = valid_xy[valid_xy['x'] >= image_center_x].copy()
        
        # CRITICAL: Right side must have substantial data (not just a few outliers)
        right_side_valid = False
        if len(right_sparks) > 20:
            right_x_span = right_sparks['x'].max() - right_sparks['x'].min()
            right_y_span = right_sparks['y'].max() - right_sparks['y'].min()
            right_area_span = right_x_span * right_y_span
            total_area_span = image_width * image_height
            
            # Right side must cover at least 5% of image area
            if (right_area_span / total_area_span) > 0.05:
                right_side_valid = True
            else:
                print(f"    → Right side sparks cover insufficient area ({right_area_span/total_area_span*100:.1f}% of image) - treating as single embryo")
        
        if len(left_sparks) > 20:
            left_centroid = (left_sparks['x'].mean(), left_sparks['y'].mean())
            
            if right_side_valid and len(right_sparks) > 20:
                # Two embryos: A (left) and B (right)
                right_centroid = (right_sparks['x'].mean(), right_sparks['y'].mean())
                
                results['A'] = {
                    'centroid': left_centroid,
                    'vector_region': {
                        'x_min': left_sparks['x'].min(),
                        'x_max': left_sparks['x'].max(),
                        'y_min': left_sparks['y'].min(),
                        'y_max': left_sparks['y'].max()
                    },
                    'vector_data': None,
                    'contour': None,
                    'head': None,
                    'tail': None
                }
                results['B'] = {
                    'centroid': right_centroid,
                    'vector_region': {
                        'x_min': right_sparks['x'].min(),
                        'x_max': right_sparks['x'].max(),
                        'y_min': right_sparks['y'].min(),
                        'y_max': right_sparks['y'].max()
                    },
                    'vector_data': None,
                    'contour': None,
                    'head': None,
                    'tail': None
                }
            else:
                # Only left side has substantial data - single embryo (A)
                results['A'] = {
                    'centroid': left_centroid,
                    'vector_region': {
                        'x_min': left_sparks['x'].min(),
                        'x_max': left_sparks['x'].max(),
                        'y_min': left_sparks['y'].min(),
                        'y_max': left_sparks['y'].max()
                    },
                    'vector_data': None,
                    'contour': None,
                    'head': None,
                    'tail': None
                }
    
    # Step 3: Refine with contour detection if TIFF available
    if tiff_path and tiff_path.exists() and len(results) > 0:
        try:
            # Get contours from TIFF
            tiff_detections = detect_embryo_from_tiff(tiff_path, embryo_id=None)
            
            if tiff_detections:
                # Match contours to our vector-based centroids
                for label in ['A', 'B']:
                    if label not in results:
                        continue
                    
                    target_centroid = results[label]['centroid']
                    
                    # Find closest contour to this centroid
                    best_contour = None
                    best_dist = float('inf')
                    best_detection = None
                    
                    for tiff_label, detection in tiff_detections.items():
                        tiff_centroid = detection.get('centroid')
                        if tiff_centroid:
                            dist = np.sqrt((tiff_centroid[0] - target_centroid[0])**2 + 
                                         (tiff_centroid[1] - target_centroid[1])**2)
                            # Use larger threshold to account for coordinate system differences
                            # TIFF centroids are in image pixels, spark centroids are in spark coordinates
                            if dist < best_dist and dist < 1000:  # Increased threshold for coordinate system differences
                                best_dist = dist
                                best_contour = detection.get('contour')
                                best_detection = detection
                    
                    if best_contour is not None:
                        results[label]['contour'] = best_contour
                        # Also get intermediate and old contours if available
                        if best_detection:
                            results[label]['contour_intermediate'] = best_detection.get('contour_intermediate')
                            results[label]['contour_old'] = best_detection.get('contour_old')
        except Exception as e:
            print(f"    ⚠ Error refining with TIFF contours: {e}")
    
    # Step 4: Determine head/tail using vector directions
    for label in ['A', 'B']:
        if label not in results:
            continue
        
        vector_data = results[label].get('vector_data')
        centroid = results[label]['centroid']
        contour = results[label].get('contour')
        
        # If we have vector data, use it to determine head/tail
        if vector_data is not None and len(vector_data) > 50:
            # Calculate average vector direction in this region
            # Waves propagate from head outward
            avg_vx = vector_data['vx'].mean()
            avg_vy = vector_data['vy'].mean()
            
            # For A (left side): if vectors point rightward, head is on left
            # For B (right side): if vectors point leftward, head is on right
            if label == 'A':
                # A is on left - head should be leftmost
                # If vectors point rightward (positive vx), head is on left (correct)
                # If vectors point leftward (negative vx), might need to check
                if avg_vx > 0.3:  # Strong rightward flow
                    # Head is on left, tail on right
                    if contour is not None:
                        contour_points = contour.reshape(-1, 2).astype(np.float32)
                        x_coords = contour_points[:, 0]
                        head = contour_points[np.argmin(x_coords)]
                        tail = contour_points[np.argmax(x_coords)]
                    else:
                        # Use vector region bounds
                        bbox = results[label].get('vector_region')
                        if bbox:
                            head = (bbox['x_min'], centroid[1])
                            tail = (bbox['x_max'], centroid[1])
                        else:
                            # Fallback to centroid
                            head = (centroid[0] - 50, centroid[1])
                            tail = (centroid[0] + 50, centroid[1])
                else:
                    # Vectors point leftward - might be reversed or single embryo
                    # Default to leftmost = head
                    if contour is not None:
                        contour_points = contour.reshape(-1, 2).astype(np.float32)
                        x_coords = contour_points[:, 0]
                        head = contour_points[np.argmin(x_coords)]
                        tail = contour_points[np.argmax(x_coords)]
                    else:
                        bbox = results[label].get('vector_region')
                        if bbox:
                            head = (bbox['x_min'], centroid[1])
                            tail = (bbox['x_max'], centroid[1])
                        else:
                            head = (centroid[0] - 50, centroid[1])
                            tail = (centroid[0] + 50, centroid[1])
            else:  # B (right side)
                # B is on right - head should be rightmost
                # If vectors point leftward (negative vx), head is on right (correct)
                if avg_vx < -0.3:  # Strong leftward flow
                    # Head is on right, tail on left
                    if contour is not None:
                        contour_points = contour.reshape(-1, 2).astype(np.float32)
                        x_coords = contour_points[:, 0]
                        head = contour_points[np.argmax(x_coords)]  # Rightmost
                        tail = contour_points[np.argmin(x_coords)]  # Leftmost
                    else:
                        bbox = results[label]['vector_region']
                        head = (bbox['x_max'], centroid[1])
                        tail = (bbox['x_min'], centroid[1])
                else:
                    # Vectors point rightward - might be reversed
                    # Default to rightmost = head
                    if contour is not None:
                        contour_points = contour.reshape(-1, 2).astype(np.float32)
                        x_coords = contour_points[:, 0]
                        head = contour_points[np.argmax(x_coords)]
                        tail = contour_points[np.argmin(x_coords)]
                    else:
                        bbox = results[label]['vector_region']
                        head = (bbox['x_max'], centroid[1])
                        tail = (bbox['x_min'], centroid[1])
            
            results[label]['head'] = (float(head[0]), float(head[1]))
            results[label]['tail'] = (float(tail[0]), float(tail[1]))
        
        # If no vector data, use contour-based head/tail (PCA + width analysis)
        elif contour is not None:
            # Use existing PCA + width analysis logic
            contour_points = contour.reshape(-1, 2).astype(np.float32)
            mean, eigenvectors, eigenvalues = cv2.PCACompute2(contour_points, mean=None)
            v = eigenvectors[0]
            v_norm = v / (np.linalg.norm(v) + 1e-9)
            
            proj = np.dot(contour_points - mean.reshape(1, 2), v_norm.reshape(2, 1)).ravel()
            min_idx = np.argmin(proj)
            max_idx = np.argmax(proj)
            end1 = contour_points[min_idx]
            end2 = contour_points[max_idx]
            
            # Simple: for A, leftmost = head, rightmost on left side = tail
            # For B, rightmost = head, leftmost on right side = tail
            x_coords = contour_points[:, 0]
            image_center_x = image_width / 2
            
            if label == 'A':
                # A: head is leftmost, tail is rightmost point on LEFT side
                head = contour_points[np.argmin(x_coords)]
                left_side_pts = contour_points[contour_points[:, 0] < image_center_x]
                if len(left_side_pts) > 0:
                    tail = left_side_pts[np.argmax(left_side_pts[:, 0])]
                else:
                    tail = head.copy()  # Fallback
            else:
                # B: head is rightmost, tail is leftmost point on RIGHT side
                head = contour_points[np.argmax(x_coords)]
                right_side_pts = contour_points[contour_points[:, 0] >= image_center_x]
                if len(right_side_pts) > 0:
                    tail = right_side_pts[np.argmin(right_side_pts[:, 0])]
                else:
                    tail = head.copy()  # Fallback
            
            results[label]['head'] = (float(head[0]), float(head[1]))
            results[label]['tail'] = (float(tail[0]), float(tail[1]))
    
    # CRITICAL VALIDATION: Enforce spatial constraints and distance requirements
    image_center_x = image_width / 2
    min_head_tail_length = max(image_width * 0.05, 50)  # 5% of image width or 50 pixels, whichever is larger
    
    for label in list(results.keys()):
        head = results[label].get('head')
        tail = results[label].get('tail')
        
        if head and tail:
            head_x = head[0]
            tail_x = tail[0]
            head_tail_dist = np.sqrt((head[0] - tail[0])**2 + (head[1] - tail[1])**2)
            
            # Validation rule 1: A head/tail must NEVER be on right side
            if label == 'A':
                if head_x >= image_center_x or tail_x >= image_center_x:
                    print(f"    [Embryo A] ⚠ CRITICAL: Violates spatial constraints")
                    print(f"    → Head x={head_x:.1f}, Tail x={tail_x:.1f}, Center={image_center_x:.1f}")
                    print(f"    → REJECTING detection - A must be entirely on left side")
                    del results[label]
                    continue
            
            # Validation rule 2: B head/tail must NEVER be on left side
            elif label == 'B':
                if head_x < image_center_x or tail_x < image_center_x:
                    print(f"    [Embryo B] ⚠ CRITICAL: Violates spatial constraints")
                    print(f"    → Head x={head_x:.1f}, Tail x={tail_x:.1f}, Center={image_center_x:.1f}")
                    print(f"    → REJECTING detection - B must be entirely on right side")
                    del results[label]
                    continue
            
            # Validation rule 3: Head-tail distance must be above minimum threshold
            if head_tail_dist < min_head_tail_length:
                print(f"    [Embryo {label}] ⚠ CRITICAL: Head-tail distance too short ({head_tail_dist:.1f}px < {min_head_tail_length:.1f}px threshold)")
                print(f"    → REJECTING detection")
                del results[label]
            else:
                # Log if it's close to the threshold
                if head_tail_dist < min_head_tail_length * 1.5:
                    print(f"    [Embryo {label}] ⚠ Head-tail distance is short ({head_tail_dist:.1f}px, threshold: {min_head_tail_length:.1f}px)")
    
    return results


def validate_head_tail_with_vectors(df_tracks, results, image_width, image_height):
    """
    Validate and correct head/tail assignments using vector wave data.
    
    Uses the direction of Ca²⁺ wave propagation to validate:
    - Waves typically propagate from head (poke location) outward
    - Left half vectors pointing rightward → head should be on left (A)
    - Right half vectors pointing leftward → head should be on right (B)
    
    Args:
        df_tracks: DataFrame with velocity data (vx, vy, x, y)
        results: Dict with 'A' and 'B' embryo detection results
        image_width: Image width in pixels
        image_height: Image height in pixels
    
    Returns:
        Modified results dict with corrected assignments if needed
    """
    if len(results) != 2 or 'A' not in results or 'B' not in results:
        return results
    
    # Filter to valid velocity data
    valid_vectors = df_tracks[(df_tracks['vx'].notna()) & (df_tracks['vy'].notna()) & 
                             (df_tracks['speed'].notna()) & (df_tracks['speed'] > 0)].copy()
    
    if len(valid_vectors) < 50:  # Need sufficient data
        return results
    
    image_center_x = image_width / 2
    
    # Split vectors into left and right halves
    left_vectors = valid_vectors[valid_vectors['x'] < image_center_x].copy()
    right_vectors = valid_vectors[valid_vectors['x'] >= image_center_x].copy()
    
    corrections_made = []
    
    # Analyze left half vectors
    if len(left_vectors) > 20:
        # Calculate average x-direction of vectors in left half
        # Positive vx = pointing right, negative vx = pointing left
        avg_vx_left = left_vectors['vx'].mean()
        
        # If vectors point predominantly rightward (positive vx), head should be on left (A)
        # If vectors point predominantly leftward (negative vx), head might be on right (B) - wrong!
        if avg_vx_left < -0.5:  # Strong leftward flow in left half
            # This suggests head is NOT on left - might be misassigned
            a_head_x = results['A']['head'][0]
            if a_head_x < image_center_x:
                # A head is on left but vectors point leftward - suspicious
                print(f"    ⚠ Vector validation: Left half vectors point leftward (avg_vx={avg_vx_left:.2f}), "
                      f"but A head is at {a_head_x:.1f} (left side)")
    
    # Analyze right half vectors
    if len(right_vectors) > 20:
        # Calculate average x-direction of vectors in right half
        avg_vx_right = right_vectors['vx'].mean()
        
        # If vectors point predominantly leftward (negative vx), head should be on right (B)
        # If vectors point predominantly rightward (positive vx), head might be on left (A) - wrong!
        if avg_vx_right > 0.5:  # Strong rightward flow in right half
            # This suggests head is NOT on right - might be misassigned
            b_head_x = results['B']['head'][0]
            if b_head_x >= image_center_x:
                # B head is on right but vectors point rightward - suspicious
                print(f"    ⚠ Vector validation: Right half vectors point rightward (avg_vx={avg_vx_right:.2f}), "
                      f"but B head is at {b_head_x:.1f} (right side)")
        
        # CRITICAL CHECK: If B head is on left side but right half vectors point leftward
        # This strongly suggests B head should be on right
        if avg_vx_right < -0.3 and b_head_x < image_center_x:
            print(f"    ⚠ CRITICAL Vector validation: B head at {b_head_x:.1f} is on LEFT, "
                  f"but right half vectors point LEFTWARD (avg_vx={avg_vx_right:.2f})")
            print(f"    → This suggests B head should be on RIGHT - forcing correction")
            
            # Force B head to rightmost point of B's contour
            contour_B = results['B']['contour']
            if len(contour_B) > 2:
                contour_points = contour_B.astype(np.float32)
                x_coords = contour_points[:, 0]
                rightmost_idx = np.argmax(x_coords)
                new_b_head = contour_points[rightmost_idx]
                results['B']['head'] = (float(new_b_head[0]), float(new_b_head[1]))
                
                # Also update tail to be leftmost
                leftmost_idx = np.argmin(x_coords)
                new_b_tail = contour_points[leftmost_idx]
                results['B']['tail'] = (float(new_b_tail[0]), float(new_b_tail[1]))
                
                corrections_made.append("B head moved to rightmost (vector-based)")
    
    # Additional check: Compare vector directions near head positions
    # Head regions should have vectors pointing AWAY from head (outward propagation)
    for label in ['A', 'B']:
        head_pos = results[label]['head']
        if head_pos:
            # Get vectors near head position (within 200 pixels)
            head_x, head_y = head_pos[0], head_pos[1]
            near_head = valid_vectors[
                ((valid_vectors['x'] - head_x)**2 + (valid_vectors['y'] - head_y)**2) < 200**2
            ].copy()
            
            if len(near_head) > 10:
                # Calculate average direction away from head
                # Vectors should point away from head
                dx = near_head['x'] - head_x
                dy = near_head['y'] - head_y
                # Normalize direction vectors
                dist = np.sqrt(dx**2 + dy**2)
                dist[dist == 0] = 1e-9
                dir_x = dx / dist
                dir_y = dy / dist
                
                # Dot product of vector direction with direction away from head
                # Positive = pointing away from head (correct)
                # Negative = pointing toward head (wrong)
                dot_products = near_head['vx'] * dir_x + near_head['vy'] * dir_y
                avg_dot = dot_products.mean()
                
                if avg_dot < -0.2:  # Vectors point toward head (wrong!)
                    print(f"    ⚠ Vector validation: {label} head at ({head_x:.1f}, {head_y:.1f}) "
                          f"has vectors pointing TOWARD it (avg_dot={avg_dot:.2f})")
                    print(f"    → This suggests head/tail might be reversed")
    
    if corrections_made:
        print(f"    ✓ Vector-based corrections: {', '.join(corrections_made)}")
    
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


def get_embryo_boundary_from_sparks(df_embryo, method='density', alpha=None):
    """
    Reconstruct approximate embryo boundary from spark locations using multiple methods.
    
    Methods:
    - 'density': Uses kernel density estimation to find high-density regions (best for sparse data)
    - 'alpha': Uses alpha shapes (concave hull) for better boundary approximation
    - 'convex': Simple convex hull (fast but less accurate)
    - 'adaptive': Tries multiple methods and picks the best
    
    Args:
        df_embryo: DataFrame with spark data (x, y columns)
        method: Method to use ('density', 'alpha', 'convex', 'adaptive')
        alpha: Alpha parameter for alpha shapes (if None, auto-calculated)
    
    Returns:
        Boundary points as numpy array, or None if insufficient data
    """
    if len(df_embryo) == 0:
        return None
    
    # Get all spark locations for this embryo
    points = df_embryo[['x', 'y']].dropna().values
    
    if len(points) < 3:
        return None
    
    if not HAS_SCIPY:
        # Fallback: use bounding box with padding
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        padding = max((x_max - x_min) * 0.1, (y_max - y_min) * 0.1, 10)
        return np.array([
            [x_min - padding, y_min - padding],
            [x_max + padding, y_min - padding],
            [x_max + padding, y_max + padding],
            [x_min - padding, y_max + padding],
            [x_min - padding, y_min - padding]
        ])
    
    # Method 1: Kernel Density Estimation (best for finding actual embryo boundaries)
    if method in ['density', 'adaptive']:
        try:
            from scipy.stats import gaussian_kde
            from scipy.ndimage import gaussian_filter
            
            # Create a grid for density estimation
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            # Adaptive grid resolution based on data density
            n_points = len(points)
            grid_res = min(100, max(50, int(np.sqrt(n_points / 10))))
            
            xi = np.linspace(x_min - x_range * 0.1, x_max + x_range * 0.1, grid_res)
            yi = np.linspace(y_min - y_range * 0.1, y_max + y_range * 0.1, grid_res)
            xi_grid, yi_grid = np.meshgrid(xi, yi)
            
            # Estimate density
            kde = gaussian_kde(points.T)
            density = kde(np.vstack([xi_grid.ravel(), yi_grid.ravel()]))
            density_grid = density.reshape(xi_grid.shape)
            
            # Smooth the density
            density_smooth = gaussian_filter(density_grid, sigma=1.0)
            
            # Find threshold (e.g., 10th percentile of density in high-density regions)
            high_density_mask = density_smooth > np.percentile(density_smooth[density_smooth > 0], 10)
            
            # Find contours at threshold
            try:
                from skimage import measure
                HAS_SKIMAGE = True
            except ImportError:
                HAS_SKIMAGE = False
            
            if HAS_SKIMAGE:
                contours = measure.find_contours(density_smooth, 
                                                threshold=np.percentile(density_smooth[high_density_mask], 20))
            else:
                # Fallback: use OpenCV to find contours
                # Normalize density to 0-255
                density_norm = ((density_smooth - density_smooth.min()) / 
                              (density_smooth.max() - density_smooth.min() + 1e-9) * 255).astype(np.uint8)
                threshold_val = int(np.percentile(density_norm[high_density_mask], 20))
                _, binary = cv2.threshold(density_norm, threshold_val, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                
                if len(contours) == 0:
                    raise ValueError("No contours found")
                
                # Convert OpenCV contours to same format as skimage
                contours_list = []
                for cnt in contours:
                    cnt_reshaped = cnt.reshape(-1, 2)
                    contours_list.append(cnt_reshaped)
                contours = contours_list
            
            if len(contours) > 0:
                # Use the largest contour
                largest_contour = max(contours, key=len)
                
                # Convert from grid coordinates to actual coordinates
                boundary_points = np.zeros_like(largest_contour)
                boundary_points[:, 0] = np.interp(largest_contour[:, 0], 
                                                 np.arange(len(yi)), yi)
                boundary_points[:, 1] = np.interp(largest_contour[:, 1], 
                                                  np.arange(len(xi)), xi)
                
                # Close the polygon
                if not np.array_equal(boundary_points[0], boundary_points[-1]):
                    boundary_points = np.vstack([boundary_points, boundary_points[0:1]])
                
                # Only return if it's a reasonable shape (not too small, not too large)
                area = cv2.contourArea(boundary_points.astype(np.float32))
                bbox_area = x_range * y_range
                if 0.01 * bbox_area < area < 0.5 * bbox_area:
                    return boundary_points
        except ImportError:
            # skimage not available, fall through to other methods
            pass
        except Exception as e:
            print(f"    ⚠ Density-based boundary failed: {e}")
    
    # Method 2: Alpha shapes (concave hull) - better than convex hull
    if method in ['alpha', 'adaptive']:
        try:
            # Calculate appropriate alpha value
            if alpha is None:
                # Use average nearest neighbor distance as basis for alpha
                from scipy.spatial.distance import cdist
                if len(points) > 10:
                    # Sample points for speed
                    sample_size = min(100, len(points))
                    sample_points = points[np.random.choice(len(points), sample_size, replace=False)]
                    distances = cdist(sample_points, sample_points)
                    # Remove diagonal (self-distances)
                    distances[distances == 0] = np.inf
                    avg_nn_dist = np.min(distances, axis=1).mean()
                    alpha = 1.0 / (avg_nn_dist * 2)  # Inverse of distance
                else:
                    alpha = 0.1
            
            # Simple alpha shape using Delaunay triangulation
            from scipy.spatial import Delaunay
            tri = Delaunay(points)
            
            # Find edges that form the alpha shape boundary
            # (edges whose circumradius is < 1/alpha)
            boundary_edges = []
            for simplex in tri.simplices:
                # Get triangle vertices
                triangle = points[simplex]
                # Calculate circumradius
                a, b, c = triangle[0], triangle[1], triangle[2]
                # Area of triangle
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                if area > 1e-9:
                    # Circumradius = (a*b*c) / (4*area)
                    side_lengths = [np.linalg.norm(b - c), np.linalg.norm(c - a), np.linalg.norm(a - b)]
                    circumradius = np.prod(side_lengths) / (4 * area)
                    
                    if circumradius < 1.0 / alpha:
                        # This triangle is part of alpha shape
                        for i in range(3):
                            edge = tuple(sorted([simplex[i], simplex[(i+1)%3]]))
                            if edge in boundary_edges:
                                boundary_edges.remove(edge)  # Interior edge
                            else:
                                boundary_edges.append(edge)
            
            if len(boundary_edges) > 0:
                # Reconstruct boundary path from edges
                edge_dict = {}
                for e1, e2 in boundary_edges:
                    if e1 not in edge_dict:
                        edge_dict[e1] = []
                    edge_dict[e1].append(e2)
                    if e2 not in edge_dict:
                        edge_dict[e2] = []
                    edge_dict[e2].append(e1)
                
                # Find starting point (point with only one neighbor)
                start = None
                for p, neighbors in edge_dict.items():
                    if len(neighbors) == 1:
                        start = p
                        break
                
                if start is None and len(edge_dict) > 0:
                    start = list(edge_dict.keys())[0]
                
                if start is not None:
                    # Traverse boundary
                    boundary_path = [start]
                    current = start
                    visited_edges = set()
                    
                    while len(boundary_path) < len(edge_dict) * 2:  # Safety limit
                        if current not in edge_dict or len(edge_dict[current]) == 0:
                            break
                        
                        next_point = edge_dict[current][0]
                        edge = tuple(sorted([current, next_point]))
                        
                        if edge in visited_edges:
                            if len(edge_dict[current]) > 1:
                                next_point = edge_dict[current][1]
                                edge = tuple(sorted([current, next_point]))
                            else:
                                break
                        
                        visited_edges.add(edge)
                        boundary_path.append(next_point)
                        current = next_point
                        
                        if current == start and len(boundary_path) > 3:
                            break
                    
                    if len(boundary_path) >= 3:
                        boundary_points = points[boundary_path]
                        # Close the polygon
                        if not np.array_equal(boundary_points[0], boundary_points[-1]):
                            boundary_points = np.vstack([boundary_points, boundary_points[0:1]])
                        return boundary_points
        except Exception as e:
            print(f"    ⚠ Alpha shape boundary failed: {e}")
    
    # Method 3: Convex hull (fallback)
    if method in ['convex', 'adaptive']:
        try:
            hull = ConvexHull(points)
            boundary_points = points[hull.vertices]
            # Close the polygon
            boundary_points = np.vstack([boundary_points, boundary_points[0]])
            return boundary_points
        except Exception as e:
            print(f"    ⚠ Convex hull boundary failed: {e}")
    
    # Final fallback: bounding box with padding
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    padding = max((x_max - x_min) * 0.1, (y_max - y_min) * 0.1, 10)
    return np.array([
        [x_min - padding, y_min - padding],
        [x_max + padding, y_min - padding],
        [x_max + padding, y_max + padding],
        [x_min - padding, y_max + padding],
        [x_min - padding, y_min - padding]
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


def find_mask_file(tiff_path, mask_base_path=None):
    """
    Find the corresponding mask PNG file for a TIFF file.
    
    Args:
        tiff_path: Path to TIFF file
        mask_base_path: Base path where masks are stored (default: looks in embryo_masks_final/)
    
    Returns:
        Path to mask file if found, None otherwise
    """
    if tiff_path is None:
        return None
    
    tiff_path_obj = Path(tiff_path)
    tiff_stem = tiff_path_obj.stem  # e.g., "B - Substack (1-301)"
    
    # Try default location first
    if mask_base_path is None:
        mask_base_path = Path(__file__).parent / "embryo_masks_final"
    else:
        mask_base_path = Path(mask_base_path)
    
    # Look for mask file: {tiff_stem}_mask_frame0.png
    mask_path = mask_base_path / f"{tiff_stem}_mask_frame0.png"
    
    if mask_path.exists():
        return mask_path
    
    # Try alternative naming (without extension variations)
    alternatives = [
        mask_base_path / f"{tiff_stem}_mask_frame0.png",
        mask_base_path / f"{tiff_path_obj.name.replace('.tif', '').replace('.tiff', '')}_mask_frame0.png",
    ]
    
    for alt_path in alternatives:
        if alt_path.exists():
            return alt_path
    
    return None


def create_embryo_visualization(df_tracks, output_dir, folder_video_key, global_bounds=None, tiff_base_path=None, mask_base_path=None):
    """
    Create visualization for a specific folder/video combination.
    
    Args:
        df_tracks: DataFrame with track data
        output_dir: Output directory
        folder_video_key: (folder, video) tuple
        global_bounds: Optional (x_min, x_max, y_min, y_max) to use consistent dimensions
        tiff_base_path: Optional base path to search for TIFF files
        mask_base_path: Optional base path to search for mask PNG files
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
    
    # Try to find corresponding mask file
    mask_path = None
    if tiff_path:
        mask_path = find_mask_file(tiff_path, mask_base_path)
        if mask_path:
            print(f"    ✓ Found mask file: {mask_path.name}")
    
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
    
    # Mask overlay will be added after all other elements are drawn
    
    # Detect all embryos from TIFF once (if available)
    # NEW APPROACH: Use vector-first detection
    tiff_detections = {}
    
    # Get image dimensions for vector-first detection
    if global_bounds:
        x_min, x_max, y_min, y_max = global_bounds
        img_width = x_max - x_min
        img_height = y_max - y_min
    else:
        valid_xy = df_file[df_file['x'].notna() & df_file['y'].notna()]
        if len(valid_xy) > 0:
            img_width = valid_xy['x'].max() - valid_xy['x'].min()
            img_height = valid_xy['y'].max() - valid_xy['y'].min()
        else:
            img_width = img_height = None
    
    # Use vector-first detection if we have data
    if len(df_file) > 50 and img_width and img_height:
        try:
            vector_first_results = detect_embryos_vector_first(
                df_file, tiff_path=tiff_path, 
                image_width=img_width, image_height=img_height
            )
            
            if len(vector_first_results) >= 2:
                # Convert to tiff_detections format for compatibility
                tiff_detections = {}
                for label in ['A', 'B']:
                    if label in vector_first_results:
                        result = vector_first_results[label]
                        tiff_detections[label] = {
                            'contour': result.get('contour'),
                            'contour_intermediate': result.get('contour_intermediate'),
                            'contour_old': result.get('contour_old'),
                            'head': result.get('head'),
                            'tail': result.get('tail'),
                            'centroid': result.get('centroid')
                        }
                print(f"    ✓ Vector-first detection: Found {len(tiff_detections)} embryos")
        except Exception as e:
            print(f"    ⚠ Error in vector-first detection: {e}")
            # Fallback to TIFF-only detection
            if tiff_path and tiff_path.exists():
                try:
                    all_detections = detect_embryo_from_tiff(tiff_path, embryo_id=None)
                    if all_detections:
                        tiff_detections = all_detections
                except Exception as e2:
                    print(f"    ⚠ Error detecting embryos from TIFF: {e2}")
    else:
        # Fallback to TIFF-only if insufficient data
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
    
    # Transform TIFF centroids to spark coordinate system before matching
    # This ensures we're comparing coordinates in the same space
    tiff_centroids_transformed = {}
    if tiff_detections and tiff_path and tiff_path.exists():
        try:
            with tiff.TiffFile(tiff_path) as tif:
                tiff_img = tif.pages[0].asarray()
                tiff_h, tiff_w = tiff_img.shape[:2]
                
                spark_x_min = df_file['x'].min()
                spark_x_max = df_file['x'].max()
                spark_y_min = df_file['y'].min()
                spark_y_max = df_file['y'].max()
                
                scale_x = (spark_x_max - spark_x_min) / tiff_w
                scale_y = (spark_y_max - spark_y_min) / tiff_h
                
                for tiff_label, detection in tiff_detections.items():
                    tiff_centroid = detection.get('centroid')
                    if tiff_centroid:
                        # Transform TIFF centroid from image pixels to spark coordinates
                        transformed_x = tiff_centroid[0] * scale_x + spark_x_min
                        transformed_y = tiff_centroid[1] * scale_y + spark_y_min
                        tiff_centroids_transformed[tiff_label] = (transformed_x, transformed_y)
        except Exception as e:
            print(f"    ⚠ Error transforming TIFF centroids for matching: {e}")
            # Fallback: use original TIFF centroids (will use larger threshold)
            for tiff_label, detection in tiff_detections.items():
                tiff_centroid = detection.get('centroid')
                if tiff_centroid:
                    tiff_centroids_transformed[tiff_label] = tiff_centroid
    
    # Match TIFF detections to spark labels by closest centroid (one-to-one mapping)
    tiff_to_spark_mapping = {}
    if tiff_centroids_transformed and spark_centroids:
        # For each spark label, find the closest unmatched TIFF detection
        used_tiff_labels = set()
        for spark_label in spark_centroids.keys():
            spark_centroid = spark_centroids[spark_label]
            min_dist = float('inf')
            best_tiff_label = None
            for tiff_label in tiff_centroids_transformed.keys():
                if tiff_label in used_tiff_labels:
                    continue  # Already matched
                tiff_centroid = tiff_centroids_transformed[tiff_label]
                dist = np.sqrt((tiff_centroid[0] - spark_centroid[0])**2 + 
                              (tiff_centroid[1] - spark_centroid[1])**2)
                # Use larger threshold if coordinates weren't transformed (fallback case)
                threshold = 200 if tiff_path and tiff_path.exists() else 500
                if dist < min_dist and dist < threshold:
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
    
    # Draw ALL TIFF contours as reference outlines (dashed gray)
    # This provides a consistent reference for greyscale data visibility
    # We draw them even when matched, so users can see both the colored matched outlines
    # and the reference TIFF outlines for comparison
    if tiff_detections and tiff_path and tiff_path.exists():
        try:
            with tiff.TiffFile(tiff_path) as tif:
                tiff_img = tif.pages[0].asarray()
                tiff_h, tiff_w = tiff_img.shape[:2]
                
                spark_x_min = df_file['x'].min()
                spark_x_max = df_file['x'].max()
                spark_y_min = df_file['y'].min()
                spark_y_max = df_file['y'].max()
                
                scale_x = (spark_x_max - spark_x_min) / tiff_w
                scale_y = (spark_y_max - spark_y_min) / tiff_h
                
                for tiff_label, detection in tiff_detections.items():
                    # Draw all TIFF contours as reference (matched or unmatched)
                    ref_contour = detection.get('contour')
                    ref_contour_intermediate = detection.get('contour_intermediate')
                    ref_contour_old = detection.get('contour_old')
                    
                    # Transform contours to spark coordinate system
                    if ref_contour is not None:
                        # Check if transformation is needed
                        boundary_x_min = ref_contour[:, 0].min()
                        boundary_x_max = ref_contour[:, 0].max()
                        boundary_y_min = ref_contour[:, 1].min()
                        boundary_y_max = ref_contour[:, 1].max()
                        
                        needs_transform = (boundary_x_max <= tiff_w * 1.1 and boundary_x_min >= -tiff_w * 0.1 and
                                        boundary_y_max <= tiff_h * 1.1 and boundary_y_min >= -tiff_h * 0.1)
                        
                        if needs_transform:
                            ref_contour = ref_contour.copy()
                            ref_contour[:, 0] = ref_contour[:, 0] * scale_x + spark_x_min
                            ref_contour[:, 1] = ref_contour[:, 1] * scale_y + spark_y_min
                        
                        ref_contour = np.vstack([ref_contour, ref_contour[0:1]])
                        poly_ref = Polygon(ref_contour, fill=False, edgecolor='gray', 
                                         linewidth=2.5, alpha=0.7, linestyle='--', zorder=5)
                        ax.add_patch(poly_ref)
                        print(f"    → Drawing reference TIFF outline for {tiff_label} (contour shape: {ref_contour.shape})")
                    
                    if ref_contour_intermediate is not None:
                        # Transform intermediate contour
                        boundary_x_min = ref_contour_intermediate[:, 0].min()
                        boundary_x_max = ref_contour_intermediate[:, 0].max()
                        boundary_y_min = ref_contour_intermediate[:, 1].min()
                        boundary_y_max = ref_contour_intermediate[:, 1].max()
                        
                        needs_transform = (boundary_x_max <= tiff_w * 1.1 and boundary_x_min >= -tiff_w * 0.1 and
                                        boundary_y_max <= tiff_h * 1.1 and boundary_y_min >= -tiff_h * 0.1)
                        
                        if needs_transform:
                            ref_contour_intermediate = ref_contour_intermediate.copy()
                            ref_contour_intermediate[:, 0] = ref_contour_intermediate[:, 0] * scale_x + spark_x_min
                            ref_contour_intermediate[:, 1] = ref_contour_intermediate[:, 1] * scale_y + spark_y_min
                        
                        ref_contour_intermediate = np.vstack([ref_contour_intermediate, ref_contour_intermediate[0:1]])
                        poly_ref_int = Polygon(ref_contour_intermediate, fill=False, edgecolor='lightgray', 
                                              linewidth=2.0, alpha=0.6, linestyle='--', zorder=4)
                        ax.add_patch(poly_ref_int)
        except Exception as e:
            import traceback
            print(f"    ⚠ Error drawing reference TIFF contours: {e}")
            print(f"    Traceback: {traceback.format_exc()}")
    
    # Create a context prefix for all warnings in this visualization
    folder_video_prefix = f"[Folder {folder}, Video {video}]"
    
    # Plot embryo outlines and labels
    for embryo_id in ['A', 'B']:
        df_embryo = df_file[df_file['embryo_id'] == embryo_id].copy()
        if len(df_embryo) == 0:
            continue
        
        # Print embryo identifier for warnings
        print(f"    {folder_video_prefix} [Embryo {embryo_id}] Processing...")
        
        colors = embryo_colors.get(embryo_id, embryo_colors['A'])
        
        # Try to get boundary from TIFF detection first (actual grey tissue)
        boundary = None
        tiff_head = None
        tiff_tail = None
        tiff_cement_gland = None
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
                
                # Transform TIFF pixel coordinates to spark data coordinates if needed
                # TIFF contours are in image pixel coordinates, but plot uses spark data coordinates
                if boundary is not None and tiff_path and tiff_path.exists():
                    try:
                        with tiff.TiffFile(tiff_path) as tif:
                            tiff_img = tif.pages[0].asarray()
                            tiff_h, tiff_w = tiff_img.shape[:2]
                            
                            # Get spark data bounds
                            spark_x_min = df_file['x'].min()
                            spark_x_max = df_file['x'].max()
                            spark_y_min = df_file['y'].min()
                            spark_y_max = df_file['y'].max()
                            
                            # Always transform TIFF pixel coordinates to spark data coordinates
                            # TIFF contours are in image pixel coordinates [0, tiff_w] x [0, tiff_h]
                            # Spark data is in a different coordinate system [spark_x_min, spark_x_max] x [spark_y_min, spark_y_max]
                            # Check if transformation is needed by comparing coordinate ranges
                            boundary_x_min = boundary[:, 0].min()
                            boundary_x_max = boundary[:, 0].max()
                            boundary_y_min = boundary[:, 1].min()
                            boundary_y_max = boundary[:, 1].max()
                            
                            # If TIFF coords are in [0, tiff_w] range, they need transformation
                            # If they're already in spark coordinate range, they might already be transformed
                            needs_transform = (boundary_x_max <= tiff_w * 1.1 and boundary_x_min >= -tiff_w * 0.1 and
                                            boundary_y_max <= tiff_h * 1.1 and boundary_y_min >= -tiff_h * 0.1)
                            
                            if needs_transform:
                                # Transform: map TIFF [0, tiff_w] -> spark [spark_x_min, spark_x_max]
                                # and TIFF [0, tiff_h] -> spark [spark_y_min, spark_y_max]
                                scale_x = (spark_x_max - spark_x_min) / tiff_w
                                scale_y = (spark_y_max - spark_y_min) / tiff_h
                                
                                boundary = boundary.copy()
                                boundary[:, 0] = boundary[:, 0] * scale_x + spark_x_min
                                boundary[:, 1] = boundary[:, 1] * scale_y + spark_y_min
                                
                                if boundary_intermediate is not None:
                                    boundary_intermediate = boundary_intermediate.copy()
                                    boundary_intermediate[:, 0] = boundary_intermediate[:, 0] * scale_x + spark_x_min
                                    boundary_intermediate[:, 1] = boundary_intermediate[:, 1] * scale_y + spark_y_min
                                
                                if boundary_old is not None:
                                    boundary_old = boundary_old.copy()
                                    boundary_old[:, 0] = boundary_old[:, 0] * scale_x + spark_x_min
                                    boundary_old[:, 1] = boundary_old[:, 1] * scale_y + spark_y_min
                                
                                # Also transform head/tail and cement gland
                                if detection.get('head'):
                                    head = detection.get('head')
                                    tiff_head = (head[0] * scale_x + spark_x_min, head[1] * scale_y + spark_y_min)
                                else:
                                    tiff_head = None
                                
                                if detection.get('tail'):
                                    tail = detection.get('tail')
                                    tiff_tail = (tail[0] * scale_x + spark_x_min, tail[1] * scale_y + spark_y_min)
                                else:
                                    tiff_tail = None
                                
                                # Transform cement gland location
                                cement_gland_raw = detection.get('cement_gland')
                                if cement_gland_raw:
                                    tiff_cement_gland = (cement_gland_raw[0] * scale_x + spark_x_min, 
                                                        cement_gland_raw[1] * scale_y + spark_y_min)
                                else:
                                    tiff_cement_gland = None
                                
                                print(f"    {folder_video_prefix} [Embryo {embryo_id}] → Transformed TIFF coordinates (scale: {scale_x:.3f}x, {scale_y:.3f}y)")
                            else:
                                # Coordinates already in spark coordinate system, use as-is
                                tiff_head = detection.get('head')
                                tiff_tail = detection.get('tail')
                                cement_gland_raw = detection.get('cement_gland')
                                if cement_gland_raw:
                                    tiff_cement_gland = (cement_gland_raw[0] * scale_x + spark_x_min, 
                                                        cement_gland_raw[1] * scale_y + spark_y_min)
                    except Exception as e:
                        print(f"    {folder_video_prefix} [Embryo {embryo_id}] ⚠ Error transforming TIFF coordinates: {e}")
                        tiff_head = detection.get('head')
                        tiff_tail = detection.get('tail')
                        tiff_cement_gland = detection.get('cement_gland')
                else:
                    tiff_head = detection.get('head')
                    tiff_tail = detection.get('tail')
                    tiff_cement_gland = detection.get('cement_gland')
                
                if boundary is not None:
                    # Close the contour
                    boundary = np.vstack([boundary, boundary[0:1]])
                if boundary_intermediate is not None:
                    # Close the intermediate contour
                    boundary_intermediate = np.vstack([boundary_intermediate, boundary_intermediate[0:1]])
                if boundary_old is not None:
                    # Close the old contour
                    boundary_old = np.vstack([boundary_old, boundary_old[0:1]])
                
                used_tiff_detection = True
        
        # If no match but we have TIFF detections, try to use the closest one anyway
        # This ensures we show greyscale contours even if matching fails
        if not used_tiff_detection and tiff_detections and len(df_embryo) > 0:
            # Get this embryo's centroid
            valid_xy = df_embryo[df_embryo['x'].notna() & df_embryo['y'].notna()]
            if len(valid_xy) > 0:
                emb_centroid = (valid_xy['x'].mean(), valid_xy['y'].mean())
                
                # Find closest TIFF detection (even if beyond normal threshold)
                best_tiff_label = None
                min_dist = float('inf')
                for tiff_label, detection in tiff_detections.items():
                    tiff_centroid = detection.get('centroid')
                    if tiff_centroid:
                        dist = np.sqrt((tiff_centroid[0] - emb_centroid[0])**2 + 
                                     (tiff_centroid[1] - emb_centroid[1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            best_tiff_label = tiff_label
                
                # Use it if within reasonable distance (increased threshold for coordinate system differences)
                if best_tiff_label and min_dist < 1000:  # More lenient threshold
                    detection = tiff_detections[best_tiff_label]
                    boundary = detection.get('contour')
                    boundary_intermediate = detection.get('contour_intermediate')
                    boundary_old = detection.get('contour_old')
                    
                    print(f"    {folder_video_prefix} [Embryo {embryo_id}] → Got TIFF detection, boundary is {'None' if boundary is None else f'shape {boundary.shape}'}, tiff_path={'None' if tiff_path is None else str(tiff_path)}")
                    
                    # Apply same coordinate transformation as above
                    if boundary is not None:
                        if tiff_path and tiff_path.exists():
                            print(f"    {folder_video_prefix} [Embryo {embryo_id}] → Found boundary and TIFF path, applying transformation...")
                            try:
                                with tiff.TiffFile(tiff_path) as tif:
                                    tiff_img = tif.pages[0].asarray()
                                    tiff_h, tiff_w = tiff_img.shape[:2]
                                    
                                    spark_x_min = df_file['x'].min()
                                    spark_x_max = df_file['x'].max()
                                    spark_y_min = df_file['y'].min()
                                    spark_y_max = df_file['y'].max()
                                    
                                    boundary_x_min = boundary[:, 0].min()
                                    boundary_x_max = boundary[:, 0].max()
                                    boundary_y_min = boundary[:, 1].min()
                                    boundary_y_max = boundary[:, 1].max()
                                    
                                    needs_transform = (boundary_x_max <= tiff_w * 1.1 and boundary_x_min >= -tiff_w * 0.1 and
                                                    boundary_y_max <= tiff_h * 1.1 and boundary_y_min >= -tiff_h * 0.1)
                                    
                                    if needs_transform:
                                        print(f"    {folder_video_prefix} [Embryo {embryo_id}] → Applying coordinate transformation (boundary: X=[{boundary_x_min:.1f}, {boundary_x_max:.1f}], Y=[{boundary_y_min:.1f}, {boundary_y_max:.1f}], TIFF: {tiff_w}x{tiff_h})")
                                        scale_x = (spark_x_max - spark_x_min) / tiff_w
                                        scale_y = (spark_y_max - spark_y_min) / tiff_h
                                        
                                        boundary = boundary.copy()
                                        boundary[:, 0] = boundary[:, 0] * scale_x + spark_x_min
                                        boundary[:, 1] = boundary[:, 1] * scale_y + spark_y_min
                                        
                                        if boundary_intermediate is not None:
                                            boundary_intermediate = boundary_intermediate.copy()
                                            boundary_intermediate[:, 0] = boundary_intermediate[:, 0] * scale_x + spark_x_min
                                            boundary_intermediate[:, 1] = boundary_intermediate[:, 1] * scale_y + spark_y_min
                                        
                                        if boundary_old is not None:
                                            boundary_old = boundary_old.copy()
                                            boundary_old[:, 0] = boundary_old[:, 0] * scale_x + spark_x_min
                                            boundary_old[:, 1] = boundary_old[:, 1] * scale_y + spark_y_min
                                        
                                        if detection.get('head'):
                                            head = detection.get('head')
                                            tiff_head = (head[0] * scale_x + spark_x_min, head[1] * scale_y + spark_y_min)
                                        else:
                                            tiff_head = None
                                        
                                        if detection.get('tail'):
                                            tail = detection.get('tail')
                                            tiff_tail = (tail[0] * scale_x + spark_x_min, tail[1] * scale_y + spark_y_min)
                                        else:
                                            tiff_tail = None
                                        
                                        print(f"    {folder_video_prefix} [Embryo {embryo_id}] → Transformed TIFF coordinates (scale: {scale_x:.3f}x, {scale_y:.3f}y)")
                                    else:
                                        print(f"    {folder_video_prefix} [Embryo {embryo_id}] → TIFF coordinates already in spark system (no transform needed)")
                                        tiff_head = detection.get('head')
                                        tiff_tail = detection.get('tail')
                                        cement_gland_raw = detection.get('cement_gland')
                                        if cement_gland_raw:
                                            tiff_cement_gland = (cement_gland_raw[0] * scale_x + spark_x_min, 
                                                                cement_gland_raw[1] * scale_y + spark_y_min)
                            except Exception as e:
                                print(f"    {folder_video_prefix} [Embryo {embryo_id}] ⚠ Error transforming TIFF coordinates: {e}")
                                tiff_head = detection.get('head')
                                tiff_tail = detection.get('tail')
                                tiff_cement_gland = detection.get('cement_gland')
                        else:
                            print(f"    {folder_video_prefix} [Embryo {embryo_id}] ⚠ Boundary found but TIFF path missing (tiff_path={tiff_path})")
                            tiff_head = detection.get('head')
                            tiff_tail = detection.get('tail')
                            tiff_cement_gland = detection.get('cement_gland')
                    else:
                        tiff_head = detection.get('head')
                        tiff_tail = detection.get('tail')
                        tiff_cement_gland = detection.get('cement_gland')
                    
                    if boundary is not None:
                        boundary = np.vstack([boundary, boundary[0:1]])
                    if boundary_intermediate is not None:
                        boundary_intermediate = np.vstack([boundary_intermediate, boundary_intermediate[0:1]])
                    if boundary_old is not None:
                        boundary_old = np.vstack([boundary_old, boundary_old[0:1]])
                    used_tiff_detection = True
                    print(f"    {folder_video_prefix} [Embryo {embryo_id}] → Using closest TIFF detection (distance: {min_dist:.1f}px)")
        
        # If still no match, show ALL TIFF contours as reference (even if not matched)
        # This ensures greyscale data is always visible
        if not used_tiff_detection and tiff_detections:
            # For unmatched embryos, we'll draw all TIFF contours as reference
            # But we'll only assign the closest one to this embryo
            if len(df_embryo) > 0:
                valid_xy = df_embryo[df_embryo['x'].notna() & df_embryo['y'].notna()]
                if len(valid_xy) > 0:
                    emb_centroid = (valid_xy['x'].mean(), valid_xy['y'].mean())
                    best_tiff_label = None
                    min_dist = float('inf')
                    for tiff_label, detection in tiff_detections.items():
                        tiff_centroid = detection.get('centroid')
                        if tiff_centroid:
                            dist = np.sqrt((tiff_centroid[0] - emb_centroid[0])**2 + 
                                         (tiff_centroid[1] - emb_centroid[1])**2)
                            if dist < min_dist:
                                min_dist = dist
                                best_tiff_label = tiff_label
                    
                    if best_tiff_label:
                        detection = tiff_detections[best_tiff_label]
                        boundary = detection.get('contour')
                        boundary_intermediate = detection.get('contour_intermediate')
                        boundary_old = detection.get('contour_old')
                        
                        # Apply coordinate transformation (same as above)
                        if boundary is not None:
                            if tiff_path and tiff_path.exists():
                                print(f"    {folder_video_prefix} [Embryo {embryo_id}] → Applying coordinate transformation...")
                                try:
                                    with tiff.TiffFile(tiff_path) as tif:
                                        tiff_img = tif.pages[0].asarray()
                                        tiff_h, tiff_w = tiff_img.shape[:2]
                                        
                                        spark_x_min = df_file['x'].min()
                                        spark_x_max = df_file['x'].max()
                                        spark_y_min = df_file['y'].min()
                                        spark_y_max = df_file['y'].max()
                                        
                                        boundary_x_min = boundary[:, 0].min()
                                        boundary_x_max = boundary[:, 0].max()
                                        boundary_y_min = boundary[:, 1].min()
                                        boundary_y_max = boundary[:, 1].max()
                                        
                                        needs_transform = (boundary_x_max <= tiff_w * 1.1 and boundary_x_min >= -tiff_w * 0.1 and
                                                        boundary_y_max <= tiff_h * 1.1 and boundary_y_min >= -tiff_h * 0.1)
                                        
                                        if needs_transform:
                                            scale_x = (spark_x_max - spark_x_min) / tiff_w
                                            scale_y = (spark_y_max - spark_y_min) / tiff_h
                                            
                                            boundary = boundary.copy()
                                            boundary[:, 0] = boundary[:, 0] * scale_x + spark_x_min
                                            boundary[:, 1] = boundary[:, 1] * scale_y + spark_y_min
                                            
                                            if boundary_intermediate is not None:
                                                boundary_intermediate = boundary_intermediate.copy()
                                                boundary_intermediate[:, 0] = boundary_intermediate[:, 0] * scale_x + spark_x_min
                                                boundary_intermediate[:, 1] = boundary_intermediate[:, 1] * scale_y + spark_y_min
                                            
                                            if boundary_old is not None:
                                                boundary_old = boundary_old.copy()
                                                boundary_old[:, 0] = boundary_old[:, 0] * scale_x + spark_x_min
                                                boundary_old[:, 1] = boundary_old[:, 1] * scale_y + spark_y_min
                                            
                                            if detection.get('head'):
                                                head = detection.get('head')
                                                tiff_head = (head[0] * scale_x + spark_x_min, head[1] * scale_y + spark_y_min)
                                            else:
                                                tiff_head = None
                                            
                                            if detection.get('tail'):
                                                tail = detection.get('tail')
                                                tiff_tail = (tail[0] * scale_x + spark_x_min, tail[1] * scale_y + spark_y_min)
                                            else:
                                                tiff_tail = None
                                            
                                            # Transform cement gland location
                                            cement_gland_raw = detection.get('cement_gland')
                                            if cement_gland_raw:
                                                tiff_cement_gland = (cement_gland_raw[0] * scale_x + spark_x_min, 
                                                                    cement_gland_raw[1] * scale_y + spark_y_min)
                                            else:
                                                tiff_cement_gland = None
                                            
                                            print(f"    {folder_video_prefix} [Embryo {embryo_id}] → Transformed TIFF coordinates (scale: {scale_x:.3f}x, {scale_y:.3f}y)")
                                        else:
                                            # Coordinates already in spark coordinate system
                                            print(f"    {folder_video_prefix} [Embryo {embryo_id}] → TIFF coordinates already in spark system (no transform needed)")
                                            tiff_head = detection.get('head')
                                            tiff_tail = detection.get('tail')
                                            tiff_cement_gland = detection.get('cement_gland')
                                except Exception as e:
                                    print(f"    {folder_video_prefix} [Embryo {embryo_id}] ⚠ Error transforming TIFF coordinates: {e}")
                                    tiff_head = detection.get('head')
                                    tiff_tail = detection.get('tail')
                                    tiff_cement_gland = detection.get('cement_gland')
                            else:
                                # No TIFF path available - use coordinates as-is
                                print(f"    {folder_video_prefix} [Embryo {embryo_id}] ⚠ No TIFF path for coordinate transformation")
                                tiff_head = detection.get('head')
                                tiff_tail = detection.get('tail')
                                tiff_cement_gland = detection.get('cement_gland')
                        else:
                            # No boundary from detection
                            tiff_head = detection.get('head')
                            tiff_tail = detection.get('tail')
                            tiff_cement_gland = detection.get('cement_gland')
                        
                        if boundary is not None:
                            boundary = np.vstack([boundary, boundary[0:1]])
                        if boundary_intermediate is not None:
                            boundary_intermediate = np.vstack([boundary_intermediate, boundary_intermediate[0:1]])
                        if boundary_old is not None:
                            boundary_old = np.vstack([boundary_old, boundary_old[0:1]])
                        used_tiff_detection = True
                        print(f"    {folder_video_prefix} [Embryo {embryo_id}] → Using closest TIFF detection (unmatched, distance: {min_dist:.1f}px)")
        
        # Fallback to spark-based boundary if TIFF detection failed
        if boundary is None:
            # If we have very few sparks for this embryo, try using all sparks in the file
            # and then extract the region near this embryo's centroid
            if len(df_embryo) < 50:
                # Use all sparks to get better density estimate
                all_sparks = df_file[df_file['x'].notna() & df_file['y'].notna()].copy()
                if len(all_sparks) > 100:
                    # Get centroid of this embryo
                    if len(df_embryo) > 0:
                        emb_centroid = (df_embryo['x'].mean(), df_embryo['y'].mean())
                        # Filter sparks near this centroid (within 2x the spread of embryo sparks)
                        x_spread = df_embryo['x'].max() - df_embryo['x'].min() if len(df_embryo) > 1 else 100
                        y_spread = df_embryo['y'].max() - df_embryo['y'].min() if len(df_embryo) > 1 else 100
                        radius = max(x_spread, y_spread) * 2
                        
                        # Get sparks within radius
                        nearby_sparks = all_sparks[
                            ((all_sparks['x'] - emb_centroid[0])**2 + 
                             (all_sparks['y'] - emb_centroid[1])**2) < radius**2
                        ].copy()
                        
                        if len(nearby_sparks) > 50:
                            # Use nearby sparks for better boundary
                            boundary = get_embryo_boundary_from_sparks(nearby_sparks, method='adaptive')
            
            # If still no boundary, try with just this embryo's sparks
            if boundary is None:
                # Try adaptive method first (tries density, then alpha, then convex)
                boundary = get_embryo_boundary_from_sparks(df_embryo, method='adaptive')
                if boundary is None:
                    # Fallback to density method
                    boundary = get_embryo_boundary_from_sparks(df_embryo, method='density')
        
        if boundary is not None:
            # Draw triple outlines: inner (old), middle (intermediate), outer (new)
            if used_tiff_detection:
                # Draw inner outline (old, restrictive) - lighter, thinner
                if boundary_old is not None:
                    try:
                        poly_inner = Polygon(boundary_old, fill=False, edgecolor=colors['outline'], 
                                            linewidth=1.5, alpha=0.6, linestyle='-')
                        ax.add_patch(poly_inner)
                    except Exception as e:
                        print(f"    [Embryo {embryo_id}] ⚠ Error drawing inner outline: {e}")
                
                # Draw middle outline (intermediate, better head capture) - medium, visible
                if boundary_intermediate is not None:
                    try:
                        poly_middle = Polygon(boundary_intermediate, fill=False, edgecolor=colors['outline'], 
                                            linewidth=2.0, alpha=0.8, linestyle='-')
                        ax.add_patch(poly_middle)
                    except Exception as e:
                        print(f"    [Embryo {embryo_id}] ⚠ Error drawing middle outline: {e}")
                
                # Draw outer outline (new, inclusive) - thicker, more visible
                try:
                    poly_outer = Polygon(boundary, fill=False, edgecolor=colors['outline'], 
                                        linewidth=2.5, alpha=0.9, linestyle='-')
                    ax.add_patch(poly_outer)
                except Exception as e:
                    print(f"    [Embryo {embryo_id}] ⚠ Error drawing outer outline: {e}")
                    print(f"    [Embryo {embryo_id}] Boundary shape: {boundary.shape if boundary is not None else None}, dtype: {boundary.dtype if boundary is not None else None}")
            else:
                # Reconstructed from sparks (approximate)
                try:
                    poly = Polygon(boundary, fill=False, edgecolor=colors['outline'], 
                                  linewidth=2, alpha=0.6, linestyle='--')
                    ax.add_patch(poly)
                except Exception as e:
                    print(f"    [Embryo {embryo_id}] ⚠ Error drawing spark-based outline: {e}")
        else:
            # Debug: why is boundary None?
            if tiff_detections:
                print(f"    [Embryo {embryo_id}] ⚠ No boundary despite {len(tiff_detections)} TIFF detections available")
                if embryo_id in tiff_to_spark_mapping:
                    print(f"    → Mapping exists: {embryo_id} -> {tiff_to_spark_mapping[embryo_id]}")
                else:
                    print(f"    → No mapping for {embryo_id}")
        
        # Get head/tail positions - prefer TIFF detection, fallback to spark data
        if tiff_head and tiff_tail:
            head_pos = tiff_head
            tail_pos = tiff_tail
        else:
            head_pos, tail_pos = get_head_tail_positions(df_embryo)
        
        # CRITICAL VALIDATION: Enforce spatial constraints before visualization
        if head_pos and tail_pos:
            # Get image bounds (use global bounds if available, otherwise from data)
            if global_bounds:
                x_min, x_max, y_min, y_max = global_bounds
                image_width = x_max - x_min
                image_center_x = (x_min + x_max) / 2
            else:
                # Estimate from spark data
                x_min = df_file['x'].min()
                x_max = df_file['x'].max()
                image_width = x_max - x_min
                image_center_x = (x_min + x_max) / 2
            
            head_x = head_pos[0]
            tail_x = tail_pos[0]
            
            # Validation rule 1: A head/tail must NEVER be on right side
            if embryo_id == 'A':
                if head_x >= image_center_x:
                    print(f"    {folder_video_prefix} [Embryo A] ⚠ CRITICAL: Head at x={head_x:.1f} is on RIGHT side (center={image_center_x:.1f})")
                    print(f"    → Correcting: finding leftmost point in A's data")
                    # Use leftmost spark point as head
                    if len(df_embryo) > 0:
                        leftmost_idx = df_embryo['x'].idxmin()
                        head_pos = (df_embryo.loc[leftmost_idx, 'x'], df_embryo.loc[leftmost_idx, 'y'])
                        head_x = head_pos[0]
                
                if tail_x >= image_center_x:
                    print(f"    {folder_video_prefix} [Embryo A] ⚠ CRITICAL: Tail at x={tail_x:.1f} is on RIGHT side (center={image_center_x:.1f})")
                    print(f"    → Correcting: finding rightmost point on LEFT side in A's data")
                    # Use rightmost point on left side as tail
                    if len(df_embryo) > 0:
                        left_side_data = df_embryo[df_embryo['x'] < image_center_x]
                        if len(left_side_data) > 0:
                            rightmost_left_idx = left_side_data['x'].idxmax()
                            tail_pos = (left_side_data.loc[rightmost_left_idx, 'x'], left_side_data.loc[rightmost_left_idx, 'y'])
                            tail_x = tail_pos[0]
                        else:
                            # No left side data - use leftmost as tail
                            leftmost_idx = df_embryo['x'].idxmin()
                            tail_pos = (df_embryo.loc[leftmost_idx, 'x'], df_embryo.loc[leftmost_idx, 'y'])
                            tail_x = tail_pos[0]
            
            # Validation rule 2: B head/tail must NEVER be on left side
            elif embryo_id == 'B':
                if head_x < image_center_x:
                    print(f"    {folder_video_prefix} [Embryo B] ⚠ CRITICAL: Head at x={head_x:.1f} is on LEFT side (center={image_center_x:.1f})")
                    print(f"    → Correcting: finding rightmost point in B's data")
                    # Use rightmost spark point as head
                    if len(df_embryo) > 0:
                        rightmost_idx = df_embryo['x'].idxmax()
                        head_pos = (df_embryo.loc[rightmost_idx, 'x'], df_embryo.loc[rightmost_idx, 'y'])
                        head_x = head_pos[0]
                
                if tail_x < image_center_x:
                    print(f"    {folder_video_prefix} [Embryo B] ⚠ CRITICAL: Tail at x={tail_x:.1f} is on LEFT side (center={image_center_x:.1f})")
                    print(f"    → Correcting: finding leftmost point on RIGHT side in B's data")
                    # Use leftmost point on right side as tail
                    if len(df_embryo) > 0:
                        right_side_data = df_embryo[df_embryo['x'] >= image_center_x]
                        if len(right_side_data) > 0:
                            leftmost_right_idx = right_side_data['x'].idxmin()
                            tail_pos = (right_side_data.loc[leftmost_right_idx, 'x'], right_side_data.loc[leftmost_right_idx, 'y'])
                            tail_x = tail_pos[0]
                        else:
                            print(f"    → No right-side data available - REJECTING this detection")
                            continue  # Skip this embryo
            
            # FINAL VALIDATION: After corrections, ensure constraints are still met
            head_x = head_pos[0]
            tail_x = tail_pos[0]
            
            # Re-check spatial constraints after corrections
            if embryo_id == 'A':
                if head_x >= image_center_x or tail_x >= image_center_x:
                    print(f"    {folder_video_prefix} [Embryo A] ⚠ CRITICAL: Still violates spatial constraints after correction")
                    print(f"    → Head x={head_x:.1f}, Tail x={tail_x:.1f}, Center={image_center_x:.1f}")
                    print(f"    → REJECTING this detection - skipping visualization")
                    continue  # Skip this embryo
            elif embryo_id == 'B':
                if head_x < image_center_x or tail_x < image_center_x:
                    print(f"    {folder_video_prefix} [Embryo B] ⚠ CRITICAL: Still violates spatial constraints after correction")
                    print(f"    → Head x={head_x:.1f}, Tail x={tail_x:.1f}, Center={image_center_x:.1f}")
                    print(f"    → REJECTING this detection - skipping visualization")
                    continue  # Skip this embryo
            
            # Validation rule 3: Head-tail distance must be above minimum threshold
            head_tail_dist = np.sqrt((head_pos[0] - tail_pos[0])**2 + (head_pos[1] - tail_pos[1])**2)
            min_head_tail_length = max(image_width * 0.05, 50)  # 5% of image width or 50 pixels
            
            if head_tail_dist < min_head_tail_length:
                print(f"    {folder_video_prefix} [Embryo {embryo_id}] ⚠ CRITICAL: Head-tail distance too short ({head_tail_dist:.1f}px < {min_head_tail_length:.1f}px)")
                print(f"    → REJECTING this detection - skipping visualization")
                continue  # Skip this embryo
        
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
        
        # Draw cement gland location if detected
        cement_gland_pos = tiff_cement_gland if tiff_cement_gland else None
        if cement_gland_pos:
            # Draw cement gland as a distinctive marker (purple/magenta circle)
            ax.plot(cement_gland_pos[0], cement_gland_pos[1], 'o', color='magenta', markersize=10,
                   markeredgecolor='white', markeredgewidth=1.5, label=f'Embryo {embryo_id} Cement Gland',
                   zorder=10, alpha=0.9)
            ax.annotate(f'{embryo_id} CG', cement_gland_pos, xytext=(10, -15), 
                       textcoords='offset points', color='magenta', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor='magenta', linewidth=1.5))
        elif head_pos:
            # If cement gland not detected but we have head position, draw a marker near head
            # This helps visualize where cement gland should be (on ventral side of head)
            # Calculate ventral direction from head-tail axis
            if tail_pos:
                axis_vec = np.array([tail_pos[0] - head_pos[0], tail_pos[1] - head_pos[1]])
                axis_len = np.linalg.norm(axis_vec)
                if axis_len > 0:
                    axis_norm = axis_vec / axis_len
                    perp_vec = np.array([-axis_norm[1], axis_norm[0]])  # Perpendicular (ventral direction)
                    # Place marker slightly ventral to head (about 5% of embryo length)
                    marker_offset = perp_vec * (axis_len * 0.05)
                    cg_marker_pos = (head_pos[0] + marker_offset[0], head_pos[1] + marker_offset[1])
                    # Draw as a smaller, lighter marker to indicate "expected location"
                    ax.plot(cg_marker_pos[0], cg_marker_pos[1], 'o', color='magenta', markersize=6,
                           markeredgecolor='white', markeredgewidth=1, label=f'Embryo {embryo_id} CG (expected)',
                           zorder=9, alpha=0.6)
                    ax.annotate(f'{embryo_id} CG?', cg_marker_pos, xytext=(8, -12), 
                               textcoords='offset points', color='magenta', fontsize=9, 
                               bbox=dict(boxstyle='round', facecolor='black', alpha=0.6, edgecolor='magenta', linewidth=1))
        
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
    
    # Add vector arrows for major waves (similar to flow_field_aurora)
    if HAS_SCIPY:
        # Filter to valid velocity data
        valid_vectors = df_file[(df_file['vx'].notna()) & (df_file['vy'].notna()) & 
                                (df_file['speed'].notna()) & (df_file['speed'] > 0)].copy()
        
        if len(valid_vectors) > 0:
            # Sample for performance if too many points
            if len(valid_vectors) > 20000:
                valid_vectors = valid_vectors.sample(n=20000, random_state=42)
            
            # Create grid for interpolation
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            # Grid resolution (adjust based on image size)
            grid_res = min(100, int(max(x_range, y_range) / 20))
            if grid_res < 20:
                grid_res = 20
            
            xi = np.linspace(x_min, x_max, grid_res)
            yi = np.linspace(y_min, y_max, grid_res)
            xi_grid, yi_grid = np.meshgrid(xi, yi)
            
            # Sample grid for quiver (every Nth point)
            sample_step = max(1, grid_res // 25)  # ~25 arrows per dimension
            x_sample = xi[::sample_step]
            y_sample = yi[::sample_step]
            x_grid_sample, y_grid_sample = np.meshgrid(x_sample, y_sample)
            
            # Interpolate velocities onto sampled grid
            try:
                vx_grid = griddata(
                    (valid_vectors['x'], valid_vectors['y']),
                    valid_vectors['vx'],
                    (x_grid_sample, y_grid_sample),
                    method='cubic',
                    fill_value=0
                )
                vy_grid = griddata(
                    (valid_vectors['x'], valid_vectors['y']),
                    valid_vectors['vy'],
                    (x_grid_sample, y_grid_sample),
                    method='cubic',
                    fill_value=0
                )
                
                # Normalize vectors for display (scale to reasonable arrow length)
                magnitude = np.sqrt(vx_grid**2 + vy_grid**2)
                magnitude[magnitude == 0] = 1  # Avoid division by zero
                
                # Scale arrows to be visible but not too large
                arrow_scale = min(x_range, y_range) / grid_res * sample_step * 0.6
                vx_norm = vx_grid / magnitude * arrow_scale
                vy_norm = vy_grid / magnitude * arrow_scale
                
                # Only draw arrows where magnitude is significant (filter out noise)
                min_magnitude = np.percentile(magnitude[magnitude > 0], 20)  # Bottom 20th percentile
                mask = magnitude > min_magnitude
                
                # Draw vectors with white color, semi-transparent
                ax.quiver(x_grid_sample[mask], y_grid_sample[mask], 
                         vx_norm[mask], vy_norm[mask],
                         angles='xy', scale_units='xy', scale=1,
                         color='white', alpha=0.4, width=0.002, 
                         headwidth=2.5, headlength=3, headaxislength=2.5,
                         zorder=4)  # Above outlines but below labels
            except Exception as e:
                # If interpolation fails, skip vector arrows
                pass
    
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
    
    # Overlay mask if available - do this LAST so it appears behind all other elements
    if mask_path and mask_path.exists() and tiff_path and tiff_path.exists():
        try:
            import cv2
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                mask_h, mask_w = mask_img.shape
                
                # Get TIFF dimensions for coordinate transformation
                with tiff.TiffFile(tiff_path) as tif:
                    tiff_img = tif.pages[0].asarray()
                    tiff_h, tiff_w = tiff_img.shape[:2]
                    
                    # Resize mask to match TIFF dimensions if they differ
                    if mask_w != tiff_w or mask_h != tiff_h:
                        mask_img = cv2.resize(mask_img, (tiff_w, tiff_h), interpolation=cv2.INTER_NEAREST)
                        print(f"    → Resized mask from {mask_w}x{mask_h} to {tiff_w}x{tiff_h} to match TIFF")
                    
                    # Get spark coordinate bounds
                    spark_x_min = df_file['x'].min()
                    spark_x_max = df_file['x'].max()
                    spark_y_min = df_file['y'].min()
                    spark_y_max = df_file['y'].max()
                    
                    # Create extent for imshow: [left, right, bottom, top] in spark coordinates
                    # Mask now matches TIFF size, so use spark bounds directly
                    extent = [
                        spark_x_min,  # left
                        spark_x_max,  # right
                        spark_y_max,  # bottom (note: y-axis is flipped for images)
                        spark_y_min   # top
                    ]
                    
                    # Create bright green overlay using filled contours (more reliable than imshow)
                    # Find contours in the mask
                    contours_mask, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if len(contours_mask) > 0:
                        # Transform contours from mask pixel coordinates to spark coordinates
                        scale_x = (spark_x_max - spark_x_min) / tiff_w
                        scale_y = (spark_y_max - spark_y_min) / tiff_h
                        
                        for contour in contours_mask:
                            # Transform contour points
                            contour_transformed = contour.reshape(-1, 2).astype(np.float32)
                            contour_transformed[:, 0] = contour_transformed[:, 0] * scale_x + spark_x_min
                            contour_transformed[:, 1] = contour_transformed[:, 1] * scale_y + spark_y_min
                            
                            # Close the contour
                            if len(contour_transformed) > 0:
                                contour_closed = np.vstack([contour_transformed, contour_transformed[0:1]])
                                
                                # Draw filled polygon with bright green
                                poly = Polygon(contour_closed, facecolor='lime', edgecolor='lime', 
                                             linewidth=0, alpha=0.5, zorder=1)
                                ax.add_patch(poly)
                        
                        print(f"    ✓ Overlaid mask: {mask_path.name} ({len(contours_mask)} contours, extent: {spark_x_min:.1f}-{spark_x_max:.1f}, {spark_y_min:.1f}-{spark_y_max:.1f})")
                    else:
                        print(f"    ⚠ No contours found in mask: {mask_path.name}")
        except Exception as e:
            print(f"    ⚠ Error overlaying mask: {e}")
            import traceback
            traceback.print_exc()
    
    ax.set_xlabel('X (pixels)', color='white', fontsize=12)
    ax.set_ylabel('Y (pixels)', color='white', fontsize=12)
    title_text = f'Folder {folder} - {video}\nEmbryo Detection & Poke Location'
    if mask_path and mask_path.exists():
        title_text += ' [Mask Overlay]'
    ax.set_title(title_text, color='white', fontsize=14, fontweight='bold')
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


def create_region_summary_page(df_tracks, pdf):
    """
    Create a summary page showing spark counts by region and add it to the PDF.
    Uses pixel-based counting (each spark = 1 pixel).
    
    Args:
        df_tracks: DataFrame with spark track data (must have 'region' column)
        pdf: PdfPages object (already open)
    """
    if 'region' not in df_tracks.columns:
        return  # No region data available
    
    # Filter out empty/unknown regions
    df_with_regions = df_tracks[
        df_tracks['region'].notna() & 
        (df_tracks['region'] != '') & 
        (df_tracks['region'] != 'unknown')
    ].copy()
    
    if len(df_with_regions) == 0:
        return  # No valid region data
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Sparks Summary by Anatomical Region (Pixel-Based)', fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Pixel count by region (bar chart) - each spark = 1 pixel
    # Count unique track_ids per region (each track_id = 1 activated pixel)
    region_pixel_counts = df_with_regions.groupby('region')['track_id'].nunique().sort_values(ascending=True)
    axes[0, 0].barh(range(len(region_pixel_counts)), region_pixel_counts.values, color='steelblue')
    axes[0, 0].set_yticks(range(len(region_pixel_counts)))
    axes[0, 0].set_yticklabels(region_pixel_counts.index)
    axes[0, 0].set_xlabel('Number of Activated Pixels', fontsize=10)
    axes[0, 0].set_title('Total Activated Pixels by Region\n(Each spark = 1 pixel)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, v in enumerate(region_pixel_counts.values):
        axes[0, 0].text(v, i, f' {v:,}', va='center', fontsize=9)
    
    # 2. Mean speed by region
    if 'speed' in df_with_regions.columns:
        speed_per_region = df_with_regions.groupby('region')['speed'].mean().sort_values(ascending=True)
        axes[0, 1].barh(range(len(speed_per_region)), speed_per_region.values, color='coral')
        axes[0, 1].set_yticks(range(len(speed_per_region)))
        axes[0, 1].set_yticklabels(speed_per_region.index)
        axes[0, 1].set_xlabel('Mean Speed (pixels/second)', fontsize=10)
        axes[0, 1].set_title('Mean Speed by Region', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, v in enumerate(speed_per_region.values):
            axes[0, 1].text(v, i, f' {v:.2f}', va='center', fontsize=9)
    else:
        axes[0, 1].text(0.5, 0.5, 'Speed data not available', 
                        ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Mean Speed by Region', fontsize=12, fontweight='bold')
    
    # 3. Region distribution by embryo (pixel counts)
    if 'embryo_id' in df_with_regions.columns:
        # Count unique pixels (track_ids) per region and embryo
        region_embryo_pixels = df_with_regions.groupby(['region', 'embryo_id'])['track_id'].nunique().unstack(fill_value=0)
        region_embryo_pixels.plot(kind='barh', ax=axes[1, 0], color=['#4CAF50', '#FF9800'], width=0.8)
        axes[1, 0].set_xlabel('Number of Activated Pixels', fontsize=10)
        axes[1, 0].set_ylabel('Region', fontsize=10)
        axes[1, 0].set_title('Activated Pixels by Region and Embryo\n(Each spark = 1 pixel)', fontsize=12, fontweight='bold')
        axes[1, 0].legend(title='Embryo', fontsize=9)
        axes[1, 0].grid(True, alpha=0.3, axis='x')
    else:
        axes[1, 0].text(0.5, 0.5, 'Embryo ID data not available', 
                        ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Activated Pixels by Region and Embryo', fontsize=12, fontweight='bold')
    
    # 4. Summary statistics table
    axes[1, 1].axis('off')
    
    # Calculate statistics using pixel counts
    total_pixels = df_tracks['track_id'].nunique()  # Total unique pixels across all data
    pixels_with_region = df_with_regions['track_id'].nunique()  # Pixels with region annotation
    unique_regions = df_with_regions['region'].nunique()
    events_total = len(df_tracks)  # Total detection events (for reference)
    region_coverage = (pixels_with_region / total_pixels * 100) if total_pixels > 0 else 0
    
    # Create summary text
    summary_text = f"""
SUMMARY STATISTICS (Pixel-Based)

Total Activated Pixels: {total_pixels:,}
Pixels with Region: {pixels_with_region:,} ({region_coverage:.1f}%)
Total Detection Events: {events_total:,}
Unique Regions: {unique_regions}

TOP 5 REGIONS BY PIXEL COUNT:
"""
    
    top_regions = region_pixel_counts.tail(5)
    for i, (region, pixel_count) in enumerate(top_regions.items(), 1):
        pct = (pixel_count / pixels_with_region * 100) if pixels_with_region > 0 else 0
        summary_text += f"{i}. {region}: {pixel_count:,} pixels ({pct:.1f}%)\n"
    
    if 'speed' in df_with_regions.columns:
        summary_text += f"\n\nSPEED STATISTICS:\n"
        summary_text += f"Overall Mean Speed: {df_with_regions['speed'].mean():.2f} px/s\n"
        summary_text += f"Overall Median Speed: {df_with_regions['speed'].median():.2f} px/s\n"
        fastest_region = df_with_regions.groupby('region')['speed'].mean().idxmax()
        fastest_speed = df_with_regions.groupby('region')['speed'].mean().max()
        summary_text += f"Fastest Region: {fastest_region} ({fastest_speed:.2f} px/s)\n"
    
    axes[1, 1].text(0.1, 0.95, summary_text, transform=axes[1, 1].transAxes,
                    fontsize=10, family='monospace', verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Add to PDF
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_pdf_from_images(image_paths, output_pdf_path, images_per_page=1, summary_data=None, warning_log_path=None, df_tracks=None):
    """
    Create a PDF from a list of image paths.
    
    Args:
        image_paths: List of (path, title) tuples
        output_pdf_path: Path to output PDF
        images_per_page: Number of images per page (1 or 2)
        summary_data: Optional list of summary data dicts for first page
        warning_log_path: Optional path to warning log file to append at end
        df_tracks: Optional DataFrame with spark track data for region summary
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
        
        # Append warning log at the end if provided
        if warning_log_path and Path(warning_log_path).exists():
            with open(warning_log_path, 'r') as log_file:
                log_content = log_file.read()
            
            if log_content.strip() and 'No warnings detected' not in log_content:
                # Create text pages with the log, split into multiple pages if needed
                lines = log_content.split('\n')
                max_lines_per_page = 50
                
                for page_start in range(0, len(lines), max_lines_per_page):
                    page_lines = lines[page_start:page_start + max_lines_per_page]
                    page_text = '\n'.join(page_lines)
                    
                    fig, ax = plt.subplots(figsize=(11, 8.5))
                    ax.axis('off')
                    if page_start == 0:
                        ax.set_title('Detection Warnings and Validation Log', fontsize=16, fontweight='bold', pad=20)
                    else:
                        ax.set_title(f'Detection Warnings and Validation Log (continued {page_start//max_lines_per_page + 1})', 
                                   fontsize=16, fontweight='bold', pad=20)
                    
                    ax.text(0.05, 0.95, page_text, transform=ax.transAxes,
                           fontsize=7, family='monospace', verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                    
                    plt.tight_layout()
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
        
        # Add region summary page at the end if spark tracks data is provided
        if df_tracks is not None:
            try:
                if 'region' in df_tracks.columns:
                    df_with_regions = df_tracks[
                        df_tracks['region'].notna() & 
                        (df_tracks['region'] != '') & 
                        (df_tracks['region'] != 'unknown')
                    ]
                    if len(df_with_regions) > 0:
                        create_region_summary_page(df_tracks, pdf)
                        print(f"  → Added region summary page to PDF ({len(df_with_regions):,} events with regions)")
                    else:
                        print(f"  → Skipping region summary: no valid region data found")
                else:
                    print(f"  → Skipping region summary: 'region' column not found in data")
            except Exception as e:
                import traceback
                print(f"  ⚠ Warning: Could not add region summary page: {e}")
                print(f"    Traceback: {traceback.format_exc()}")
    
    print(f"✓ Generated PDF: {output_pdf_path}")


def generate_summary_table(df_tracks, output_path, output_dir, image_paths_dict=None, tiff_base_path=None):
    """
    Generate markdown table summarizing all detections.
    
    Args:
        df_tracks: DataFrame with track data
        output_path: Path to output markdown file
        output_dir: Directory containing images
        image_paths_dict: Dict mapping (folder, video) to image path
        tiff_base_path: Optional base path to check for single-frame TIFFs
    """
    # Group by folder and video
    df_tracks = df_tracks.copy()
    df_tracks['base_filename'] = df_tracks['filename'].str.replace(r' \(page \d+\)', '', regex=True)
    
    # Extract folder and video, filtering out single-frame TIFFs
    folder_video_data = []
    for base_file in df_tracks['base_filename'].unique():
        folder, video = extract_folder_and_video(base_file)
        if folder and video:
            # Check if this is a single-frame TIFF and skip it
            if video.endswith(('.tif', '.tiff')):
                tiff_path = find_tiff_file(folder, video, tiff_base_path)
                if not tiff_path or not tiff_path.exists():
                    # Try direct paths
                    if tiff_base_path:
                        direct_path = Path(tiff_base_path) / folder / video
                        if direct_path.exists():
                            tiff_path = direct_path
                    if not tiff_path or not tiff_path.exists():
                        direct_path = Path(folder) / video
                        if direct_path.exists():
                            tiff_path = direct_path
                
                if tiff_path and tiff_path.exists():
                    try:
                        with tiff.TiffFile(tiff_path) as tif:
                            if len(tif.pages) <= 1:
                                # Skip single-frame TIFFs
                                continue
                    except:
                        # If we can't read it, include it (might not be a TIFF issue)
                        pass
            
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
    parser.add_argument('--mask-base-path', default=None,
                       help='Base path to search for mask PNG files (default: wave-vector-analysis/embryo_masks_final)')
    
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
    
    # Create log file for warnings - will capture all warning messages
    log_file_path = output_dir / "detection_warnings.log"
    warnings_log = []
    
    # Create a custom print function that captures warnings
    import sys
    from io import StringIO
    
    class WarningCapture:
        def __init__(self, original_stdout, warnings_list):
            self.original_stdout = original_stdout
            self.warnings_list = warnings_list
            self.buffer = StringIO()
        
        def write(self, text):
            # Write to original stdout
            self.original_stdout.write(text)
            # Also capture if it's a warning
            if '⚠' in text or 'WARNING' in text or 'CRITICAL' in text or 'FORCED' in text:
                self.warnings_list.append(text.rstrip())
            # Also write to buffer for full log
            self.buffer.write(text)
        
        def flush(self):
            self.original_stdout.flush()
            self.buffer.flush()
    
    # Replace stdout temporarily (we'll restore it)
    original_stdout = sys.stdout
    warning_capture = WarningCapture(original_stdout, warnings_log)
    sys.stdout = warning_capture
    
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
            
            # Check if TIFF is single-frame and skip it (check multiple possible locations)
            tiff_path = None
            if args.tiff_base_path:
                tiff_path = find_tiff_file(folder, video, args.tiff_base_path)
            else:
                # Try to find TIFF even without base path
                tiff_path = find_tiff_file(folder, video, None)
            
            # Also try direct paths if not found
            if not tiff_path or not tiff_path.exists():
                if args.tiff_base_path:
                    direct_path = Path(args.tiff_base_path) / folder / video
                    if direct_path.exists():
                        tiff_path = direct_path
                if not tiff_path or not tiff_path.exists():
                    direct_path = Path(folder) / video
                    if direct_path.exists():
                        tiff_path = direct_path
            
            # Check frame count if we found a TIFF file
            if tiff_path and tiff_path.exists() and video.endswith(('.tif', '.tiff')):
                try:
                    with tiff.TiffFile(tiff_path) as tif:
                        if len(tif.pages) <= 1:
                            print(f"    ⚠ Skipping single-frame TIFF: {video} ({len(tif.pages)} frame(s))")
                            continue
                except Exception as e:
                    # If we can't read it, continue anyway (might not be a TIFF issue)
                    pass
            
            try:
                # Determine mask base path (default to embryo_masks_final)
                mask_base = args.mask_base_path if hasattr(args, 'mask_base_path') and args.mask_base_path else None
                output_path = create_embryo_visualization(df_tracks, output_dir, folder_video_key, global_bounds, args.tiff_base_path, mask_base)
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
            # Prepare summary data for PDF table (only include videos that have visualizations)
            # Use image_paths_dict.keys() directly to ensure we only include multi-frame TIFFs
            summary_data = []
            for folder, video in sorted(image_paths_dict.keys(), key=lambda x: (int(x[0]) if x[0].isdigit() else 999, x[1])):
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
            
            # Store image paths and summary data for later PDF creation with log included
            # We'll create the PDF after warnings are collected
            stored_image_paths = image_paths_for_pdf.copy()
            stored_summary_data = summary_data.copy()
        else:
            # Initialize empty lists if no images were created
            stored_image_paths = []
            stored_summary_data = []
    
    # Generate summary table (after visualizations so we can include image references)
    print("\nGenerating summary table...")
    generate_summary_table(df_tracks, output_dir / "detection_summary.md", output_dir, image_paths_dict, args.tiff_base_path)
    
    # Restore original stdout
    sys.stdout = original_stdout
    
    # Write warnings to log file
    if warnings_log:
        with open(log_file_path, 'w') as log_file:
            log_file.write("=" * 80 + "\n")
            log_file.write("Embryo Detection Warnings and Validation Log\n")
            log_file.write("=" * 80 + "\n\n")
            log_file.write("All warnings and validation messages from embryo detection:\n\n")
            for warning in warnings_log:
                log_file.write(warning + "\n")
        print(f"\n✓ Warning log saved to: {log_file_path} ({len(warnings_log)} warnings logged)")
    else:
        # Create empty log file to indicate no warnings
        with open(log_file_path, 'w') as log_file:
            log_file.write("=" * 80 + "\n")
            log_file.write("Embryo Detection Warnings and Validation Log\n")
            log_file.write("=" * 80 + "\n\n")
            log_file.write("No warnings detected.\n")
        print(f"\n✓ Warning log saved to: {log_file_path} (no warnings)")
    
    # Create PDF if we have images (regardless of warnings)
    pdf_path = output_dir / "detection_visualizations.pdf"
    if stored_image_paths and len(stored_image_paths) > 0:
        log_path_for_pdf = log_file_path if log_file_path.exists() else None
        print(f"\nCreating PDF with {len(stored_image_paths)} images...")
        if warnings_log:
            print(f"  → Including warning log ({len(warnings_log)} warnings)")
        # Check if region data is available
        if 'region' in df_tracks.columns:
            df_with_regions = df_tracks[
                df_tracks['region'].notna() & 
                (df_tracks['region'] != '') & 
                (df_tracks['region'] != 'unknown')
            ]
            if len(df_with_regions) > 0:
                print(f"  → Will include region summary ({len(df_with_regions):,} events with regions)")
            else:
                print(f"  → No valid region data found - skipping region summary")
        else:
            print(f"  → 'region' column not found in data - skipping region summary")
            print(f"    (Re-run wave-vector-tiff-parser.py to generate region data)")
        
        # Create PDF with everything included (including region summary)
        create_pdf_from_images(stored_image_paths, pdf_path, images_per_page=1, 
                             summary_data=stored_summary_data, warning_log_path=log_path_for_pdf,
                             df_tracks=df_tracks)
    else:
        print(f"\n⚠ No images to include in PDF - skipping PDF creation")
    
    print(f"\n✓ Complete!")


if __name__ == '__main__':
    main()

