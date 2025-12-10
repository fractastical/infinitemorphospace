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
    
    # Find background threshold
    background_percentile = 10
    background_threshold = np.percentile(flat_intensities, background_percentile)
    
    # Calculate embryo threshold
    median_intensity = np.median(flat_intensities)
    p25 = np.percentile(flat_intensities, 25)
    mean_intensity = flat_intensities.mean()
    
    embryo_threshold = max(median_intensity * 0.7, p25, mean_intensity * 0.8, background_threshold * 1.2)
    
    # Create binary mask
    embryo_mask = (blur >= embryo_threshold).astype(np.uint8) * 255
    
    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    embryo_mask = cv2.morphologyEx(embryo_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    embryo_mask = cv2.morphologyEx(embryo_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(embryo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by minimum area
    min_area = 0.005 * h * w
    emb_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    
    if not emb_contours:
        return None
    
    # Sort by centroid x position (leftmost = A, rightmost = B)
    def contour_cx(c):
        M = cv2.moments(c)
        if M["m00"] == 0:
            return 0
        return M["m10"] / M["m00"]
    
    emb_contours.sort(key=contour_cx)
    
    # Process up to 2 embryos
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
        
        results[label] = {
            'contour': contour_points,
            'head': (float(head[0]), float(head[1])),
            'tail': (float(tail[0]), float(tail[1])),
            'centroid': (float(cx), float(cy))
        }
    
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
    
    # Plot embryo outlines and labels
    for embryo_id in ['A', 'B']:
        df_embryo = df_file[df_file['embryo_id'] == embryo_id].copy()
        if len(df_embryo) == 0:
            continue
        
        # Try to get boundary from TIFF detection first (actual grey tissue)
        boundary = None
        tiff_head = None
        tiff_tail = None
        used_tiff_detection = False
        
        if tiff_path and tiff_path.exists():
            try:
                detection = detect_embryo_from_tiff(tiff_path, embryo_id)
                if detection:
                    boundary = detection.get('contour')
                    if boundary is not None:
                        # Close the contour
                        boundary = np.vstack([boundary, boundary[0:1]])
                    tiff_head = detection.get('head')
                    tiff_tail = detection.get('tail')
                    used_tiff_detection = True
            except Exception as e:
                pass  # Fall back to spark-based detection
        
        # Fallback to spark-based boundary if TIFF detection failed
        if boundary is None:
            boundary = get_embryo_boundary_from_sparks(df_embryo)
        
        if boundary is not None:
            # Draw outline - use different color/style to indicate detection method
            if used_tiff_detection:
                # Actual embryo tissue detection (from TIFF)
                poly = Polygon(boundary, fill=False, edgecolor='cyan', linewidth=2.5, alpha=0.9, linestyle='-')
            else:
                # Reconstructed from sparks (approximate)
                poly = Polygon(boundary, fill=False, edgecolor='lightblue', linewidth=2, alpha=0.6, linestyle='--')
            ax.add_patch(poly)
        
        # Get head/tail positions - prefer TIFF detection, fallback to spark data
        if tiff_head and tiff_tail:
            head_pos = tiff_head
            tail_pos = tiff_tail
        else:
            head_pos, tail_pos = get_head_tail_positions(df_embryo)
        
        # Draw head
        if head_pos:
            ax.plot(head_pos[0], head_pos[1], 'go', markersize=12, label=f'Embryo {embryo_id} Head')
            ax.annotate(f'{embryo_id} Head', head_pos, xytext=(10, 10), 
                       textcoords='offset points', color='green', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='green'))
        
        # Draw tail
        if tail_pos:
            ax.plot(tail_pos[0], tail_pos[1], 'ro', markersize=12, label=f'Embryo {embryo_id} Tail')
            ax.annotate(f'{embryo_id} Tail', tail_pos, xytext=(10, -10), 
                       textcoords='offset points', color='red', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='red'))
        
        # Draw head-tail axis
        if head_pos and tail_pos:
            ax.plot([head_pos[0], tail_pos[0]], [head_pos[1], tail_pos[1]], 
                   'y--', linewidth=1, alpha=0.5)
    
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


def create_pdf_from_images(image_paths, output_pdf_path, images_per_page=1):
    """
    Create a PDF from a list of image paths.
    
    Args:
        image_paths: List of (path, title) tuples
        output_pdf_path: Path to output PDF
        images_per_page: Number of images per page (1 or 2)
    """
    with PdfPages(output_pdf_path) as pdf:
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
            create_pdf_from_images(image_paths_for_pdf, pdf_path, images_per_page=1)
    
    # Generate summary table (after visualizations so we can include image references)
    print("\nGenerating summary table...")
    generate_summary_table(df_tracks, output_dir / "detection_summary.md", output_dir, image_paths_dict)
    
    print("\n✓ Complete!")


if __name__ == '__main__':
    main()

