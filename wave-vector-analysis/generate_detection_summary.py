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
from pathlib import Path
import argparse
import re
from collections import defaultdict

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


def get_embryo_boundary_from_sparks(df_embryo):
    """
    Reconstruct approximate embryo boundary from spark locations.
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


def create_embryo_visualization(df_tracks, output_dir, folder_video_key):
    """
    Create visualization for a specific folder/video combination.
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
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Get spatial bounds
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
        
        # Get boundary
        boundary = get_embryo_boundary_from_sparks(df_embryo)
        if boundary is not None:
            # Draw outline
            poly = Polygon(boundary, fill=False, edgecolor='cyan', linewidth=2, alpha=0.8)
            ax.add_patch(poly)
        
        # Get head/tail positions
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
    
    # Save
    safe_video_name = re.sub(r'[^\w\-_\.]', '_', video)
    output_path = output_dir / f"folder_{folder}_{safe_video_name}_detection.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black', edgecolor='none')
    plt.close()
    
    return output_path


def generate_summary_table(df_tracks, output_path):
    """
    Generate markdown table summarizing all detections.
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
    table_rows.append("| Folder | Video | Embryo A | Embryo B | Poke Location | Healed Wound |")
    table_rows.append("|--------|-------|----------|----------|---------------|--------------|")
    
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
        
        table_rows.append(f"| {video_label} | {video} | {emb_a_orient} | {emb_b_orient} | {poke_str} | {healed_wound} |")
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write("# Embryo Detection Summary\n\n")
        f.write("This table summarizes embryo detection results for manual fact-checking.\n\n")
        f.write("## Legend\n\n")
        f.write("- **Embryo A/B**: Orientation format is 'head [direction], tail [direction]'\n")
        f.write("- **Poke Location**: Coordinates in pixels (x, y)\n")
        f.write("- **Healed Wound**: Location of previously healed wounds (if detected)\n\n")
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
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading {args.tracks_csv}...")
    df_tracks = pd.read_csv(args.tracks_csv)
    print(f"  → Loaded {len(df_tracks):,} track states")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate summary table
    print("\nGenerating summary table...")
    generate_summary_table(df_tracks, output_dir / "detection_summary.md")
    
    # Generate visualizations
    if not args.skip_visualizations:
        print("\nGenerating visualizations...")
        df_tracks['base_filename'] = df_tracks['filename'].str.replace(r' \(page \d+\)', '', regex=True)
        
        folder_video_keys = set()
        for base_file in df_tracks['base_filename'].unique():
            folder, video = extract_folder_and_video(base_file)
            if folder and video:
                folder_video_keys.add((folder, video))
        
        print(f"  → Found {len(folder_video_keys)} unique folder/video combinations")
        
        for folder_video_key in sorted(folder_video_keys, key=lambda x: (int(x[0]) if x[0].isdigit() else 999, x[1])):
            folder, video = folder_video_key
            print(f"  → Processing folder {folder}, video {video}...")
            try:
                output_path = create_embryo_visualization(df_tracks, output_dir, folder_video_key)
                if output_path:
                    print(f"    ✓ Saved: {output_path.name}")
                else:
                    print(f"    ⚠ No data found for this combination")
            except Exception as e:
                print(f"    ✗ Error: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n✓ Complete!")


if __name__ == '__main__':
    main()

