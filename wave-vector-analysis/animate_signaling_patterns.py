#!/usr/bin/env python3
"""
Create animated heatmaps showing calcium signaling patterns aggregated by poke region.

This script:
1. Groups videos by poke location regions (spatial clusters or AP positions)
2. Aggregates spark tracks from videos with similar poke locations
3. Creates animated heatmaps showing signal propagation over time
4. Exports as animated GIF or MP4
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import argparse
from scipy.cluster.vq import kmeans2, whiten
from scipy.spatial.distance import cdist
from plot_poke_locations import infer_poke_from_early_sparks
import warnings
warnings.filterwarnings('ignore')


def group_poke_locations_by_region(poke_locations_df, method='spatial', n_regions=3, eps=200):
    """
    Group poke locations into regions for aggregation.
    
    Args:
        poke_locations_df: DataFrame with poke locations (columns: filename, poke_x, poke_y)
        method: 'spatial' (K-means clustering), 'ap' (by AP position), 'density' (DBSCAN)
        n_regions: Number of regions for K-means (default: 3)
        eps: Distance threshold for DBSCAN (pixels, default: 200)
    
    Returns:
        DataFrame with added 'region' column
    """
    poke_df = poke_locations_df.copy()
    
    if method == 'ap':
        # Group by AP position if available
        if 'ap_norm' in poke_df.columns:
            # Define regions: anterior (0-0.33), mid (0.33-0.67), posterior (0.67-1.0)
            poke_df['region'] = pd.cut(poke_df['ap_norm'], 
                                      bins=[0, 0.33, 0.67, 1.0],
                                      labels=['anterior', 'mid', 'posterior'],
                                      include_lowest=True)
            return poke_df
        else:
            print("Warning: AP positions not available, falling back to spatial clustering")
            method = 'spatial'
    
    if method == 'spatial':
        # K-means clustering on spatial coordinates
        coords = poke_df[['poke_x', 'poke_y']].values
        
        # Normalize coordinates for clustering
        coords_normalized = whiten(coords)
        
        # Perform K-means clustering
        centroids, labels = kmeans2(coords_normalized, n_regions, iter=50, minit='points')
        
        # Convert back to original coordinates
        poke_df['region'] = labels
        poke_df['region_label'] = poke_df['region'].apply(lambda x: f'Region {x+1}')
        
        # Sort regions by position (left to right, or top to bottom)
        region_centers = poke_df.groupby('region')[['poke_x', 'poke_y']].mean()
        region_order = region_centers.sort_values('poke_x').index
        
        # Rename regions based on position
        region_map = {old: new for new, old in enumerate(region_order)}
        poke_df['region'] = poke_df['region'].map(region_map)
        poke_df['region_label'] = poke_df['region'].apply(lambda x: f'Region {x+1}')
        
        # Post-processing: Move pokes in the left third of Region 1 to Region 0
        if n_regions == 3:
            # Find Region 1's X range to determine left third of Region 1
            region1_pokes = poke_df[poke_df['region'] == 1]
            if len(region1_pokes) > 0:
                r1_x_min = region1_pokes['poke_x'].min()
                r1_x_max = region1_pokes['poke_x'].max()
                r1_x_range = r1_x_max - r1_x_min
                r1_left_third_boundary = r1_x_min + (r1_x_range / 3)
                
                # Find pokes in Region 1 that are in the left third of Region 1's range
                region1_mask = (poke_df['region'] == 1) & (poke_df['poke_x'] < r1_left_third_boundary)
                n_moved = region1_mask.sum()
                
                if n_moved > 0:
                    print(f"  Moving {n_moved} pokes from Region 1 to Region 0 (left third of Region 1)")
                    poke_df.loc[region1_mask, 'region'] = 0
                    poke_df['region_label'] = poke_df['region'].apply(lambda x: f'Region {x+1}')
        
        return poke_df
    
    elif method == 'density':
        # Simple distance-based clustering for variable number of regions
        # Group pokes that are within eps distance of each other
        coords = poke_df[['poke_x', 'poke_y']].values
        n_pokes = len(coords)
        regions = np.full(n_pokes, -1, dtype=int)
        current_region = 0
        
        for i in range(n_pokes):
            if regions[i] == -1:
                # Start new region
                regions[i] = current_region
                # Find nearby points
                distances = np.sqrt(np.sum((coords - coords[i])**2, axis=1))
                nearby = distances < eps
                regions[nearby] = current_region
                current_region += 1
        
        poke_df['region'] = regions
        poke_df['region_label'] = poke_df['region'].apply(lambda x: f'Region {x+1}' if x >= 0 else 'Noise')
        
        return poke_df
    
    else:
        raise ValueError(f"Unknown method: {method}")


def aggregate_tracks_by_region(tracks_df, poke_regions_df, region_id):
    """
    Aggregate spark tracks from all videos in a specific poke region.
    
    Args:
        tracks_df: DataFrame with spark tracks
        poke_regions_df: DataFrame with poke locations and region assignments
        region_id: Region ID to aggregate
    
    Returns:
        Filtered DataFrame with tracks from videos in this region
    """
    # Get filenames for this region
    region_files = poke_regions_df[poke_regions_df['region'] == region_id]['filename'].values
    
    # Filter tracks from these files
    tracks_df = tracks_df.copy()
    if 'filename' in tracks_df.columns:
        tracks_df['base_filename'] = tracks_df['filename'].str.replace(r' \(page \d+\)', '', regex=True)
        filtered = tracks_df[tracks_df['base_filename'].isin(region_files)].copy()
    else:
        filtered = tracks_df.copy()
    
    return filtered


def create_animated_heatmap(tracks_df, poke_locations_df, output_path, 
                           region_method='spatial', time_window_s=60, 
                           frame_interval_s=1, fps=5, normalize_by_region=True,
                           colormap='hot', bins=50, vmax_percentile=98):
    """
    Create animated heatmap showing signal propagation from aggregated poke regions.
    
    Args:
        tracks_df: DataFrame with spark tracks
        poke_locations_df: DataFrame with poke locations
        output_path: Path to save animation (GIF or MP4)
        region_method: Method to group poke locations ('spatial', 'ap', 'density')
        time_window_s: Time window to animate (seconds, default: 60)
        frame_interval_s: Time between frames (seconds, default: 1)
        fps: Frames per second for animation (default: 5)
        normalize_by_region: Whether to normalize heatmap intensity per region
        colormap: Colormap name (default: 'hot')
        bins: Number of bins for 2D histogram (default: 50)
        vmax_percentile: Percentile for max color value (default: 98)
    """
    print("Creating animated signaling pattern heatmap...")
    print(f"  Time window: {time_window_s}s, Frame interval: {frame_interval_s}s")
    
    # Group poke locations into regions
    print(f"\nGrouping poke locations by {region_method} method...")
    poke_regions_df = group_poke_locations_by_region(poke_locations_df, method=region_method)
    
    unique_regions = sorted(poke_regions_df['region'].dropna().unique())
    n_regions = len(unique_regions)
    print(f"  → Found {n_regions} poke regions")
    
    for region_id in unique_regions:
        region_files = poke_regions_df[poke_regions_df['region'] == region_id]
        print(f"  Region {region_id}: {len(region_files)} videos")
    
    # Align videos by poke time only (keep absolute spatial coordinates)
    tracks_df = tracks_df.copy()
    if 'filename' in tracks_df.columns:
        tracks_df['base_filename'] = tracks_df['filename'].str.replace(r' \(page \d+\)', '', regex=True)
    
    # For each file, align time relative to poke (keep spatial coordinates absolute)
    aligned_tracks = []
    for base_file in tracks_df['base_filename'].unique():
        file_data = tracks_df[tracks_df['base_filename'] == base_file].copy()
        
        if len(file_data) > 0 and 'time_s' in file_data.columns:
            # Find the earliest time in this file (this is when poke occurs)
            poke_time = file_data['time_s'].min()
            # Shift all times to be relative to poke (poke_time becomes t=0)
            file_data['time_s_aligned'] = file_data['time_s'] - poke_time
            aligned_tracks.append(file_data)
    
    if len(aligned_tracks) > 0:
        tracks_df = pd.concat(aligned_tracks, ignore_index=True)
        tracks_df['time_s'] = tracks_df['time_s_aligned']
        # Keep x and y as absolute coordinates (no spatial alignment)
    
    # Filter to post-poke data within time window (now aligned)
    post_poke = tracks_df[(tracks_df['time_s'] >= 0) & (tracks_df['time_s'] <= time_window_s)].copy()
    
    print(f"\nPost-poke events in time window (aligned): {len(post_poke):,}")
    
    # Prepare data for each region
    region_data = {}
    for region_id in unique_regions:
        region_tracks = aggregate_tracks_by_region(post_poke, poke_regions_df, region_id)
        region_data[region_id] = region_tracks
        print(f"  Region {region_id}: {len(region_tracks):,} events")
    
    # Determine spatial extent using absolute coordinates (shared across all regions)
    all_x = post_poke['x'].dropna()
    all_y = post_poke['y'].dropna()
    
    if len(all_x) == 0 or len(all_y) == 0:
        print("ERROR: No valid spatial coordinates found!")
        return
    
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    
    # Add padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * 0.05
    x_max += x_range * 0.05
    y_min -= y_range * 0.05
    y_max += y_range * 0.05
    
    print(f"\nSpatial extent (absolute coordinates):")
    print(f"  X: {x_min:.1f} to {x_max:.1f} pixels")
    print(f"  Y: {y_min:.1f} to {y_max:.1f} pixels")
    
    # Create time bins
    time_bins = np.arange(0, time_window_s + frame_interval_s, frame_interval_s)
    n_frames = len(time_bins) - 1
    
    print(f"\nCreating {n_frames} frames...")
    
    # Pre-compute histograms for each frame and region
    frames_data = {}
    for region_id in unique_regions:
        region_tracks = region_data[region_id]
        frames_data[region_id] = []
        
        for i in range(n_frames):
            t_start = time_bins[i]
            t_end = time_bins[i + 1]
            
            # Filter tracks CUMULATIVELY from start (all sparks up to this time)
            frame_tracks = region_tracks[(region_tracks['time_s'] >= 0) & 
                                        (region_tracks['time_s'] <= t_end)].copy()
            
            if len(frame_tracks) > 0 and frame_tracks['x'].notna().sum() > 0:
                # Create 2D histogram of spark density (cumulative)
                valid_tracks = frame_tracks[frame_tracks['x'].notna() & frame_tracks['y'].notna()]
                if len(valid_tracks) > 0:
                    H, xedges, yedges = np.histogram2d(
                        valid_tracks['x'], valid_tracks['y'],
                        bins=bins, range=[[x_min, x_max], [y_min, y_max]]
                    )
                    frames_data[region_id].append((H.T, t_end, len(valid_tracks)))
                else:
                    H = np.zeros((bins, bins))
                    frames_data[region_id].append((H, t_end, 0))
            else:
                # Empty frame
                H = np.zeros((bins, bins))
                frames_data[region_id].append((H, t_end, 0))
    
    # Determine color scale (use 98th percentile across all frames)
    # Use the final frame of each region (maximum accumulation) to set scale
    all_values = []
    for region_id in unique_regions:
        if len(frames_data[region_id]) > 0:
            # Use the last frame (maximum cumulative accumulation) for scaling
            H, _, _ = frames_data[region_id][-1]
            all_values.extend(H.flatten())
    
    vmax = np.percentile(all_values, vmax_percentile) if len(all_values) > 0 else 1
    if vmax == 0:
        vmax = 1  # Avoid division by zero
    
    # Create figure with subplots for each region
    if n_regions == 1:
        fig, axes = plt.subplots(1, 1, figsize=(10, 10))
        axes = [axes]
    elif n_regions <= 3:
        fig, axes = plt.subplots(1, n_regions, figsize=(5*n_regions, 5))
        if n_regions == 1:
            axes = [axes]
    elif n_regions <= 6:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
    else:
        # Too many regions, use first 9
        print(f"Warning: Showing first 9 of {n_regions} regions")
        unique_regions = unique_regions[:9]
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
    
    # Initialize images and poke location markers
    images = []
    text_annotations = []
    poke_markers = []  # Store poke location markers separately
    
    for idx, region_id in enumerate(unique_regions):
        ax = axes[idx]
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_xlabel('X (pixels)', fontsize=10)
        ax.set_ylabel('Y (pixels)', fontsize=10)
        
        region_label = poke_regions_df[poke_regions_df['region'] == region_id]['region_label'].iloc[0] if 'region_label' in poke_regions_df.columns else f'Region {region_id}'
        ax.set_title(region_label, fontsize=12, fontweight='bold')
        
        # Show poke locations for this region at their absolute positions
        region_pokes = poke_regions_df[poke_regions_df['region'] == region_id]
        poke_x_list = []
        poke_y_list = []
        for _, poke_row in region_pokes.iterrows():
            poke_x_list.append(poke_row['poke_x'])
            poke_y_list.append(poke_row['poke_y'])
        
        # Draw poke locations at their actual absolute positions
        if len(poke_x_list) > 0:
            ax.scatter(poke_x_list, poke_y_list, 
                      s=40, c='cyan', marker='X', 
                      edgecolors='darkblue', linewidths=1, 
                      zorder=20, alpha=0.9, label='Poke locations')
        
        # Initial empty heatmap
        im = ax.imshow(np.zeros((bins, bins)), origin='lower',
                      extent=[x_min, x_max, y_min, y_max],
                      cmap=colormap, vmin=0, vmax=vmax,
                      interpolation='bilinear', alpha=0.9)
        images.append(im)
        
        # Add colorbar for first subplot
        if idx == 0:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Spark Count', fontsize=10)
        
        # Time annotation
        text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                      fontsize=11, fontweight='bold',
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        text_annotations.append(text)
    
    # Remove extra subplots
    for idx in range(len(unique_regions), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    
    # Animation function
    def animate(frame):
        if frame >= n_frames:
            frame = n_frames - 1
        
        t_end = time_bins[frame + 1]
        
        for idx, region_id in enumerate(unique_regions):
            if frame < len(frames_data[region_id]):
                H, t, n_sparks = frames_data[region_id][frame]
            else:
                # Use last frame if index out of bounds
                H, t, n_sparks = frames_data[region_id][-1]
            
            # Display the heatmap (cumulative spark accumulation)
            # H is the histogram of all sparks from t=0 to current time
            images[idx].set_data(H)
            
            # Set color scale - use global vmax so all regions are comparable
            images[idx].set_clim(0, vmax)
            
            # Update time annotation with spark count
            text_annotations[idx].set_text(f't = {t:.1f}s\n{n_sparks:,} sparks')
            text_annotations[idx].set_fontsize(10)
        
        return images + text_annotations
    
    # Create animation
    print(f"\nRendering animation ({n_frames} frames, {fps} fps)...")
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, 
                                   interval=1000/fps, blit=False, repeat=True)
    
    # Save animation
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix.lower() == '.gif':
        print(f"Saving as GIF to {output_path}...")
        anim.save(output_path, writer='pillow', fps=fps)
    elif output_path.suffix.lower() == '.mp4':
        print(f"Saving as MP4 to {output_path}...")
        anim.save(output_path, writer='ffmpeg', fps=fps, bitrate=1800)
    else:
        # Default to GIF
        output_path = output_path.with_suffix('.gif')
        print(f"Saving as GIF to {output_path}...")
        anim.save(output_path, writer='pillow', fps=fps)
    
    print(f"✓ Animation saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Create animated heatmaps of calcium signaling patterns by poke region'
    )
    parser.add_argument('tracks_csv', help='Path to spark_tracks.csv')
    parser.add_argument('--output', '-o', default='analysis_results/signaling_animation.gif',
                       help='Output path for animation (GIF or MP4)')
    parser.add_argument('--poke-csv', help='Path to poke_locations.csv (optional, will infer if not provided)')
    parser.add_argument('--region-method', choices=['spatial', 'ap', 'density'], default='spatial',
                       help='Method to group poke locations (default: spatial)')
    parser.add_argument('--n-regions', type=int, default=3,
                       help='Number of regions for spatial clustering (default: 3)')
    parser.add_argument('--time-window', type=float, default=60,
                       help='Time window to animate in seconds (default: 60)')
    parser.add_argument('--frame-interval', type=float, default=1.0,
                       help='Time between frames in seconds (default: 1.0)')
    parser.add_argument('--fps', type=int, default=5,
                       help='Frames per second for animation (default: 5)')
    parser.add_argument('--bins', type=int, default=50,
                       help='Number of bins for heatmap (default: 50)')
    parser.add_argument('--colormap', default='hot',
                       help='Colormap name (default: hot)')
    
    args = parser.parse_args()
    
    print(f"Loading {args.tracks_csv}...")
    tracks_df = pd.read_csv(args.tracks_csv)
    print(f"  → Loaded {len(tracks_df):,} track states")
    
    # Get poke locations
    if args.poke_csv and Path(args.poke_csv).exists():
        print(f"\nLoading poke locations from {args.poke_csv}...")
        poke_locations_df = pd.read_csv(args.poke_csv)
        print(f"  → Loaded {len(poke_locations_df)} poke locations")
    else:
        print(f"\nInferring poke locations from early spark clusters...")
        poke_locations_df = infer_poke_from_early_sparks(tracks_df)
        print(f"  → Inferred {len(poke_locations_df)} poke locations")
    
    if len(poke_locations_df) == 0:
        print("\nERROR: No poke locations found!")
        return
    
    # Create animation
    create_animated_heatmap(
        tracks_df, poke_locations_df, args.output,
        region_method=args.region_method,
        time_window_s=args.time_window,
        frame_interval_s=args.frame_interval,
        fps=args.fps,
        bins=args.bins,
        colormap=args.colormap
    )
    
    print("\n✓ Complete!")


if __name__ == '__main__':
    main()

