#!/usr/bin/env python3
"""
Create animated heatmaps showing simulated calcium signaling patterns.

This script creates animated visualizations of simulated spark data,
showing wave propagation over time, similar to the experimental animation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import argparse
import json
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for poke inference function
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from plot_poke_locations import infer_poke_from_early_sparks
except ImportError:
    # Fallback: simple poke inference
    def infer_poke_from_early_sparks(df):
        """Infer poke locations from early sparks in simulation data."""
        df = df.copy()
        df['base_filename'] = df['filename'].str.replace(r' \(page \d+\)', '', regex=True)
        
        poke_locs = []
        for base_file in df['base_filename'].unique():
            file_data = df[df['base_filename'] == base_file].copy()
            
            # For simulations, pokes are at time_s=0, find earliest sparks
            early_sparks = file_data[file_data['time_s'] <= 1.0]  # First second
            
            if len(early_sparks) > 0 and early_sparks['x'].notna().sum() > 0:
                # Use the first spark location as poke location
                first_spark = early_sparks.nsmallest(1, 'time_s').iloc[0]
                poke_locs.append({
                    'filename': base_file,
                    'poke_x': first_spark['x'],
                    'poke_y': first_spark['y']
                })
        
        return pd.DataFrame(poke_locs)


def create_animated_heatmap(tracks_df, output_path, time_window_s=60, 
                           frame_interval_s=1, fps=5, colormap='hot', 
                           bins=50, vmax_percentile=98):
    """
    Create animated heatmap showing signal propagation from simulation.
    
    Args:
        tracks_df: DataFrame with simulated spark tracks
        output_path: Path to save animation (GIF or MP4)
        time_window_s: Time window to animate (seconds, default: 60)
        frame_interval_s: Time between frames (seconds, default: 1)
        fps: Frames per second for animation (default: 5)
        colormap: Colormap name (default: 'hot')
        bins: Number of bins for 2D histogram (default: 50)
        vmax_percentile: Percentile for max color value (default: 98)
    """
    print("Creating animated signaling pattern heatmap for simulation...")
    print(f"  Time window: {time_window_s}s, Frame interval: {frame_interval_s}s")
    
    tracks_df = tracks_df.copy()
    
    # Infer poke locations from early sparks
    print("\nInferring poke locations from early spark clusters...")
    poke_locations_df = infer_poke_from_early_sparks(tracks_df)
    print(f"  → Found {len(poke_locations_df)} poke locations")
    
    # Group by embryo or poke location
    tracks_df['base_filename'] = tracks_df['filename'].str.replace(r' \(page \d+\)', '', regex=True)
    
    if 'embryo_id' in tracks_df.columns:
        # Group by embryo
        unique_embryos = sorted(tracks_df['embryo_id'].dropna().unique())
        n_regions = len(unique_embryos)
        print(f"\nFound {n_regions} embryo(s) in simulation")
        
        # Create region mapping
        tracks_df['region'] = tracks_df['embryo_id'].map(
            {emb: i for i, emb in enumerate(unique_embryos)}
        )
        
        # Map poke locations to regions via embryo_id
        if 'embryo_id' in poke_locations_df.columns:
            poke_locations_df['region'] = poke_locations_df['embryo_id'].map(
                {emb: i for i, emb in enumerate(unique_embryos)}
            )
        else:
            # Infer embryo from filename
            for idx, row in poke_locations_df.iterrows():
                filename = row['filename']
                matching_tracks = tracks_df[tracks_df['base_filename'] == filename]
                if len(matching_tracks) > 0:
                    embryo = matching_tracks['embryo_id'].iloc[0]
                    poke_locations_df.at[idx, 'region'] = unique_embryos.index(embryo) if embryo in unique_embryos else 0
                else:
                    poke_locations_df.at[idx, 'region'] = 0
    else:
        # Group by poke location (spatial clustering)
        from scipy.cluster.vq import kmeans2, whiten
        
        coords = poke_locations_df[['poke_x', 'poke_y']].values
        n_pokes = len(coords)
        n_regions = min(3, n_pokes)  # Up to 3 regions
        
        if n_pokes > 1:
            coords_norm = whiten(coords)
            centroids, labels = kmeans2(coords_norm, n_regions, iter=50, minit='points')
            
            # Map poke locations to regions
            poke_locations_df['region'] = labels
            
            # Map tracks to regions via filename
            poke_region_map = dict(zip(poke_locations_df['filename'], poke_locations_df['region']))
            tracks_df['region'] = tracks_df['base_filename'].map(poke_region_map).fillna(0).astype(int)
        else:
            poke_locations_df['region'] = 0
            tracks_df['region'] = 0
            n_regions = 1
    
    # Align time to poke (time_s should already be relative to poke in simulations)
    tracks_df['base_filename'] = tracks_df['filename'].str.replace(r' \(page \d+\)', '', regex=True)
    
    # Filter to post-poke data within time window
    post_poke = tracks_df[(tracks_df['time_s'] >= 0) & (tracks_df['time_s'] <= time_window_s)].copy()
    
    print(f"\nPost-poke events in time window: {len(post_poke):,}")
    
    # Prepare data for each region
    region_data = {}
    unique_regions = sorted(post_poke['region'].dropna().unique())
    
    for region_id in unique_regions:
        region_tracks = post_poke[post_poke['region'] == region_id]
        region_data[region_id] = region_tracks
        print(f"  Region {region_id}: {len(region_tracks):,} events")
    
    # Determine spatial extent
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
    
    print(f"\nSpatial extent:")
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
            
            # Cumulative: all sparks from t=0 to t_end
            frame_tracks = region_tracks[region_tracks['time_s'] <= t_end]
            
            if len(frame_tracks) > 0 and frame_tracks['x'].notna().sum() > 0:
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
                H = np.zeros((bins, bins))
                frames_data[region_id].append((H, t_end, 0))
    
    # Determine color scale
    all_values = []
    for region_id in unique_regions:
        if len(frames_data[region_id]) > 0:
            H, _, _ = frames_data[region_id][-1]  # Last frame
            all_values.extend(H.flatten())
    
    vmax = np.percentile(all_values, vmax_percentile) if len(all_values) > 0 else 1
    if vmax == 0:
        vmax = 1
    
    # Create figure with subplots for each region
    if n_regions == 1:
        fig, axes = plt.subplots(1, 1, figsize=(10, 10))
        axes = [axes]
    elif n_regions <= 3:
        fig, axes = plt.subplots(1, n_regions, figsize=(5*n_regions, 5))
        if n_regions == 1:
            axes = [axes]
    else:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
    
    images = []
    text_annotations = []
    
    for idx, region_id in enumerate(unique_regions):
        ax = axes[idx]
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_xlabel('X (pixels)', fontsize=10)
        ax.set_ylabel('Y (pixels)', fontsize=10)
        
        # Get region label
        if 'embryo_id' in tracks_df.columns:
            embryo_ids = post_poke[post_poke['region'] == region_id]['embryo_id'].unique()
            region_label = f"Embryo {', '.join(sorted(embryo_ids))}"
        else:
            region_label = f'Region {region_id}'
        
        ax.set_title(region_label, fontsize=12, fontweight='bold')
        
        # Show poke location(s) for this region
        if 'region' in poke_locations_df.columns:
            region_pokes = poke_locations_df[poke_locations_df['region'] == region_id]
        else:
            # Fallback: find pokes by filename
            region_files = post_poke[post_poke['region'] == region_id]['base_filename'].unique()
            region_pokes = poke_locations_df[poke_locations_df['filename'].isin(region_files)]
        
        if len(region_pokes) > 0:
            ax.scatter(region_pokes['poke_x'], region_pokes['poke_y'],
                      s=40, c='cyan', marker='X',
                      edgecolors='darkblue', linewidths=1,
                      zorder=20, alpha=0.9, label='Poke location')
        
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
    
    # Animation function
    def animate(frame):
        if frame >= n_frames:
            frame = n_frames - 1
        
        t_end = time_bins[frame + 1]
        
        for idx, region_id in enumerate(unique_regions):
            if frame < len(frames_data[region_id]):
                H, t, n_sparks = frames_data[region_id][frame]
            else:
                H, t, n_sparks = frames_data[region_id][-1]
            
            images[idx].set_data(H)
            images[idx].set_clim(0, vmax)
            
            text_annotations[idx].set_text(f't = {t:.1f}s\n{n_sparks:,} sparks')
            text_annotations[idx].set_fontsize(10)
        
        return images + text_annotations
    
    # Create animation
    print(f"\nRendering animation ({n_frames} frames, {fps} fps)...")
    anim = animation.FuncAnimation(fig, animate, frames=n_frames,
                                  interval=1000/fps, blit=True, repeat=True)
    
    # Save animation
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix.lower() == '.gif':
        print(f"Saving as GIF to {output_path}...")
        anim.save(output_path, writer='pillow', fps=fps)
    elif output_path.suffix.lower() in ['.mp4', '.mov']:
        print(f"Saving as MP4 to {output_path}...")
        anim.save(output_path, writer='ffmpeg', fps=fps, bitrate=1800)
    else:
        output_path = output_path.with_suffix('.gif')
        print(f"Saving as GIF to {output_path}...")
        anim.save(output_path, writer='pillow', fps=fps)
    
    print(f"✓ Animation saved to {output_path}")
    plt.close()
    
    print("\n✓ Complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Create animated heatmap from simulated spark tracks',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('input_csv', type=str,
                       help='Path to simulated spark tracks CSV file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for animation (GIF or MP4). Default: input_csv with .gif extension')
    parser.add_argument('--time-window', type=float, default=60.0,
                       help='Time window to animate (seconds, default: 60)')
    parser.add_argument('--frame-interval', type=float, default=1.0,
                       help='Time between frames (seconds, default: 1)')
    parser.add_argument('--fps', type=int, default=5,
                       help='Frames per second for animation (default: 5)')
    parser.add_argument('--colormap', type=str, default='hot',
                       help='Colormap name (default: hot)')
    parser.add_argument('--bins', type=int, default=50,
                       help='Number of bins for 2D histogram (default: 50)')
    
    args = parser.parse_args()
    
    # Load simulation data
    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return 1
    
    print(f"Loading {input_path}...")
    tracks_df = pd.read_csv(input_path)
    print(f"  → Loaded {len(tracks_df):,} track states")
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix('.gif')
    
    # Create animation
    create_animated_heatmap(
        tracks_df, output_path,
        time_window_s=args.time_window,
        frame_interval_s=args.frame_interval,
        fps=args.fps,
        colormap=args.colormap,
        bins=args.bins
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

