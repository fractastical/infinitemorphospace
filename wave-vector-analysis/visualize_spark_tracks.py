#!/usr/bin/env python3
"""
Visualization and analysis tools for spark_tracks.csv and vector_clusters.csv.

This script provides various visualizations of Ca²⁺ wave dynamics:
- Trajectory plots
- Speed/direction distributions
- Time series
- Spatial heatmaps
- Per-embryo analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import sys
import argparse


def plot_trajectories(df_tracks, output_path=None, max_tracks=100):
    """Plot spark trajectories overlaid on a spatial map."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Sample tracks if too many
    unique_tracks = df_tracks['track_id'].unique()
    if len(unique_tracks) > max_tracks:
        print(f"Plotting {max_tracks} of {len(unique_tracks)} tracks...")
        np.random.seed(42)
        sampled_tracks = np.random.choice(unique_tracks, max_tracks, replace=False)
        df_plot = df_tracks[df_tracks['track_id'].isin(sampled_tracks)]
    else:
        df_plot = df_tracks
    
    # Plot each track
    for track_id, df_track in df_plot.groupby('track_id'):
        df_sorted = df_track.sort_values('frame_idx')
        ax.plot(df_sorted['x'], df_sorted['y'], alpha=0.3, linewidth=0.5)
        # Mark start
        ax.plot(df_sorted['x'].iloc[0], df_sorted['y'].iloc[0], 'go', markersize=3, alpha=0.5)
        # Mark end
        ax.plot(df_sorted['x'].iloc[-1], df_sorted['y'].iloc[-1], 'ro', markersize=3, alpha=0.5)
    
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title('Ca²⁺ Event Trajectories\n(Green=start, Red=end)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory plot to {output_path}")
    else:
        plt.show()


def plot_speed_distribution(df_clusters, output_path=None):
    """Plot distribution of propagation speeds."""
    if df_clusters is None or len(df_clusters) == 0:
        print("No cluster data available for speed plot.")
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    speeds = df_clusters['mean_speed_px_per_s'].dropna()
    
    if len(speeds) == 0:
        print("No speed data available.")
        return
    
    # Histogram - define bins explicitly to avoid issues
    speed_min, speed_max = speeds.min(), speeds.max()
    if speed_max > speed_min:
        bins = np.linspace(speed_min, speed_max, 51)
        axes[0].hist(speeds.values, bins=bins, edgecolor='black', alpha=0.7)
    else:
        axes[0].hist(speeds.values, bins=10, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Mean Speed (pixels/second)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Mean Propagation Speeds')
    axes[0].grid(True, alpha=0.3)
    
    # Box plot - convert to list to avoid issues
    axes[1].boxplot([speeds.values], vert=True, tick_labels=[''])
    axes[1].set_ylabel('Mean Speed (pixels/second)')
    axes[1].set_title('Speed Distribution (Box Plot)')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved speed distribution to {output_path}")
    else:
        plt.show()


def plot_time_series(df_tracks, output_path=None):
    """Plot number of active tracks over time."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Count tracks per frame
    tracks_per_frame = df_tracks.groupby('time_s')['track_id'].nunique()
    
    axes[0].plot(tracks_per_frame.index, tracks_per_frame.values, linewidth=1.5)
    axes[0].axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Poke time')
    axes[0].set_xlabel('Time (seconds, relative to poke)')
    axes[0].set_ylabel('Number of Active Tracks')
    axes[0].set_title('Ca²⁺ Event Activity Over Time')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Total area per frame (proxy for Ca²⁺ signal intensity)
    area_per_frame = df_tracks.groupby('time_s')['area'].sum()
    axes[1].plot(area_per_frame.index, area_per_frame.values, linewidth=1.5, color='orange')
    axes[1].axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Poke time')
    axes[1].set_xlabel('Time (seconds, relative to poke)')
    axes[1].set_ylabel('Total Area (pixels²)')
    axes[1].set_title('Integrated Ca²⁺ Signal Over Time')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved time series to {output_path}")
    else:
        plt.show()


def plot_spatial_heatmap(df_tracks, output_path=None):
    """Create a spatial heatmap of spark density."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create 2D histogram
    H, xedges, yedges = np.histogram2d(
        df_tracks['x'], df_tracks['y'],
        bins=50
    )
    
    # Plot heatmap
    im = ax.imshow(
        H.T, origin='lower',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap='hot', interpolation='nearest'
    )
    
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title('Spatial Density of Ca²⁺ Events')
    plt.colorbar(im, ax=ax, label='Event Count')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved spatial heatmap to {output_path}")
    else:
        plt.show()


def plot_embryo_comparison(df_tracks, output_path=None):
    """Compare Ca²⁺ dynamics between embryos."""
    if 'embryo_id' not in df_tracks.columns:
        print("No embryo_id column found. Skipping embryo comparison.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Count tracks per embryo
    tracks_per_embryo = df_tracks.groupby('embryo_id')['track_id'].nunique()
    axes[0, 0].bar(tracks_per_embryo.index, tracks_per_embryo.values)
    axes[0, 0].set_xlabel('Embryo ID')
    axes[0, 0].set_ylabel('Number of Tracks')
    axes[0, 0].set_title('Tracks per Embryo')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Mean speed per embryo
    if 'speed' in df_tracks.columns:
        speed_per_embryo = df_tracks.groupby('embryo_id')['speed'].mean()
        axes[0, 1].bar(speed_per_embryo.index, speed_per_embryo.values)
        axes[0, 1].set_xlabel('Embryo ID')
        axes[0, 1].set_ylabel('Mean Speed (pixels/second)')
        axes[0, 1].set_title('Mean Speed per Embryo')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Activity over time per embryo
    for embryo_id in df_tracks['embryo_id'].dropna().unique():
        df_emb = df_tracks[df_tracks['embryo_id'] == embryo_id]
        tracks_per_frame = df_emb.groupby('time_s')['track_id'].nunique()
        axes[1, 0].plot(tracks_per_frame.index, tracks_per_frame.values, 
                       label=f'Embryo {embryo_id}', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Time (seconds)')
    axes[1, 0].set_ylabel('Active Tracks')
    axes[1, 0].set_title('Activity Over Time by Embryo')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # AP position distribution
    if 'ap_norm' in df_tracks.columns:
        # Collect all AP values to determine consistent bin range
        all_ap_vals = df_tracks['ap_norm'].dropna()
        if len(all_ap_vals) > 0:
            ap_min, ap_max = all_ap_vals.min(), all_ap_vals.max()
            # Define consistent bins for all histograms
            bins = np.linspace(ap_min, ap_max, 21) if ap_max > ap_min else 20
            
            for embryo_id in df_tracks['embryo_id'].dropna().unique():
                df_emb = df_tracks[df_tracks['embryo_id'] == embryo_id]
                ap_vals = df_emb['ap_norm'].dropna()
                if len(ap_vals) > 0:
                    axes[1, 1].hist(ap_vals.values, bins=bins, alpha=0.5, label=f'Embryo {embryo_id}')
            axes[1, 1].set_xlabel('AP Position (0=head, 1=tail)')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('AP Position Distribution')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved embryo comparison to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize spark track data')
    parser.add_argument('tracks_csv', help='Path to spark_tracks.csv')
    parser.add_argument('--clusters-csv', help='Path to vector_clusters.csv (optional)')
    parser.add_argument('--output-dir', help='Directory to save plots (default: show plots)')
    parser.add_argument('--plot', choices=['all', 'trajectories', 'speed', 'time', 'heatmap', 'embryo'],
                       default='all', help='Which plot to generate')
    
    args = parser.parse_args()
    
    print(f"Loading {args.tracks_csv}...")
    df_tracks = pd.read_csv(args.tracks_csv)
    print(f"Loaded {len(df_tracks)} track states")
    
    df_clusters = None
    if args.clusters_csv:
        print(f"Loading {args.clusters_csv}...")
        df_clusters = pd.read_csv(args.clusters_csv)
        print(f"Loaded {len(df_clusters)} clusters")
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_funcs = {
        'trajectories': lambda: plot_trajectories(
            df_tracks, 
            output_path=output_dir / 'trajectories.png' if output_dir else None
        ),
        'speed': lambda: plot_speed_distribution(
            df_clusters if df_clusters is not None else None,
            output_path=output_dir / 'speed_distribution.png' if output_dir else None
        ) if df_clusters is not None else print("Need clusters CSV for speed plot"),
        'time': lambda: plot_time_series(
            df_tracks,
            output_path=output_dir / 'time_series.png' if output_dir else None
        ),
        'heatmap': lambda: plot_spatial_heatmap(
            df_tracks,
            output_path=output_dir / 'spatial_heatmap.png' if output_dir else None
        ),
        'embryo': lambda: plot_embryo_comparison(
            df_tracks,
            output_path=output_dir / 'embryo_comparison.png' if output_dir else None
        ),
    }
    
    if args.plot == 'all':
        for plot_name, plot_func in plot_funcs.items():
            try:
                plot_func()
            except Exception as e:
                print(f"Error generating {plot_name} plot: {e}")
    else:
        if args.plot in plot_funcs:
            plot_funcs[args.plot]()
        else:
            print(f"Unknown plot type: {args.plot}")


if __name__ == '__main__':
    main()

