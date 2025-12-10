#!/usr/bin/env python3
"""
Create scientifically useful visualizations from Ca²⁺ wave data.

This script generates visualizations that provide new insights not available
in standard analysis tools:
1. Flow field paintings - spatial vector fields showing wave propagation directions
2. 3D time-space sculptures - temporal-spatial dynamics with time as z-axis
3. Speed gradient flow - spatial mapping of propagation speeds

These visualizations complement existing analysis tools by providing:
- Spatial vector field representations (not just direction distributions)
- Combined temporal-spatial views (not just separate 2D/1D views)
- Spatial speed mapping (not just speed histograms)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import argparse
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')


def create_flow_field_painting(df_tracks, output_path, style='aurora'):
    """
    Create a beautiful flow field visualization showing wave directions and speeds.
    Style options: 'aurora', 'fire', 'ocean', 'neon'
    """
    print(f"Creating flow field painting (style: {style})...")
    
    # Filter valid velocity data
    valid = df_tracks[(df_tracks['vx'].notna()) & (df_tracks['vy'].notna()) & 
                      (df_tracks['speed'].notna()) & (df_tracks['speed'] > 0)]
    
    if len(valid) == 0:
        print("No valid velocity data found")
        return
    
    # Sample for performance if too many points
    if len(valid) > 50000:
        valid = valid.sample(n=50000, random_state=42)
    
    fig, ax = plt.subplots(figsize=(16, 16), facecolor='black')
    ax.set_facecolor('black')
    
    # Define color schemes
    color_schemes = {
        'aurora': ['#000000', '#1a0033', '#330066', '#4d0099', '#6600cc', '#7f00ff', '#9933ff', '#ff00ff', '#ff66ff', '#ffffff'],
        'fire': ['#000000', '#330000', '#660000', '#990000', '#cc0000', '#ff3300', '#ff6600', '#ff9900', '#ffcc00', '#ffff00'],
        'ocean': ['#000033', '#000066', '#000099', '#0033cc', '#0066ff', '#0099ff', '#00ccff', '#00ffff', '#66ffff', '#ffffff'],
        'neon': ['#000000', '#003300', '#006600', '#009900', '#00cc00', '#00ff00', '#33ff33', '#66ff66', '#99ff99', '#ffffff']
    }
    
    colors = color_schemes.get(style, color_schemes['aurora'])
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)
    
    # Create grid for interpolation
    x_min, x_max = valid['x'].min(), valid['x'].max()
    y_min, y_max = valid['y'].min(), valid['y'].max()
    
    # Extend bounds
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * 0.1
    x_max += x_range * 0.1
    y_min -= y_range * 0.1
    y_max += y_range * 0.1
    
    # Create grid
    grid_res = 200
    xi = np.linspace(x_min, x_max, grid_res)
    yi = np.linspace(y_min, y_max, grid_res)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Interpolate speed
    speed_grid = griddata(
        (valid['x'], valid['y']),
        valid['speed'],
        (xi_grid, yi_grid),
        method='cubic',
        fill_value=0
    )
    
    # Smooth
    speed_grid = gaussian_filter(speed_grid, sigma=2)
    
    # Normalize for color mapping
    speed_norm = (speed_grid - speed_grid.min()) / (speed_grid.max() - speed_grid.min() + 1e-10)
    
    # Create base image
    im = ax.imshow(speed_norm, extent=[x_min, x_max, y_min, y_max],
                   origin='lower', cmap=cmap, alpha=0.9, interpolation='bilinear')
    
    # Overlay vector field (sample for clarity)
    sample_step = 15
    x_sample = xi[::sample_step]
    y_sample = yi[::sample_step]
    x_grid_sample, y_grid_sample = np.meshgrid(x_sample, y_sample)
    
    # Interpolate velocities
    vx_grid = griddata(
        (valid['x'], valid['y']),
        valid['vx'],
        (x_grid_sample, y_grid_sample),
        method='cubic',
        fill_value=0
    )
    vy_grid = griddata(
        (valid['x'], valid['y']),
        valid['vy'],
        (x_grid_sample, y_grid_sample),
        method='cubic',
        fill_value=0
    )
    
    # Normalize vectors for display
    magnitude = np.sqrt(vx_grid**2 + vy_grid**2)
    magnitude[magnitude == 0] = 1
    vx_norm = vx_grid / magnitude * (x_range / grid_res * sample_step * 0.8)
    vy_norm = vy_grid / magnitude * (y_range / grid_res * sample_step * 0.8)
    
    # Draw vectors
    ax.quiver(x_grid_sample, y_grid_sample, vx_norm, vy_norm,
              angles='xy', scale_units='xy', scale=1,
              color='white', alpha=0.3, width=0.003, headwidth=3, headlength=4)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axis('off')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black', edgecolor='none')
    print(f"✓ Saved to {output_path}")
    plt.close()


def create_3d_time_sculpture(df_tracks, output_path, time_window=60):
    """
    Create a 3D visualization with time as the z-axis.
    """
    print("Creating 3D time-space sculpture...")
    
    # Filter to time window
    post_poke = df_tracks[(df_tracks['time_s'] >= 0) & (df_tracks['time_s'] <= time_window)].copy()
    
    if len(post_poke) == 0:
        print("No data in time window")
        return
    
    # Sample for performance
    if len(post_poke) > 100000:
        post_poke = post_poke.sample(n=100000, random_state=42)
    
    fig = plt.figure(figsize=(16, 12), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    
    # Color by time
    times = post_poke['time_s'].values
    colors = plt.cm.plasma(times / times.max())
    
    # Scatter plot
    scatter = ax.scatter(post_poke['x'], post_poke['y'], post_poke['time_s'],
                        c=colors, s=0.5, alpha=0.6, edgecolors='none')
    
    ax.set_xlabel('X (pixels)', color='white', fontsize=12)
    ax.set_ylabel('Y (pixels)', color='white', fontsize=12)
    ax.set_zlabel('Time (seconds)', color='white', fontsize=12)
    ax.tick_params(colors='white')
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black', edgecolor='none')
    print(f"✓ Saved to {output_path}")
    plt.close()


def create_speed_gradient_flow(df_tracks, output_path):
    """
    Create a flowing gradient visualization based on speed.
    """
    print("Creating speed gradient flow...")
    
    valid = df_tracks[(df_tracks['x'].notna()) & (df_tracks['y'].notna()) & 
                     (df_tracks['speed'].notna()) & (df_tracks['speed'] > 0)]
    
    if len(valid) == 0:
        print("No valid speed data")
        return
    
    # Sample for performance
    if len(valid) > 100000:
        valid = valid.sample(n=100000, random_state=42)
    
    fig, ax = plt.subplots(figsize=(20, 20), facecolor='black')
    ax.set_facecolor('black')
    
    # Create scatter plot with speed-based colors and sizes
    speeds = valid['speed'].values
    speed_norm = (speeds - speeds.min()) / (speeds.max() - speeds.min() + 1e-10)
    
    scatter = ax.scatter(valid['x'], valid['y'],
                        c=speed_norm, s=speeds * 2,
                        cmap='turbo', alpha=0.6, edgecolors='none')
    
    ax.set_xlim(valid['x'].min() - 100, valid['x'].max() + 100)
    ax.set_ylim(valid['y'].min() - 100, valid['y'].max() + 100)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.colorbar(scatter, ax=ax, label='Speed (normalized)', pad=0.01)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black', edgecolor='none')
    print(f"✓ Saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Create artistic visualizations from Ca²⁺ wave data'
    )
    parser.add_argument('tracks_csv', help='Path to spark_tracks.csv')
    parser.add_argument('--clusters-csv', help='Path to vector_clusters.csv (optional)')
    parser.add_argument('--output-dir', default='analysis_results/artistic',
                       help='Output directory for visualizations')
    parser.add_argument('--visualizations', nargs='+',
                       choices=['all', 'flow', '3d', 'gradient'],
                       default=['all'],
                       help='Which visualizations to create')
    parser.add_argument('--style', choices=['aurora', 'fire', 'ocean', 'neon'],
                       default='aurora', help='Color style for flow field')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {args.tracks_csv}...")
    df_tracks = pd.read_csv(args.tracks_csv)
    print(f"  → Loaded {len(df_tracks):,} track states")
    
    df_clusters = None
    if args.clusters_csv:
        print(f"\nLoading {args.clusters_csv}...")
        df_clusters = pd.read_csv(args.clusters_csv)
        print(f"  → Loaded {len(df_clusters):,} clusters")
    
    visualizations = args.visualizations
    if 'all' in visualizations:
        visualizations = ['flow', '3d', 'gradient']
    
    print(f"\nCreating {len(visualizations)} visualization(s)...\n")
    
    if 'flow' in visualizations:
        create_flow_field_painting(df_tracks, output_dir / f'flow_field_{args.style}.png', style=args.style)
    
    if '3d' in visualizations:
        create_3d_time_sculpture(df_tracks, output_dir / '3d_time_sculpture.png')
    
    if 'gradient' in visualizations:
        create_speed_gradient_flow(df_tracks, output_dir / 'speed_gradient_flow.png')
    
    print("\n✓ All visualizations complete!")


if __name__ == '__main__':
    main()

