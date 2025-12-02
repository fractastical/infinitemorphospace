#!/usr/bin/env python3
"""
Visualize simulated Ca²⁺ wave data.

Compare simulated trajectories, wave propagation patterns, and spatial
distributions to help validate simulation parameters against real data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, Arrow
import argparse
from pathlib import Path
import json


def load_simulation_metadata(csv_path):
    """Load metadata JSON file if it exists."""
    metadata_path = csv_path.replace('.csv', '_metadata.json')
    if Path(metadata_path).exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None


def plot_simulation_overview(df, metadata=None, output_path=None):
    """Create overview plot of simulated data."""
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Spatial distribution with trajectories
    ax1 = plt.subplot(2, 3, 1)
    for track_id in df['track_id'].unique()[:50]:  # Limit to 50 tracks for clarity
        track = df[df['track_id'] == track_id]
        if len(track) > 0:
            ax1.plot(track['x'], track['y'], alpha=0.3, linewidth=0.5)
    
    # Mark poke locations
    if metadata and 'pokes' in metadata:
        for poke in metadata['pokes']:
            ax1.scatter([poke['x']], [poke['y']], s=200, c='red', marker='X', 
                       zorder=10, edgecolors='black', linewidths=2)
    
    # Mark embryo centers
    if metadata and 'embryos' in metadata:
        for emb in metadata['embryos']:
            ax1.scatter([emb['center_x']], [emb['center_y']], s=100, 
                       c='blue', marker='o', zorder=10, edgecolors='black', linewidths=1)
            ax1.text(emb['center_x'], emb['center_y'], f"  {emb['id']}", 
                    fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.set_title('Spatial Trajectories', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 2. Speed distribution
    ax2 = plt.subplot(2, 3, 2)
    speeds = df['speed'].dropna()
    if len(speeds) > 0:
        ax2.hist(speeds, bins=50, edgecolor='black', alpha=0.7, color='orange')
        ax2.axvline(speeds.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {speeds.mean():.2f} px/s')
        ax2.set_xlabel('Speed (pixels/second)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Speed Distribution', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Activity over time
    ax3 = plt.subplot(2, 3, 3)
    activity_over_time = df.groupby('time_s')['area'].sum()
    ax3.plot(activity_over_time.index, activity_over_time.values, 
            linewidth=2, color='green')
    ax3.axvline(0, color='red', linestyle='--', linewidth=1, label='Poke time')
    ax3.set_xlabel('Time (seconds, relative to poke)')
    ax3.set_ylabel('Total Activity (pixels²)')
    ax3.set_title('Activity Over Time', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Directionality (rose plot)
    ax4 = plt.subplot(2, 3, 4, projection='polar')
    angles = df['angle_deg'].dropna()
    if len(angles) > 0:
        angles_rad = np.radians(angles)
        n_bins = 36
        counts, bins = np.histogram(angles_rad, bins=n_bins, range=(0, 2*np.pi))
        theta = (bins[:-1] + bins[1:]) / 2
        ax4.bar(theta, counts, width=2*np.pi/n_bins, alpha=0.7, color='purple')
        ax4.set_title('Wave Directionality', fontweight='bold', pad=20)
        ax4.set_theta_zero_location('E')
        ax4.set_theta_direction(1)
    
    # 5. Number of active tracks over time
    ax5 = plt.subplot(2, 3, 5)
    active_tracks = df.groupby('time_s')['track_id'].nunique()
    ax5.plot(active_tracks.index, active_tracks.values, 
            linewidth=2, color='blue')
    ax5.axvline(0, color='red', linestyle='--', linewidth=1, label='Poke time')
    ax5.set_xlabel('Time (seconds, relative to poke)')
    ax5.set_ylabel('Number of Active Tracks')
    ax5.set_title('Active Tracks Over Time', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. AP position distribution
    ax6 = plt.subplot(2, 3, 6)
    ap_vals = df['ap_norm'].dropna()
    if len(ap_vals) > 0:
        ax6.hist(ap_vals, bins=30, edgecolor='black', alpha=0.7, color='cyan')
        ax6.axvline(0.7, color='red', linestyle='--', linewidth=2, 
                   label='Tail threshold (0.7)')
        ax6.set_xlabel('AP Position (0=head, 1=tail)')
        ax6.set_ylabel('Frequency')
        ax6.set_title('AP Position Distribution', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved simulation visualization to {output_path}")
    else:
        plt.show()


def compare_simulated_vs_real(sim_df, real_df, output_path=None):
    """Compare simulated data against real experimental data."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 1. Speed distribution comparison
    ax = axes[0, 0]
    sim_speeds = sim_df['speed'].dropna()
    real_speeds = real_df['speed'].dropna()
    if len(sim_speeds) > 0 and len(real_speeds) > 0:
        bins = np.linspace(0, max(sim_speeds.max(), real_speeds.max()), 50)
        ax.hist(sim_speeds, bins=bins, alpha=0.5, label='Simulated', color='blue', density=True)
        ax.hist(real_speeds, bins=bins, alpha=0.5, label='Real', color='orange', density=True)
        ax.set_xlabel('Speed (px/s)')
        ax.set_ylabel('Density')
        ax.set_title('Speed Distribution Comparison', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. Activity over time comparison
    ax = axes[0, 1]
    sim_activity = sim_df.groupby('time_s')['area'].sum()
    real_activity = real_df.groupby('time_s')['area'].sum()
    if len(sim_activity) > 0:
        ax.plot(sim_activity.index, sim_activity.values, label='Simulated', 
               linewidth=2, color='blue')
    if len(real_activity) > 0:
        ax.plot(real_activity.index, real_activity.values, label='Real', 
               linewidth=2, color='orange', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Total Activity')
    ax.set_title('Activity Over Time', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Track duration comparison
    ax = axes[0, 2]
    sim_durations = sim_df.groupby('track_id')['time_s'].apply(lambda x: x.max() - x.min())
    real_durations = real_df.groupby('track_id')['time_s'].apply(lambda x: x.max() - x.min())
    if len(sim_durations) > 0 and len(real_durations) > 0:
        bins = np.linspace(0, max(sim_durations.max(), real_durations.max()), 30)
        ax.hist(sim_durations, bins=bins, alpha=0.5, label='Simulated', color='blue', density=True)
        ax.hist(real_durations, bins=bins, alpha=0.5, label='Real', color='orange', density=True)
        ax.set_xlabel('Track Duration (s)')
        ax.set_ylabel('Density')
        ax.set_title('Track Duration Comparison', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 4. Spatial spread comparison
    ax = axes[1, 0]
    # Calculate spatial spread (std dev of positions)
    sim_spread = sim_df.groupby('time_s').agg({
        'x': lambda x: x.std(),
        'y': lambda y: y.std()
    }).mean(axis=1)
    real_spread = real_df.groupby('time_s').agg({
        'x': lambda x: x.std(),
        'y': lambda y: y.std()
    }).mean(axis=1)
    if len(sim_spread) > 0:
        ax.plot(sim_spread.index, sim_spread.values, label='Simulated', 
               linewidth=2, color='blue')
    if len(real_spread) > 0:
        ax.plot(real_spread.index, real_spread.values, label='Real', 
               linewidth=2, color='orange', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Spatial Spread (px)')
    ax.set_title('Spatial Spread Over Time', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Number of tracks comparison
    ax = axes[1, 1]
    sim_n_tracks = sim_df.groupby('time_s')['track_id'].nunique()
    real_n_tracks = real_df.groupby('time_s')['track_id'].nunique()
    if len(sim_n_tracks) > 0:
        ax.plot(sim_n_tracks.index, sim_n_tracks.values, label='Simulated', 
               linewidth=2, color='blue')
    if len(real_n_tracks) > 0:
        ax.plot(real_n_tracks.index, real_n_tracks.values, label='Real', 
               linewidth=2, color='orange', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Number of Tracks')
    ax.set_title('Active Tracks Over Time', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Summary statistics table
    ax = axes[1, 2]
    ax.axis('off')
    
    stats_data = []
    stats_data.append(['Metric', 'Simulated', 'Real'])
    stats_data.append(['---', '---', '---'])
    
    if len(sim_df) > 0:
        stats_data.append(['Total events', len(sim_df), len(real_df)])
        stats_data.append(['Unique tracks', sim_df['track_id'].nunique(), 
                          real_df['track_id'].nunique()])
        
        if 'speed' in sim_df.columns and 'speed' in real_df.columns:
            stats_data.append(['Mean speed (px/s)', 
                             f"{sim_df['speed'].mean():.2f}", 
                             f"{real_df['speed'].mean():.2f}"])
    
    table = ax.table(cellText=stats_data, cellLoc='left', loc='center',
                    colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax.set_title('Summary Statistics', fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize simulated Ca²⁺ wave data"
    )
    parser.add_argument(
        'sim_csv',
        help='Path to simulated spark_tracks.csv'
    )
    parser.add_argument(
        '--real-csv',
        help='Path to real spark_tracks.csv for comparison'
    )
    parser.add_argument(
        '--output',
        help='Output plot path (default: simulation_visualization.png)',
        default=None
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Generate comparison plot with real data'
    )
    
    args = parser.parse_args()
    
    # Load simulated data
    print(f"Loading simulated data from {args.sim_csv}...")
    sim_df = pd.read_csv(args.sim_csv)
    metadata = load_simulation_metadata(args.sim_csv)
    
    if args.compare and args.real_csv:
        print(f"Loading real data from {args.real_csv}...")
        real_df = pd.read_csv(args.real_csv)
        
        output = args.output or 'simulation_comparison.png'
        compare_simulated_vs_real(sim_df, real_df, output_path=output)
    else:
        output = args.output or 'simulation_visualization.png'
        plot_simulation_overview(sim_df, metadata=metadata, output_path=output)
    
    print(f"\n✓ Visualization complete!")


if __name__ == "__main__":
    main()

