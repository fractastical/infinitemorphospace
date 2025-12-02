#!/usr/bin/env python3
"""
Generate comparison plots for all simulation scenarios.

This script creates comprehensive visualizations comparing different
scenarios, especially focusing on multi-embryo configurations (3-4 embryos).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import argparse
import json
import glob


def load_simulation_data(csv_path):
    """Load simulation CSV and metadata."""
    df = pd.read_csv(csv_path)
    metadata_path = csv_path.replace('.csv', '_metadata.json')
    metadata = None
    if Path(metadata_path).exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    return df, metadata


def plot_multi_embryo_comparison(scenario_data, output_path=None):
    """
    Create comprehensive comparison plot for multiple scenarios.
    
    scenario_data: dict mapping scenario_name -> (df, metadata)
    """
    n_scenarios = len(scenario_data)
    
    fig = plt.figure(figsize=(20, 5 * n_scenarios))
    gs = gridspec.GridSpec(n_scenarios, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    for idx, (scenario_name, (df, metadata)) in enumerate(scenario_data.items()):
        if df is None or len(df) == 0:
            continue
        
        # 1. Spatial trajectories
        ax1 = fig.add_subplot(gs[idx, 0])
        for track_id in df['track_id'].unique()[:30]:  # Limit tracks for clarity
            track = df[df['track_id'] == track_id]
            if len(track) > 0:
                ax1.plot(track['x'], track['y'], alpha=0.3, linewidth=0.5)
        
        # Mark embryos
        if metadata and 'embryos' in metadata:
            for emb in metadata['embryos']:
                ax1.scatter([emb['center_x']], [emb['center_y']], s=100,
                           c='blue', marker='o', zorder=10, edgecolors='black', linewidths=1)
                ax1.text(emb['center_x'], emb['center_y'], f"  {emb['id']}",
                        fontsize=8, fontweight='bold')
        
        # Mark pokes
        if metadata and 'pokes' in metadata:
            for poke in metadata['pokes']:
                ax1.scatter([poke['x']], [poke['y']], s=150, c='red', marker='X',
                           zorder=10, edgecolors='black', linewidths=2)
        
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        ax1.set_title(f'{scenario_name}\nSpatial Trajectories', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # 2. Activity over time
        ax2 = fig.add_subplot(gs[idx, 1])
        if 'time_s' in df.columns and 'area' in df.columns:
            activity = df.groupby('time_s')['area'].sum()
            ax2.plot(activity.index, activity.values, linewidth=2, color='green')
            ax2.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Total Activity (pixels²)')
            ax2.set_title('Activity Over Time', fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        # 3. Number of active tracks
        ax3 = fig.add_subplot(gs[idx, 2])
        if 'time_s' in df.columns:
            active_tracks = df.groupby('time_s')['track_id'].nunique()
            ax3.plot(active_tracks.index, active_tracks.values, linewidth=2, color='blue')
            ax3.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax3.set_xlabel('Time (seconds)')
            ax3.set_ylabel('Active Tracks')
            ax3.set_title('Active Tracks Over Time', fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # 4. Summary statistics
        ax4 = fig.add_subplot(gs[idx, 3])
        ax4.axis('off')
        
        stats = []
        stats.append(['Metric', 'Value'])
        stats.append(['---', '---'])
        stats.append(['Total tracks', df['track_id'].nunique()])
        stats.append(['Total events', len(df)])
        
        if 'embryo_id' in df.columns:
            stats.append(['Embryos', df['embryo_id'].nunique()])
            embryo_counts = df['embryo_id'].value_counts()
            for emb_id, count in embryo_counts.items():
                stats.append([f'  {emb_id} events', count])
        
        if 'speed' in df.columns:
            speeds = df['speed'].dropna()
            if len(speeds) > 0:
                stats.append(['Mean speed (px/s)', f"{speeds.mean():.2f}"])
        
        if metadata:
            stats.append(['Pokes', len(metadata.get('pokes', []))])
        
        table = ax4.table(cellText=stats, cellLoc='left', loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        ax4.set_title('Summary Statistics', fontweight='bold', pad=10)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved comparison plot to {output_path}")
    else:
        plt.show()


def plot_embryo_count_comparison(scenario_data, output_path=None):
    """Compare scenarios by number of embryos."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Group by embryo count
    by_embryo_count = {}
    for name, (df, metadata) in scenario_data.items():
        if df is None or len(df) == 0:
            continue
        n_embryos = df['embryo_id'].nunique() if 'embryo_id' in df.columns else 0
        if n_embryos not in by_embryo_count:
            by_embryo_count[n_embryos] = []
        by_embryo_count[n_embryos].append((name, df, metadata))
    
    # 1. Activity over time by embryo count
    ax = axes[0, 0]
    for n_embryos in sorted(by_embryo_count.keys()):
        for name, df, _ in by_embryo_count[n_embryos]:
            if 'time_s' in df.columns and 'area' in df.columns:
                activity = df.groupby('time_s')['area'].sum()
                label = f"{n_embryos} embryos: {name}"
                ax.plot(activity.index, activity.values, linewidth=2, label=label, alpha=0.7)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Total Activity')
    ax.set_title('Activity Over Time by Embryo Count', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2. Active tracks over time by embryo count
    ax = axes[0, 1]
    for n_embryos in sorted(by_embryo_count.keys()):
        for name, df, _ in by_embryo_count[n_embryos]:
            if 'time_s' in df.columns:
                active = df.groupby('time_s')['track_id'].nunique()
                label = f"{n_embryos} embryos: {name}"
                ax.plot(active.index, active.values, linewidth=2, label=label, alpha=0.7)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Active Tracks')
    ax.set_title('Active Tracks Over Time by Embryo Count', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. Peak activity by scenario
    ax = axes[1, 0]
    scenario_names = []
    peak_activities = []
    for name, (df, _) in sorted(scenario_data.items()):
        if df is None or len(df) == 0:
            continue
        if 'time_s' in df.columns and 'area' in df.columns:
            activity = df.groupby('time_s')['area'].sum()
            peak = activity.max()
            scenario_names.append(name.replace('_', '\n'))
            peak_activities.append(peak)
    
    if scenario_names:
        bars = ax.bar(range(len(scenario_names)), peak_activities, alpha=0.7)
        ax.set_xticks(range(len(scenario_names)))
        ax.set_xticklabels(scenario_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Peak Activity (pixels²)')
        ax.set_title('Peak Activity by Scenario', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Total tracks by scenario
    ax = axes[1, 1]
    scenario_names = []
    total_tracks = []
    for name, (df, _) in sorted(scenario_data.items()):
        if df is None or len(df) == 0:
            continue
        n_tracks = df['track_id'].nunique()
        scenario_names.append(name.replace('_', '\n'))
        total_tracks.append(n_tracks)
    
    if scenario_names:
        bars = ax.bar(range(len(scenario_names)), total_tracks, alpha=0.7, color='orange')
        ax.set_xticks(range(len(scenario_names)))
        ax.set_xticklabels(scenario_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Total Tracks')
        ax.set_title('Total Tracks by Scenario', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved embryo count comparison to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison plots for all simulation scenarios"
    )
    parser.add_argument(
        '--input-dir',
        default='simulation_outputs',
        help='Directory containing simulated CSV files'
    )
    parser.add_argument(
        '--output-dir',
        default='simulation_outputs/comparisons',
        help='Output directory for plots'
    )
    parser.add_argument(
        '--scenarios',
        nargs='+',
        help='Specific scenarios to include (default: all found)'
    )
    
    args = parser.parse_args()
    
    # Find all simulation CSV files
    input_dir = Path(args.input_dir)
    csv_files = list(input_dir.glob('simulated_*.csv'))
    
    if not csv_files:
        print(f"No simulation files found in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} simulation files")
    
    # Load all scenarios
    scenario_data = {}
    for csv_file in csv_files:
        # Extract scenario name from filename
        scenario_name = csv_file.stem.replace('simulated_', '')
        if args.scenarios and scenario_name not in args.scenarios:
            continue
        
        try:
            df, metadata = load_simulation_data(str(csv_file))
            scenario_data[scenario_name] = (df, metadata)
            print(f"  ✓ Loaded {scenario_name}: {len(df)} events, {df['track_id'].nunique()} tracks")
        except Exception as e:
            print(f"  ✗ Failed to load {scenario_name}: {e}")
    
    if not scenario_data:
        print("No valid scenarios loaded")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    
    # 1. Multi-embryo comparison (detailed view)
    plot_multi_embryo_comparison(
        scenario_data,
        output_path=str(output_dir / 'all_scenarios_comparison.png')
    )
    
    # 2. Embryo count comparison
    plot_embryo_count_comparison(
        scenario_data,
        output_path=str(output_dir / 'embryo_count_comparison.png')
    )
    
    print(f"\n✓ All plots saved to {output_dir}/")


if __name__ == "__main__":
    main()

