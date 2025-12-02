#!/usr/bin/env python3
"""
Visualize all poke locations across multiple processing runs.

This script can:
1. Read poke locations from a poke_locations.csv file (if created by parser)
2. Infer poke locations from spark_tracks.csv by finding earliest spark clusters
3. Plot all poke locations with labels and statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import argparse
from pathlib import Path
import sys

def infer_poke_from_early_sparks(tracks_df, time_window_s=2.0, min_sparks=1):
    """
    Infer poke location from the first spark cluster that appears in each file.
    
    Args:
        tracks_df: DataFrame with spark_tracks.csv data
        time_window_s: Not used anymore - kept for compatibility
        min_sparks: Minimum number of sparks needed to infer location (default 1, just need first cluster)
    
    Returns:
        DataFrame with columns: filename, poke_x, poke_y, n_sparks, method
    """
    # Group by base filename (strip page numbers for multi-page files)
    poke_locations = []
    
    # Get base filenames (without page numbers)
    if 'filename' in tracks_df.columns:
        tracks_df = tracks_df.copy()
        tracks_df['base_filename'] = tracks_df['filename'].str.replace(r' \(page \d+\)', '', regex=True)
        unique_base_files = sorted(tracks_df['base_filename'].unique())
        print(f"  → Found {len(unique_base_files)} unique base files (excluding page variants)")
    else:
        unique_base_files = ['unknown']
        tracks_df['base_filename'] = 'unknown'
    
    for base_file in unique_base_files:
        file_data = tracks_df[tracks_df['base_filename'] == base_file].copy()
        
        if len(file_data) == 0:
            continue
        
        # Find the first spark cluster(s) that appear in this file
        # Use frame_idx to find the earliest frame, or time_s if frame_idx not available
        if 'frame_idx' in file_data.columns:
            min_frame = file_data['frame_idx'].min()
            first_sparks = file_data[file_data['frame_idx'] == min_frame].copy()
        elif 'time_s' in file_data.columns:
            min_time = file_data['time_s'].min()
            # Use first 10 seconds worth of data, or first frame's worth
            first_sparks = file_data[file_data['time_s'] <= min_time + 10].copy()
        else:
            # No way to determine first - skip
            continue
        
        if len(first_sparks) < 1:  # Changed from min_sparks to just need at least 1
            continue
        
        # If we have multiple sparks in first frame, use the largest/earliest ones
        # Sort by area (largest first) or by x,y position
        if 'area' in first_sparks.columns:
            first_sparks = first_sparks.sort_values('area', ascending=False)
        
        # Use first few sparks (or all if few)
        n_to_use = min(min_sparks, len(first_sparks)) if min_sparks > 1 else len(first_sparks)
        use_sparks = first_sparks.head(n_to_use)
        
        # Calculate weighted centroid (weight by area if available)
        if 'area' in use_sparks.columns:
            weights = use_sparks['area'].fillna(1)
        else:
            weights = pd.Series(1, index=use_sparks.index)
        
        total_weight = weights.sum()
        if total_weight == 0:
            weights = pd.Series(1, index=use_sparks.index)
            total_weight = len(use_sparks)
        
        poke_x = (use_sparks['x'] * weights).sum() / total_weight
        poke_y = (use_sparks['y'] * weights).sum() / total_weight
        n_sparks = len(use_sparks)
        
        # Get frame/time info
        if 'frame_idx' in use_sparks.columns:
            frame_info = f"frame {use_sparks['frame_idx'].min()}"
        elif 'time_s' in use_sparks.columns:
            min_time = use_sparks['time_s'].min()
            max_time = use_sparks['time_s'].max()
            frame_info = f"time {min_time:.2f}-{max_time:.2f}s"
        else:
            frame_info = "unknown"
        
        poke_locations.append({
            'filename': base_file,
            'poke_x': poke_x,
            'poke_y': poke_y,
            'n_sparks': n_sparks,
            'time_range_s': frame_info,
            'method': 'inferred_from_first_cluster'
        })
    
    return pd.DataFrame(poke_locations)


def read_poke_locations_csv(poke_csv_path):
    """Read poke locations from a CSV file."""
    df = pd.read_csv(poke_csv_path)
    required_cols = ['poke_x', 'poke_y']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"poke_locations.csv must have columns: {required_cols}")
    
    # Ensure filename column exists
    if 'filename' not in df.columns:
        df['filename'] = f"poke_{df.index}"
    
    return df


def plot_poke_locations(poke_df, output_path=None, show_embryos=True, tracks_df=None):
    """
    Plot all poke locations.
    
    Args:
        poke_df: DataFrame with poke locations (columns: filename, poke_x, poke_y, ...)
        output_path: Path to save plot (if None, displays interactively)
        show_embryos: Whether to overlay embryo outlines (if available)
        tracks_df: Optional DataFrame with spark tracks (for embryo visualization)
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Extract poke locations
    poke_x = poke_df['poke_x'].values
    poke_y = poke_df['poke_y'].values
    
    # Get unique filenames for coloring
    if 'filename' in poke_df.columns:
        unique_files = poke_df['filename'].unique()
        n_files = len(unique_files)
        colors = plt.cm.tab20(np.linspace(0, 1, min(n_files, 20)))
        file_to_color = {f: colors[i % len(colors)] for i, f in enumerate(unique_files)}
        poke_colors = [file_to_color.get(f, 'black') for f in poke_df['filename']]
    else:
        poke_colors = 'red'
    
    # Plot poke locations
    scatter = ax.scatter(poke_x, poke_y, c=poke_colors, s=100, alpha=0.7, 
                        edgecolors='black', linewidths=1.5, zorder=10)
    
    # Add labels if not too many
    if len(poke_df) <= 50 and 'filename' in poke_df.columns:
        for idx, row in poke_df.iterrows():
            label = Path(row['filename']).stem  # Just filename, no path
            if len(label) > 30:
                label = label[:27] + "..."
            ax.annotate(label, (row['poke_x'], row['poke_y']), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
    
    # Optionally show embryo outlines from tracks
    if show_embryos and tracks_df is not None:
        if 'embryo_id' in tracks_df.columns and 'x' in tracks_df.columns:
            # Plot all spark locations colored by embryo
            for embryo_id in tracks_df['embryo_id'].dropna().unique():
                emb_data = tracks_df[tracks_df['embryo_id'] == embryo_id]
                if len(emb_data) > 0:
                    # Sample points to avoid overcrowding
                    sample_size = min(1000, len(emb_data))
                    sample = emb_data.sample(n=sample_size) if len(emb_data) > sample_size else emb_data
                    
                    ax.scatter(sample['x'], sample['y'], alpha=0.1, s=1,
                             c='lightblue', label=f'Embryo {embryo_id} sparks' if embryo_id else 'Sparks')
    
    ax.set_xlabel('X Position (pixels)', fontsize=12)
    ax.set_ylabel('Y Position (pixels)', fontsize=12)
    ax.set_title('Poke Locations Across All Processing Runs', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Add statistics text box
    stats_text = f"Total pokes: {len(poke_df)}\n"
    if 'n_sparks' in poke_df.columns:
        stats_text += f"Avg sparks per poke: {poke_df['n_sparks'].mean():.1f}\n"
        stats_text += f"Median sparks: {poke_df['n_sparks'].median():.0f}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add legend if multiple files
    if len(poke_df) > 1 and 'filename' in poke_df.columns:
        n_legend_items = min(10, len(unique_files))
        if n_legend_items < len(unique_files):
            legend_text = f"{n_legend_items} of {len(unique_files)} files shown"
            ax.text(0.98, 0.02, legend_text, transform=ax.transAxes,
                   fontsize=8, horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved poke locations plot to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize all poke locations from processing runs"
    )
    parser.add_argument(
        'tracks_csv',
        help='Path to spark_tracks.csv file'
    )
    parser.add_argument(
        '--poke-csv',
        help='Optional path to poke_locations.csv (if available)',
        default=None
    )
    parser.add_argument(
        '--output',
        help='Output plot path (default: poke_locations.png)',
        default='poke_locations.png'
    )
    parser.add_argument(
        '--time-window',
        type=float,
        default=2.0,
        help='Time window in seconds for inferring poke from sparks (default: 2.0)'
    )
    parser.add_argument(
        '--min-sparks',
        type=int,
        default=1,
        help='Minimum number of sparks to use from first cluster (default: 1)'
    )
    parser.add_argument(
        '--no-embryos',
        action='store_true',
        help='Do not show embryo spark overlays'
    )
    
    args = parser.parse_args()
    
    # Read tracks data
    print(f"Reading spark tracks from {args.tracks_csv}...")
    try:
        tracks_df = pd.read_csv(args.tracks_csv)
        print(f"  → Loaded {len(tracks_df):,} track states")
    except Exception as e:
        print(f"Error reading {args.tracks_csv}: {e}")
        sys.exit(1)
    
    # Get poke locations
    if args.poke_csv and Path(args.poke_csv).exists():
        print(f"\nReading poke locations from {args.poke_csv}...")
        try:
            poke_df = read_poke_locations_csv(args.poke_csv)
            print(f"  → Loaded {len(poke_df)} poke location(s)")
        except Exception as e:
            print(f"Error reading {args.poke_csv}: {e}")
            print("  → Falling back to inference from spark tracks...")
            poke_df = None
    else:
        poke_df = None
    
    if poke_df is None:
        print(f"\nInferring poke locations from first spark cluster in each file...")
        print(f"  → Using first cluster(s) in each file (min: {args.min_sparks})")
        poke_df = infer_poke_from_early_sparks(
            tracks_df, 
            time_window_s=args.time_window,
            min_sparks=args.min_sparks
        )
        print(f"  → Inferred {len(poke_df)} poke location(s)")
    
    if len(poke_df) == 0:
        print("\n❌ No poke locations found. Try:")
        print("  - Adjusting --time-window (increase if sparks appear later)")
        print("  - Adjusting --min-sparks (decrease if fewer sparks)")
        print("  - Providing --poke-csv with manually specified locations")
        sys.exit(1)
    
    # Plot
    print(f"\nGenerating plot...")
    plot_poke_locations(
        poke_df,
        output_path=args.output,
        show_embryos=not args.no_embryos,
        tracks_df=tracks_df if not args.no_embryos else None
    )
    
    print(f"\n✓ Complete! Plotted {len(poke_df)} poke location(s)")


if __name__ == "__main__":
    main()

