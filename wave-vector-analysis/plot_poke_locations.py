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

def infer_poke_from_early_sparks(tracks_df, time_window_s=2.0, min_sparks=3):
    """
    Infer poke location from the earliest spark clusters after time_s = 0.
    
    Args:
        tracks_df: DataFrame with spark_tracks.csv data
        time_window_s: Maximum time after poke to consider (default 2 seconds)
        min_sparks: Minimum number of sparks needed to infer location
    
    Returns:
        DataFrame with columns: filename, poke_x, poke_y, n_sparks, method
    """
    # Group by filename (each file/processing run may have different poke)
    poke_locations = []
    
    # Get unique files (processing runs)
    unique_files = tracks_df['filename'].unique() if 'filename' in tracks_df.columns else ['unknown']
    
    for filename in unique_files:
        file_data = tracks_df[tracks_df['filename'] == filename] if 'filename' in tracks_df.columns else tracks_df
        
        # Find early post-poke sparks (time_s > 0 and small)
        early_sparks = file_data[
            (file_data['time_s'] > 0) & 
            (file_data['time_s'] <= time_window_s)
        ].copy()
        
        if len(early_sparks) < min_sparks:
            # Try to find sparks at exactly time_s = 0
            zero_sparks = file_data[file_data['time_s'] == 0].copy()
            if len(zero_sparks) >= min_sparks:
                early_sparks = zero_sparks
            else:
                continue
        
        # Calculate weighted centroid (weight by area if available)
        if 'area' in early_sparks.columns:
            weights = early_sparks['area'].fillna(1)
        else:
            weights = pd.Series(1, index=early_sparks.index)
        
        total_weight = weights.sum()
        if total_weight == 0:
            weights = pd.Series(1, index=early_sparks.index)
            total_weight = len(early_sparks)
        
        poke_x = (early_sparks['x'] * weights).sum() / total_weight
        poke_y = (early_sparks['y'] * weights).sum() / total_weight
        n_sparks = len(early_sparks)
        
        # Get the time range
        min_time = early_sparks['time_s'].min()
        max_time = early_sparks['time_s'].max()
        
        poke_locations.append({
            'filename': filename,
            'poke_x': poke_x,
            'poke_y': poke_y,
            'n_sparks': n_sparks,
            'time_range_s': f"{min_time:.2f}-{max_time:.2f}",
            'method': 'inferred_from_sparks'
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
        default=3,
        help='Minimum number of sparks needed to infer poke location (default: 3)'
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
        print(f"\nInferring poke locations from early spark clusters...")
        print(f"  → Time window: {args.time_window_s} seconds after poke")
        print(f"  → Minimum sparks: {args.min_sparks}")
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

