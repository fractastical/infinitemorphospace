#!/usr/bin/env python3
"""
Pixel-based wave analysis: treat each spark as a single activated pixel.

This script refactors the wave analysis to:
1. Count total activated pixels (each spark = 1 pixel) instead of using area
2. Measure wave propagation speed between embryos
3. Track region-specific propagation (e.g., Heart activity spreading from embryo A to B)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import argparse
from pathlib import Path
from collections import defaultdict
import sys


def count_activated_pixels(df_tracks, time_bin_s=0.1):
    """
    Count activated pixels per time bin, treating each spark as 1 pixel.
    
    Args:
        df_tracks: DataFrame with spark_tracks.csv data
        time_bin_s: Time bin size in seconds for aggregation
    
    Returns:
        DataFrame with columns: time_s, n_pixels, n_sparks, embryo_id (optional)
    """
    df = df_tracks.copy()
    
    # Each row represents one spark detection at one time point
    # Count unique sparks per time bin (each spark = 1 pixel)
    # Note: We count unique track_ids per time bin, where each track_id represents one activated pixel
    df['time_bin'] = (df['time_s'] / time_bin_s).round() * time_bin_s
    
    # Group by time bin and count unique sparks (pixels)
    if 'embryo_id' in df.columns:
        pixel_counts = df.groupby(['time_bin', 'embryo_id']).agg({
            'track_id': 'nunique',  # Count unique sparks = unique pixels
            'x': 'count'  # Total detections (for reference)
        }).reset_index()
        pixel_counts.columns = ['time_s', 'embryo_id', 'n_pixels', 'n_detections']
    else:
        pixel_counts = df.groupby('time_bin').agg({
            'track_id': 'nunique',  # Count unique sparks = unique pixels
            'x': 'count'  # Total detections (for reference)
        }).reset_index()
        pixel_counts.columns = ['time_s', 'n_pixels', 'n_detections']
    
    return pixel_counts


def measure_wave_propagation(df_tracks, source_embryo='A', target_embryo='B', 
                            region_filter=None, min_activation_threshold=3):
    """
    Measure how rapidly a wave spreads from one embryo to another.
    
    Specifically: if source_embryo has activity in a region, how long until
    target_embryo gets activity in the same region?
    
    Args:
        df_tracks: DataFrame with spark_tracks.csv data
        source_embryo: Embryo ID where wave originates ('A' or 'B')
        target_embryo: Embryo ID where wave propagates to ('A' or 'B')
        region_filter: Optional region name to filter (e.g., 'Heart')
        min_activation_threshold: Minimum number of pixels to consider as "activation"
    
    Returns:
        Dictionary with propagation metrics
    """
    df = df_tracks.copy()
    
    # Filter to post-poke only
    df = df[df['time_s'] >= 0].copy()
    
    # Filter by region if specified
    if region_filter and 'region' in df.columns:
        df = df[df['region'] == region_filter].copy()
    
    # Separate by embryo
    df_source = df[df['embryo_id'] == source_embryo].copy()
    df_target = df[df['embryo_id'] == target_embryo].copy()
    
    if len(df_source) == 0:
        return {'error': f'No data for source embryo {source_embryo}'}
    if len(df_target) == 0:
        return {'error': f'No data for target embryo {target_embryo}'}
    
    # Count activated pixels per time point for each embryo
    source_counts = df_source.groupby('time_s')['track_id'].nunique()
    target_counts = df_target.groupby('time_s')['track_id'].nunique()
    
    # Find first time when source has significant activation
    source_activation_times = source_counts[source_counts >= min_activation_threshold].index
    if len(source_activation_times) == 0:
        return {'error': f'Source embryo {source_embryo} never reaches activation threshold'}
    
    first_source_time = source_activation_times.min()
    
    # Find first time when target has significant activation after source
    target_activation_times = target_counts[target_counts >= min_activation_threshold].index
    target_after_source = target_activation_times[target_activation_times >= first_source_time]
    
    if len(target_after_source) == 0:
        return {
            'source_activation_time': first_source_time,
            'target_activation_time': None,
            'propagation_delay_s': None,
            'propagation_detected': False
        }
    
    first_target_time = target_after_source.min()
    propagation_delay = first_target_time - first_source_time
    
    # Calculate peak activation times
    source_peak_time = source_counts.idxmax()
    target_peak_time = target_counts.idxmax()
    
    # Calculate activation rates (pixels per second)
    source_activation_rate = source_counts.max() / (source_peak_time - first_source_time + 1e-6)
    target_activation_rate = target_counts.max() / (target_peak_time - first_target_time + 1e-6)
    
    return {
        'source_embryo': source_embryo,
        'target_embryo': target_embryo,
        'region': region_filter or 'all',
        'source_activation_time': first_source_time,
        'target_activation_time': first_target_time,
        'propagation_delay_s': propagation_delay,
        'propagation_detected': True,
        'source_peak_time': source_peak_time,
        'target_peak_time': target_peak_time,
        'source_peak_pixels': source_counts.max(),
        'target_peak_pixels': target_counts.max(),
        'source_activation_rate': source_activation_rate,
        'target_activation_rate': target_activation_rate,
    }


def analyze_region_propagation(df_tracks, regions_of_interest=None):
    """
    Analyze wave propagation for multiple regions between embryos.
    
    Args:
        df_tracks: DataFrame with spark_tracks.csv data
        regions_of_interest: List of region names to analyze (default: ['Heart'])
    
    Returns:
        DataFrame with propagation metrics for each region
    """
    if regions_of_interest is None:
        regions_of_interest = ['Heart']
    
    results = []
    
    for region in regions_of_interest:
        # A -> B propagation
        prop_ab = measure_wave_propagation(df_tracks, 'A', 'B', region_filter=region)
        if 'error' not in prop_ab:
            prop_ab['direction'] = 'A->B'
            results.append(prop_ab)
        
        # B -> A propagation
        prop_ba = measure_wave_propagation(df_tracks, 'B', 'A', region_filter=region)
        if 'error' not in prop_ba:
            prop_ba['direction'] = 'B->A'
            results.append(prop_ba)
    
    if len(results) == 0:
        return pd.DataFrame()
    
    return pd.DataFrame(results)


def plot_pixel_activation_timecourse(df_tracks, output_path=None, time_bin_s=0.1):
    """
    Plot activated pixel counts over time for each embryo.
    
    Args:
        df_tracks: DataFrame with spark_tracks.csv data
        output_path: Optional path to save plot
        time_bin_s: Time bin size in seconds
    """
    pixel_counts = count_activated_pixels(df_tracks, time_bin_s=time_bin_s)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Total pixels over time
    if 'embryo_id' in pixel_counts.columns:
        for embryo_id in pixel_counts['embryo_id'].unique():
            df_emb = pixel_counts[pixel_counts['embryo_id'] == embryo_id]
            axes[0].plot(df_emb['time_s'], df_emb['n_pixels'], 
                        label=f'Embryo {embryo_id}', linewidth=2, alpha=0.7)
    else:
        axes[0].plot(pixel_counts['time_s'], pixel_counts['n_pixels'], 
                    linewidth=2, label='All embryos')
    
    axes[0].axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Poke time')
    axes[0].set_xlabel('Time (seconds, relative to poke)')
    axes[0].set_ylabel('Activated Pixels')
    axes[0].set_title('Activated Pixel Count Over Time\n(Each spark = 1 pixel)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Cumulative pixels
    if 'embryo_id' in pixel_counts.columns:
        for embryo_id in pixel_counts['embryo_id'].unique():
            df_emb = pixel_counts[pixel_counts['embryo_id'] == embryo_id].sort_values('time_s')
            df_emb['cumulative'] = df_emb['n_pixels'].cumsum()
            axes[1].plot(df_emb['time_s'], df_emb['cumulative'], 
                        label=f'Embryo {embryo_id}', linewidth=2, alpha=0.7)
    else:
        df_sorted = pixel_counts.sort_values('time_s')
        df_sorted['cumulative'] = df_sorted['n_pixels'].cumsum()
        axes[1].plot(df_sorted['time_s'], df_sorted['cumulative'], 
                    linewidth=2, label='All embryos')
    
    axes[1].axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Poke time')
    axes[1].set_xlabel('Time (seconds, relative to poke)')
    axes[1].set_ylabel('Cumulative Activated Pixels')
    axes[1].set_title('Cumulative Activated Pixel Count')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved pixel activation plot to {output_path}")
    else:
        plt.show()


def plot_propagation_analysis(df_tracks, output_path=None):
    """
    Plot wave propagation analysis between embryos.
    
    Args:
        df_tracks: DataFrame with spark_tracks.csv data
        output_path: Optional path to save plot
    """
    # Analyze propagation for key regions
    regions = ['Heart', 'Brain', 'Tail']
    prop_df = analyze_region_propagation(df_tracks, regions_of_interest=regions)
    
    if len(prop_df) == 0:
        print("No propagation data found")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Propagation delays by region
    if len(prop_df) > 0:
        delays = prop_df[prop_df['propagation_detected'] == True]['propagation_delay_s'].dropna()
        if len(delays) > 0:
            axes[0, 0].bar(range(len(delays)), delays.values)
            axes[0, 0].set_xticks(range(len(delays)))
            axes[0, 0].set_xticklabels([f"{r['region']} {r['direction']}" 
                                       for _, r in prop_df[prop_df['propagation_detected'] == True].iterrows()],
                                      rotation=45, ha='right')
            axes[0, 0].set_ylabel('Propagation Delay (seconds)')
            axes[0, 0].set_title('Wave Propagation Delay Between Embryos')
            axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Activation times comparison
    if len(prop_df) > 0:
        valid_prop = prop_df[prop_df['propagation_detected'] == True]
        if len(valid_prop) > 0:
            x_pos = np.arange(len(valid_prop))
            width = 0.35
            axes[0, 1].bar(x_pos - width/2, valid_prop['source_activation_time'], 
                          width, label='Source', alpha=0.7)
            axes[0, 1].bar(x_pos + width/2, valid_prop['target_activation_time'], 
                          width, label='Target', alpha=0.7)
            axes[0, 1].set_xticks(x_pos)
            axes[0, 1].set_xticklabels([f"{r['region']} {r['direction']}" 
                                       for _, r in valid_prop.iterrows()],
                                      rotation=45, ha='right')
            axes[0, 1].set_ylabel('Activation Time (seconds)')
            axes[0, 1].set_title('First Activation Time by Region')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Activation rates
    if len(prop_df) > 0:
        valid_prop = prop_df[prop_df['propagation_detected'] == True]
        if len(valid_prop) > 0:
            x_pos = np.arange(len(valid_prop))
            width = 0.35
            axes[1, 0].bar(x_pos - width/2, valid_prop['source_activation_rate'], 
                          width, label='Source', alpha=0.7)
            axes[1, 0].bar(x_pos + width/2, valid_prop['target_activation_rate'], 
                          width, label='Target', alpha=0.7)
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels([f"{r['region']} {r['direction']}" 
                                       for _, r in valid_prop.iterrows()],
                                      rotation=45, ha='right')
            axes[1, 0].set_ylabel('Activation Rate (pixels/second)')
            axes[1, 0].set_title('Activation Rate by Region')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Time series for Heart region (if available)
    if 'region' in df_tracks.columns:
        df_heart = df_tracks[df_tracks['region'] == 'Heart'].copy()
        if len(df_heart) > 0:
            for embryo_id in ['A', 'B']:
                df_emb = df_heart[df_heart['embryo_id'] == embryo_id]
                if len(df_emb) > 0:
                    pixel_counts = df_emb.groupby('time_s')['track_id'].nunique()
                    axes[1, 1].plot(pixel_counts.index, pixel_counts.values, 
                                   label=f'Embryo {embryo_id}', linewidth=2, alpha=0.7)
            axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Poke time')
            axes[1, 1].set_xlabel('Time (seconds)')
            axes[1, 1].set_ylabel('Activated Pixels')
            axes[1, 1].set_title('Heart Region Activity Over Time')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved propagation analysis plot to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze wave propagation using pixel-based spark counting'
    )
    parser.add_argument('tracks_csv', help='Path to spark_tracks.csv')
    parser.add_argument('--output-dir', help='Directory to save plots and results')
    parser.add_argument('--region', help='Specific region to analyze (e.g., Heart)')
    parser.add_argument('--time-bin', type=float, default=0.1,
                       help='Time bin size in seconds (default: 0.1)')
    
    args = parser.parse_args()
    
    print(f"Loading {args.tracks_csv}...")
    df_tracks = pd.read_csv(args.tracks_csv)
    print(f"Loaded {len(df_tracks)} track states")
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Count activated pixels
    print("\nCounting activated pixels...")
    pixel_counts = count_activated_pixels(df_tracks, time_bin_s=args.time_bin)
    print(f"Found {pixel_counts['n_pixels'].sum()} total activated pixels")
    
    if output_dir:
        pixel_counts.to_csv(output_dir / 'pixel_counts.csv', index=False)
        print(f"Saved pixel counts to {output_dir / 'pixel_counts.csv'}")
    
    # Analyze wave propagation
    print("\nAnalyzing wave propagation...")
    if args.region:
        regions = [args.region]
    else:
        regions = ['Heart', 'Brain', 'Tail']
    
    prop_df = analyze_region_propagation(df_tracks, regions_of_interest=regions)
    
    if len(prop_df) > 0:
        print("\nPropagation Results:")
        print(prop_df.to_string(index=False))
        
        if output_dir:
            prop_df.to_csv(output_dir / 'wave_propagation.csv', index=False)
            print(f"\nSaved propagation results to {output_dir / 'wave_propagation.csv'}")
    else:
        print("No propagation data found")
    
    # Generate plots
    if output_dir:
        plot_pixel_activation_timecourse(
            df_tracks, 
            output_path=output_dir / 'pixel_activation_timecourse.png',
            time_bin_s=args.time_bin
        )
        plot_propagation_analysis(
            df_tracks,
            output_path=output_dir / 'wave_propagation_analysis.png'
        )
        print(f"\nPlots saved to {output_dir}")
    else:
        plot_pixel_activation_timecourse(df_tracks, time_bin_s=args.time_bin)
        plot_propagation_analysis(df_tracks)


if __name__ == '__main__':
    main()
