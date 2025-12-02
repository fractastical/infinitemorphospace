#!/usr/bin/env python3
"""
Generate comprehensive plots for all testable experimental hypotheses.

This script creates publication-ready plots for each hypothesis that can be tested
with the current data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow
import sys
from pathlib import Path

# Import plotting functions from analyze_experimental_hypotheses
sys.path.insert(0, str(Path(__file__).parent))
from analyze_experimental_hypotheses import (
    plot_activity_comparison,
    plot_wave_directionality,
    plot_tail_response,
    analyze_calcium_activity,
    analyze_wave_directionality,
    analyze_tail_response
)
from visualize_spark_tracks import (
    plot_trajectories,
    plot_time_series,
    plot_spatial_heatmap,
    plot_embryo_comparison
)


def plot_activity_with_std(tracks_df, output_path=None):
    """
    Plot activity with standard deviation - more detailed for Hypothesis 1.
    Shows variability in response.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 1. Activity over time with error bars (if multiple embryos)
    if 'embryo_id' in tracks_df.columns:
        for embryo_id in tracks_df['embryo_id'].dropna().unique():
            df_emb = tracks_df[tracks_df['embryo_id'] == embryo_id]
            activity_per_time = df_emb.groupby('time_s')['area'].sum()
            axes[0].plot(activity_per_time.index, activity_per_time.values, 
                        label=f'Embryo {embryo_id}', linewidth=2, alpha=0.8)
    else:
        activity_per_time = tracks_df.groupby('time_s')['area'].sum()
        axes[0].plot(activity_per_time.index, activity_per_time.values, 
                    label='All embryos', linewidth=2, alpha=0.8)
    
    axes[0].axvline(x=0, color='r', linestyle='--', alpha=0.7, linewidth=2, label='Poke time')
    axes[0].set_xlabel('Time (seconds, relative to poke)', fontsize=12)
    axes[0].set_ylabel('Total Activity (pixels²)', fontsize=12)
    axes[0].set_title('Calcium Activity Over Time\n(ΔF/F₀ proxy: Total Area)', 
                     fontweight='bold', fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Standard deviation of activity over time
    if 'embryo_id' in tracks_df.columns:
        for embryo_id in tracks_df['embryo_id'].dropna().unique():
            df_emb = tracks_df[tracks_df['embryo_id'] == embryo_id]
            std_per_time = df_emb.groupby('time_s')['area'].std()
            axes[1].plot(std_per_time.index, std_per_time.values, 
                        label=f'Embryo {embryo_id}', linewidth=2, alpha=0.8)
    else:
        std_per_time = tracks_df.groupby('time_s')['area'].std()
        axes[1].plot(std_per_time.index, std_per_time.values, 
                    label='All embryos', linewidth=2, alpha=0.8)
    
    axes[1].axvline(x=0, color='r', linestyle='--', alpha=0.7, linewidth=2, label='Poke time')
    axes[1].set_xlabel('Time (seconds, relative to poke)', fontsize=12)
    axes[1].set_ylabel('Standard Deviation (pixels²)', fontsize=12)
    axes[1].set_title('Variability in Calcium Activity Over Time\n(Standard Deviation)', 
                     fontweight='bold', fontsize=13)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Activity ratio (post/pre) by embryo
    pre_poke = tracks_df[tracks_df['time_s'] < 0]
    post_poke = tracks_df[tracks_df['time_s'] > 0]
    
    if 'embryo_id' in tracks_df.columns:
        embryos = sorted(tracks_df['embryo_id'].dropna().unique())
        ratios = []
        labels = []
        for eid in embryos:
            pre_total = pre_poke[pre_poke['embryo_id'] == eid]['area'].sum()
            post_total = post_poke[post_poke['embryo_id'] == eid]['area'].sum()
            if pre_total > 0:
                ratio = post_total / pre_total
                ratios.append(ratio)
                labels.append(f'Embryo {eid}')
        
        if ratios:
            bars = axes[2].bar(labels, ratios, alpha=0.7, color=['steelblue', 'coral'][:len(ratios)])
            axes[2].set_ylabel('Activity Ratio (Post / Pre)', fontsize=12)
            axes[2].set_title('Activity Increase Ratio by Embryo\n(Hypothesis 1: Presence of Activity)', 
                            fontweight='bold', fontsize=13)
            axes[2].grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, ratio in zip(bars, ratios):
                height = bar.get_height()
                axes[2].text(bar.get_x() + bar.get_width()/2., height,
                           f'{ratio:.0f}x', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved activity with std plot to {output_path}")
    else:
        plt.show()


def plot_inter_embryo_comparison(tracks_df, clusters_df, output_path=None):
    """
    Compare wave propagation between embryos A and B.
    For Hypothesis 4 - inter-embryo wave directionality.
    """
    if 'embryo_id' not in tracks_df.columns:
        print("No embryo_id column found")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Filter by embryo
    df_A = tracks_df[tracks_df['embryo_id'] == 'A']
    df_B = tracks_df[tracks_df['embryo_id'] == 'B']
    
    # Get clusters for each embryo
    if clusters_df is not None:
        track_ids_A = df_A['track_id'].unique()
        track_ids_B = df_B['track_id'].unique()
        clusters_A = clusters_df[clusters_df['cluster_id'].isin(track_ids_A)]
        clusters_B = clusters_df[clusters_df['cluster_id'].isin(track_ids_B)]
    else:
        clusters_A = None
        clusters_B = None
    
    # 1. Activity over time comparison
    activity_A = df_A.groupby('time_s')['area'].sum()
    activity_B = df_B.groupby('time_s')['area'].sum()
    
    axes[0, 0].plot(activity_A.index, activity_A.values, label='Embryo A', 
                   linewidth=2, alpha=0.8, color='blue')
    axes[0, 0].plot(activity_B.index, activity_B.values, label='Embryo B', 
                   linewidth=2, alpha=0.8, color='orange')
    axes[0, 0].axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Poke time')
    axes[0, 0].set_xlabel('Time (seconds)', fontsize=11)
    axes[0, 0].set_ylabel('Total Activity (pixels²)', fontsize=11)
    axes[0, 0].set_title('Activity Over Time: Embryo A vs B', fontweight='bold', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Number of active tracks over time
    tracks_A = df_A.groupby('time_s')['track_id'].nunique()
    tracks_B = df_B.groupby('time_s')['track_id'].nunique()
    
    axes[0, 1].plot(tracks_A.index, tracks_A.values, label='Embryo A', 
                   linewidth=2, alpha=0.8, color='blue')
    axes[0, 1].plot(tracks_B.index, tracks_B.values, label='Embryo B', 
                   linewidth=2, alpha=0.8, color='orange')
    axes[0, 1].axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Poke time')
    axes[0, 1].set_xlabel('Time (seconds)', fontsize=11)
    axes[0, 1].set_ylabel('Number of Active Tracks', fontsize=11)
    axes[0, 1].set_title('Number of Active Tracks Over Time', fontweight='bold', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Speed distribution comparison
    if clusters_A is not None and clusters_B is not None:
        speeds_A = clusters_A['mean_speed_px_per_s'].dropna()
        speeds_B = clusters_B['mean_speed_px_per_s'].dropna()
        
        axes[0, 2].hist(speeds_A.values, bins=50, alpha=0.6, label='Embryo A', color='blue', edgecolor='black')
        axes[0, 2].hist(speeds_B.values, bins=50, alpha=0.6, label='Embryo B', color='orange', edgecolor='black')
        axes[0, 2].set_xlabel('Mean Speed (pixels/second)', fontsize=11)
        axes[0, 2].set_ylabel('Count', fontsize=11)
        axes[0, 2].set_title('Wave Speed Distribution Comparison', fontweight='bold', fontsize=12)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4. AP position distribution
    if 'ap_norm' in tracks_df.columns:
        ap_A = df_A['ap_norm'].dropna()
        ap_B = df_B['ap_norm'].dropna()
        
        if len(ap_A) > 0 and len(ap_B) > 0:
            # Use consistent bins for both histograms
            all_ap = pd.concat([ap_A, ap_B])
            ap_min, ap_max = all_ap.min(), all_ap.max()
            bins = np.linspace(ap_min, ap_max, 31) if ap_max > ap_min else 30
            
            axes[1, 0].hist(ap_A.values, bins=bins, alpha=0.6, label='Embryo A', color='blue', edgecolor='black')
            axes[1, 0].hist(ap_B.values, bins=bins, alpha=0.6, label='Embryo B', color='orange', edgecolor='black')
            axes[1, 0].set_xlabel('AP Position (0=head, 1=tail)', fontsize=11)
            axes[1, 0].set_ylabel('Count', fontsize=11)
            axes[1, 0].set_title('AP Position Distribution', fontweight='bold', fontsize=12)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].axvline(0.7, color='r', linestyle='--', alpha=0.5, label='Tail threshold')
    
    # 5. Cumulative activity by embryo
    cumulative_A = activity_A.cumsum()
    cumulative_B = activity_B.cumsum()
    
    axes[1, 1].plot(cumulative_A.index, cumulative_A.values, label='Embryo A', 
                   linewidth=2, alpha=0.8, color='blue')
    axes[1, 1].plot(cumulative_B.index, cumulative_B.values, label='Embryo B', 
                   linewidth=2, alpha=0.8, color='orange')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Poke time')
    axes[1, 1].set_xlabel('Time (seconds)', fontsize=11)
    axes[1, 1].set_ylabel('Cumulative Activity (pixels²)', fontsize=11)
    axes[1, 1].set_title('Cumulative Activity Over Time', fontweight='bold', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Summary statistics
    axes[1, 2].axis('off')
    
    stats_text = "Summary Statistics:\n\n"
    stats_text += f"Embryo A:\n"
    stats_text += f"  Total events: {len(df_A):,}\n"
    stats_text += f"  Total clusters: {df_A['track_id'].nunique():,}\n"
    if clusters_A is not None:
        stats_text += f"  Mean speed: {clusters_A['mean_speed_px_per_s'].mean():.2f} px/s\n"
    stats_text += f"  Total activity: {df_A['area'].sum():,} px²\n\n"
    
    stats_text += f"Embryo B:\n"
    stats_text += f"  Total events: {len(df_B):,}\n"
    stats_text += f"  Total clusters: {df_B['track_id'].nunique():,}\n"
    if clusters_B is not None:
        stats_text += f"  Mean speed: {clusters_B['mean_speed_px_per_s'].mean():.2f} px/s\n"
    stats_text += f"  Total activity: {df_B['area'].sum():,} px²\n"
    
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                   verticalalignment='center', transform=axes[1, 2].transAxes)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved inter-embryo comparison to {output_path}")
    else:
        plt.show()


def plot_ap_position_analysis(tracks_df, output_path=None):
    """
    Analyze activity by AP position - relevant for multiple hypotheses.
    """
    if 'ap_norm' not in tracks_df.columns:
        print("No AP position data available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    df_with_ap = tracks_df[tracks_df['ap_norm'].notna()]
    
    # 1. Activity distribution by AP position
    ap_values = df_with_ap['ap_norm'].values
    ap_min, ap_max = ap_values.min(), ap_values.max()
    bins = np.linspace(ap_min, ap_max, 51) if ap_max > ap_min else 50
    axes[0, 0].hist(ap_values, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].axvline(0.7, color='r', linestyle='--', linewidth=2, label='Tail threshold (0.7)')
    axes[0, 0].set_xlabel('AP Position (0=head, 1=tail)', fontsize=11)
    axes[0, 0].set_ylabel('Count', fontsize=11)
    axes[0, 0].set_title('Event Distribution by AP Position', fontweight='bold', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Mean activity by AP position bin
    df_with_ap = df_with_ap.copy()
    ap_min, ap_max = df_with_ap['ap_norm'].min(), df_with_ap['ap_norm'].max()
    df_with_ap['ap_bin'] = pd.cut(df_with_ap['ap_norm'], bins=20, labels=False)
    activity_by_bin = df_with_ap.groupby('ap_bin')['area'].mean()
    
    bin_centers = np.linspace(ap_min, ap_max, len(activity_by_bin))
    axes[0, 1].plot(bin_centers, activity_by_bin.values, marker='o', linewidth=2, markersize=4)
    axes[0, 1].axvline(0.7, color='r', linestyle='--', alpha=0.5, label='Tail threshold')
    axes[0, 1].set_xlabel('AP Position (0=head, 1=tail)', fontsize=11)
    axes[0, 1].set_ylabel('Mean Activity (pixels²)', fontsize=11)
    axes[0, 1].set_title('Mean Activity by AP Position', fontweight='bold', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Activity over time for different AP regions
    head_events = df_with_ap[df_with_ap['ap_norm'] < 0.3]
    mid_events = df_with_ap[(df_with_ap['ap_norm'] >= 0.3) & (df_with_ap['ap_norm'] < 0.7)]
    tail_events = df_with_ap[df_with_ap['ap_norm'] >= 0.7]
    
    for label, df_region in [('Head (AP < 0.3)', head_events), 
                             ('Mid (0.3 ≤ AP < 0.7)', mid_events),
                             ('Tail (AP ≥ 0.7)', tail_events)]:
        if len(df_region) > 0:
            activity = df_region.groupby('time_s')['area'].sum()
            axes[1, 0].plot(activity.index, activity.values, label=label, linewidth=2, alpha=0.8)
    
    axes[1, 0].axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Poke time')
    axes[1, 0].set_xlabel('Time (seconds)', fontsize=11)
    axes[1, 0].set_ylabel('Total Activity (pixels²)', fontsize=11)
    axes[1, 0].set_title('Activity Over Time by AP Region', fontweight='bold', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. AP position over time (show spatial progression)
    time_bins = pd.cut(df_with_ap['time_s'], bins=50)
    mean_ap_by_time = df_with_ap.groupby(time_bins)['ap_norm'].mean()
    time_centers = [interval.mid for interval in mean_ap_by_time.index]
    
    axes[1, 1].plot(time_centers, mean_ap_by_time.values, marker='o', linewidth=2, markersize=3, alpha=0.7)
    axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Poke time')
    axes[1, 1].axhline(0.7, color='orange', linestyle='--', alpha=0.5, label='Tail threshold')
    axes[1, 1].set_xlabel('Time (seconds)', fontsize=11)
    axes[1, 1].set_ylabel('Mean AP Position', fontsize=11)
    axes[1, 1].set_title('Mean AP Position of Events Over Time\n(Spatial Progression)', 
                        fontweight='bold', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved AP position analysis to {output_path}")
    else:
        plt.show()


def plot_speed_vs_time(tracks_df, clusters_df, output_path=None):
    """
    Plot speed over time - relevant for wave propagation analysis.
    """
    if clusters_df is None:
        print("Clusters data required")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Merge clusters with time info
    track_times = tracks_df.groupby('track_id')['time_s'].agg(['min', 'max', 'mean']).reset_index()
    clusters_with_time = clusters_df.merge(track_times, left_on='cluster_id', right_on='track_id', how='left')
    
    # 1. Speed distribution by time period
    pre_poke = clusters_with_time[clusters_with_time['mean'] < 0]
    post_poke = clusters_with_time[clusters_with_time['mean'] > 0]
    
    speeds_pre = pre_poke['mean_speed_px_per_s'].dropna()
    speeds_post = post_poke['mean_speed_px_per_s'].dropna()
    
    # Use consistent bins
    all_speeds = pd.concat([speeds_pre, speeds_post])
    if len(all_speeds) > 0:
        speed_min, speed_max = all_speeds.min(), all_speeds.max()
        bins = np.linspace(speed_min, speed_max, 51) if speed_max > speed_min else 50
        
        if len(speeds_pre) > 0:
            axes[0, 0].hist(speeds_pre.values, bins=bins, alpha=0.6, label='Pre-poke', 
                           color='blue', edgecolor='black')
        if len(speeds_post) > 0:
            axes[0, 0].hist(speeds_post.values, bins=bins, alpha=0.6, label='Post-poke', 
                           color='orange', edgecolor='black')
    axes[0, 0].set_xlabel('Mean Speed (pixels/second)', fontsize=11)
    axes[0, 0].set_ylabel('Count', fontsize=11)
    axes[0, 0].set_title('Speed Distribution: Pre vs Post Poke', fontweight='bold', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Mean speed over time
    clusters_with_time = clusters_with_time[clusters_with_time['mean_speed_px_per_s'].notna()]
    if len(clusters_with_time) > 0:
        time_bins = pd.cut(clusters_with_time['mean'], bins=100)
        mean_speed_by_time = clusters_with_time.groupby(time_bins)['mean_speed_px_per_s'].mean()
        time_centers = [interval.mid for interval in mean_speed_by_time.index]
        
        axes[0, 1].plot(time_centers, mean_speed_by_time.values, linewidth=2, alpha=0.8)
        axes[0, 1].axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Poke time')
        axes[0, 1].set_xlabel('Time (seconds)', fontsize=11)
        axes[0, 1].set_ylabel('Mean Speed (pixels/second)', fontsize=11)
        axes[0, 1].set_title('Mean Wave Speed Over Time', fontweight='bold', fontsize=12)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Peak speed over time
    if len(clusters_with_time) > 0:
        peak_speed_by_time = clusters_with_time.groupby(time_bins)['peak_speed_px_per_s'].max()
        axes[1, 0].plot(time_centers, peak_speed_by_time.values, linewidth=2, alpha=0.8, color='purple')
        axes[1, 0].axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Poke time')
        axes[1, 0].set_xlabel('Time (seconds)', fontsize=11)
        axes[1, 0].set_ylabel('Peak Speed (pixels/second)', fontsize=11)
        axes[1, 0].set_title('Peak Wave Speed Over Time', fontweight='bold', fontsize=12)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Speed by embryo
    if 'embryo_id' in tracks_df.columns:
        all_emb_speeds = []
        for embryo_id in ['A', 'B']:
            df_emb = tracks_df[tracks_df['embryo_id'] == embryo_id]
            track_ids = df_emb['track_id'].unique()
            clusters_emb = clusters_df[clusters_df['cluster_id'].isin(track_ids)]
            speeds = clusters_emb['mean_speed_px_per_s'].dropna()
            if len(speeds) > 0:
                all_emb_speeds.append(speeds.values)
        
        # Use consistent bins across embryos
        if all_emb_speeds:
            all_speeds_combined = np.concatenate(all_emb_speeds)
            speed_min, speed_max = all_speeds_combined.min(), all_speeds_combined.max()
            bins = np.linspace(speed_min, speed_max, 51) if speed_max > speed_min else 50
            
            colors = ['blue', 'orange']
            for idx, embryo_id in enumerate(['A', 'B']):
                df_emb = tracks_df[tracks_df['embryo_id'] == embryo_id]
                track_ids = df_emb['track_id'].unique()
                clusters_emb = clusters_df[clusters_df['cluster_id'].isin(track_ids)]
                speeds = clusters_emb['mean_speed_px_per_s'].dropna()
                if len(speeds) > 0:
                    axes[1, 1].hist(speeds.values, bins=bins, alpha=0.6, 
                                   label=f'Embryo {embryo_id}', edgecolor='black', color=colors[idx])
        
        axes[1, 1].set_xlabel('Mean Speed (pixels/second)', fontsize=11)
        axes[1, 1].set_ylabel('Count', fontsize=11)
        axes[1, 1].set_title('Speed Distribution by Embryo', fontweight='bold', fontsize=12)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved speed vs time plot to {output_path}")
    else:
        plt.show()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate all hypothesis plots')
    parser.add_argument('tracks_csv', help='Path to spark_tracks.csv')
    parser.add_argument('--clusters-csv', help='Path to vector_clusters.csv')
    parser.add_argument('--output-dir', default='analysis_results', 
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    print(f"Loading {args.tracks_csv}...")
    tracks_df = pd.read_csv(args.tracks_csv)
    print(f"Loaded {len(tracks_df)} track states")
    
    clusters_df = None
    if args.clusters_csv:
        print(f"Loading {args.clusters_csv}...")
        clusters_df = pd.read_csv(args.clusters_csv)
        print(f"Loaded {len(clusters_df)} clusters")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating plots...")
    
    # Generate all plots
    plots_generated = []
    
    # 1. Detailed activity plot with std
    print("  → Activity with standard deviation...")
    plot_activity_with_std(tracks_df, output_dir / 'hypothesis1_activity_detailed.png')
    plots_generated.append('hypothesis1_activity_detailed.png')
    
    # 2. Inter-embryo comparison
    if 'embryo_id' in tracks_df.columns and clusters_df is not None:
        print("  → Inter-embryo comparison...")
        plot_inter_embryo_comparison(tracks_df, clusters_df, 
                                    output_dir / 'hypothesis4_inter_embryo_comparison.png')
        plots_generated.append('hypothesis4_inter_embryo_comparison.png')
    
    # 3. AP position analysis
    if 'ap_norm' in tracks_df.columns:
        print("  → AP position analysis...")
        plot_ap_position_analysis(tracks_df, output_dir / 'hypothesis_ap_position_analysis.png')
        plots_generated.append('hypothesis_ap_position_analysis.png')
    
    # 4. Speed vs time
    if clusters_df is not None:
        print("  → Speed vs time analysis...")
        plot_speed_vs_time(tracks_df, clusters_df, output_dir / 'hypothesis_speed_vs_time.png')
        plots_generated.append('hypothesis_speed_vs_time.png')
    
    # 5. Time series (if not already generated)
    print("  → Time series plots...")
    plot_time_series(tracks_df, output_dir / 'hypothesis_time_series_detailed.png')
    plots_generated.append('hypothesis_time_series_detailed.png')
    
    # 6. Embryo comparison (if not already generated)
    if 'embryo_id' in tracks_df.columns:
        print("  → Embryo comparison plots...")
        plot_embryo_comparison(tracks_df, output_dir / 'hypothesis_embryo_comparison_detailed.png')
        plots_generated.append('hypothesis_embryo_comparison_detailed.png')
    
    print(f"\n✓ Generated {len(plots_generated)} plots:")
    for plot in plots_generated:
        print(f"  - {plot}")


if __name__ == '__main__':
    main()

