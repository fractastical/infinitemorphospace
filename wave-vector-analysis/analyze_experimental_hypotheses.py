#!/usr/bin/env python3
"""
Analysis script for testing experimental hypotheses about Ca²⁺ wave dynamics.

This script provides analysis functions to test the specific hypotheses outlined
in EXPERIMENTAL_HYPOTHESES.md.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import argparse
from pathlib import Path
import sys


def analyze_calcium_activity(tracks_df, clusters_df, embryo_id=None, time_window=None):
    """
    Analyze calcium activity levels over time.
    
    Tests: Hypothesis 1 (presence of calcium activity)
    
    Args:
        tracks_df: DataFrame with spark_tracks.csv data
        clusters_df: DataFrame with vector_clusters.csv data
        embryo_id: Filter to specific embryo ('A', 'B', or None for both)
        time_window: Tuple of (start_time, end_time) in seconds, or None for all
    
    Returns:
        Dictionary with activity metrics
    """
    df = tracks_df.copy()
    
    # Filter by embryo if specified
    if embryo_id and 'embryo_id' in df.columns:
        df = df[df['embryo_id'] == embryo_id]
    
    # Filter by time window if specified
    if time_window:
        start, end = time_window
        df = df[(df['time_s'] >= start) & (df['time_s'] <= end)]
    
    # Separate pre-poke and post-poke
    pre_poke = df[df['time_s'] < 0]
    post_poke = df[df['time_s'] > 0]
    
    # Calculate activity metrics
    # Activity proxy: sum of areas per time point
    activity_per_time = df.groupby('time_s')['area'].sum()
    
    results = {
        'pre_poke_mean_activity': pre_poke['area'].mean() if len(pre_poke) > 0 else 0,
        'post_poke_mean_activity': post_poke['area'].mean() if len(post_poke) > 0 else 0,
        'pre_poke_total_activity': pre_poke['area'].sum(),
        'post_poke_total_activity': post_poke['area'].sum(),
        'activity_ratio': (post_poke['area'].sum() / pre_poke['area'].sum() 
                          if len(pre_poke) > 0 and pre_poke['area'].sum() > 0 else np.nan),
        'activity_per_time': activity_per_time,
        'peak_activity_time': activity_per_time.idxmax() if len(activity_per_time) > 0 else np.nan,
        'peak_activity_value': activity_per_time.max() if len(activity_per_time) > 0 else np.nan,
        'pre_poke_std': pre_poke['area'].std() if len(pre_poke) > 0 else np.nan,
        'post_poke_std': post_poke['area'].std() if len(post_poke) > 0 else np.nan,
    }
    
    return results


def analyze_wave_directionality(tracks_df, clusters_df, embryo_id=None, poke_region=None):
    """
    Analyze wave directionality within or between embryos.
    
    Tests: Hypotheses 3, 4 (wave directionality)
    
    Args:
        tracks_df: DataFrame with spark_tracks.csv data
        clusters_df: DataFrame with vector_clusters.csv data
        embryo_id: Filter to specific embryo ('A', 'B', or None)
        poke_region: Filter by poke region ('anterior', 'mid', 'posterior', or None)
    
    Returns:
        Dictionary with directionality metrics
    """
    df_tracks = tracks_df.copy()
    df_clusters = clusters_df.copy()
    
    # Filter by embryo
    if embryo_id and 'embryo_id' in df_tracks.columns:
        df_tracks = df_tracks[df_tracks['embryo_id'] == embryo_id]
        # Get track_ids for this embryo
        track_ids = df_tracks['track_id'].unique()
        df_clusters = df_clusters[df_clusters['cluster_id'].isin(track_ids)]
    
    # Filter clusters with valid angle data
    valid_clusters = df_clusters[df_clusters['mean_angle_deg'].notna()]
    
    if len(valid_clusters) == 0:
        return {'error': 'No clusters with angle data found'}
    
    # Calculate directionality metrics
    angles = valid_clusters['mean_angle_deg'].values
    
    # Convert to radians for circular statistics
    angles_rad = np.deg2rad(angles)
    
    # Circular mean
    mean_sin = np.sin(angles_rad).mean()
    mean_cos = np.cos(angles_rad).mean()
    mean_angle_rad = np.arctan2(mean_sin, mean_cos)
    mean_angle_deg = np.rad2deg(mean_angle_rad)
    
    # Circular variance (1 - R, where R is mean resultant length)
    R = np.sqrt(mean_sin**2 + mean_cos**2)
    circular_variance = 1 - R
    
    # Speed statistics
    speeds = valid_clusters['mean_speed_px_per_s'].dropna()
    
    results = {
        'n_clusters': len(valid_clusters),
        'mean_angle_deg': mean_angle_deg,
        'circular_variance': circular_variance,
        'mean_speed': speeds.mean() if len(speeds) > 0 else np.nan,
        'peak_speed': valid_clusters['peak_speed_px_per_s'].max() if 'peak_speed_px_per_s' in valid_clusters.columns else np.nan,
        'angle_distribution': angles,
        'speed_distribution': speeds.values if len(speeds) > 0 else [],
    }
    
    return results


def analyze_spatial_matching(tracks_df, poke_x, poke_y):
    """
    Analyze spatial matching between wound location and response location.
    
    Tests: Spatial matching hypothesis
    
    Args:
        tracks_df: DataFrame with spark_tracks.csv data
        poke_x, poke_y: Coordinates of poke/wound site
    
    Returns:
        Dictionary with spatial matching metrics
    """
    df = tracks_df.copy()
    
    # Filter to post-poke events
    df_post = df[df['time_s'] > 0]
    
    if len(df_post) == 0:
        return {'error': 'No post-poke events found'}
    
    # Calculate distances from poke
    if 'dist_from_poke_px' in df_post.columns:
        distances = df_post['dist_from_poke_px']
    else:
        dx = df_post['x'] - poke_x
        dy = df_post['y'] - poke_y
        distances = np.sqrt(dx**2 + dy**2)
    
    # Find closest response (potential spatial match)
    closest_idx = distances.idxmin()
    closest_event = df_post.loc[closest_idx]
    
    # Calculate AP position matching if available
    ap_match = None
    if 'ap_norm' in df_post.columns and 'dist_from_poke_px' in df_post.columns:
        # Get AP position of poke (would need to be provided or estimated)
        # For now, analyze distribution of responses
        ap_match = {
            'response_ap_distribution': df_post['ap_norm'].describe(),
            'mean_distance_from_poke': distances.mean(),
            'min_distance_from_poke': distances.min(),
        }
    
    results = {
        'poke_location': (poke_x, poke_y),
        'closest_response_location': (closest_event['x'], closest_event['y']),
        'min_distance_px': distances.min(),
        'mean_distance_px': distances.mean(),
        'median_distance_px': distances.median(),
        'ap_matching': ap_match,
        'n_responses': len(df_post),
    }
    
    return results


def analyze_tail_response(tracks_df, clusters_df, tail_threshold=0.7):
    """
    Analyze local tail response.
    
    Tests: Hypothesis 7, 8 (local tail response)
    
    Args:
        tracks_df: DataFrame with spark_tracks.csv data
        clusters_df: DataFrame with vector_clusters.csv data
        tail_threshold: AP position threshold for tail region (0=head, 1=tail)
    
    Returns:
        Dictionary with tail response metrics
    """
    df_tracks = tracks_df.copy()
    df_clusters = clusters_df.copy()
    
    # Filter to tail region
    if 'ap_norm' in df_tracks.columns:
        tail_events = df_tracks[df_tracks['ap_norm'] >= tail_threshold]
    else:
        return {'error': 'AP position data not available'}
    
    if len(tail_events) == 0:
        return {'error': 'No events in tail region'}
    
    # Get tail clusters
    tail_track_ids = tail_events['track_id'].unique()
    tail_clusters = df_clusters[df_clusters['cluster_id'].isin(tail_track_ids)]
    
    # Filter to post-poke
    tail_post = tail_events[tail_events['time_s'] > 0]
    
    # Calculate metrics
    activity_per_time = tail_post.groupby('time_s')['area'].sum()
    
    # Speed analysis
    tail_speeds = tail_clusters['mean_speed_px_per_s'].dropna()
    
    results = {
        'n_tail_events': len(tail_post),
        'n_tail_clusters': len(tail_clusters),
        'total_tail_activity': tail_post['area'].sum(),
        'peak_tail_activity_time': activity_per_time.idxmax() if len(activity_per_time) > 0 else np.nan,
        'peak_tail_activity': activity_per_time.max() if len(activity_per_time) > 0 else np.nan,
        'mean_tail_speed': tail_speeds.mean() if len(tail_speeds) > 0 else np.nan,
        'peak_tail_speed': tail_clusters['peak_speed_px_per_s'].max() if 'peak_speed_px_per_s' in tail_clusters.columns else np.nan,
        'tail_activity_over_time': activity_per_time,
    }
    
    return results


def compare_conditions(tracks_df1, tracks_df2, label1="Condition 1", label2="Condition 2"):
    """
    Compare calcium activity between two conditions.
    
    Tests: Hypothesis 2 (distance effect - contact vs non-contact)
    
    Args:
        tracks_df1, tracks_df2: DataFrames for two conditions
        label1, label2: Labels for the conditions
    
    Returns:
        Dictionary with comparison metrics
    """
    # Analyze each condition
    cond1 = analyze_calcium_activity(tracks_df1, None)
    cond2 = analyze_calcium_activity(tracks_df2, None)
    
    # Filter to embryo B for inter-embryo analysis
    if 'embryo_id' in tracks_df1.columns and 'embryo_id' in tracks_df2.columns:
        cond1_B = analyze_calcium_activity(tracks_df1, None, embryo_id='B')
        cond2_B = analyze_calcium_activity(tracks_df2, None, embryo_id='B')
    else:
        cond1_B = cond1
        cond2_B = cond2
    
    results = {
        f'{label1}_total_activity': cond1['post_poke_total_activity'],
        f'{label2}_total_activity': cond2['post_poke_total_activity'],
        f'{label1}_embryo_B_activity': cond1_B['post_poke_total_activity'],
        f'{label2}_embryo_B_activity': cond2_B['post_poke_total_activity'],
        'embryo_B_activity_ratio': (cond2_B['post_poke_total_activity'] / cond1_B['post_poke_total_activity']
                                     if cond1_B['post_poke_total_activity'] > 0 else np.nan),
        'embryo_B_difference': cond2_B['post_poke_total_activity'] - cond1_B['post_poke_total_activity'],
    }
    
    return results


def plot_activity_comparison(tracks_df, output_path=None):
    """
    Plot activity comparison between embryos and pre/post poke.
    Uses pixel-based counting (each spark = 1 pixel).
    Tests: Hypothesis 1 (Presence of calcium activity)
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Activity over time per embryo (pixel counts)
    if 'embryo_id' in tracks_df.columns:
        for embryo_id in tracks_df['embryo_id'].dropna().unique():
            df_emb = tracks_df[tracks_df['embryo_id'] == embryo_id]
            # Count unique pixels (track_ids) per time point
            activity = df_emb.groupby('time_s')['track_id'].nunique()
            axes[0].plot(activity.index, activity.values, label=f'Embryo {embryo_id}', alpha=0.7, linewidth=2)
    else:
        # Count unique pixels per time point
        activity = tracks_df.groupby('time_s')['track_id'].nunique()
        axes[0].plot(activity.index, activity.values, label='All embryos', alpha=0.7, linewidth=2)
    
    axes[0].axvline(x=0, color='r', linestyle='--', alpha=0.5, linewidth=2, label='Poke time')
    axes[0].set_xlabel('Time (seconds, relative to poke)', fontsize=11)
    axes[0].set_ylabel('Activated Pixels', fontsize=11)
    axes[0].set_title('Calcium Activity Over Time: Pre-poke vs Post-poke\n(Each spark = 1 pixel)', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Pre/post comparison (pixel counts)
    pre_poke = tracks_df[tracks_df['time_s'] < 0]
    post_poke = tracks_df[tracks_df['time_s'] > 0]
    
    if 'embryo_id' in tracks_df.columns:
        embryos = tracks_df['embryo_id'].dropna().unique()
        x_pos = np.arange(len(embryos))
        width = 0.35
        
        # Count unique pixels per embryo for pre/post
        pre_means = [pre_poke[pre_poke['embryo_id'] == e]['track_id'].nunique() if len(pre_poke[pre_poke['embryo_id'] == e]) > 0 else 0 for e in embryos]
        post_means = [post_poke[post_poke['embryo_id'] == e]['track_id'].nunique() if len(post_poke[post_poke['embryo_id'] == e]) > 0 else 0 for e in embryos]
        
        axes[1].bar(x_pos - width/2, pre_means, width, label='Pre-poke', alpha=0.7, color='blue')
        axes[1].bar(x_pos + width/2, post_means, width, label='Post-poke', alpha=0.7, color='orange')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(embryos)
    else:
        pre_mean = pre_poke['track_id'].nunique() if len(pre_poke) > 0 else 0
        post_mean = post_poke['track_id'].nunique() if len(post_poke) > 0 else 0
        axes[1].bar(['Pre-poke', 'Post-poke'], [pre_mean, post_mean], alpha=0.7)
    
    axes[1].set_ylabel('Activated Pixels', fontsize=11)
    axes[1].set_title('Pre-poke vs Post-poke Activity by Embryo\n(Each spark = 1 pixel)', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved activity comparison to {output_path}")
    else:
        plt.show()


def plot_wave_directionality(tracks_df, clusters_df, embryo_id=None, output_path=None):
    """
    Plot wave directionality analysis.
    Tests: Hypotheses 3, 4 (Wave directionality within and between embryos)
    """
    fig = plt.figure(figsize=(14, 12))
    
    # Filter data
    df_clusters = clusters_df.copy()
    if embryo_id and 'embryo_id' in tracks_df.columns:
        track_ids = tracks_df[tracks_df['embryo_id'] == embryo_id]['track_id'].unique()
        df_clusters = df_clusters[df_clusters['cluster_id'].isin(track_ids)]
    
    valid_clusters = df_clusters[df_clusters['mean_angle_deg'].notna()]
    
    if len(valid_clusters) == 0:
        print("No clusters with angle data found")
        return
    
    angles = valid_clusters['mean_angle_deg'].values
    speeds = valid_clusters['mean_speed_px_per_s'].dropna().values
    
    # 1. Rose plot (circular histogram) of wave directions
    ax1 = plt.subplot(2, 2, 1, projection='polar')
    # Convert to radians and create histogram
    angles_rad = np.deg2rad(angles)
    n_bins = 36  # 10 degree bins
    hist, bins = np.histogram(angles_rad, bins=n_bins, range=(0, 2*np.pi))
    width = 2 * np.pi / n_bins
    bars = ax1.bar(bins[:-1], hist, width=width, alpha=0.7, color='steelblue')
    ax1.set_theta_zero_location('E')
    ax1.set_theta_direction(1)
    ax1.set_thetagrids(np.arange(0, 360, 45), ['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'])
    ax1.set_title(f'Wave Direction Distribution\n(Embryo {embryo_id if embryo_id else "All"})', 
                 fontweight='bold', pad=20)
    
    # 2. Speed distribution
    ax2 = plt.subplot(2, 2, 2)
    if len(speeds) > 0:
        ax2.hist(speeds, bins=50, edgecolor='black', alpha=0.7, color='coral')
        ax2.set_xlabel('Mean Speed (pixels/second)', fontsize=10)
        ax2.set_ylabel('Count', fontsize=10)
        ax2.set_title('Wave Speed Distribution', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(np.mean(speeds), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(speeds):.2f}')
        ax2.legend()
    
    # 3. Angle vs Speed scatter
    ax3 = plt.subplot(2, 2, 3)
    angle_speed = valid_clusters[valid_clusters['mean_speed_px_per_s'].notna()]
    if len(angle_speed) > 0:
        scatter = ax3.scatter(angle_speed['mean_angle_deg'], angle_speed['mean_speed_px_per_s'],
                            c=angle_speed['total_area_px2_frames'] if 'total_area_px2_frames' in angle_speed.columns else None,
                            alpha=0.6, s=50, cmap='viridis')
        ax3.set_xlabel('Wave Direction (degrees)', fontsize=10)
        ax3.set_ylabel('Speed (pixels/second)', fontsize=10)
        ax3.set_title('Wave Direction vs Speed', fontweight='bold')
        if 'total_area_px2_frames' in angle_speed.columns:
            plt.colorbar(scatter, ax=ax3, label='Total Activity')
        ax3.grid(True, alpha=0.3)
    
    # 4. Activity over time by direction quadrant
    ax4 = plt.subplot(2, 2, 4)
    df_tracks = tracks_df.copy()
    if embryo_id and 'embryo_id' in df_tracks.columns:
        df_tracks = df_tracks[df_tracks['embryo_id'] == embryo_id]
    
    # Group by direction quadrants
    df_tracks_with_angle = df_tracks[df_tracks['angle_deg'].notna()].copy()
    if len(df_tracks_with_angle) > 0:
        df_tracks_with_angle['direction'] = pd.cut(df_tracks_with_angle['angle_deg'], 
                                                    bins=[-180, -90, 0, 90, 180],
                                                    labels=['Left', 'Up', 'Right', 'Down'])
        
        for direction in df_tracks_with_angle['direction'].dropna().unique():
            df_dir = df_tracks_with_angle[df_tracks_with_angle['direction'] == direction]
            activity = df_dir.groupby('time_s')['area'].sum()
            if len(activity) > 0:
                ax4.plot(activity.index, activity.values, label=f'{direction}', alpha=0.7, linewidth=2)
        
        ax4.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Poke time')
        ax4.set_xlabel('Time (seconds)', fontsize=10)
        ax4.set_ylabel('Activity (pixels²)', fontsize=10)
        ax4.set_title('Activity Over Time by Direction', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved directionality plot to {output_path}")
    else:
        plt.show()


def plot_spatial_matching(tracks_df, poke_x, poke_y, output_path=None):
    """
    Plot spatial matching between wound location and response locations.
    Tests: Spatial matching hypothesis
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    df_post = tracks_df[tracks_df['time_s'] > 0].copy()
    
    if len(df_post) == 0:
        print("No post-poke events found")
        return
    
    # Calculate distances
    if 'dist_from_poke_px' not in df_post.columns:
        dx = df_post['x'] - poke_x
        dy = df_post['y'] - poke_y
        df_post['dist_from_poke_px'] = np.sqrt(dx**2 + dy**2)
    
    # 1. Spatial overlay showing poke and responses
    ax = axes[0]
    
    # Plot all post-poke events
    scatter = ax.scatter(df_post['x'], df_post['y'], 
                        c=df_post['time_s'], 
                        s=df_post['area'], 
                        alpha=0.5, 
                        cmap='hot',
                        edgecolors='black', 
                        linewidths=0.5)
    
    # Mark poke location
    ax.plot(poke_x, poke_y, 'r*', markersize=20, markeredgecolor='black', 
            markeredgewidth=2, label='Poke site', zorder=10)
    
    # Mark closest response
    closest_idx = df_post['dist_from_poke_px'].idxmin()
    closest_event = df_post.loc[closest_idx]
    ax.plot(closest_event['x'], closest_event['y'], 'go', markersize=15,
            markeredgecolor='black', markeredgewidth=2, label='Closest response', zorder=10)
    
    # Draw line from poke to closest
    ax.plot([poke_x, closest_event['x']], [poke_y, closest_event['y']], 
            'g--', alpha=0.5, linewidth=2, label=f'Distance: {df_post.loc[closest_idx, "dist_from_poke_px"]:.1f} px')
    
    plt.colorbar(scatter, ax=ax, label='Time (seconds)')
    ax.set_xlabel('X (pixels)', fontsize=11)
    ax.set_ylabel('Y (pixels)', fontsize=11)
    ax.set_title('Spatial Matching: Poke Site and Responses', fontweight='bold', fontsize=12)
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 2. Distance distribution
    ax = axes[1]
    distances = df_post['dist_from_poke_px']
    ax.hist(distances, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(distances.min(), color='green', linestyle='--', linewidth=2, 
               label=f'Min: {distances.min():.1f} px')
    ax.axvline(distances.mean(), color='orange', linestyle='--', linewidth=2,
               label=f'Mean: {distances.mean():.1f} px')
    ax.set_xlabel('Distance from Poke (pixels)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Distribution of Response Distances from Poke', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved spatial matching plot to {output_path}")
    else:
        plt.show()


def plot_tail_response(tracks_df, clusters_df, tail_threshold=0.7, output_path=None):
    """
    Plot tail response analysis.
    Tests: Hypotheses 7, 8 (Local tail response)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    if 'ap_norm' not in tracks_df.columns:
        print("AP position data not available")
        return
    
    tail_events = tracks_df[tracks_df['ap_norm'] >= tail_threshold].copy()
    tail_post = tail_events[tail_events['time_s'] > 0]
    
    if len(tail_post) == 0:
        print("No tail events found")
        return
    
    # 1. Activity over time in tail region
    ax = axes[0, 0]
    activity_per_time = tail_post.groupby('time_s')['area'].sum()
    ax.plot(activity_per_time.index, activity_per_time.values, 
            color='purple', linewidth=2, marker='o', markersize=4)
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Poke time')
    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('Tail Activity (pixels²)', fontsize=11)
    ax.set_title(f'Tail Response Over Time\n(AP > {tail_threshold})', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Spatial distribution in tail
    ax = axes[0, 1]
    scatter = ax.scatter(tail_post['x'], tail_post['y'],
                        c=tail_post['time_s'],
                        s=tail_post['area'],
                        alpha=0.6,
                        cmap='plasma',
                        edgecolors='black',
                        linewidths=0.5)
    plt.colorbar(scatter, ax=ax, label='Time (seconds)')
    ax.set_xlabel('X (pixels)', fontsize=11)
    ax.set_ylabel('Y (pixels)', fontsize=11)
    ax.set_title('Spatial Distribution of Tail Responses', fontweight='bold', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 3. Speed analysis
    ax = axes[1, 0]
    tail_track_ids = tail_post['track_id'].unique()
    tail_clusters = clusters_df[clusters_df['cluster_id'].isin(tail_track_ids)]
    tail_speeds = tail_clusters['mean_speed_px_per_s'].dropna()
    
    if len(tail_speeds) > 0:
        ax.hist(tail_speeds.values, bins=30, edgecolor='black', alpha=0.7, color='orange')
        ax.axvline(tail_speeds.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {tail_speeds.mean():.2f} px/s')
        ax.set_xlabel('Speed (pixels/second)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Tail Response Speed Distribution', fontweight='bold', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 4. AP position distribution
    ax = axes[1, 1]
    if 'ap_norm' in tail_post.columns:
        ap_vals = tail_post['ap_norm'].dropna().values
        if len(ap_vals) > 0:
            ap_min, ap_max = ap_vals.min(), ap_vals.max()
            bins = np.linspace(ap_min, ap_max, 31) if ap_max > ap_min else 30
            ax.hist(ap_vals, bins=bins, edgecolor='black', alpha=0.7, color='teal')
        ax.axvline(tail_threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold: {tail_threshold}')
        ax.set_xlabel('AP Position (0=head, 1=tail)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('AP Position Distribution of Tail Events', fontweight='bold', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved tail response plot to {output_path}")
    else:
        plt.show()


def plot_condition_comparison(tracks_df1, tracks_df2, label1="Condition 1", label2="Condition 2", output_path=None):
    """
    Plot comparison between two experimental conditions.
    Tests: Hypothesis 2 (Distance effect - contact vs non-contact)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Filter to embryo B for inter-embryo analysis
    if 'embryo_id' in tracks_df1.columns:
        df1_B = tracks_df1[tracks_df1['embryo_id'] == 'B']
        df2_B = tracks_df2[tracks_df2['embryo_id'] == 'B']
    else:
        df1_B = tracks_df1
        df2_B = tracks_df2
    
    # 1. Activity over time comparison (embryo B)
    ax = axes[0, 0]
    activity1 = df1_B.groupby('time_s')['area'].sum()
    activity2 = df2_B.groupby('time_s')['area'].sum()
    ax.plot(activity1.index, activity1.values, label=label1, linewidth=2, alpha=0.8)
    ax.plot(activity2.index, activity2.values, label=label2, linewidth=2, alpha=0.8)
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Poke time')
    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('Activity in Embryo B (pixels²)', fontsize=11)
    ax.set_title('Activity Comparison: Embryo B', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Total activity comparison
    ax = axes[0, 1]
    post1 = df1_B[df1_B['time_s'] > 0]
    post2 = df2_B[df2_B['time_s'] > 0]
    totals = [post1['area'].sum(), post2['area'].sum()]
    bars = ax.bar([label1, label2], totals, alpha=0.7, color=['blue', 'orange'])
    ax.set_ylabel('Total Activity (pixels²)', fontsize=11)
    ax.set_title('Total Activity Comparison\n(Embryo B, Post-poke)', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom', fontsize=10)
    
    # 3. Peak activity time comparison
    ax = axes[1, 0]
    peak_times = [activity1.idxmax() if len(activity1) > 0 else 0,
                  activity2.idxmax() if len(activity2) > 0 else 0]
    bars = ax.bar([label1, label2], peak_times, alpha=0.7, color=['blue', 'orange'])
    ax.set_ylabel('Peak Activity Time (seconds)', fontsize=11)
    ax.set_title('Time to Peak Activity\n(Embryo B)', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom', fontsize=10)
    
    # 4. Activity ratio
    ax = axes[1, 1]
    ratio = totals[1] / totals[0] if totals[0] > 0 else np.nan
    ax.bar(['Activity Ratio'], [ratio], alpha=0.7, color='green' if ratio < 1 else 'red')
    ax.set_ylabel(f'{label2} / {label1}', fontsize=11)
    ax.set_title('Activity Ratio Comparison\n(Embryo B)', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    
    # Add value label
    ax.text(0, ratio, f'{ratio:.2f}x', ha='center', va='bottom' if ratio > 1 else 'top', 
            fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved condition comparison to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze experimental hypotheses')
    parser.add_argument('tracks_csv', help='Path to spark_tracks.csv')
    parser.add_argument('--clusters-csv', help='Path to vector_clusters.csv')
    parser.add_argument('--analysis', choices=['activity', 'directionality', 'spatial', 'tail', 'compare'],
                       default='activity', help='Type of analysis to perform')
    parser.add_argument('--embryo', choices=['A', 'B'], help='Filter to specific embryo')
    parser.add_argument('--output', help='Output file path for plots')
    parser.add_argument('--poke-x', type=float, help='X coordinate of poke site (for spatial analysis)')
    parser.add_argument('--poke-y', type=float, help='Y coordinate of poke site (for spatial analysis)')
    parser.add_argument('--compare-csv2', help='Second CSV file for condition comparison')
    parser.add_argument('--label1', default='Condition 1', help='Label for first condition')
    parser.add_argument('--label2', default='Condition 2', help='Label for second condition')
    
    args = parser.parse_args()
    
    print(f"Loading {args.tracks_csv}...")
    tracks_df = pd.read_csv(args.tracks_csv)
    print(f"Loaded {len(tracks_df)} track states")
    
    clusters_df = None
    if args.clusters_csv:
        print(f"Loading {args.clusters_csv}...")
        clusters_df = pd.read_csv(args.clusters_csv)
        print(f"Loaded {len(clusters_df)} clusters")
    
    # Run analysis
    if args.analysis == 'activity':
        results = analyze_calcium_activity(tracks_df, clusters_df, embryo_id=args.embryo)
        print("\n=== Calcium Activity Analysis ===")
        for key, value in results.items():
            if key != 'activity_per_time':
                print(f"{key}: {value}")
        
        plot_activity_comparison(tracks_df, args.output)
    
    elif args.analysis == 'directionality':
        if clusters_df is None:
            print("Error: --clusters-csv required for directionality analysis")
            sys.exit(1)
        results = analyze_wave_directionality(tracks_df, clusters_df, embryo_id=args.embryo)
        print("\n=== Wave Directionality Analysis ===")
        for key, value in results.items():
            if key not in ['angle_distribution', 'speed_distribution']:
                print(f"{key}: {value}")
        
        plot_wave_directionality(tracks_df, clusters_df, embryo_id=args.embryo, output_path=args.output)
    
    elif args.analysis == 'tail':
        if clusters_df is None:
            print("Error: --clusters-csv required for tail analysis")
            sys.exit(1)
        results = analyze_tail_response(tracks_df, clusters_df)
        print("\n=== Tail Response Analysis ===")
        for key, value in results.items():
            if key != 'tail_activity_over_time':
                print(f"{key}: {value}")
        
        plot_tail_response(tracks_df, clusters_df, output_path=args.output)
    
    elif args.analysis == 'spatial':
        # For spatial matching, need poke coordinates
        if args.poke_x is None or args.poke_y is None:
            print("Error: --poke-x and --poke-y required for spatial analysis")
            print("Alternatively, extract from spark_tracks.csv if dist_from_poke_px is available")
            sys.exit(1)
        plot_spatial_matching(tracks_df, args.poke_x, args.poke_y, output_path=args.output)
    
    elif args.analysis == 'compare':
        if args.compare_csv2 is None:
            print("Error: --compare-csv2 required for condition comparison")
            sys.exit(1)
        print(f"Loading {args.compare_csv2}...")
        tracks_df2 = pd.read_csv(args.compare_csv2)
        print(f"Loaded {len(tracks_df2)} track states")
        plot_condition_comparison(tracks_df, tracks_df2, 
                                 label1=args.label1, label2=args.label2,
                                 output_path=args.output)


if __name__ == '__main__':
    main()

