#!/usr/bin/env python3
"""
Generate vector_clusters.csv from spark_tracks.csv.

This script aggregates per-frame spark track data into per-cluster summaries
suitable for downstream analysis and clustering.
"""

import pandas as pd
import numpy as np
import math
import sys
from pathlib import Path


def circular_mean(angles_deg):
    """Compute circular mean of angles in degrees."""
    if len(angles_deg) == 0:
        return np.nan
    
    # Filter out NaN values
    angles_deg = angles_deg.dropna()
    if len(angles_deg) == 0:
        return np.nan
    
    # Convert to radians
    angles_rad = np.deg2rad(angles_deg)
    
    # Compute mean direction
    mean_sin = np.sin(angles_rad).mean()
    mean_cos = np.cos(angles_rad).mean()
    
    mean_rad = np.arctan2(mean_sin, mean_cos)
    return np.rad2deg(mean_rad)


def circular_std(angles_deg):
    """Compute circular standard deviation (dispersion) of angles in degrees."""
    if len(angles_deg) == 0:
        return np.nan
    
    angles_deg = angles_deg.dropna()
    if len(angles_deg) == 0:
        return np.nan
    
    angles_rad = np.deg2rad(angles_deg)
    mean_sin = np.sin(angles_rad).mean()
    mean_cos = np.cos(angles_rad).mean()
    
    # Mean resultant length (R)
    R = np.sqrt(mean_sin**2 + mean_cos**2)
    
    # Circular standard deviation
    # For small R, use approximation; for large R, use proper formula
    if R < 1e-6:
        return 180.0  # Uniform distribution
    else:
        # Circular variance = 1 - R, convert to degrees
        circular_var = 1 - R
        # Approximate std in degrees (this is a simplified version)
        circular_std_rad = np.sqrt(-2 * np.log(R))
        return np.rad2deg(circular_std_rad)


def compute_path_length(df_track):
    """Compute total path length by summing stepwise distances."""
    if len(df_track) < 2:
        return 0.0
    
    # Sort by frame_idx to ensure correct order
    df_sorted = df_track.sort_values('frame_idx')
    
    path_length = 0.0
    for i in range(1, len(df_sorted)):
        prev = df_sorted.iloc[i-1]
        curr = df_sorted.iloc[i]
        dx = curr['x'] - prev['x']
        dy = curr['y'] - prev['y']
        path_length += math.hypot(dx, dy)
    
    return path_length


def process_tracks_to_clusters(csv_path, output_path=None):
    """
    Convert spark_tracks.csv to vector_clusters.csv.
    
    Args:
        csv_path: Path to spark_tracks.csv
        output_path: Path for output CSV (default: vector_clusters.csv in same directory)
    """
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Processing {len(df)} rows...")
    print(f"Unique tracks: {df['track_id'].nunique()}")
    
    clusters = []
    
    for track_id, df_track in df.groupby('track_id'):
        # Basic counts and durations
        n_frames = len(df_track)
        
        start_frame_idx = df_track['frame_idx'].min()
        end_frame_idx = df_track['frame_idx'].max()
        duration_frames = end_frame_idx - start_frame_idx + 1
        
        start_time_s = df_track['time_s'].min()
        end_time_s = df_track['time_s'].max()
        duration_s = end_time_s - start_time_s
        
        # Positions
        start_x = df_track.loc[df_track['frame_idx'].idxmin(), 'x']
        start_y = df_track.loc[df_track['frame_idx'].idxmin(), 'y']
        end_x = df_track.loc[df_track['frame_idx'].idxmax(), 'x']
        end_y = df_track.loc[df_track['frame_idx'].idxmax(), 'y']
        
        # Net displacement
        net_displacement_px = math.hypot(end_x - start_x, end_y - start_y)
        
        # Path length
        path_length_px = compute_path_length(df_track)
        
        # Speeds
        speeds = df_track['speed'].dropna()
        mean_speed = speeds.mean() if len(speeds) > 0 else np.nan
        peak_speed = speeds.max() if len(speeds) > 0 else np.nan
        
        # Net speed (displacement / duration)
        net_speed = net_displacement_px / duration_s if duration_s > 0 else np.nan
        
        # Angles
        angles = df_track['angle_deg'].dropna()
        mean_angle = circular_mean(angles) if len(angles) > 0 else np.nan
        angle_dispersion = circular_std(angles) if len(angles) > 0 else np.nan
        
        # Areas
        areas = df_track['area'].dropna()
        mean_area = areas.mean() if len(areas) > 0 else np.nan
        total_area_frames = areas.sum() if len(areas) > 0 else np.nan
        
        # Distance from poke (if available)
        dist_from_poke_start = np.nan
        dist_from_poke_end = np.nan
        if 'dist_from_poke_px' in df_track.columns:
            poke_dists = df_track['dist_from_poke_px'].dropna()
            if len(poke_dists) > 0:
                dist_from_poke_start = poke_dists.iloc[0]
                dist_from_poke_end = poke_dists.iloc[-1]
        
        # Region information (if available)
        primary_region = ""
        region_counts = {}
        if 'region' in df_track.columns:
            regions = df_track['region'].dropna()
            regions = regions[regions != ""]  # Filter out empty strings
            if len(regions) > 0:
                # Count occurrences of each region
                region_counts = regions.value_counts().to_dict()
                # Primary region is the most common one
                primary_region = regions.value_counts().index[0] if len(regions) > 0 else ""
        
        cluster = {
            'cluster_id': track_id,
            'n_frames': n_frames,
            'duration_frames': duration_frames,
            'duration_s': duration_s,
            'start_frame_idx': start_frame_idx,
            'end_frame_idx': end_frame_idx,
            'start_time_s': start_time_s,
            'end_time_s': end_time_s,
            'start_x_px': start_x,
            'start_y_px': start_y,
            'end_x_px': end_x,
            'end_y_px': end_y,
            'net_displacement_px': net_displacement_px,
            'path_length_px': path_length_px,
            'net_speed_px_per_s': net_speed,
            'mean_speed_px_per_s': mean_speed,
            'peak_speed_px_per_s': peak_speed,
            'mean_angle_deg': mean_angle,
            'angle_dispersion_deg': angle_dispersion,
            'mean_area_px2': mean_area,
            'total_area_px2_frames': total_area_frames,
        }
        
        # Add optional poke distance columns if available
        if not np.isnan(dist_from_poke_start):
            cluster['dist_from_poke_start_px'] = dist_from_poke_start
        if not np.isnan(dist_from_poke_end):
            cluster['dist_from_poke_end_px'] = dist_from_poke_end
        
        # Add region information if available
        if primary_region:
            cluster['primary_region'] = primary_region
            # Add region distribution as a string (for CSV compatibility)
            if region_counts:
                region_str = "; ".join([f"{r}:{c}" for r, c in sorted(region_counts.items(), key=lambda x: -x[1])])
                cluster['region_distribution'] = region_str
        
        clusters.append(cluster)
    
    # Create DataFrame
    df_clusters = pd.DataFrame(clusters)
    
    # Determine output path
    if output_path is None:
        csv_file = Path(csv_path)
        output_path = csv_file.parent / 'vector_clusters.csv'
    
    print(f"\nWriting {len(df_clusters)} clusters to {output_path}...")
    df_clusters.to_csv(output_path, index=False)
    
    print(f"\nSummary:")
    print(f"  • Total clusters: {len(df_clusters)}")
    print(f"  • Clusters with speed data: {(~df_clusters['mean_speed_px_per_s'].isna()).sum()}")
    print(f"  • Mean duration: {df_clusters['duration_s'].mean():.2f} seconds")
    print(f"  • Mean path length: {df_clusters['path_length_px'].mean():.1f} pixels")
    
    return df_clusters


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python spark_tracks_to_clusters.py <spark_tracks.csv> [output.csv]")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    process_tracks_to_clusters(csv_path, output_path)

