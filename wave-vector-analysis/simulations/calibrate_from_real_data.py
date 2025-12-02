#!/usr/bin/env python3
"""
Extract parameters from real experimental data to calibrate simulations.

This script analyzes real spark_tracks.csv and vector_clusters.csv to extract:
- Wave propagation speeds
- Activity decay rates
- Spatial distributions
- Temporal patterns
- Embryo geometries

These parameters can then be used to generate more realistic simulations.
"""

import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
from scipy import stats


def extract_wave_parameters(tracks_df, clusters_df):
    """Extract wave propagation parameters from real data."""
    params = {}
    
    # Speed statistics
    if 'speed' in tracks_df.columns:
        speeds = tracks_df['speed'].dropna()
        if len(speeds) > 0:
            params['mean_speed'] = float(speeds.mean())
            params['median_speed'] = float(speeds.median())
            params['std_speed'] = float(speeds.std())
            params['min_speed'] = float(speeds.min())
            params['max_speed'] = float(speeds.max())
    
    # Cluster-level speed statistics
    if 'mean_speed_px_per_s' in clusters_df.columns:
        cluster_speeds = clusters_df['mean_speed_px_per_s'].dropna()
        if len(cluster_speeds) > 0:
            params['mean_cluster_speed'] = float(cluster_speeds.mean())
            params['peak_cluster_speed'] = float(cluster_speeds.max())
    
    # Activity over time (decay rate)
    if 'time_s' in tracks_df.columns and 'area' in tracks_df.columns:
        post_poke = tracks_df[tracks_df['time_s'] > 0].copy()
        if len(post_poke) > 0:
            activity_over_time = post_poke.groupby('time_s')['area'].sum()
            
            # Fit exponential decay
            if len(activity_over_time) > 5:
                times = activity_over_time.index.values
                values = activity_over_time.values
                
                # Only use non-zero values
                mask = values > 0
                if mask.sum() > 3:
                    times_fit = times[mask]
                    values_fit = values[mask]
                    
                    # Log-linear fit for decay rate
                    log_values = np.log(values_fit + 1)
                    try:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(times_fit, log_values)
                        decay_rate = -slope  # Negative because decay
                        params['decay_rate'] = float(decay_rate)
                        params['decay_r_squared'] = float(r_value ** 2)
                    except:
                        pass
    
    # Track duration statistics
    if 'duration_s' in clusters_df.columns:
        durations = clusters_df['duration_s'].dropna()
        if len(durations) > 0:
            params['mean_duration'] = float(durations.mean())
            params['median_duration'] = float(durations.median())
            params['max_duration'] = float(durations.max())
    
    # Spatial spread
    if 'x' in tracks_df.columns and 'y' in tracks_df.columns:
        x_std = tracks_df['x'].std()
        y_std = tracks_df['y'].std()
        params['spatial_spread_x'] = float(x_std)
        params['spatial_spread_y'] = float(y_std)
    
    # Embryo statistics
    if 'embryo_id' in tracks_df.columns:
        embryo_counts = tracks_df['embryo_id'].dropna().value_counts()
        params['embryos_detected'] = embryo_counts.to_dict()
        params['n_embryos'] = len(embryo_counts)
    
    # AP position distribution
    if 'ap_norm' in tracks_df.columns:
        ap_vals = tracks_df['ap_norm'].dropna()
        if len(ap_vals) > 0:
            params['ap_mean'] = float(ap_vals.mean())
            params['ap_std'] = float(ap_vals.std())
            
            # Tail region (ap_norm >= 0.7)
            tail_fraction = (ap_vals >= 0.7).sum() / len(ap_vals)
            params['tail_fraction'] = float(tail_fraction)
    
    return params


def extract_embryo_geometries(tracks_df):
    """Extract embryo positions and sizes from real data."""
    geometries = []
    
    if 'embryo_id' not in tracks_df.columns:
        return geometries
    
    for embryo_id in tracks_df['embryo_id'].dropna().unique():
        emb_data = tracks_df[tracks_df['embryo_id'] == embryo_id].copy()
        
        if len(emb_data) == 0:
            continue
        
        # Estimate embryo center from spark positions
        center_x = emb_data['x'].mean()
        center_y = emb_data['y'].mean()
        
        # Estimate size from spread
        x_range = emb_data['x'].max() - emb_data['x'].min()
        y_range = emb_data['y'].max() - emb_data['y'].min()
        
        # Estimate length and width
        length = max(x_range, y_range) * 1.2  # Add margin
        width = min(x_range, y_range) * 1.2
        
        # Estimate orientation from AP distribution if available
        angle = 0.0
        if 'ap_norm' in emb_data.columns:
            # Simple heuristic: if AP varies, embryo is elongated
            ap_std = emb_data['ap_norm'].std()
            if ap_std > 0.1:
                # Estimate angle from spatial distribution
                # This is simplified - could use PCA for better estimate
                angle = 0.0  # Default horizontal
        
        # Estimate head/tail positions (simplified)
        if 'ap_norm' in emb_data.columns:
            # Find points at ap_norm ~ 0 (head) and ~ 1 (tail)
            head_points = emb_data[emb_data['ap_norm'] < 0.2]
            tail_points = emb_data[emb_data['ap_norm'] > 0.8]
            
            if len(head_points) > 0:
                head_x = head_points['x'].mean()
                head_y = head_points['y'].mean()
            else:
                head_x = center_x - length / 2
                head_y = center_y
            
            if len(tail_points) > 0:
                tail_x = tail_points['x'].mean()
                tail_y = tail_points['y'].mean()
            else:
                tail_x = center_x + length / 2
                tail_y = center_y
        else:
            # Default: horizontal orientation
            head_x = center_x - length / 2
            head_y = center_y
            tail_x = center_x + length / 2
            tail_y = center_y
        
        geometries.append({
            'id': str(embryo_id),
            'center_x': float(center_x),
            'center_y': float(center_y),
            'length': float(length),
            'width': float(width),
            'angle': float(angle),
            'head_x': float(head_x),
            'head_y': float(head_y),
            'tail_x': float(tail_x),
            'tail_y': float(tail_y)
        })
    
    return geometries


def extract_poke_locations(tracks_df, clusters_df):
    """Extract poke locations from real data (using first spark clusters)."""
    pokes = []
    
    # Group by base filename to get one poke per file
    if 'filename' in tracks_df.columns:
        tracks_df = tracks_df.copy()
        tracks_df['base_filename'] = tracks_df['filename'].str.replace(r'\s*\(page\s+\d+\)', '', regex=True, case=False)
        
        for base_file in tracks_df['base_filename'].unique():
            file_data = tracks_df[tracks_df['base_filename'] == base_file]
            
            if len(file_data) == 0:
                continue
            
            # Find first frame with sparks
            if 'frame_idx' in file_data.columns:
                min_frame = file_data['frame_idx'].min()
                first_sparks = file_data[file_data['frame_idx'] == min_frame]
            elif 'time_s' in file_data.columns:
                min_time = file_data['time_s'].min()
                first_sparks = file_data[file_data['time_s'] <= min_time + 5]
            else:
                continue
            
            if len(first_sparks) > 0:
                # Use centroid of first sparks as poke location
                poke_x = first_sparks['x'].mean()
                poke_y = first_sparks['y'].mean()
                
                # Determine which embryo (if any)
                embryo_id = None
                if 'embryo_id' in first_sparks.columns:
                    embryo_ids = first_sparks['embryo_id'].dropna().unique()
                    if len(embryo_ids) > 0:
                        embryo_id = str(embryo_ids[0])  # Use most common
                
                pokes.append({
                    'x': float(poke_x),
                    'y': float(poke_y),
                    'embryo_id': embryo_id,
                    'time': float(file_data['time_s'].min()) if 'time_s' in file_data.columns else 0.0
                })
    
    return pokes


def generate_simulation_config(tracks_csv, clusters_csv=None, output_json=None):
    """Generate a simulation configuration file from real data."""
    print(f"Loading real data from {tracks_csv}...")
    tracks_df = pd.read_csv(tracks_csv)
    print(f"  → Loaded {len(tracks_df)} track states")
    
    clusters_df = None
    if clusters_csv and Path(clusters_csv).exists():
        print(f"Loading clusters from {clusters_csv}...")
        clusters_df = pd.read_csv(clusters_csv)
        print(f"  → Loaded {len(clusters_df)} clusters")
    
    print("\nExtracting parameters from real data...")
    
    # Extract wave parameters
    wave_params = extract_wave_parameters(tracks_df, clusters_df if clusters_df is not None else pd.DataFrame())
    print(f"  ✓ Wave parameters: mean speed = {wave_params.get('mean_speed', 'N/A'):.2f} px/s")
    
    # Extract embryo geometries
    geometries = extract_embryo_geometries(tracks_df)
    print(f"  ✓ Found {len(geometries)} embryo geometry(ies)")
    
    # Extract poke locations
    pokes = extract_poke_locations(tracks_df, clusters_df if clusters_df is not None else pd.DataFrame())
    print(f"  ✓ Found {len(pokes)} poke location(s)")
    
    # Build configuration
    config = {
        'source_data': {
            'tracks_csv': str(Path(tracks_csv).absolute()),
            'clusters_csv': str(Path(clusters_csv).absolute()) if clusters_csv else None
        },
        'wave_config': {
            'speed_px_per_s': wave_params.get('mean_speed', 5.0),
            'duration_s': wave_params.get('mean_duration', 10.0),
            'decay_rate': wave_params.get('decay_rate', 0.1),
            'radial': True
        },
        'embryos': geometries,
        'pokes': pokes,
        'extracted_parameters': wave_params
    }
    
    # Save configuration
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\n✓ Saved simulation configuration to {output_json}")
    
    # Print summary
    print("\n=== Extracted Parameters ===")
    print(f"Wave speed: {wave_params.get('mean_speed', 'N/A'):.2f} px/s (mean)")
    print(f"Wave duration: {wave_params.get('mean_duration', 'N/A'):.2f} s (mean)")
    print(f"Decay rate: {wave_params.get('decay_rate', 'N/A'):.4f}")
    print(f"Embryos: {len(geometries)}")
    print(f"Poke locations: {len(pokes)}")
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Extract parameters from real data to calibrate simulations"
    )
    parser.add_argument(
        'tracks_csv',
        help='Path to spark_tracks.csv from real data'
    )
    parser.add_argument(
        '--clusters-csv',
        help='Path to vector_clusters.csv (optional)'
    )
    parser.add_argument(
        '--output',
        default='simulation_config_from_data.json',
        help='Output JSON configuration file'
    )
    
    args = parser.parse_args()
    
    config = generate_simulation_config(args.tracks_csv, args.clusters_csv, args.output)
    
    print(f"\n✓ Configuration ready!")
    print(f"\nTo use this configuration:")
    print(f"  python3 generate_simulated_data.py --config {args.output}")


if __name__ == "__main__":
    main()

