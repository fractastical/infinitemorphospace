#!/usr/bin/env python3
"""
Add distance from poke calculations to spark_tracks.csv using inferred poke locations.

This script:
1. Infers poke locations from early spark clusters (using plot_poke_locations logic)
2. Calculates dist_from_poke_px for all sparks based on inferred poke locations
3. Optionally updates the spark_tracks.csv file or creates an enhanced version
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import sys

# Import the poke inference function from plot_poke_locations
try:
    from plot_poke_locations import infer_poke_from_early_sparks
except ImportError:
    # If import fails, define a simplified version
    def infer_poke_from_early_sparks(tracks_df, time_window_s=2.0, min_sparks=1):
        """Infer poke location from the first spark cluster that appears in each file."""
        poke_locations = []
        
        if 'filename' not in tracks_df.columns:
            return pd.DataFrame(columns=['filename', 'poke_x', 'poke_y', 'n_sparks', 'method'])
        
        tracks_df = tracks_df.copy()
        tracks_df['base_filename'] = tracks_df['filename'].str.replace(r' \(page \d+\)', '', regex=True)
        unique_base_files = sorted(tracks_df['base_filename'].unique())
        
        for base_file in unique_base_files:
            file_data = tracks_df[tracks_df['base_filename'] == base_file].copy()
            
            if len(file_data) == 0:
                continue
            
            # Find the first spark cluster(s) that appear in this file
            if 'frame_idx' in file_data.columns:
                min_frame = file_data['frame_idx'].min()
                first_sparks = file_data[file_data['frame_idx'] == min_frame].copy()
            elif 'time_s' in file_data.columns:
                min_time = file_data['time_s'].min()
                first_sparks = file_data[file_data['time_s'] <= min_time + 10].copy()
            else:
                continue
            
            if len(first_sparks) < 1:
                continue
            
            if 'area' in first_sparks.columns:
                first_sparks = first_sparks.sort_values('area', ascending=False)
            
            use_sparks = first_sparks.head(min(min_sparks, len(first_sparks)) if min_sparks > 1 else len(first_sparks))
            
            # Calculate weighted centroid
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
            
            poke_locations.append({
                'filename': base_file,
                'poke_x': poke_x,
                'poke_y': poke_y,
                'n_sparks': len(use_sparks),
                'method': 'inferred_from_first_cluster'
            })
        
        return pd.DataFrame(poke_locations)


def calculate_distances_from_poke(tracks_df, poke_locations_df):
    """
    Calculate distance from poke for all sparks based on poke locations.
    
    Args:
        tracks_df: DataFrame with spark tracks
        poke_locations_df: DataFrame with poke locations (columns: filename, poke_x, poke_y)
    
    Returns:
        DataFrame with dist_from_poke_px calculated
    """
    tracks_df = tracks_df.copy()
    
    # Prepare base filename matching
    if 'filename' in tracks_df.columns:
        tracks_df['base_filename'] = tracks_df['filename'].str.replace(r' \(page \d+\)', '', regex=True)
    else:
        print("ERROR: 'filename' column not found in tracks_df")
        return tracks_df
    
    # Initialize distance column if it doesn't exist
    if 'dist_from_poke_px' not in tracks_df.columns:
        tracks_df['dist_from_poke_px'] = np.nan
    
    # Create a lookup dictionary for poke locations by base filename
    poke_lookup = {}
    for _, row in poke_locations_df.iterrows():
        base_file = row['filename'].replace(' (page \\d+)', '') if 'filename' in row else str(row.get('filename', ''))
        poke_lookup[base_file] = (row['poke_x'], row['poke_y'])
    
    # Also create lookup by full filename (in case it matches)
    for _, row in poke_locations_df.iterrows():
        filename = row.get('filename', '')
        poke_lookup[filename] = (row['poke_x'], row['poke_y'])
    
    # Calculate distances for each row
    distances = []
    matched_files = set()
    
    for idx, row in tracks_df.iterrows():
        base_file = row.get('base_filename', row.get('filename', ''))
        
        # Try to find poke location
        poke_coords = None
        if base_file in poke_lookup:
            poke_coords = poke_lookup[base_file]
            matched_files.add(base_file)
        elif 'filename' in row and row['filename'] in poke_lookup:
            poke_coords = poke_lookup[row['filename']]
            matched_files.add(row['filename'])
        
        if poke_coords and pd.notna(row['x']) and pd.notna(row['y']):
            dx = row['x'] - poke_coords[0]
            dy = row['y'] - poke_coords[1]
            distance = np.sqrt(dx**2 + dy**2)
            distances.append(distance)
        else:
            # Keep existing value if present, otherwise NaN
            existing_dist = row.get('dist_from_poke_px', np.nan)
            distances.append(existing_dist)
    
    tracks_df['dist_from_poke_px'] = distances
    
    # Report statistics
    valid_distances = tracks_df['dist_from_poke_px'].notna().sum()
    total_rows = len(tracks_df)
    print(f"\nDistance calculation complete:")
    print(f"  - Total spark events: {total_rows:,}")
    print(f"  - Events with valid distances: {valid_distances:,} ({100*valid_distances/total_rows:.1f}%)")
    print(f"  - Matched files: {len(matched_files)}")
    
    if valid_distances > 0:
        dist_stats = tracks_df['dist_from_poke_px'].describe()
        print(f"\nDistance statistics:")
        print(f"  - Mean: {dist_stats['mean']:.1f} px")
        print(f"  - Median: {dist_stats['50%']:.1f} px")
        print(f"  - Min: {dist_stats['min']:.1f} px")
        print(f"  - Max: {dist_stats['max']:.1f} px")
    
    return tracks_df


def main():
    parser = argparse.ArgumentParser(
        description='Add distance from poke calculations to spark_tracks.csv'
    )
    parser.add_argument('tracks_csv', help='Path to spark_tracks.csv')
    parser.add_argument('--output', '-o', help='Output CSV file (default: overwrite input)')
    parser.add_argument('--poke-csv', help='Path to poke_locations.csv (optional, will infer if not provided)')
    parser.add_argument('--chunk-size', type=int, default=100000,
                       help='Process CSV in chunks of this size (default: 100000)')
    
    args = parser.parse_args()
    
    tracks_path = Path(args.tracks_csv)
    if not tracks_path.exists():
        print(f"ERROR: File not found: {tracks_path}")
        sys.exit(1)
    
    print(f"Loading {tracks_path}...")
    
    # Load in chunks to handle large files
    print(f"Reading tracks CSV...")
    chunks = []
    for chunk in pd.read_csv(tracks_path, chunksize=args.chunk_size):
        chunks.append(chunk)
    
    if len(chunks) == 1:
        tracks_df = chunks[0]
    else:
        print(f"  → Concatenating {len(chunks)} chunks...")
        tracks_df = pd.concat(chunks, ignore_index=True)
    
    print(f"  → Loaded {len(tracks_df):,} spark events from {tracks_df['filename'].nunique()} files")
    
    # Get poke locations
    if args.poke_csv and Path(args.poke_csv).exists():
        print(f"\nLoading poke locations from {args.poke_csv}...")
        poke_locations_df = pd.read_csv(args.poke_csv)
        print(f"  → Loaded {len(poke_locations_df)} poke locations")
    else:
        print(f"\nInferring poke locations from early spark clusters...")
        poke_locations_df = infer_poke_from_early_sparks(tracks_df)
        print(f"  → Inferred {len(poke_locations_df)} poke locations")
    
    if len(poke_locations_df) == 0:
        print("\nERROR: No poke locations found!")
        sys.exit(1)
    
    # Calculate distances
    print(f"\nCalculating distances from poke locations...")
    tracks_df = calculate_distances_from_poke(tracks_df, poke_locations_df)
    
    # Save output
    output_path = Path(args.output) if args.output else tracks_path
    print(f"\nSaving to {output_path}...")
    tracks_df.to_csv(output_path, index=False)
    print(f"  → Saved {len(tracks_df):,} rows")
    
    print(f"\n✓ Complete!")


if __name__ == '__main__':
    main()


