#!/usr/bin/env python3
"""
Analyze spatial matching hypotheses using inferred poke locations.

Hypotheses:
- Hypothesis 9: Embryo A/B shows a local calcium response in a similar region as the wound site
- Hypothesis 10: No local response when poked further posterior

This script:
1. Uses inferred poke locations to calculate distances
2. Analyzes spatial matching between poke location and response location
3. Compares responses based on poke location (anterior vs posterior)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from plot_poke_locations import infer_poke_from_early_sparks

def calculate_distances_from_inferred_pokes(tracks_df):
    """
    Calculate distances from inferred poke locations for all sparks.
    
    Returns:
        tracks_df with dist_from_poke_px calculated
    """
    print("Inferring poke locations from early spark clusters...")
    poke_locations_df = infer_poke_from_early_sparks(tracks_df)
    print(f"  → Found {len(poke_locations_df)} poke locations")
    
    if len(poke_locations_df) == 0:
        print("  WARNING: No poke locations found!")
        return tracks_df
    
    tracks_df = tracks_df.copy()
    
    # Prepare base filename matching
    if 'filename' in tracks_df.columns:
        tracks_df['base_filename'] = tracks_df['filename'].str.replace(r' \(page \d+\)', '', regex=True)
    else:
        print("ERROR: 'filename' column not found")
        return tracks_df
    
    # Initialize distance column
    if 'dist_from_poke_px' not in tracks_df.columns:
        tracks_df['dist_from_poke_px'] = np.nan
    
    # Create lookup dictionary
    poke_lookup = {}
    for _, row in poke_locations_df.iterrows():
        base_file = row['filename']
        poke_lookup[base_file] = (row['poke_x'], row['poke_y'])
    
    # Calculate distances
    distances = []
    matched_count = 0
    
    for idx, row in tracks_df.iterrows():
        base_file = row.get('base_filename', '')
        
        if base_file in poke_lookup and pd.notna(row['x']) and pd.notna(row['y']):
            poke_x, poke_y = poke_lookup[base_file]
            dx = row['x'] - poke_x
            dy = row['y'] - poke_y
            distance = np.sqrt(dx**2 + dy**2)
            distances.append(distance)
            matched_count += 1
        else:
            # Keep existing value if present
            existing_dist = row.get('dist_from_poke_px', np.nan)
            distances.append(existing_dist)
    
    tracks_df['dist_from_poke_px'] = distances
    
    print(f"  → Calculated distances for {matched_count:,} events")
    
    return tracks_df


def analyze_spatial_matching(tracks_df, clusters_df=None, output_dir='analysis_results'):
    """
    Analyze spatial matching between poke location and response location.
    
    Hypothesis: Embryo A/B shows a local calcium response in a similar region 
    as the wound site of embryo A/B.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("SPATIAL MATCHING ANALYSIS")
    print("="*80)
    
    # Calculate distances if not already present
    if 'dist_from_poke_px' not in tracks_df.columns or tracks_df['dist_from_poke_px'].notna().sum() == 0:
        print("\nCalculating distances from inferred poke locations...")
        tracks_df = calculate_distances_from_inferred_pokes(tracks_df)
    
    # Filter to post-poke events only
    post_poke = tracks_df[tracks_df['time_s'] > 0].copy()
    print(f"\nPost-poke events: {len(post_poke):,}")
    
    # Check data availability
    valid_dist = post_poke['dist_from_poke_px'].notna().sum()
    print(f"Events with valid distances: {valid_dist:,} ({100*valid_dist/len(post_poke):.1f}%)")
    
    if valid_dist == 0:
        print("\nERROR: No valid distance data! Cannot analyze spatial matching.")
        return
    
    # Filter to events with valid distances
    valid_data = post_poke[post_poke['dist_from_poke_px'].notna()].copy()
    
    # Group by embryo
    if 'embryo_id' in valid_data.columns:
        embryo_groups = valid_data.groupby('embryo_id')
        print(f"\nEvents by embryo:")
        for embryo_id, group in embryo_groups:
            print(f"  Embryo {embryo_id}: {len(group):,} events")
    else:
        print("\nWARNING: No embryo_id column - analyzing all events together")
        embryo_groups = [('All', valid_data)]
    
    # Create comprehensive plot
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Distance distribution (all embryos)
    ax1 = fig.add_subplot(gs[0, 0])
    distances = valid_data['dist_from_poke_px']
    # Explicitly define bins to avoid broadcasting issues
    dist_min, dist_max = distances.min(), distances.max()
    bins = np.linspace(dist_min, dist_max, 51)  # 51 edges = 50 bins
    ax1.hist(distances, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(distances.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {distances.median():.1f} px')
    ax1.axvline(distances.mean(), color='orange', linestyle='--', linewidth=2, label=f'Mean: {distances.mean():.1f} px')
    ax1.set_xlabel('Distance from Poke (pixels)', fontsize=11)
    ax1.set_ylabel('Number of Events', fontsize=11)
    ax1.set_title('Distance Distribution (All Events)', fontweight='bold', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Distance by embryo (if available)
    ax2 = fig.add_subplot(gs[0, 1])
    if 'embryo_id' in valid_data.columns:
        embryo_ids = valid_data['embryo_id'].dropna().unique()
        data_by_embryo = [valid_data[valid_data['embryo_id'] == eid]['dist_from_poke_px'].values 
                         for eid in embryo_ids]
        bp = ax2.boxplot(data_by_embryo, tick_labels=[f'Embryo {eid}' for eid in embryo_ids],
                        patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax2.set_ylabel('Distance from Poke (pixels)', fontsize=11)
        ax2.set_title('Distance by Embryo', fontweight='bold', fontsize=12)
        ax2.grid(alpha=0.3, axis='y')
    else:
        ax2.text(0.5, 0.5, 'No embryo_id data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Distance by Embryo (No Data)', fontweight='bold', fontsize=12)
    
    # 3. Distance over time
    ax3 = fig.add_subplot(gs[0, 2])
    time_max = valid_data['time_s'].max()
    time_bins = np.linspace(0, time_max, 51)  # 51 edges = 50 bins
    valid_data['time_bin'] = pd.cut(valid_data['time_s'], bins=time_bins, observed=False)
    time_grouped = valid_data.groupby('time_bin', observed=False)['dist_from_poke_px'].agg(['mean', 'std', 'count'])
    time_centers = [(interval.left + interval.right) / 2 for interval in time_grouped.index]
    ax3.plot(time_centers, time_grouped['mean'], 'o-', color='steelblue', label='Mean distance')
    ax3.fill_between(time_centers, 
                     time_grouped['mean'] - time_grouped['std'],
                     time_grouped['mean'] + time_grouped['std'],
                     alpha=0.3, color='steelblue', label='±1 std dev')
    ax3.set_xlabel('Time (seconds post-poke)', fontsize=11)
    ax3.set_ylabel('Mean Distance from Poke (pixels)', fontsize=11)
    ax3.set_title('Distance Over Time', fontweight='bold', fontsize=12)
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Spatial scatter: poke locations and early responses
    ax4 = fig.add_subplot(gs[1, 0])
    # Get early responses (first 5 seconds)
    early_responses = valid_data[(valid_data['time_s'] > 0) & (valid_data['time_s'] <= 5)].copy()
    if len(early_responses) > 0:
        # Sample for visualization if too many points
        if len(early_responses) > 5000:
            early_responses = early_responses.sample(n=5000, random_state=42)
        ax4.scatter(early_responses['x'], early_responses['y'], 
                   c=early_responses['dist_from_poke_px'], cmap='viridis',
                   s=10, alpha=0.5, label='Early responses')
        # Add poke locations
        poke_locs = infer_poke_from_early_sparks(tracks_df)
        if len(poke_locs) > 0:
            # Match poke locations to visible data
            for _, poke_row in poke_locs.iterrows():
                ax4.scatter(poke_row['poke_x'], poke_row['poke_y'], 
                           s=200, c='red', marker='X', 
                           edgecolors='black', linewidths=2, 
                           label='Poke locations', zorder=10)
        ax4.set_xlabel('X (pixels)', fontsize=11)
        ax4.set_ylabel('Y (pixels)', fontsize=11)
        ax4.set_title('Poke Locations and Early Responses', fontweight='bold', fontsize=12)
        ax4.legend()
        ax4.set_aspect('equal', adjustable='box')
    else:
        ax4.text(0.5, 0.5, 'No early response data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Poke Locations and Early Responses', fontweight='bold', fontsize=12)
    
    # 5. Local vs distant responses
    ax5 = fig.add_subplot(gs[1, 1])
    # Define "local" as within 50 pixels
    valid_data['is_local'] = valid_data['dist_from_poke_px'] <= 50
    local_responses = valid_data[valid_data['is_local']]
    distant_responses = valid_data[~valid_data['is_local']]
    
    categories = ['Local (≤50px)', 'Distant (>50px)']
    counts = [len(local_responses), len(distant_responses)]
    colors = ['coral', 'steelblue']
    bars = ax5.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    ax5.set_ylabel('Number of Events', fontsize=11)
    ax5.set_title('Local vs Distant Responses', fontweight='bold', fontsize=12)
    ax5.grid(alpha=0.3, axis='y')
    
    # Add count labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # 6. Activity by distance bins
    ax6 = fig.add_subplot(gs[1, 2])
    distance_bins = [0, 25, 50, 100, 200, 500, np.inf]
    valid_data['dist_bin'] = pd.cut(valid_data['dist_from_poke_px'], bins=distance_bins,
                                    labels=['0-25', '25-50', '50-100', '100-200', '200-500', '500+'])
    if 'area' in valid_data.columns:
        activity_by_dist = valid_data.groupby('dist_bin', observed=False)['area'].agg(['sum', 'mean', 'count'])
        x_pos = np.arange(len(activity_by_dist))
        ax6.bar(x_pos, activity_by_dist['sum'], alpha=0.7, color='steelblue', edgecolor='black')
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(activity_by_dist.index, rotation=45, ha='right')
        ax6.set_ylabel('Total Activity (pixels²)', fontsize=11)
        ax6.set_title('Activity by Distance from Poke', fontweight='bold', fontsize=12)
        ax6.grid(alpha=0.3, axis='y')
    else:
        ax6.text(0.5, 0.5, 'No area data', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Activity by Distance (No Data)', fontweight='bold', fontsize=12)
    
    # 7. Distance vs AP position (if available)
    ax7 = fig.add_subplot(gs[2, 0])
    if 'ap_norm' in valid_data.columns:
        ap_valid = valid_data[valid_data['ap_norm'].notna()]
        if len(ap_valid) > 0:
            ax7.scatter(ap_valid['ap_norm'], ap_valid['dist_from_poke_px'],
                       alpha=0.3, s=10, color='steelblue')
            ax7.set_xlabel('AP Position (normalized, 0=head, 1=tail)', fontsize=11)
            ax7.set_ylabel('Distance from Poke (pixels)', fontsize=11)
            ax7.set_title('Distance vs AP Position', fontweight='bold', fontsize=12)
            ax7.grid(alpha=0.3)
    else:
        ax7.text(0.5, 0.5, 'No AP position data', ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title('Distance vs AP Position (No Data)', fontweight='bold', fontsize=12)
    
    # 8. Inter-embryo spatial matching
    ax8 = fig.add_subplot(gs[2, 1])
    if 'embryo_id' in valid_data.columns:
        # Compare distances between embryos
        embryo_ids = valid_data['embryo_id'].dropna().unique()
        if len(embryo_ids) >= 2:
            data_to_plot = []
            labels = []
            for eid in sorted(embryo_ids):
                eid_data = valid_data[valid_data['embryo_id'] == eid]['dist_from_poke_px'].values
                if len(eid_data) > 0:
                    data_to_plot.append(eid_data)
                    labels.append(f'Embryo {eid}\n(n={len(eid_data):,})')
            
            if len(data_to_plot) > 0:
                bp = ax8.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightgreen')
                ax8.set_ylabel('Distance from Poke (pixels)', fontsize=11)
                ax8.set_title('Spatial Matching: Embryo Comparison', fontweight='bold', fontsize=12)
                ax8.grid(alpha=0.3, axis='y')
            else:
                ax8.text(0.5, 0.5, 'No data for comparison', ha='center', va='center', transform=ax8.transAxes)
        else:
            ax8.text(0.5, 0.5, 'Need 2+ embryos for comparison', ha='center', va='center', transform=ax8.transAxes)
    else:
        ax8.text(0.5, 0.5, 'No embryo_id data', ha='center', va='center', transform=ax8.transAxes)
    
    # 9. Summary statistics text
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    stats_text = "SPATIAL MATCHING STATISTICS\n" + "="*40 + "\n\n"
    stats_text += f"Total events analyzed: {len(valid_data):,}\n"
    stats_text += f"\nDistance from poke:\n"
    stats_text += f"  Mean: {distances.mean():.1f} px\n"
    stats_text += f"  Median: {distances.median():.1f} px\n"
    stats_text += f"  Std dev: {distances.std():.1f} px\n"
    stats_text += f"  Min: {distances.min():.1f} px\n"
    stats_text += f"  Max: {distances.max():.1f} px\n"
    
    if len(local_responses) > 0:
        stats_text += f"\nLocal responses (≤50px): {len(local_responses):,} ({100*len(local_responses)/len(valid_data):.1f}%)\n"
    if len(distant_responses) > 0:
        stats_text += f"Distant responses (>50px): {len(distant_responses):,} ({100*len(distant_responses)/len(valid_data):.1f}%)\n"
    
    if 'embryo_id' in valid_data.columns:
        stats_text += f"\nBy embryo:\n"
        for eid in sorted(valid_data['embryo_id'].dropna().unique()):
            eid_data = valid_data[valid_data['embryo_id'] == eid]
            eid_mean = eid_data['dist_from_poke_px'].mean()
            stats_text += f"  Embryo {eid}: {len(eid_data):,} events, mean dist: {eid_mean:.1f} px\n"
    
    ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Spatial Matching Analysis: Local vs Distant Responses', 
                fontsize=16, fontweight='bold', y=0.995)
    
    output_path = output_dir / 'spatial_matching_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to {output_path}")
    plt.close()
    
    # Generate summary
    print(f"\n{'='*80}")
    print("SPATIAL MATCHING SUMMARY")
    print(f"{'='*80}")
    print(f"Total events analyzed: {len(valid_data):,}")
    print(f"Mean distance from poke: {distances.mean():.1f} px")
    print(f"Median distance from poke: {distances.median():.1f} px")
    print(f"Local responses (≤50px): {len(local_responses):,} ({100*len(local_responses)/len(valid_data):.1f}%)")
    print(f"Distant responses (>50px): {len(distant_responses):,} ({100*len(distant_responses)/len(valid_data):.1f}%)")


def analyze_posterior_poke_effect(tracks_df, clusters_df=None, output_dir='analysis_results'):
    """
    Analyze effect of posterior poke location.
    
    Hypothesis: Embryo A/B does not show a local response in a similar region 
    when poked further posterior.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("POSTERIOR POKE EFFECT ANALYSIS")
    print("="*80)
    
    # Get poke locations with AP positions
    print("\nInferring poke locations...")
    poke_locs_df = infer_poke_from_early_sparks(tracks_df)
    
    if len(poke_locs_df) == 0:
        print("ERROR: No poke locations found!")
        return
    
    # Match poke locations to tracks to get AP positions
    # We need to find the AP position of each poke location
    tracks_df = tracks_df.copy()
    if 'filename' in tracks_df.columns:
        tracks_df['base_filename'] = tracks_df['filename'].str.replace(r' \(page \d+\)', '', regex=True)
    
    # For each poke location, find nearby tracks to determine AP position
    poke_ap_positions = {}
    for _, poke_row in poke_locs_df.iterrows():
        base_file = poke_row['filename']
        file_tracks = tracks_df[tracks_df['base_filename'] == base_file].copy()
        
        if len(file_tracks) == 0:
            continue
        
        # Find tracks near poke location
        file_tracks['dist_to_poke'] = np.sqrt(
            (file_tracks['x'] - poke_row['poke_x'])**2 + 
            (file_tracks['y'] - poke_row['poke_y'])**2
        )
        
        # Get AP position from nearest tracks
        if 'ap_norm' in file_tracks.columns:
            nearest_tracks = file_tracks.nsmallest(10, 'dist_to_poke')
            ap_values = nearest_tracks['ap_norm'].dropna()
            if len(ap_values) > 0:
                poke_ap_positions[base_file] = ap_values.mean()
    
    print(f"  → Found AP positions for {len(poke_ap_positions)} poke locations")
    
    # Classify pokes as anterior vs posterior
    posterior_threshold = 0.7  # ap_norm >= 0.7 is posterior
    anterior_pokes = [f for f, ap in poke_ap_positions.items() if ap < posterior_threshold]
    posterior_pokes = [f for f, ap in poke_ap_positions.items() if ap >= posterior_threshold]
    
    print(f"  → Anterior pokes: {len(anterior_pokes)}")
    print(f"  → Posterior pokes: {len(posterior_pokes)}")
    
    # Analyze responses
    post_poke = tracks_df[tracks_df['time_s'] > 0].copy()
    
    # Add poke classification to tracks
    post_poke['poke_type'] = 'unknown'
    if 'base_filename' in post_poke.columns:
        post_poke.loc[post_poke['base_filename'].isin(anterior_pokes), 'poke_type'] = 'anterior'
        post_poke.loc[post_poke['base_filename'].isin(posterior_pokes), 'poke_type'] = 'posterior'
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Activity comparison
    ax = axes[0, 0]
    if 'poke_type' in post_poke.columns and 'area' in post_poke.columns:
        anterior_data = post_poke[post_poke['poke_type'] == 'anterior']
        posterior_data = post_poke[post_poke['poke_type'] == 'posterior']
        
        if len(anterior_data) > 0 and len(posterior_data) > 0:
            categories = ['Anterior\nPoke', 'Posterior\nPoke']
            activity = [
                anterior_data['area'].sum(),
                posterior_data['area'].sum()
            ]
            counts = [len(anterior_data), len(posterior_data)]
            
            ax2 = ax.twinx()
            bars1 = ax.bar(categories, activity, alpha=0.7, color='steelblue', label='Total Activity')
            bars2 = ax2.bar(categories, counts, alpha=0.7, color='coral', width=0.6, label='Event Count')
            
            ax.set_ylabel('Total Activity (pixels²)', fontsize=11, color='steelblue')
            ax2.set_ylabel('Number of Events', fontsize=11, color='coral')
            ax.set_title('Response to Anterior vs Posterior Pokes', fontweight='bold', fontsize=12)
            ax.tick_params(axis='y', labelcolor='steelblue')
            ax2.tick_params(axis='y', labelcolor='coral')
            ax.grid(alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
    
    # 2. Time course comparison
    ax = axes[0, 1]
    if 'poke_type' in post_poke.columns:
        for poke_type in ['anterior', 'posterior']:
            type_data = post_poke[post_poke['poke_type'] == poke_type]
            if len(type_data) > 0 and 'time_s' in type_data.columns and 'area' in type_data.columns:
                time_bins = np.linspace(0, type_data['time_s'].max(), 30)
                type_data['time_bin'] = pd.cut(type_data['time_s'], bins=time_bins)
                grouped = type_data.groupby('time_bin')['area'].sum()
                time_centers = [(interval.left + interval.right) / 2 for interval in grouped.index]
                label = 'Anterior' if poke_type == 'anterior' else 'Posterior'
                ax.plot(time_centers, grouped.values, 'o-', label=label, linewidth=2)
        ax.set_xlabel('Time (seconds post-poke)', fontsize=11)
        ax.set_ylabel('Activity (pixels²)', fontsize=11)
        ax.set_title('Time Course: Anterior vs Posterior', fontweight='bold', fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
    
    # 3. Spatial distribution
    ax = axes[1, 0]
    if 'poke_type' in post_poke.columns:
        for poke_type in ['anterior', 'posterior']:
            type_data = post_poke[post_poke['poke_type'] == poke_type]
            if len(type_data) > 0:
                # Sample for visualization
                sample_size = min(1000, len(type_data))
                type_data_sample = type_data.sample(n=sample_size, random_state=42) if len(type_data) > sample_size else type_data
                label = 'Anterior poke responses' if poke_type == 'anterior' else 'Posterior poke responses'
                ax.scatter(type_data_sample['x'], type_data_sample['y'], 
                          alpha=0.3, s=10, label=label)
        ax.set_xlabel('X (pixels)', fontsize=11)
        ax.set_ylabel('Y (pixels)', fontsize=11)
        ax.set_title('Spatial Distribution of Responses', fontweight='bold', fontsize=12)
        ax.legend()
        ax.set_aspect('equal', adjustable='box')
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
    
    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = "POSTERIOR POKE EFFECT\n" + "="*30 + "\n\n"
    stats_text += f"Anterior pokes: {len(anterior_pokes)}\n"
    stats_text += f"Posterior pokes: {len(posterior_pokes)}\n\n"
    
    if 'poke_type' in post_poke.columns:
        for poke_type in ['anterior', 'posterior']:
            type_data = post_poke[post_poke['poke_type'] == poke_type]
            if len(type_data) > 0:
                label = 'Anterior' if poke_type == 'anterior' else 'Posterior'
                stats_text += f"{label} poke responses:\n"
                stats_text += f"  Events: {len(type_data):,}\n"
                if 'area' in type_data.columns:
                    stats_text += f"  Total activity: {type_data['area'].sum():,.0f} px²\n"
                    stats_text += f"  Mean activity: {type_data['area'].mean():.1f} px²\n"
                stats_text += "\n"
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Posterior Poke Effect: Local Response Analysis', 
                fontsize=16, fontweight='bold', y=0.995)
    
    output_path = output_dir / 'posterior_poke_effect.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze spatial matching hypotheses using inferred poke locations'
    )
    parser.add_argument('tracks_csv', help='Path to spark_tracks.csv')
    parser.add_argument('--clusters-csv', help='Path to vector_clusters.csv (optional)')
    parser.add_argument('--output-dir', default='analysis_results',
                       help='Output directory for plots (default: analysis_results)')
    parser.add_argument('--analysis', choices=['spatial', 'posterior', 'both'],
                       default='both', help='Which analysis to run')
    
    args = parser.parse_args()
    
    print(f"Loading {args.tracks_csv}...")
    tracks_df = pd.read_csv(args.tracks_csv)
    print(f"  → Loaded {len(tracks_df):,} track states")
    
    clusters_df = None
    if args.clusters_csv:
        print(f"\nLoading {args.clusters_csv}...")
        clusters_df = pd.read_csv(args.clusters_csv)
        print(f"  → Loaded {len(clusters_df):,} clusters")
    
    if args.analysis in ['spatial', 'both']:
        analyze_spatial_matching(tracks_df, clusters_df, args.output_dir)
    
    if args.analysis in ['posterior', 'both']:
        analyze_posterior_poke_effect(tracks_df, clusters_df, args.output_dir)
    
    print("\n✓ Analysis complete!")


if __name__ == '__main__':
    main()


