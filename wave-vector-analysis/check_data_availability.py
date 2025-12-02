#!/usr/bin/env python3
"""
Analyze data availability for testing experimental hypotheses.

This script checks whether the current spark_tracks.csv and vector_clusters.csv
contain sufficient data to test each experimental hypothesis.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path


# Define required data fields for each hypothesis
HYPOTHESIS_REQUIREMENTS = {
    "1. Presence of Calcium Activity": {
        "required_fields": ["time_s", "area", "embryo_id"],
        "optional_fields": [],
        "required_conditions": ["Pre-poke data (time < 0)", "Post-poke data (time > 0)"],
        "can_test": True,
        "notes": "Can compare activity levels if both pre and post poke data exist"
    },
    "2. Distance Effect (Contact vs Non-contact)": {
        "required_fields": ["time_s", "area", "embryo_id"],
        "optional_fields": [],
        "required_conditions": ["Data from two conditions (contact and non-contact)"],
        "can_test": False,
        "notes": "Requires data from multiple experimental conditions - need separate CSV files"
    },
    "3. Wave Directionality Within Embryo": {
        "required_fields": ["angle_deg", "embryo_id", "speed"],
        "optional_fields": ["mean_angle_deg", "angle_dispersion_deg"],
        "required_conditions": ["Events in specific embryo", "Angle data available"],
        "can_test": True,
        "notes": "Can test if angle_deg column has valid data"
    },
    "4. Wave Directionality Between Embryos": {
        "required_fields": ["angle_deg", "embryo_id", "time_s", "area"],
        "optional_fields": ["mean_angle_deg", "mean_speed_px_per_s"],
        "required_conditions": ["Events in embryo B", "Wave propagation data"],
        "can_test": True,
        "notes": "Can test if events propagate from A to B"
    },
    "5. Posterior Damage Effect": {
        "required_fields": ["embryo_id", "area", "time_s"],
        "optional_fields": ["ap_norm", "dist_from_poke_px"],
        "required_conditions": ["Poke location in posterior region", "Events in embryo B"],
        "can_test": "Partial",
        "notes": "Can test if AP position or distance data available to identify posterior pokes"
    },
    "6. Spatial Patterning": {
        "required_fields": ["x", "y", "area", "time_s"],
        "optional_fields": [],
        "required_conditions": ["Spatial coordinates", "Temporal data"],
        "can_test": True,
        "notes": "Can create spatial heatmaps and pattern analysis"
    },
    "7. Local Tail Response": {
        "required_fields": ["ap_norm", "area", "time_s", "speed"],
        "optional_fields": ["mean_speed_px_per_s"],
        "required_conditions": ["AP position data", "Tail region events"],
        "can_test": True,
        "notes": "Can test if ap_norm column has valid data for tail region"
    },
    "8. Age-Dependent Localization": {
        "required_fields": ["ap_norm", "x", "y"],
        "optional_fields": [],
        "required_conditions": ["Data from different embryo ages/stages", "Spatial spread data"],
        "can_test": False,
        "notes": "Requires data from multiple time points or embryo stages - need metadata"
    },
    "Spatial Matching": {
        "required_fields": ["x", "y", "dist_from_poke_px"],
        "optional_fields": ["ap_norm"],
        "required_conditions": ["Poke coordinates", "Response locations"],
        "can_test": "Partial",
        "notes": "Can test if dist_from_poke_px or poke coordinates available"
    },
    "Contraction": {
        "required_fields": [],
        "optional_fields": [],
        "required_conditions": ["Contraction detection data"],
        "can_test": False,
        "notes": "Requires separate contraction analysis - not in current pipeline"
    },
    "Wound Memory - Increased Activity": {
        "required_fields": ["time_s", "area"],
        "optional_fields": ["x", "y"],
        "required_conditions": ["Data from healed wound locations", "Time course data"],
        "can_test": "Partial",
        "notes": "Would need to identify healed wound locations from metadata or coordinates"
    },
    "Wound Memory - Local Response": {
        "required_fields": ["x", "y", "time_s", "area"],
        "optional_fields": ["ap_norm"],
        "required_conditions": ["Healed wound coordinates", "Response timing"],
        "can_test": "Partial",
        "notes": "Would need healed wound location data to match responses"
    }
}


def check_data_availability(tracks_csv, clusters_csv=None):
    """
    Check what data is available and which hypotheses can be tested.
    
    Returns:
        Dictionary with analysis results for each hypothesis
    """
    print(f"Loading {tracks_csv}...")
    tracks_df = pd.read_csv(tracks_csv)
    print(f"Loaded {len(tracks_df)} track states\n")
    
    clusters_df = None
    if clusters_csv:
        print(f"Loading {clusters_csv}...")
        clusters_df = pd.read_csv(clusters_csv)
        print(f"Loaded {len(clusters_df)} clusters\n")
    
    # Get available columns
    tracks_columns = set(tracks_df.columns)
    clusters_columns = set(clusters_df.columns) if clusters_df is not None else set()
    all_columns = tracks_columns | clusters_columns
    
    # Analyze data quality
    data_quality = {
        "tracks_columns": sorted(tracks_columns),
        "clusters_columns": sorted(clusters_columns),
        "missing_values": {},
        "data_ranges": {},
        "data_counts": {}
    }
    
    # Check for missing values in key columns
    key_columns = ["time_s", "area", "embryo_id", "angle_deg", "speed", "ap_norm", "dist_from_poke_px"]
    for col in key_columns:
        if col in tracks_df.columns:
            missing = tracks_df[col].isna().sum()
            total = len(tracks_df)
            data_quality["missing_values"][col] = {
                "missing": missing,
                "total": total,
                "percent_missing": 100 * missing / total if total > 0 else 0
            }
            
            if tracks_df[col].dtype in [np.float64, np.int64]:
                data_quality["data_ranges"][col] = {
                    "min": tracks_df[col].min() if not tracks_df[col].isna().all() else None,
                    "max": tracks_df[col].max() if not tracks_df[col].isna().all() else None,
                    "mean": tracks_df[col].mean() if not tracks_df[col].isna().all() else None
                }
    
    # Check time ranges
    if "time_s" in tracks_df.columns:
        time_range = tracks_df["time_s"].agg(["min", "max"])
        data_quality["data_ranges"]["time_s"] = {
            "min": time_range["min"],
            "max": time_range["max"],
            "has_pre_poke": (tracks_df["time_s"] < 0).any(),
            "has_post_poke": (tracks_df["time_s"] > 0).any(),
            "has_poke_time": (tracks_df["time_s"] == 0).any()
        }
    
    # Check embryo IDs
    if "embryo_id" in tracks_df.columns:
        embryo_ids = tracks_df["embryo_id"].dropna().unique()
        data_quality["data_counts"]["embryo_ids"] = sorted([str(e) for e in embryo_ids])
        for eid in embryo_ids:
            count = (tracks_df["embryo_id"] == eid).sum()
            data_quality["data_counts"][f"embryo_{eid}_count"] = count
    
    # Analyze each hypothesis
    results = {}
    for hypothesis, requirements in HYPOTHESIS_REQUIREMENTS.items():
        result = {
            "hypothesis": hypothesis,
            "required_fields": requirements["required_fields"],
            "available_fields": [],
            "missing_fields": [],
            "can_test": requirements.get("can_test", False),
            "confidence": "Unknown",
            "notes": requirements.get("notes", ""),
            "data_checks": {}
        }
        
        # Check field availability
        for field in requirements["required_fields"]:
            if field in all_columns:
                result["available_fields"].append(field)
                
                # Check data quality for this field
                if field in tracks_df.columns:
                    non_null = tracks_df[field].notna().sum()
                    result["data_checks"][field] = {
                        "available": True,
                        "non_null_count": int(non_null),
                        "total_count": len(tracks_df),
                        "percent_valid": 100 * non_null / len(tracks_df) if len(tracks_df) > 0 else 0
                    }
                elif clusters_df is not None and field in clusters_df.columns:
                    non_null = clusters_df[field].notna().sum()
                    result["data_checks"][field] = {
                        "available": True,
                        "non_null_count": int(non_null),
                        "total_count": len(clusters_df),
                        "percent_valid": 100 * non_null / len(clusters_df) if len(clusters_df) > 0 else 0
                    }
            else:
                result["missing_fields"].append(field)
        
        # Determine confidence level
        if len(result["missing_fields"]) == 0:
            if result["can_test"] is True:
                result["confidence"] = "High"
            elif result["can_test"] == "Partial":
                result["confidence"] = "Medium"
            else:
                result["confidence"] = "Low - requires additional data"
        else:
            result["confidence"] = f"Missing fields: {', '.join(result['missing_fields'])}"
            result["can_test"] = False
        
        # Special checks for specific hypotheses
        if hypothesis == "1. Presence of Calcium Activity":
            if "time_s" in tracks_df.columns:
                has_pre = (tracks_df["time_s"] < 0).any()
                has_post = (tracks_df["time_s"] > 0).any()
                if has_pre and has_post:
                    result["confidence"] = "High - can compare pre/post"
                elif has_post:
                    result["confidence"] = "Medium - only post-poke data available"
                    result["notes"] += " Warning: No pre-poke baseline data"
        
        if hypothesis == "5. Posterior Damage Effect":
            if "ap_norm" in tracks_df.columns or "dist_from_poke_px" in tracks_df.columns:
                result["confidence"] = "Medium - can identify posterior region if poke location known"
            else:
                result["confidence"] = "Low - need AP position or distance data"
        
        if hypothesis == "Spatial Matching":
            if "dist_from_poke_px" in tracks_df.columns:
                valid_dist = tracks_df["dist_from_poke_px"].notna().sum()
                if valid_dist > 0:
                    result["confidence"] = "High - distance data available"
                else:
                    result["confidence"] = "Low - distance column exists but is empty"
            elif "x" in tracks_df.columns and "y" in tracks_df.columns:
                result["confidence"] = "Medium - can calculate distances if poke coordinates provided"
            else:
                result["confidence"] = "Low - need spatial coordinates"
        
        results[hypothesis] = result
    
    return {
        "data_quality": data_quality,
        "hypothesis_results": results,
        "summary": {
            "total_hypotheses": len(HYPOTHESIS_REQUIREMENTS),
            "can_test_fully": sum(1 for r in results.values() if r["can_test"] is True),
            "can_test_partially": sum(1 for r in results.values() if r["can_test"] == "Partial"),
            "cannot_test": sum(1 for r in results.values() if r["can_test"] is False)
        }
    }


def print_report(analysis_results):
    """Print a formatted report of data availability."""
    print("=" * 80)
    print("DATA AVAILABILITY ANALYSIS REPORT")
    print("=" * 80)
    print()
    
    # Summary
    summary = analysis_results["summary"]
    print(f"SUMMARY:")
    print(f"  Total hypotheses: {summary['total_hypotheses']}")
    print(f"  Can test fully: {summary['can_test_fully']}")
    print(f"  Can test partially: {summary['can_test_partially']}")
    print(f"  Cannot test: {summary['cannot_test']}")
    print()
    
    # Data quality
    print("=" * 80)
    print("DATA QUALITY OVERVIEW")
    print("=" * 80)
    
    dq = analysis_results["data_quality"]
    
    print(f"\nAvailable columns in spark_tracks.csv: {len(dq['tracks_columns'])}")
    print(f"  {', '.join(dq['tracks_columns'])}")
    
    if dq['clusters_columns']:
        print(f"\nAvailable columns in vector_clusters.csv: {len(dq['clusters_columns'])}")
        print(f"  {', '.join(dq['clusters_columns'])}")
    
    print("\nMissing Data:")
    for col, info in dq['missing_values'].items():
        if info['percent_missing'] > 0:
            print(f"  {col}: {info['missing']}/{info['total']} ({info['percent_missing']:.1f}% missing)")
    
    if 'time_s' in dq['data_ranges']:
        tr = dq['data_ranges']['time_s']
        print(f"\nTime Range: {tr['min']:.2f} to {tr['max']:.2f} seconds")
        print(f"  Has pre-poke data: {tr.get('has_pre_poke', False)}")
        print(f"  Has post-poke data: {tr.get('has_post_poke', False)}")
        print(f"  Has poke time (t=0): {tr.get('has_poke_time', False)}")
    
    if 'embryo_id' in dq.get('data_counts', {}):
        print(f"\nEmbryos in dataset: {', '.join(dq['data_counts'].get('embryo_ids', []))}")
        for key, count in dq['data_counts'].items():
            if key.startswith('embryo_'):
                print(f"  {key}: {count} events")
    
    # Hypothesis analysis
    print("\n" + "=" * 80)
    print("HYPOTHESIS-BY-HYPOTHESIS ANALYSIS")
    print("=" * 80)
    print()
    
    for hypothesis, result in analysis_results["hypothesis_results"].items():
        print(f"{hypothesis}")
        print(f"  Status: {'✓ CAN TEST' if result['can_test'] else '✗ CANNOT TEST'}")
        print(f"  Confidence: {result['confidence']}")
        
        if result['available_fields']:
            print(f"  Available fields: {', '.join(result['available_fields'])}")
        
        if result['missing_fields']:
            print(f"  Missing fields: {', '.join(result['missing_fields'])}")
        
        # Show data quality for key fields
        if result['data_checks']:
            print(f"  Data quality:")
            for field, check in result['data_checks'].items():
                if check['percent_valid'] < 100:
                    print(f"    {field}: {check['percent_valid']:.1f}% valid ({check['non_null_count']}/{check['total_count']})")
        
        if result['notes']:
            print(f"  Notes: {result['notes']}")
        
        print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Check data availability for experimental hypotheses')
    parser.add_argument('tracks_csv', help='Path to spark_tracks.csv')
    parser.add_argument('--clusters-csv', help='Path to vector_clusters.csv (optional)')
    parser.add_argument('--output', help='Save report to file (JSON format)')
    
    args = parser.parse_args()
    
    # Run analysis
    results = check_data_availability(args.tracks_csv, args.clusters_csv)
    
    # Print report
    print_report(results)
    
    # Save if requested
    if args.output:
        import json
        # Convert numpy types to native Python types for JSON
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj
        
        results_json = convert_to_native(results)
        with open(args.output, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"\nReport saved to {args.output}")


if __name__ == '__main__':
    main()

