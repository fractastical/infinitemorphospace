#!/usr/bin/env python3
"""
Quick test of cement gland detection on a small subset of data.
"""

import subprocess
import re
import sys
import os

# Configuration
TIFF_BASE = "/Users/jdietz/Documents/Levin/Embryos"
FPS = 1.0
POKE_FRAME = 0

# Test on more folders for better statistics
TEST_FOLDERS = list(range(1, 26))  # Folders 1-25 for comprehensive test

def run_parser_on_folder(folder_num):
    """Run parser on a single folder and capture output."""
    folder_path = os.path.join(TIFF_BASE, str(folder_num))
    
    if not os.path.exists(folder_path):
        return None, f"Folder {folder_num} not found"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    
    cmd = [
        sys.executable,
        os.path.join(repo_root, "wave-vector-analysis", "wave-vector-tiff-parser.py"),
        folder_path,
        str(POKE_FRAME),
        "--fps", str(FPS),
        "--csv", f"/tmp/test_folder_{folder_num}.csv"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout per folder
            cwd=repo_root
        )
        return result.stdout + result.stderr, None
    except subprocess.TimeoutExpired:
        return None, f"Timeout processing folder {folder_num}"
    except Exception as e:
        return None, f"Error: {str(e)}"

def parse_cement_gland_results(output):
    """Parse output to extract cement gland detection results."""
    results = {
        'found_end1': [],
        'found_end2': [],
        'not_found': [],
        'total_embryos': 0,
        'width_conflicts': []
    }
    
    # Pattern to match cement gland detection messages
    pattern_found_end1 = r'\[Embryo ([AB])\] ✓ Cement gland detected near end1'
    pattern_found_end2 = r'\[Embryo ([AB])\] ✓ Cement gland detected near end2'
    pattern_not_found = r'\[Embryo ([AB])\] ⚠ Cement gland not detected'
    pattern_width_conflict = r'→ Note: Width analysis suggests.*but cement gland'
    
    for line in output.split('\n'):
        match1 = re.search(pattern_found_end1, line)
        match2 = re.search(pattern_found_end2, line)
        match_not = re.search(pattern_not_found, line)
        match_conflict = re.search(pattern_width_conflict, line)
        
        if match1:
            results['found_end1'].append(match1.group(1))
            results['total_embryos'] += 1
        elif match2:
            results['found_end2'].append(match2.group(1))
            results['total_embryos'] += 1
        elif match_not:
            results['not_found'].append(match_not.group(1))
            results['total_embryos'] += 1
        elif match_conflict:
            results['width_conflicts'].append(line.strip())
    
    return results

def main():
    print("=" * 70)
    print("Quick Cement Gland Detection Test")
    print("=" * 70)
    print(f"Testing on folders: {TEST_FOLDERS}")
    print(f"TIFF base: {TIFF_BASE}")
    print()
    
    all_results = []
    total_found = 0
    total_not_found = 0
    total_embryos = 0
    total_conflicts = 0
    
    for folder_num in TEST_FOLDERS:
        print(f"Processing folder {folder_num}...", end=" ", flush=True)
        output, error = run_parser_on_folder(folder_num)
        
        if error:
            print(f"❌ {error}")
            continue
        
        results = parse_cement_gland_results(output)
        all_results.append((folder_num, results, output))
        
        found_count = len(results['found_end1']) + len(results['found_end2'])
        not_found_count = len(results['not_found'])
        conflicts = len(results['width_conflicts'])
        
        total_found += found_count
        total_not_found += not_found_count
        total_embryos += results['total_embryos']
        total_conflicts += conflicts
        
        if found_count > 0:
            print(f"✓ Found: {found_count}, Not found: {not_found_count}", end="")
            if conflicts > 0:
                print(f", Conflicts: {conflicts}")
            else:
                print()
        else:
            print(f"⚠ Not found: {not_found_count}")
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total embryos tested: {total_embryos}")
    if total_embryos > 0:
        print(f"Cement gland detected: {total_found} ({100*total_found/total_embryos:.1f}%)")
        print(f"Cement gland NOT detected: {total_not_found} ({100*total_not_found/total_embryos:.1f}%)")
        print(f"Width analysis conflicts: {total_conflicts} (cases where width suggested different head)")
    print()
    
    print("Detailed results by folder:")
    print("-" * 70)
    for folder_num, results, output in all_results:
        found_end1 = len(results['found_end1'])
        found_end2 = len(results['found_end2'])
        not_found = len(results['not_found'])
        total = results['total_embryos']
        conflicts = len(results['width_conflicts'])
        
        if total > 0:
            print(f"Folder {folder_num:2d}: Found={found_end1+found_end2}/{total} "
                  f"(end1: {found_end1}, end2: {found_end2}), Not found: {not_found}", end="")
            if conflicts > 0:
                print(f", Conflicts: {conflicts}")
            else:
                print()
    
    # Show conflicts if any
    if total_conflicts > 0:
        print()
        print("Width analysis conflicts (cement gland overrode width analysis):")
        print("-" * 70)
        for folder_num, results, output in all_results:
            if results['width_conflicts']:
                print(f"\nFolder {folder_num}:")
                for conflict in results['width_conflicts']:
                    print(f"  {conflict}")
    
    print()
    print("=" * 70)

if __name__ == "__main__":
    main()
