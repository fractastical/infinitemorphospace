#!/usr/bin/env python3
"""
Generate a report comparing current parser thresholding vs. permissive thresholding
to show how much of each TIFF file qualifies as embryo body.
"""

import cv2
import numpy as np
import tifffile as tiff
from pathlib import Path
import csv
import sys

def current_parser_threshold(gray):
    """Replicate the current parser's thresholding logic."""
    blur = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    flat_intensities = blur.flatten()
    
    background_percentile = 10
    background_threshold = np.percentile(flat_intensities, background_percentile)
    
    median_intensity = np.median(flat_intensities)
    p25 = np.percentile(flat_intensities, 25)
    mean_intensity = flat_intensities.mean()
    
    # Current parser logic
    embryo_threshold = max(median_intensity * 0.7, p25, mean_intensity * 0.8, background_threshold * 1.2)
    
    mask = (blur >= embryo_threshold).astype(np.uint8) * 255
    
    # Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return mask, embryo_threshold, background_threshold


def permissive_multi_method(gray, percentile_low=5):
    """Replicate the multi method from create_embryo_masks.py."""
    blur = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    flat_intensities = blur.flatten()
    
    background_percentile = 10
    background_threshold = np.percentile(flat_intensities, background_percentile)
    
    median_intensity = np.median(flat_intensities)
    p25 = np.percentile(flat_intensities, 25)
    
    # Multiple thresholds
    percentile_threshold = np.percentile(flat_intensities, percentile_low)
    adaptive_threshold = max(median_intensity * 0.5, p25 * 0.7, background_threshold * 1.2)
    
    mask1 = (blur >= percentile_threshold).astype(np.uint8) * 255
    mask2 = (blur >= adaptive_threshold).astype(np.uint8) * 255
    
    # Otsu
    if gray.dtype == np.uint16:
        gray_8bit = (gray.astype(np.float32) / 65535.0 * 255).astype(np.uint8)
    else:
        gray_8bit = gray.astype(np.uint8)
    _, mask3 = cv2.threshold(gray_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Union of all masks
    mask = cv2.bitwise_or(mask1, cv2.bitwise_or(mask2, mask3))
    
    # Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return mask, percentile_threshold, adaptive_threshold, background_threshold


def process_file(tiff_path, frame_idx=0):
    """Process a single file and return statistics."""
    try:
        with tiff.TiffFile(tiff_path) as tif:
            raw_data = tif.pages[frame_idx].asarray()
    except Exception as e:
        return None
    
    # Convert to grayscale
    if raw_data.ndim == 3:
        if raw_data.shape[2] == 3:
            gray = cv2.cvtColor(raw_data, cv2.COLOR_RGB2GRAY)
        else:
            gray = raw_data[:, :, 0]
    else:
        gray = raw_data
    
    h, w = gray.shape
    total_pixels = h * w
    
    # Current parser
    mask_current, thresh_current, bg_current = current_parser_threshold(gray)
    area_current = np.sum(mask_current > 0)
    coverage_current = (area_current / total_pixels) * 100
    
    # Permissive multi
    mask_permissive, thresh_perc, thresh_adapt, bg_permissive = permissive_multi_method(gray, percentile_low=5)
    area_permissive = np.sum(mask_permissive > 0)
    coverage_permissive = (area_permissive / total_pixels) * 100
    
    diff_coverage = coverage_permissive - coverage_current
    
    return {
        'filename': Path(tiff_path).name,
        'path': str(tiff_path),
        'width': w,
        'height': h,
        'total_pixels': total_pixels,
        'current_coverage': coverage_current,
        'permissive_coverage': coverage_permissive,
        'difference': diff_coverage,
        'current_area': area_current,
        'permissive_area': area_permissive,
        'current_threshold': thresh_current,
        'permissive_percentile_threshold': thresh_perc,
        'permissive_adaptive_threshold': thresh_adapt,
        'background_threshold': bg_current,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_coverage_report.py <tiff_directory> [output_csv]")
        print("Example: python generate_coverage_report.py /path/to/tiffs coverage_report.csv")
        sys.exit(1)
    
    tiff_dir = Path(sys.argv[1])
    output_csv = sys.argv[2] if len(sys.argv) > 2 else 'embryo_coverage_comparison.csv'
    
    if not tiff_dir.exists():
        print(f"✗ Error: Directory does not exist: {tiff_dir}")
        sys.exit(1)
    
    # Find all TIFF files
    tiff_files = list(tiff_dir.rglob('*.tif')) + list(tiff_dir.rglob('*.tiff'))
    
    if not tiff_files:
        print(f"✗ No TIFF files found in {tiff_dir}")
        sys.exit(1)
    
    print(f"Found {len(tiff_files)} TIFF files")
    print("Processing...")
    
    results = []
    for i, tiff_file in enumerate(sorted(tiff_files)):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(tiff_files)}...")
        
        result = process_file(tiff_file, frame_idx=0)
        if result:
            results.append(result)
    
    print(f"\nProcessed {len(results)} files")
    
    # Write CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'File', 'Path', 'Width', 'Height', 'Total_Pixels',
            'Current_Coverage_%', 'Permissive_Coverage_%', 'Difference_%',
            'Current_Area', 'Permissive_Area',
            'Current_Threshold', 'Permissive_Percentile_Threshold', 'Permissive_Adaptive_Threshold', 'Background_Threshold'
        ])
        
        for r in results:
            writer.writerow([
                r['filename'], r['path'], r['width'], r['height'], r['total_pixels'],
                f"{r['current_coverage']:.2f}", f"{r['permissive_coverage']:.2f}", f"{r['difference']:.2f}",
                r['current_area'], r['permissive_area'],
                f"{r['current_threshold']:.2f}", f"{r['permissive_percentile_threshold']:.2f}",
                f"{r['permissive_adaptive_threshold']:.2f}", f"{r['background_threshold']:.2f}"
            ])
    
    # Print summary
    if results:
        current_avg = np.mean([r['current_coverage'] for r in results])
        permissive_avg = np.mean([r['permissive_coverage'] for r in results])
        diff_avg = np.mean([r['difference'] for r in results])
        
        print(f"\n{'='*80}")
        print("SUMMARY STATISTICS")
        print(f"{'='*80}")
        print(f"Current Parser Method:")
        print(f"  Average coverage: {current_avg:.2f}%")
        print(f"  Median coverage:  {np.median([r['current_coverage'] for r in results]):.2f}%")
        print(f"  Min coverage:     {np.min([r['current_coverage'] for r in results]):.2f}%")
        print(f"  Max coverage:     {np.max([r['current_coverage'] for r in results]):.2f}%")
        
        print(f"\nPermissive Multi Method:")
        print(f"  Average coverage: {permissive_avg:.2f}%")
        print(f"  Median coverage:  {np.median([r['permissive_coverage'] for r in results]):.2f}%")
        print(f"  Min coverage:     {np.min([r['permissive_coverage'] for r in results]):.2f}%")
        print(f"  Max coverage:     {np.max([r['permissive_coverage'] for r in results]):.2f}%")
        
        print(f"\nDifference:")
        print(f"  Average additional coverage: {diff_avg:.2f}%")
        print(f"  Median additional coverage:  {np.median([r['difference'] for r in results]):.2f}%")
        
        print(f"\n✓ Report saved to: {output_csv}")


if __name__ == '__main__':
    main()
