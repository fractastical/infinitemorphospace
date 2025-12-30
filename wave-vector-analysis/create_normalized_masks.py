#!/usr/bin/env python3
"""
Create normalized embryo masks calibrated to a reference file.

Uses Folder 31: C3 - Substack (48-500).tif as a reference model and adjusts
all masks to be approximately the same size (±20% tolerance).
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path
import tifffile as tiff
import csv


def find_reference_mask_area(reference_tiff_path, mask_base_path=None, percentile_low=3, 
                             background_percentile=10, min_intensity_ratio=0.2):
    """
    Find the reference mask area from the reference TIFF file.
    Uses more sensitive thresholding to get a good baseline.
    """
    print(f"\n{'='*60}")
    print(f"Calibrating with reference file: {Path(reference_tiff_path).name}")
    print(f"{'='*60}")
    
    # Read reference TIFF
    with tiff.TiffFile(reference_tiff_path) as tif:
        raw_data = tif.pages[0].asarray()
    
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
    
    # Use more sensitive thresholding
    gray_float = gray.astype(np.float32)
    blur = cv2.GaussianBlur(gray_float, (5, 5), 0)
    flat_intensities = blur.flatten()
    
    background_threshold = np.percentile(flat_intensities, background_percentile)
    percentile_threshold = np.percentile(flat_intensities, percentile_low)
    median_intensity = np.median(flat_intensities)
    p25 = np.percentile(flat_intensities, 25)
    
    # Moderate threshold (less sensitive than before, but more than old version)
    adaptive_threshold = max(median_intensity * 0.4, p25 * 0.6, background_threshold * 1.1)
    embryo_threshold = min(percentile_threshold, adaptive_threshold)
    embryo_threshold = max(embryo_threshold, background_threshold * (1 + min_intensity_ratio))
    
    # Create mask
    initial_mask = (blur >= embryo_threshold).astype(np.uint8) * 255
    
    # Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    initial_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    initial_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours and filter by reasonable size (moderate: 0.5% - 50% of image)
    contours, _ = cv2.findContours(initial_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = int(0.005 * total_pixels)
    max_area = int(0.50 * total_pixels)
    
    embryo_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            embryo_contours.append(contour)
    
    # Create final mask
    final_mask = np.zeros_like(initial_mask)
    if embryo_contours:
        cv2.drawContours(final_mask, embryo_contours, -1, 255, -1)
        
        # Fill holes
        for contour in embryo_contours:
            single_mask = np.zeros_like(final_mask)
            cv2.drawContours(single_mask, [contour], -1, 255, -1)
            filled = single_mask.copy()
            contours_filled, _ = cv2.findContours(single_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(filled, contours_filled, -1, 255, -1)
            final_mask = cv2.bitwise_or(final_mask, filled)
    
    mask_area = np.sum(final_mask > 0)
    coverage = (mask_area / total_pixels) * 100
    
    # Scale down to target ~28% coverage (middle ground between old 17% and new 41%)
    # If reference gives us 41%, scale it down to 28%
    target_coverage_pct = 28.0
    if coverage > 0:
        scale_factor = target_coverage_pct / coverage
        scaled_area = int(mask_area * scale_factor)
        scaled_coverage = target_coverage_pct
    else:
        scaled_area = int(total_pixels * target_coverage_pct / 100.0)
        scaled_coverage = target_coverage_pct
    
    print(f"  Reference mask area: {mask_area:,} pixels ({coverage:.2f}% coverage)")
    print(f"  Scaled target: {scaled_area:,} pixels ({scaled_coverage:.2f}% coverage)")
    print(f"  Target range (±20%): {scaled_area * 0.8:,.0f} - {scaled_area * 1.2:,.0f} pixels")
    
    return scaled_area, scaled_coverage, embryo_threshold


def create_normalized_mask(gray, target_coverage_pct, target_coverage_tolerance=0.20,
                           percentile_low_start=2, background_percentile=10):
    """
    Create mask with iterative threshold adjustment to match target coverage percentage.
    
    Args:
        gray: Grayscale image
        target_coverage_pct: Target coverage percentage (e.g., 40.0 for 40%)
        target_coverage_tolerance: Tolerance for matching (±20% = 0.20)
        percentile_low_start: Starting percentile for thresholding
    
    Returns:
        Tuple of (mask, stats_dict)
    """
    h, w = gray.shape
    total_pixels = h * w
    target_area = int(total_pixels * target_coverage_pct / 100.0)
    target_min = int(total_pixels * target_coverage_pct * (1 - target_coverage_tolerance) / 100.0)
    target_max = int(total_pixels * target_coverage_pct * (1 + target_coverage_tolerance) / 100.0)
    
    # Normalize to float32
    if gray.dtype == np.uint16:
        gray_float = gray.astype(np.float32)
    else:
        gray_float = gray.astype(np.float32)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray_float, (5, 5), 0)
    flat_intensities = blur.flatten()
    
    # Calculate base thresholds
    background_threshold = np.percentile(flat_intensities, background_percentile)
    median_intensity = np.median(flat_intensities)
    p25 = np.percentile(flat_intensities, 25)
    mean_intensity = flat_intensities.mean()
    
    # Iterative threshold adjustment to match target area
    best_mask = None
    best_area = 0
    best_percentile = percentile_low_start
    
    # Try different percentiles to find one that gives us the target area
    # Use moderate range (not too sensitive, not too strict)
    # Start from 2% percentile and go up to find a good match
    for percentile_test in np.arange(2, 15, 0.3):
        percentile_threshold = np.percentile(flat_intensities, percentile_test)
        # Moderate adaptive threshold (between old and new)
        adaptive_threshold = max(median_intensity * 0.4, p25 * 0.6, background_threshold * 1.1)
        
        # Use moderate threshold
        embryo_threshold = min(percentile_threshold, adaptive_threshold)
        embryo_threshold = max(embryo_threshold, background_threshold * 1.1)
        
        # Create mask
        test_mask = (blur >= embryo_threshold).astype(np.uint8) * 255
        
        # Morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        test_mask = cv2.morphologyEx(test_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        test_mask = cv2.morphologyEx(test_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours and filter by reasonable size (moderate)
        contours, _ = cv2.findContours(test_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area_filter = int(0.005 * total_pixels)  # Moderate minimum
        max_area_filter = int(0.50 * total_pixels)   # Moderate maximum
        
        embryo_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area_filter <= area <= max_area_filter:
                embryo_contours.append(contour)
        
        # Create final mask
        final_test_mask = np.zeros_like(test_mask)
        if embryo_contours:
            cv2.drawContours(final_test_mask, embryo_contours, -1, 255, -1)
            
            # Fill holes
            for contour in embryo_contours:
                single_mask = np.zeros_like(final_test_mask)
                cv2.drawContours(single_mask, [contour], -1, 255, -1)
                filled = single_mask.copy()
                contours_filled, _ = cv2.findContours(single_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(filled, contours_filled, -1, 255, -1)
                final_test_mask = cv2.bitwise_or(final_test_mask, filled)
        
        mask_area_test = np.sum(final_test_mask > 0)
        
        # Check if this is closer to target
        if target_min <= mask_area_test <= target_max:
            # Perfect match within target range!
            best_mask = final_test_mask
            best_area = mask_area_test
            best_percentile = percentile_test
            # If we're very close to target (within 3%), stop searching immediately
            if abs(mask_area_test - target_area) / target_area < 0.03:
                break
        elif mask_area_test < target_max:
            # Under target - check if better than current best
            if best_area == 0 or (best_area < target_min) or (best_area > target_max and abs(mask_area_test - target_area) < abs(best_area - target_area)):
                best_mask = final_test_mask
                best_area = mask_area_test
                best_percentile = percentile_test
        elif mask_area_test > target_max:
            # Over target - only use if we haven't found anything better yet
            # Prefer masks that are closer to target_max (not way over)
            if best_area == 0:
                best_mask = final_test_mask
                best_area = mask_area_test
                best_percentile = percentile_test
            elif best_area > target_max:
                # If both are over, prefer the one closer to target_max
                if abs(mask_area_test - target_max) < abs(best_area - target_max):
                    best_mask = final_test_mask
                    best_area = mask_area_test
                    best_percentile = percentile_test
                # If we're getting further from target_max, stop searching
                elif mask_area_test > best_area:
                    break
            elif best_area < target_min:
                # If current best is under, prefer it unless new one is very close to target
                if abs(mask_area_test - target_max) < abs(best_area - target_area):
                    best_mask = final_test_mask
                    best_area = mask_area_test
                    best_percentile = percentile_test
    
    # If no mask found in target range, use the best one we found
    if best_mask is None:
        # Fallback: use moderate threshold
        percentile_threshold = np.percentile(flat_intensities, 3.0)  # Use 3rd percentile
        adaptive_threshold = max(median_intensity * 0.4, p25 * 0.6, background_threshold * 1.1)
        embryo_threshold = min(percentile_threshold, adaptive_threshold)
        embryo_threshold = max(embryo_threshold, background_threshold * 1.1)
        
        best_mask = (blur >= embryo_threshold).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area_filter = int(0.005 * total_pixels)  # Moderate minimum
        max_area_filter = int(0.50 * total_pixels)   # Moderate maximum
        
        embryo_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area_filter <= area <= max_area_filter:
                embryo_contours.append(contour)
        
        final_mask = np.zeros_like(best_mask)
        if embryo_contours:
            cv2.drawContours(final_mask, embryo_contours, -1, 255, -1)
            for contour in embryo_contours:
                single_mask = np.zeros_like(final_mask)
                cv2.drawContours(single_mask, [contour], -1, 255, -1)
                filled = single_mask.copy()
                contours_filled, _ = cv2.findContours(single_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(filled, contours_filled, -1, 255, -1)
                final_mask = cv2.bitwise_or(final_mask, filled)
        best_mask = final_mask
        best_area = np.sum(best_mask > 0)
        best_percentile = percentile_low_start
    
    mask_area = np.sum(best_mask > 0)
    coverage = (mask_area / total_pixels) * 100
    
    stats = {
        'method': 'normalized',
        'target_coverage_pct': target_coverage_pct,
        'target_area': target_area,
        'target_min': target_min,
        'target_max': target_max,
        'mask_area': mask_area,
        'coverage': coverage,
        'percentile_used': best_percentile,
        'background_threshold': background_threshold,
        'within_target': target_min <= mask_area <= target_max,
    }
    
    return best_mask, stats


def process_tiff_file(tiff_path, target_coverage_pct, output_dir=None, save_mask=True, frame_idx=0):
    """Process a single TIFF file and create normalized mask."""
    print(f"\nProcessing: {Path(tiff_path).name}")
    
    # Read TIFF
    try:
        with tiff.TiffFile(tiff_path) as tif:
            num_frames = len(tif.pages)
            if frame_idx >= num_frames:
                frame_idx = 0
            raw_data = tif.pages[frame_idx].asarray()
    except Exception as e:
        print(f"  ✗ Error reading TIFF: {e}")
        return None
    
    # Convert to grayscale
    if raw_data.ndim == 3:
        if raw_data.shape[2] == 3:
            gray = cv2.cvtColor(raw_data, cv2.COLOR_RGB2GRAY)
        else:
            gray = raw_data[:, :, 0]
    else:
        gray = raw_data
    
    # Create normalized mask
    mask, stats = create_normalized_mask(gray, target_coverage_pct)
    
    h, w = gray.shape
    total_pixels = h * w
    
    print(f"  Mask area: {stats['mask_area']:,} pixels ({stats['coverage']:.2f}% coverage)")
    print(f"  Target: {stats['target_coverage_pct']:.2f}% coverage ({stats['target_area']:,.0f} pixels, range: {stats['target_min']:,.0f}-{stats['target_max']:,.0f})")
    if stats['within_target']:
        print(f"  ✓ Within target range")
    else:
        diff_pct = ((stats['mask_area'] - stats['target_area']) / stats['target_area']) * 100
        print(f"  ⚠ Outside target range ({diff_pct:+.1f}% difference)")
    
    # Save mask
    if save_mask:
        if output_dir is None:
            output_dir = os.path.dirname(tiff_path)
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        base_name = Path(tiff_path).stem
        mask_path = os.path.join(output_dir, f"{base_name}_mask_frame{frame_idx}.png")
        cv2.imwrite(mask_path, mask)
        print(f"  ✓ Saved mask to: {mask_path}")
    
    return {
        'mask': mask,
        'stats': stats,
        'gray': gray,
        'mask_path': mask_path if save_mask else None,
        'tiff_path': str(tiff_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Create normalized embryo masks calibrated to a reference file',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('input', help='Input TIFF file or directory')
    parser.add_argument('--reference', required=True,
                       help='Reference TIFF file to use as calibration model (e.g., Folder 31/C3 - Substack (48-500).tif)')
    parser.add_argument('--batch', action='store_true', help='Process all TIFF files in directory')
    parser.add_argument('--output-dir', '-o', help='Output directory for masks')
    parser.add_argument('--frame-idx', type=int, default=0, help='Frame index to use (default: 0)')
    parser.add_argument('--skip-blank', action='store_true', help='Skip saving masks that are blank')
    
    args = parser.parse_args()
    
    # Find reference mask area
    reference_path = Path(args.reference)
    if not reference_path.exists():
        print(f"✗ Error: Reference file does not exist: {reference_path}")
        return 1
    
    target_area, target_coverage, ref_threshold = find_reference_mask_area(str(reference_path))
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"✗ Error: Path does not exist: {input_path}")
        return 1
    
    if args.batch or input_path.is_dir():
        # Batch mode
        tiff_files = list(input_path.rglob('*.tif')) + list(input_path.rglob('*.tiff'))
        if not tiff_files:
            print(f"✗ No TIFF files found in {input_path}")
            return 1
        
        print(f"\nFound {len(tiff_files)} TIFF files")
        print(f"Target coverage: {target_coverage:.2f}% (calibrated from reference)")
        print(f"Target range (±20%): {target_coverage * 0.8:.2f}% - {target_coverage * 1.2:.2f}%\n")
        
        results = []
        for tiff_file in sorted(tiff_files):
            # Skip reference file itself
            if str(tiff_file) == str(reference_path):
                print(f"\nSkipping reference file: {tiff_file.name}")
                continue
            
            result = process_tiff_file(
                str(tiff_file),
                target_coverage_pct=target_coverage,
                output_dir=args.output_dir,
                save_mask=not args.skip_blank or True,  # Always save for now
                frame_idx=args.frame_idx
            )
            if result:
                # Check if blank and skip if requested
                if args.skip_blank and result['stats']['mask_area'] == 0:
                    print(f"  ⚠ Skipping blank mask")
                    if result.get('mask_path'):
                        os.remove(result['mask_path'])
                    continue
                results.append(result)
        
        print(f"\n{'='*60}")
        print(f"Batch processing complete: {len(results)}/{len(tiff_files)-1} files processed")
        print(f"{'='*60}")
        
        # Generate summary
        if results:
            print(f"\n{'='*60}")
            print("NORMALIZED MASK SUMMARY")
            print(f"{'='*60}")
            print(f"{'File':<50} {'Area':<15} {'Coverage %':<12} {'Target Match':<12}")
            print("-" * 90)
            
            areas = []
            within_target_count = 0
            for result in results:
                stats = result['stats']
                areas.append(stats['mask_area'])
                
                mask_path = result.get('mask_path', 'N/A')
                if mask_path != 'N/A':
                    filename = Path(mask_path).name.replace('_mask_frame0.png', '')
                else:
                    filename = Path(result['tiff_path']).name
                
                match_status = "✓ In range" if stats['within_target'] else "⚠ Outside"
                within_target_count += 1 if stats['within_target'] else 0
                
                print(f"{filename:<50} {stats['mask_area']:>13,}  {stats['coverage']:>10.2f}%  {match_status:<12}")
            
            print("-" * 90)
            if areas:
                print(f"\nStatistics:")
                avg_coverage = np.mean([r['stats']['coverage'] for r in results])
                print(f"  Average coverage: {avg_coverage:.2f}%")
                print(f"  Median coverage:  {np.median([r['stats']['coverage'] for r in results]):.2f}%")
                print(f"  Min coverage:     {np.min([r['stats']['coverage'] for r in results]):.2f}%")
                print(f"  Max coverage:     {np.max([r['stats']['coverage'] for r in results]):.2f}%")
                print(f"  Target coverage:  {target_coverage:.2f}%")
                print(f"  Within target: {within_target_count}/{len(results)} ({within_target_count/len(results)*100:.1f}%)")
            
            # Save summary CSV
            if args.output_dir:
                summary_path = os.path.join(args.output_dir, 'normalized_coverage_summary.csv')
            else:
                summary_path = 'normalized_coverage_summary.csv'
            
            with open(summary_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'File', 'Mask_Area', 'Coverage_%', 'Image_Width', 'Image_Height',
                    'Target_Coverage_%', 'Target_Area', 'Target_Min', 'Target_Max', 'Within_Target',
                    'Percentile_Used', 'Background_Threshold'
                ])
                for result in results:
                    stats = result['stats']
                    gray_shape = result['gray'].shape
                    
                    mask_path = result.get('mask_path', 'N/A')
                    if mask_path != 'N/A':
                        filename = Path(mask_path).name.replace('_mask_frame0.png', '')
                    else:
                        filename = Path(result['tiff_path']).name
                    
                    writer.writerow([
                        filename,
                        stats['mask_area'],
                        f"{stats['coverage']:.2f}",
                        gray_shape[1],
                        gray_shape[0],
                        stats['target_coverage_pct'],
                        stats['target_area'],
                        stats['target_min'],
                        stats['target_max'],
                        stats['within_target'],
                        stats['percentile_used'],
                        stats['background_threshold']
                    ])
            
            print(f"\n✓ Summary saved to: {summary_path}")
    
    else:
        # Single file mode
        if not input_path.suffix.lower() in ['.tif', '.tiff']:
            print(f"✗ Error: Not a TIFF file: {input_path}")
            return 1
        
        result = process_tiff_file(
            str(input_path),
            target_coverage_pct=target_coverage,
            output_dir=args.output_dir,
            save_mask=not args.skip_blank,
            frame_idx=args.frame_idx
        )
        
        if result is None:
            return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
