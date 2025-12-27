#!/usr/bin/env python3
"""
Create embryo masks using size constraints - embryos are always roughly the same size,
so we can filter by expected embryo area to exclude background and noise.

This script:
1. Uses adaptive thresholding with a wider greyscale range
2. Filters contours by expected embryo size (min/max area)
3. Excludes regions that are too large (background) or too small (noise)
4. Outputs masks that capture the actual embryo body, not the whole image
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path
import tifffile as tiff
import matplotlib.pyplot as plt
import csv


def create_size_constrained_mask(gray, min_area_ratio=0.005, max_area_ratio=0.15, 
                                 percentile_low=5, background_percentile=10,
                                 min_intensity_ratio=0.3, expand_percent=10.0):
    """
    Create mask using size constraints to identify embryo-sized regions.
    
    Args:
        gray: Grayscale image
        min_area_ratio: Minimum embryo area as fraction of image (default: 0.5%)
        max_area_ratio: Maximum embryo area as fraction of image (default: 15%)
        percentile_low: Lower percentile for thresholding (default: 5)
        background_percentile: Percentile to define background (default: 10)
        min_intensity_ratio: Minimum ratio relative to background
    
    Returns:
        Tuple of (mask, stats_dict)
    """
    h, w = gray.shape
    total_pixels = h * w
    min_area = int(min_area_ratio * total_pixels)
    max_area = int(max_area_ratio * total_pixels)
    
    # Normalize to float32
    if gray.dtype == np.uint16:
        gray_float = gray.astype(np.float32)
    elif gray.dtype == np.uint8:
        gray_float = gray.astype(np.float32)
    else:
        gray_float = gray.astype(np.float32)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray_float, (5, 5), 0)
    flat_intensities = blur.flatten()
    
    # Calculate thresholds
    background_threshold = np.percentile(flat_intensities, background_percentile)
    median_intensity = np.median(flat_intensities)
    p25 = np.percentile(flat_intensities, 25)
    mean_intensity = flat_intensities.mean()
    
    # Use lower percentile but ensure it's above background
    percentile_threshold = np.percentile(flat_intensities, percentile_low)
    adaptive_threshold = max(median_intensity * 0.5, p25 * 0.7, background_threshold * 1.2)
    
    # Use the more permissive threshold (lower value = more inclusive)
    embryo_threshold = min(percentile_threshold, adaptive_threshold)
    embryo_threshold = max(embryo_threshold, background_threshold * (1 + min_intensity_ratio))
    
    # Create initial mask
    initial_mask = (blur >= embryo_threshold).astype(np.uint8) * 255
    
    # Morphology to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    initial_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    initial_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours and filter by size
    contours, _ = cv2.findContours(initial_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    embryo_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            embryo_contours.append(contour)
    
    # If no contours found with strict size filter, try more lenient approach
    # This handles cases where embryos might be slightly larger or smaller than expected
    if not embryo_contours and len(contours) > 0:
        # Try with more lenient min_area (half of original)
        lenient_min_area = max(min_area * 0.5, 100)  # At least 100 pixels
        lenient_max_area = max_area * 1.5  # Allow up to 1.5x max
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if lenient_min_area <= area <= lenient_max_area:
                embryo_contours.append(contour)
        
        if embryo_contours:
            print(f"    ⚠ Using lenient size filter: {lenient_min_area:.0f} - {lenient_max_area:.0f} pixels")
    
    # Create final mask from filtered contours
    final_mask = np.zeros_like(initial_mask)
    if embryo_contours:
        cv2.drawContours(final_mask, embryo_contours, -1, 255, -1)
    
    # Fill holes within embryo contours
    if embryo_contours:
        # Create a mask for each embryo and fill holes
        for contour in embryo_contours:
            # Create individual mask
            single_mask = np.zeros_like(final_mask)
            cv2.drawContours(single_mask, [contour], -1, 255, -1)
            
            # Fill holes
            filled = single_mask.copy()
            contours_filled, _ = cv2.findContours(single_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(filled, contours_filled, -1, 255, -1)
            
            # Combine
            final_mask = cv2.bitwise_or(final_mask, filled)
    
    # Expand mask by specified percentage using dilation
    # Calculate dilation kernel size based on mask dimensions
    if np.sum(final_mask > 0) > 0 and expand_percent > 0:
        # Estimate average dimension of mask regions
        mask_area_before = np.sum(final_mask > 0)
        
        # To increase area by expand_percent%, we need to increase radius by sqrt(1 + expand_percent/100) - 1
        # For area = πr²: new_area = (1 + expand_percent/100) * old_area
        # => new_r = sqrt(1 + expand_percent/100) * old_r
        # => dilation_radius ≈ (sqrt(1 + expand_percent/100) - 1) * old_r
        radius_factor = np.sqrt(1 + expand_percent / 100.0) - 1
        
        # Estimate average radius of mask regions
        if embryo_contours:
            # Use average of contour bounding box dimensions
            avg_radius = 0
            for contour in embryo_contours:
                (x, y), (w, h), angle = cv2.minAreaRect(contour)
                avg_radius += np.sqrt(w * h) / 2
            avg_radius = avg_radius / len(embryo_contours)
        else:
            # Fallback: estimate from total area
            avg_radius = np.sqrt(mask_area_before / np.pi)
        
        dilation_radius = max(2, int(avg_radius * radius_factor))  # At least 2 pixels
        
        # Create circular kernel for dilation
        kernel_size = dilation_radius * 2 + 1
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        final_mask = cv2.dilate(final_mask, dilation_kernel, iterations=1)
    
    # Final cleanup
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Calculate statistics
    mask_area = np.sum(final_mask > 0)
    coverage = (mask_area / total_pixels) * 100
    
    stats = {
        'method': 'size_constrained',
        'background_threshold': background_threshold,
        'embryo_threshold': embryo_threshold,
        'percentile_threshold': percentile_threshold,
        'adaptive_threshold': adaptive_threshold,
        'min_area': min_area,
        'max_area': max_area,
        'mask_area': mask_area,
        'coverage': coverage,
        'num_contours_before_filter': len(contours),
        'num_contours_after_filter': len(embryo_contours),
        'total_pixels': total_pixels,
    }
    
    return final_mask, stats


def process_tiff_file(tiff_path, output_dir=None, min_area_ratio=0.005, max_area_ratio=0.15,
                      percentile_low=5, expand_percent=10.0, visualize=True, save_mask=True, 
                      frame_idx=0, skip_blank=False):
    """Process a single TIFF file and create size-constrained embryo masks."""
    print(f"\n{'='*60}")
    print(f"Processing: {Path(tiff_path).name}")
    print(f"{'='*60}")
    
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
    
    # Create size-constrained mask
    mask, stats = create_size_constrained_mask(
        gray, 
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        percentile_low=percentile_low,
        expand_percent=expand_percent
    )
    
    print(f"\n  Mask Statistics:")
    print(f"    Embryo threshold: {stats['embryo_threshold']:.1f}")
    print(f"    Area range: {stats['min_area']:,} - {stats['max_area']:,} pixels")
    print(f"    Contours before filter: {stats['num_contours_before_filter']}")
    print(f"    Contours after filter: {stats['num_contours_after_filter']}")
    print(f"    Mask area: {stats['mask_area']:,} pixels ({stats['coverage']:.2f}% of image)")
    
    # Save mask (only if not blank, or if skip_blank is False)
    mask_area_check = np.sum(mask > 0)
    is_blank = mask_area_check == 0
    
    if save_mask:
        if skip_blank and is_blank:
            print(f"    ⚠ Skipping blank mask (0% coverage)")
            mask_path = None
        else:
            if output_dir is None:
                output_dir = os.path.dirname(tiff_path)
            else:
                os.makedirs(output_dir, exist_ok=True)
            
            base_name = Path(tiff_path).stem
            mask_path = os.path.join(output_dir, f"{base_name}_mask_frame{frame_idx}.png")
            cv2.imwrite(mask_path, mask)
            print(f"    ✓ Saved mask to: {mask_path}")
    else:
        mask_path = None
    
    # Create visualization
    if visualize:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Original grayscale
        axes[0, 0].imshow(gray, cmap='gray')
        axes[0, 0].set_title('Original Grayscale Image')
        axes[0, 0].axis('off')
        
        # Mask overlay
        overlay = gray.copy()
        if gray.dtype == np.uint16:
            overlay = (overlay / 65535.0 * 255).astype(np.uint8)
        overlay_colored = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        overlay_colored[mask > 0] = [0, 255, 0]  # Green overlay
        
        axes[0, 1].imshow(overlay_colored)
        axes[0, 1].set_title(f'Mask Overlay (Green = Embryo, {stats["coverage"]:.2f}%)')
        axes[0, 1].axis('off')
        
        # Mask only
        axes[1, 0].imshow(mask, cmap='gray')
        axes[1, 0].set_title('Binary Mask')
        axes[1, 0].axis('off')
        
        # Intensity histogram
        axes[1, 1].hist(gray.flatten(), bins=100, alpha=0.7, label='All pixels', density=True)
        axes[1, 1].hist(gray[mask > 0].flatten(), bins=100, alpha=0.7, label='Embryo pixels', color='green', density=True)
        
        bg_thresh = stats['background_threshold']
        emb_thresh = stats['embryo_threshold']
        axes[1, 1].axvline(bg_thresh, color='red', linestyle='--', label=f'Background ({bg_thresh:.1f})')
        axes[1, 1].axvline(emb_thresh, color='blue', linestyle='--', label=f'Embryo ({emb_thresh:.1f})')
        
        axes[1, 1].set_xlabel('Intensity')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Intensity Histogram')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir is None:
            output_dir = os.path.dirname(tiff_path)
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        vis_path = os.path.join(output_dir, f"{base_name}_mask_vis_frame{frame_idx}.png")
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        print(f"    ✓ Saved visualization to: {vis_path}")
        plt.close()
    
    return {
        'mask': mask,
        'stats': stats,
        'gray': gray,
        'mask_path': mask_path if save_mask else None,
        'tiff_path': str(tiff_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Create size-constrained embryo masks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file
  python create_size_constrained_masks.py path/to/image.tif
  
  # Process directory
  python create_size_constrained_masks.py path/to/tiff_dir --batch
  
  # Adjust size constraints (embryo should be 1-10% of image)
  python create_size_constrained_masks.py image.tif --min-area 0.01 --max-area 0.10
        """
    )
    
    parser.add_argument('input', help='Input TIFF file or directory')
    parser.add_argument('--batch', action='store_true', help='Process all TIFF files in directory')
    parser.add_argument('--output-dir', '-o', help='Output directory for masks')
    parser.add_argument('--min-area', type=float, default=0.005,
                       help='Minimum embryo area as fraction of image (default: 0.005 = 0.5%%)')
    parser.add_argument('--max-area', type=float, default=0.15,
                       help='Maximum embryo area as fraction of image (default: 0.15 = 15%%)')
    parser.add_argument('--percentile-low', type=float, default=5.0,
                       help='Lower percentile for thresholding (default: 5.0)')
    parser.add_argument('--expand-percent', type=float, default=10.0,
                       help='Expand mask size by this percentage (default: 10.0%%)')
    parser.add_argument('--frame-idx', type=int, default=0,
                       help='Frame index to use (default: 0)')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--no-save', action='store_true',
                       help='Skip saving mask files')
    parser.add_argument('--skip-blank', action='store_true',
                       help='Skip saving masks that are blank (0%% coverage)')
    
    args = parser.parse_args()
    
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
        
        print(f"Found {len(tiff_files)} TIFF files")
        results = []
        
        for tiff_file in sorted(tiff_files):
            result = process_tiff_file(
                str(tiff_file),
                output_dir=args.output_dir,
                min_area_ratio=args.min_area,
                max_area_ratio=args.max_area,
                percentile_low=args.percentile_low,
                expand_percent=args.expand_percent,
                visualize=not args.no_visualize,
                save_mask=not args.no_save,
                frame_idx=args.frame_idx,
                skip_blank=args.skip_blank
            )
            if result:
                results.append(result)
        
        print(f"\n{'='*60}")
        print(f"Batch processing complete: {len(results)}/{len(tiff_files)} files processed")
        print(f"{'='*60}")
        
        # Generate summary
        if results:
            print(f"\n{'='*60}")
            print("EMBRYO BODY COVERAGE SUMMARY (Size-Constrained)")
            print(f"{'='*60}")
            print(f"{'File':<50} {'Coverage %':<12} {'Mask Area':<15} {'Contours':<10}")
            print("-" * 90)
            
            coverages = []
            for result in results:
                stats = result['stats']
                coverages.append(stats['coverage'])
                
                mask_path = result.get('mask_path')
                if mask_path and mask_path != 'N/A':
                    filename = Path(mask_path).name.replace('_mask_frame0.png', '')
                else:
                    # Extract filename from tiff path if available
                    if 'tiff_path' in result:
                        filename = Path(result['tiff_path']).name
                    else:
                        filename = "Unknown"
                
                print(f"{filename:<50} {stats['coverage']:>10.2f}%  {stats['mask_area']:>13,}  {stats['num_contours_after_filter']:>8}")
            
            print("-" * 90)
            if coverages:
                print(f"\nStatistics:")
                print(f"  Average coverage: {np.mean(coverages):.2f}%")
                print(f"  Median coverage:  {np.median(coverages):.2f}%")
                print(f"  Min coverage:     {np.min(coverages):.2f}%")
                print(f"  Max coverage:     {np.max(coverages):.2f}%")
            
            # Save summary CSV (only include files with masks if skip_blank is True)
            if args.output_dir:
                summary_path = os.path.join(args.output_dir, 'size_constrained_coverage_summary.csv')
            else:
                summary_path = 'size_constrained_coverage_summary.csv'
            
            with open(summary_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'File', 'Coverage_%', 'Mask_Area', 'Image_Width', 'Image_Height',
                    'Min_Area', 'Max_Area', 'Contours_Before_Filter', 'Contours_After_Filter',
                    'Embryo_Threshold', 'Background_Threshold'
                ])
                for result in results:
                    stats = result['stats']
                    
                    # Skip blank masks in CSV if skip_blank is enabled
                    if args.skip_blank and stats['coverage'] == 0.0:
                        continue
                    
                    gray_shape = result['gray'].shape
                    
                    mask_path = result.get('mask_path')
                    if mask_path and mask_path != 'N/A':
                        filename = Path(mask_path).name.replace('_mask_frame0.png', '')
                    else:
                        # Extract filename from tiff path if available
                        if 'tiff_path' in result:
                            filename = Path(result['tiff_path']).name
                        else:
                            filename = "Unknown"
                    
                    writer.writerow([
                        filename,
                        f"{stats['coverage']:.2f}",
                        stats['mask_area'],
                        gray_shape[1],
                        gray_shape[0],
                        stats['min_area'],
                        stats['max_area'],
                        stats['num_contours_before_filter'],
                        stats['num_contours_after_filter'],
                        stats['embryo_threshold'],
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
            output_dir=args.output_dir,
            min_area_ratio=args.min_area,
            max_area_ratio=args.max_area,
            percentile_low=args.percentile_low,
            expand_percent=args.expand_percent,
            visualize=not args.no_visualize,
            save_mask=not args.no_save,
            frame_idx=args.frame_idx,
            skip_blank=args.skip_blank
        )
        
        if result is None:
            return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
