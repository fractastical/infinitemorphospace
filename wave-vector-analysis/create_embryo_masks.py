#!/usr/bin/env python3
"""
Create improved embryo masks with wider greyscale range to capture the full embryo body.

This script generates masks that:
1. Use more permissive greyscale thresholds to capture dimmer embryo regions
2. Mask out background (non-embryo) pixels for training purposes
3. Can output masks as images or binary files for use in training pipelines

Usage:
    python create_embryo_masks.py <tiff_file> [options]
    python create_embryo_masks.py <tiff_directory> --batch [options]
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def create_adaptive_mask(gray, method='percentile', percentile_low=5, percentile_high=95, 
                        background_percentile=10, min_intensity_ratio=0.3, use_otsu=False):
    """
    Create an adaptive mask using multiple strategies to capture full embryo body.
    
    Args:
        gray: Grayscale image (can be 8-bit or 16-bit)
        method: 'percentile', 'adaptive', 'otsu', or 'multi'
        percentile_low: Lower percentile for embryo range (default: 5)
        percentile_high: Upper percentile for embryo range (default: 95)
        background_percentile: Percentile to define background (default: 10)
        min_intensity_ratio: Minimum ratio relative to background (default: 0.3)
        use_otsu: Whether to use Otsu's method as additional constraint
    
    Returns:
        Tuple of (mask, stats_dict) where mask is binary uint8 and stats contains threshold info
    """
    # Normalize to float32 for processing
    if gray.dtype == np.uint16:
        gray_float = gray.astype(np.float32)
        max_val = 65535.0
    elif gray.dtype == np.uint8:
        gray_float = gray.astype(np.float32)
        max_val = 255.0
    else:
        gray_float = gray.astype(np.float32)
        max_val = gray_float.max()
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray_float, (5, 5), 0)
    flat_intensities = blur.flatten()
    
    # Calculate statistics
    background_threshold = np.percentile(flat_intensities, background_percentile)
    median_intensity = np.median(flat_intensities)
    mean_intensity = flat_intensities.mean()
    p25 = np.percentile(flat_intensities, 25)
    p75 = np.percentile(flat_intensities, 75)
    
    stats = {
        'background_threshold': background_threshold,
        'median': median_intensity,
        'mean': mean_intensity,
        'p25': p25,
        'p75': p75,
        'min': flat_intensities.min(),
        'max': flat_intensities.max(),
    }
    
    # Strategy 1: Percentile-based (most permissive)
    if method == 'percentile':
        # Use lower percentile to capture dimmer regions
        embryo_threshold = np.percentile(flat_intensities, percentile_low)
        # Ensure it's above background but not too restrictive
        embryo_threshold = max(embryo_threshold, background_threshold * (1 + min_intensity_ratio))
        stats['method'] = 'percentile'
        stats['threshold'] = embryo_threshold
        mask = (blur >= embryo_threshold).astype(np.uint8) * 255
    
    # Strategy 2: Adaptive multi-threshold
    elif method == 'adaptive':
        # Use multiple thresholds and combine
        threshold1 = max(median_intensity * 0.5, p25 * 0.7, background_threshold * 1.2)
        threshold2 = max(mean_intensity * 0.6, background_threshold * 1.5)
        threshold3 = np.percentile(flat_intensities, percentile_low)
        
        # Use the most permissive (lowest) threshold
        embryo_threshold = min(threshold1, threshold2, threshold3)
        embryo_threshold = max(embryo_threshold, background_threshold * (1 + min_intensity_ratio))
        
        stats['method'] = 'adaptive'
        stats['threshold'] = embryo_threshold
        stats['thresholds'] = [threshold1, threshold2, threshold3]
        mask = (blur >= embryo_threshold).astype(np.uint8) * 255
    
    # Strategy 3: Otsu's method (for bimodal distributions)
    elif method == 'otsu':
        # Convert to uint8 for Otsu if needed
        if gray.dtype == np.uint16:
            gray_8bit = (gray_float / max_val * 255).astype(np.uint8)
        else:
            gray_8bit = gray_float.astype(np.uint8)
        
        _, otsu_mask = cv2.threshold(gray_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Also apply percentile constraint to be more permissive
        percentile_threshold = np.percentile(flat_intensities, percentile_low)
        percentile_mask = (blur >= percentile_threshold).astype(np.uint8) * 255
        
        # Combine: union of both masks (most permissive)
        mask = cv2.bitwise_or(otsu_mask, percentile_mask)
        
        stats['method'] = 'otsu'
        stats['otsu_threshold'] = _
        stats['percentile_threshold'] = percentile_threshold
    
    # Strategy 4: Multi-method combination (most permissive)
    elif method == 'multi':
        # Try multiple methods and combine
        percentile_threshold = np.percentile(flat_intensities, percentile_low)
        adaptive_threshold = max(median_intensity * 0.5, p25 * 0.7, background_threshold * 1.2)
        
        mask1 = (blur >= percentile_threshold).astype(np.uint8) * 255
        mask2 = (blur >= adaptive_threshold).astype(np.uint8) * 255
        
        if use_otsu and gray.dtype in [np.uint8, np.uint16]:
            if gray.dtype == np.uint16:
                gray_8bit = (gray_float / max_val * 255).astype(np.uint8)
            else:
                gray_8bit = gray_float.astype(np.uint8)
            _, mask3 = cv2.threshold(gray_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            mask3 = np.zeros_like(mask1)
        
        # Union of all masks (most permissive)
        mask = cv2.bitwise_or(mask1, cv2.bitwise_or(mask2, mask3))
        
        stats['method'] = 'multi'
        stats['percentile_threshold'] = percentile_threshold
        stats['adaptive_threshold'] = adaptive_threshold
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Post-processing: fill holes and smooth
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Fill holes
    mask_filled = mask.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask_filled, contours, -1, 255, -1)
    
    stats['mask_area'] = np.sum(mask_filled > 0)
    stats['num_contours'] = len(contours)
    
    return mask_filled, stats


def process_tiff_file(tiff_path, output_dir=None, method='multi', percentile_low=5, 
                      visualize=True, save_mask=True, frame_idx=0):
    """
    Process a single TIFF file and create embryo masks.
    
    Args:
        tiff_path: Path to TIFF file
        output_dir: Directory to save outputs (if None, uses same dir as input)
        method: Mask creation method
        percentile_low: Lower percentile for thresholding
        visualize: Whether to create visualization
        save_mask: Whether to save mask as image
        frame_idx: Which frame to use (0 = first frame)
    
    Returns:
        Dictionary with results and stats
    """
    print(f"\n{'='*60}")
    print(f"Processing: {tiff_path}")
    print(f"{'='*60}")
    
    # Read TIFF
    try:
        with tiff.TiffFile(tiff_path) as tif:
            num_frames = len(tif.pages)
            print(f"  Found {num_frames} frames")
            
            if frame_idx >= num_frames:
                print(f"  ⚠ Warning: frame_idx {frame_idx} >= {num_frames}, using frame 0")
                frame_idx = 0
            
            # Read specified frame
            raw_data = tif.pages[frame_idx].asarray()
            print(f"  Using frame {frame_idx}/{num_frames}")
            print(f"  Image shape: {raw_data.shape}, dtype: {raw_data.dtype}")
            
    except Exception as e:
        print(f"  ✗ Error reading TIFF: {e}")
        return None
    
    # Convert to grayscale if needed
    if raw_data.ndim == 3:
        if raw_data.shape[2] == 3:
            # RGB/BGR - convert to grayscale
            gray = cv2.cvtColor(raw_data, cv2.COLOR_RGB2GRAY)
        else:
            gray = raw_data[:, :, 0]  # Take first channel
    else:
        gray = raw_data
    
    # Create mask
    mask, stats = create_adaptive_mask(gray, method=method, percentile_low=percentile_low)
    
    print(f"\n  Mask Statistics:")
    print(f"    Method: {stats.get('method', 'unknown')}")
    print(f"    Background threshold: {stats.get('background_threshold', 0):.1f}")
    print(f"    Embryo threshold: {stats.get('threshold', stats.get('percentile_threshold', 0)):.1f}")
    print(f"    Mask area: {stats['mask_area']:,} pixels ({stats['mask_area']/(gray.shape[0]*gray.shape[1])*100:.1f}% of image)")
    print(f"    Contours found: {stats['num_contours']}")
    
    # Save mask
    if save_mask:
        if output_dir is None:
            output_dir = os.path.dirname(tiff_path)
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        base_name = Path(tiff_path).stem
        mask_path = os.path.join(output_dir, f"{base_name}_mask_frame{frame_idx}.png")
        cv2.imwrite(mask_path, mask)
        print(f"    ✓ Saved mask to: {mask_path}")
    
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
        overlay_colored[mask > 0] = [0, 255, 0]  # Green overlay for embryo
        
        axes[0, 1].imshow(overlay_colored)
        axes[0, 1].set_title('Mask Overlay (Green = Embryo)')
        axes[0, 1].axis('off')
        
        # Mask only
        axes[1, 0].imshow(mask, cmap='gray')
        axes[1, 0].set_title('Binary Mask')
        axes[1, 0].axis('off')
        
        # Intensity histogram with thresholds
        axes[1, 1].hist(gray.flatten(), bins=100, alpha=0.7, label='All pixels')
        axes[1, 1].hist(gray[mask > 0].flatten(), bins=100, alpha=0.7, label='Embryo pixels', color='green')
        
        bg_thresh = stats.get('background_threshold', 0)
        emb_thresh = stats.get('threshold', stats.get('percentile_threshold', 0))
        axes[1, 1].axvline(bg_thresh, color='red', linestyle='--', label=f'Background threshold ({bg_thresh:.1f})')
        axes[1, 1].axvline(emb_thresh, color='blue', linestyle='--', label=f'Embryo threshold ({emb_thresh:.1f})')
        
        axes[1, 1].set_xlabel('Intensity')
        axes[1, 1].set_ylabel('Frequency')
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
    }


def main():
    parser = argparse.ArgumentParser(
        description='Create improved embryo masks with wider greyscale range',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file
  python create_embryo_masks.py path/to/image.tif
  
  # Process directory with batch mode
  python create_embryo_masks.py path/to/tiff_dir --batch
  
  # Use more permissive thresholding (lower percentile)
  python create_embryo_masks.py image.tif --percentile-low 2
  
  # Use Otsu method
  python create_embryo_masks.py image.tif --method otsu
  
  # Save masks only, no visualization
  python create_embryo_masks.py image.tif --no-visualize
        """
    )
    
    parser.add_argument('input', help='Input TIFF file or directory')
    parser.add_argument('--batch', action='store_true', help='Process all TIFF files in directory')
    parser.add_argument('--output-dir', '-o', help='Output directory for masks and visualizations')
    parser.add_argument('--method', choices=['percentile', 'adaptive', 'otsu', 'multi'], 
                       default='multi', help='Mask creation method (default: multi)')
    parser.add_argument('--percentile-low', type=float, default=5.0,
                       help='Lower percentile for embryo threshold (default: 5.0)')
    parser.add_argument('--frame-idx', type=int, default=0,
                       help='Frame index to use (default: 0)')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--no-save', action='store_true',
                       help='Skip saving mask files')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"✗ Error: Path does not exist: {input_path}")
        return 1
    
    if args.batch or input_path.is_dir():
        # Batch mode - search recursively
        tiff_files = list(input_path.rglob('*.tif')) + list(input_path.rglob('*.tiff'))
        if not tiff_files:
            print(f"✗ No TIFF files found in {input_path}")
            print(f"  (Searched recursively for *.tif and *.tiff)")
            return 1
        
        print(f"Found {len(tiff_files)} TIFF files")
        results = []
        
        for tiff_file in sorted(tiff_files):
            result = process_tiff_file(
                str(tiff_file),
                output_dir=args.output_dir,
                method=args.method,
                percentile_low=args.percentile_low,
                visualize=not args.no_visualize,
                save_mask=not args.no_save,
                frame_idx=args.frame_idx
            )
            if result:
                results.append(result)
        
        print(f"\n{'='*60}")
        print(f"Batch processing complete: {len(results)}/{len(tiff_files)} files processed")
        print(f"{'='*60}")
        
        # Generate summary report
        if results:
            print(f"\n{'='*60}")
            print("EMBRYO BODY COVERAGE SUMMARY")
            print(f"{'='*60}")
            print(f"{'File':<50} {'Coverage %':<12} {'Mask Area':<15} {'Image Size':<15} {'Method':<10}")
            print("-" * 100)
            
            total_coverage = []
            for result in results:
                stats = result['stats']
                mask_area = stats['mask_area']
                gray_shape = result['gray'].shape
                image_size = gray_shape[0] * gray_shape[1]
                coverage_pct = (mask_area / image_size) * 100
                total_coverage.append(coverage_pct)
                
                # Extract filename from path
                mask_path = result.get('mask_path', 'N/A')
                if mask_path != 'N/A':
                    filename = Path(mask_path).name.replace('_mask_frame0.png', '')
                else:
                    filename = "Unknown"
                
                method = stats.get('method', 'unknown')
                print(f"{filename:<50} {coverage_pct:>10.2f}%  {mask_area:>13,}  {image_size:>13,}  {method:<10}")
            
            print("-" * 100)
            if total_coverage:
                avg_coverage = np.mean(total_coverage)
                median_coverage = np.median(total_coverage)
                min_coverage = np.min(total_coverage)
                max_coverage = np.max(total_coverage)
                print(f"\nStatistics:")
                print(f"  Average coverage: {avg_coverage:.2f}%")
                print(f"  Median coverage:  {median_coverage:.2f}%")
                print(f"  Min coverage:     {min_coverage:.2f}%")
                print(f"  Max coverage:     {max_coverage:.2f}%")
            
            # Save summary to CSV
            if args.output_dir:
                summary_path = os.path.join(args.output_dir, 'embryo_coverage_summary.csv')
            else:
                summary_path = 'embryo_coverage_summary.csv'
            
            import csv
            with open(summary_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['File', 'Coverage_%', 'Mask_Area', 'Image_Width', 'Image_Height', 'Method', 'Background_Threshold', 'Embryo_Threshold'])
                for result in results:
                    stats = result['stats']
                    mask_area = stats['mask_area']
                    gray_shape = result['gray'].shape
                    image_size = gray_shape[0] * gray_shape[1]
                    coverage_pct = (mask_area / image_size) * 100
                    
                    mask_path = result.get('mask_path', 'N/A')
                    if mask_path != 'N/A':
                        filename = Path(mask_path).name.replace('_mask_frame0.png', '')
                    else:
                        filename = "Unknown"
                    
                    writer.writerow([
                        filename,
                        f"{coverage_pct:.2f}",
                        mask_area,
                        gray_shape[1],
                        gray_shape[0],
                        stats.get('method', 'unknown'),
                        stats.get('background_threshold', 0),
                        stats.get('threshold', stats.get('percentile_threshold', 0))
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
            method=args.method,
            percentile_low=args.percentile_low,
            visualize=not args.no_visualize,
            save_mask=not args.no_save,
            frame_idx=args.frame_idx
        )
        
        if result is None:
            return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
