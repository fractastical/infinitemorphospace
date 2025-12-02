#!/usr/bin/env python3
"""
Diagnostic script to check TIFF file properties and why they might appear black.
"""
import os
import sys
import numpy as np
import tifffile as tiff
from pathlib import Path


def analyze_tiff(path):
    """Analyze a single TIFF file and report its properties."""
    print(f"\n{'='*60}")
    print(f"File: {os.path.basename(path)}")
    print(f"{'='*60}")
    
    try:
        with tiff.TiffFile(path) as tif:
            num_pages = len(tif.pages)
            print(f"Number of pages: {num_pages}")
            
            # Analyze first page
            img = tif.asarray(key=0)
            print(f"\nImage properties:")
            print(f"  Shape: {img.shape}")
            print(f"  Dtype: {img.dtype}")
            print(f"  Min value: {img.min()}")
            print(f"  Max value: {img.max()}")
            print(f"  Mean value: {img.mean():.2f}")
            print(f"  Median value: {np.median(img):.2f}")
            
            # Check bit depth
            if img.dtype == np.uint16:
                print(f"  Bit depth: 16-bit (range: 0-65535)")
                # Check if values are in a narrow range
                value_range = img.max() - img.min()
                if value_range < 1000:
                    print(f"  ⚠️  WARNING: Very narrow dynamic range ({value_range} out of 65535 possible)")
                    print(f"     This might make the image appear black in standard viewers")
            elif img.dtype == np.uint8:
                print(f"  Bit depth: 8-bit (range: 0-255)")
                value_range = img.max() - img.min()
                if value_range < 50:
                    print(f"  ⚠️  WARNING: Very narrow dynamic range ({value_range} out of 255 possible)")
            else:
                print(f"  ⚠️  Unusual dtype: {img.dtype}")
            
            # Check if mostly black
            if img.dtype == np.uint16:
                threshold = 100  # Very low threshold for 16-bit
            else:
                threshold = 10  # Low threshold for 8-bit
                
            black_pixels = np.sum(img < threshold)
            total_pixels = img.size
            black_percent = (black_pixels / total_pixels) * 100
            print(f"\nPixel distribution:")
            print(f"  Pixels < {threshold}: {black_pixels:,} ({black_percent:.1f}%)")
            
            if black_percent > 90:
                print(f"  ⚠️  WARNING: Image is >90% dark/black")
            
            # Histogram info
            if img.dtype == np.uint16:
                hist, bins = np.histogram(img, bins=256, range=(0, 65536))
            else:
                hist, bins = np.histogram(img, bins=256, range=(0, 256))
            
            # Find where most values are
            max_bin_idx = np.argmax(hist)
            print(f"  Peak histogram bin: {bins[max_bin_idx]:.0f} (contains {hist[max_bin_idx]:,} pixels)")
            
            # Check file size
            file_size = os.path.getsize(path)
            print(f"\nFile size: {file_size / (1024*1024):.2f} MB")
            
            return {
                'dtype': str(img.dtype),
                'min': int(img.min()),
                'max': int(img.max()),
                'mean': float(img.mean()),
                'shape': img.shape,
                'num_pages': num_pages,
                'black_percent': float(black_percent)
            }
            
    except Exception as e:
        print(f"  ✗ ERROR: {str(e)}")
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_tiffs.py <folder_path> [num_files]")
        print("  Analyzes TIFF files to diagnose why they might appear black")
        print("  If num_files is provided, only analyzes that many files")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    num_files = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    print(f"Analyzing TIFF files in: {folder_path}")
    
    # Collect all TIFF files
    paths = []
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith((".tif", ".tiff")):
                paths.append(os.path.join(root, f))
    
    if not paths:
        print(f"✗ No TIFF files found in {folder_path}")
        sys.exit(1)
    
    paths.sort()
    
    if num_files:
        paths = paths[:num_files]
    
    print(f"\nFound {len(paths)} TIFF file(s) to analyze\n")
    
    results = []
    for i, path in enumerate(paths, 1):
        print(f"\n[{i}/{len(paths)}] Analyzing...")
        result = analyze_tiff(path)
        if result:
            results.append(result)
    
    # Summary statistics
    if results:
        print(f"\n\n{'='*60}")
        print("SUMMARY STATISTICS")
        print(f"{'='*60}")
        
        dtypes = [r['dtype'] for r in results]
        dtype_counts = {d: dtypes.count(d) for d in set(dtypes)}
        print(f"\nData types found:")
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} file(s)")
        
        mins = [r['min'] for r in results]
        maxs = [r['max'] for r in results]
        means = [r['mean'] for r in results]
        black_percents = [r['black_percent'] for r in results]
        
        print(f"\nValue ranges across all files:")
        print(f"  Min values: {min(mins)} to {max(mins)} (avg: {np.mean(mins):.1f})")
        print(f"  Max values: {min(maxs)} to {max(maxs)} (avg: {np.mean(maxs):.1f})")
        print(f"  Mean values: {min(means):.1f} to {max(means):.1f} (avg: {np.mean(means):.1f})")
        
        avg_black = np.mean(black_percents)
        print(f"\nDark pixel analysis:")
        print(f"  Average % dark pixels: {avg_black:.1f}%")
        
        very_dark = sum(1 for p in black_percents if p > 90)
        print(f"  Files with >90% dark pixels: {very_dark}/{len(results)}")
        
        # Recommendations
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS")
        print(f"{'='*60}")
        
        if np.any(np.array([r['dtype'] == 'uint16' for r in results])):
            print("\n✓ Files are 16-bit TIFF format")
            print("  • macOS Preview may not display 16-bit TIFFs correctly")
            print("  • Try using ImageJ, Fiji, or other scientific image viewers")
            print("  • Or convert to 8-bit for viewing (data will be preserved in 16-bit files)")
        
        if avg_black > 50:
            print(f"\n⚠️  Files are predominantly dark ({avg_black:.1f}% dark pixels)")
            print("  • This is normal for low-light/scientific imaging")
            print("  • The spark detection algorithm should still work on bright regions")
            print("  • Consider adjusting display gamma/contrast in your viewer")
        
        narrow_range_files = [r for r in results if r['max'] - r['min'] < (65536 if r['dtype'] == 'uint16' else 256) * 0.1]
        if narrow_range_files:
            print(f"\n⚠️  {len(narrow_range_files)} file(s) have narrow dynamic range")
            print("  • These may appear all black/gray in standard viewers")
            print("  • The actual data is fine - use auto-contrast/stretch in your viewer")


if __name__ == "__main__":
    main()

