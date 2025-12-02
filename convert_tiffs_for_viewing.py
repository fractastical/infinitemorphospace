#!/usr/bin/env python3
"""
Convert 16-bit TIFF files to properly scaled 8-bit versions for viewing.

The issue: 16-bit scientific TIFFs often have narrow dynamic ranges that
appear black in standard viewers. This script converts them to 8-bit with
proper contrast stretching so they're viewable.
"""
import os
import sys
import numpy as np
import tifffile as tiff
from pathlib import Path


def convert_tiff_page(img, method='percentile'):
    """
    Convert a 16-bit image to 8-bit with contrast enhancement.
    
    Args:
        img: numpy array (16-bit)
        method: 'percentile' (default) or 'linear' or 'minmax'
    
    Returns:
        8-bit numpy array
    """
    if img.dtype != np.uint16:
        # Already 8-bit or other, just convert
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        return img
    
    if method == 'percentile':
        # Use percentile-based stretching (clips outliers)
        p2 = np.percentile(img, 2)   # 2nd percentile
        p98 = np.percentile(img, 98) # 98th percentile
        # Stretch contrast to full 8-bit range
        img_scaled = np.clip((img.astype(np.float32) - p2) / (p98 - p2 + 1e-6) * 255, 0, 255)
        return img_scaled.astype(np.uint8)
    
    elif method == 'linear':
        # Simple linear stretch from min to max
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img_scaled = ((img.astype(np.float32) - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_scaled = np.zeros_like(img, dtype=np.uint8)
        return img_scaled
    
    elif method == 'minmax':
        # Stretch to full 16-bit range first, then convert to 8-bit
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img_scaled = ((img.astype(np.float32) - img_min) / (img_max - img_min) * 65535).astype(np.uint16)
            # Then convert 16-bit to 8-bit
            return (img_scaled >> 8).astype(np.uint8)
        else:
            return np.zeros_like(img, dtype=np.uint8)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def convert_tiff_file(input_path, output_path, method='percentile', first_page_only=False):
    """Convert a TIFF file (possibly multi-page) to 8-bit."""
    try:
        with tiff.TiffFile(input_path) as tif:
            num_pages = len(tif.pages)
            
            if first_page_only:
                # Just convert first page
                img = tif.asarray(key=0)
                img_8bit = convert_tiff_page(img, method)
                tiff.imwrite(output_path, img_8bit, photometric='minisblack')
                print(f"  Converted 1 page -> {output_path}")
                return 1
            
            # Convert all pages
            pages_8bit = []
            for page_idx in range(num_pages):
                img = tif.asarray(key=page_idx)
                img_8bit = convert_tiff_page(img, method)
                pages_8bit.append(img_8bit)
            
            # Write as multi-page TIFF
            tiff.imwrite(output_path, np.array(pages_8bit), photometric='minisblack')
            print(f"  Converted {num_pages} pages -> {output_path}")
            return num_pages
            
    except Exception as e:
        print(f"  ✗ ERROR: {str(e)}")
        return 0


def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_tiffs_for_viewing.py <input_folder> <output_folder> [method] [--first-only]")
        print("")
        print("Convert 16-bit TIFF files to viewable 8-bit versions.")
        print("")
        print("Arguments:")
        print("  input_folder  : Folder containing TIFF files")
        print("  output_folder : Where to save converted files")
        print("  method        : 'percentile' (default, clips outliers), 'linear', or 'minmax'")
        print("  --first-only  : Only convert first page of multi-page files")
        print("")
        print("Example:")
        print("  python convert_tiffs_for_viewing.py /path/to/tiffs /path/to/output percentile")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    method = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else 'percentile'
    first_only = '--first-only' in sys.argv
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    print(f"Converting TIFF files from: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Method: {method}")
    if first_only:
        print("Mode: First page only")
    print()
    
    # Collect all TIFF files
    paths = []
    for root, dirs, files in os.walk(input_folder):
        for f in files:
            if f.lower().endswith((".tif", ".tiff")):
                paths.append(os.path.join(root, f))
    
    if not paths:
        print(f"✗ No TIFF files found in {input_folder}")
        sys.exit(1)
    
    paths.sort()
    print(f"Found {len(paths)} TIFF file(s)\n")
    
    total_pages = 0
    for i, input_path in enumerate(paths, 1):
        rel_path = os.path.relpath(input_path, input_folder)
        output_path = os.path.join(output_folder, os.path.basename(input_path))
        
        # Preserve directory structure if in subfolder
        if os.path.dirname(rel_path):
            subfolder = os.path.join(output_folder, os.path.dirname(rel_path))
            os.makedirs(subfolder, exist_ok=True)
            output_path = os.path.join(subfolder, os.path.basename(input_path))
        
        print(f"[{i}/{len(paths)}] {rel_path}")
        pages = convert_tiff_file(input_path, output_path, method, first_only)
        total_pages += pages
    
    print(f"\n✓ Conversion complete!")
    print(f"  Files converted: {len(paths)}")
    print(f"  Total pages: {total_pages}")
    print(f"\nYou can now view the 8-bit TIFF files in any image viewer.")
    print(f"Original 16-bit files are unchanged.")


if __name__ == "__main__":
    main()

