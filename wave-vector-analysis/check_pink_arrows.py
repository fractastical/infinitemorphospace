#!/usr/bin/env python3
"""
Quick diagnostic script to check if TIFF files contain pink arrows
that could be used for poke detection.
"""

import cv2
import numpy as np
import sys
from pathlib import Path
import tifffile

def check_pink_arrows(image_path, page_idx=0):
    """Check if an image contains pink/magenta pixels (potential arrow overlay)."""
    try:
        # Try reading with tifffile first (handles multi-page)
        try:
            with tifffile.TiffFile(image_path) as tif:
                if page_idx < len(tif.pages):
                    raw = tif.pages[page_idx].asarray()
                    is_bgr = False
                else:
                    return None
        except:
            # Fallback to OpenCV
            raw = cv2.imread(str(image_path))
            if raw is None:
                return None
            is_bgr = True
        
        # Convert to BGR if needed
        if raw.ndim == 2:
            # Grayscale - no color overlay possible
            return None
        
        if is_bgr:
            bgr = raw
        else:
            # tifffile gives RGB, convert to BGR
            if raw.shape[2] >= 3:
                rgb = raw[:, :, :3]
                bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
            else:
                return None
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        
        # Pink / magenta range in HSV (same as parser uses)
        lower_pink = np.array([140, 50, 80], dtype=np.uint8)
        upper_pink = np.array([179, 255, 255], dtype=np.uint8)
        pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)
        
        pink_pixels = np.sum(pink_mask > 0)
        total_pixels = pink_mask.size
        pink_percentage = (pink_pixels / total_pixels) * 100
        
        return {
            'pink_pixels': pink_pixels,
            'total_pixels': total_pixels,
            'percentage': pink_percentage,
            'has_pink': pink_pixels > 100  # Threshold: at least 100 pink pixels
        }
    except Exception as e:
        return {'error': str(e)}

def scan_folder(folder_path, max_files=20):
    """Scan a folder for TIFF files and check for pink arrows."""
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Error: Folder not found: {folder_path}")
        return
    
    tiff_files = sorted(list(folder.rglob("*.tif*")))
    if not tiff_files:
        print(f"No TIFF files found in {folder_path}")
        return
    
    print(f"Found {len(tiff_files)} TIFF file(s)")
    print(f"Checking first {min(max_files, len(tiff_files))} file(s)...\n")
    
    files_with_pink = []
    
    for i, tif_path in enumerate(tiff_files[:max_files]):
        print(f"[{i+1}/{min(max_files, len(tiff_files))}] {tif_path.name}")
        
        # Check first page
        result = check_pink_arrows(tif_path, page_idx=0)
        
        if result is None:
            print("  → Grayscale or couldn't read\n")
            continue
        
        if 'error' in result:
            print(f"  → Error: {result['error']}\n")
            continue
        
        if result['has_pink']:
            files_with_pink.append(tif_path)
            print(f"  ✓ PINK DETECTED: {result['pink_pixels']:,} pixels ({result['percentage']:.2f}%)\n")
        else:
            print(f"  ✗ No pink detected ({result['pink_pixels']} pixels, {result['percentage']:.4f}%)\n")
        
        # If multi-page, also check a few other pages
        try:
            with tifffile.TiffFile(tif_path) as tif:
                num_pages = len(tif.pages)
                if num_pages > 1:
                    print(f"     (Multi-page: {num_pages} pages)")
                    # Check middle and last page
                    for check_idx in [num_pages // 2, num_pages - 1]:
                        if check_idx != 0 and check_idx < num_pages:
                            page_result = check_pink_arrows(tif_path, page_idx=check_idx)
                            if page_result and 'error' not in page_result:
                                if page_result['has_pink']:
                                    print(f"     ✓ Page {check_idx + 1}: {page_result['pink_pixels']:,} pink pixels")
                                else:
                                    print(f"     ✗ Page {check_idx + 1}: no pink")
                    print()
        except:
            pass
    
    print("\n" + "="*60)
    print(f"SUMMARY: {len(files_with_pink)}/{min(max_files, len(tiff_files))} files contain pink arrows")
    
    if files_with_pink:
        print("\nFiles with pink arrows:")
        for f in files_with_pink:
            print(f"  - {f}")
        print("\n✓ Poke detection should work automatically!")
    else:
        print("\n⚠ No pink arrows detected in sampled files.")
        print("  → Poke detection may require manual coordinates (--poke-x/--poke-y)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_pink_arrows.py <tiff_folder> [max_files]")
        print("\nExample:")
        print("  python check_pink_arrows.py embryos/2 10")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    max_files = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    
    scan_folder(folder_path, max_files)

