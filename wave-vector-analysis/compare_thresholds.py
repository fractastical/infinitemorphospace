#!/usr/bin/env python3
"""
Compare current parser thresholding vs. more permissive thresholding to see
how much of the embryo body we're capturing.
"""

import cv2
import numpy as np
import tifffile as tiff
from pathlib import Path
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


def permissive_threshold(gray, percentile_low=5):
    """More permissive thresholding using lower percentile."""
    blur = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    flat_intensities = blur.flatten()
    
    background_percentile = 10
    background_threshold = np.percentile(flat_intensities, background_percentile)
    
    # Use lower percentile
    embryo_threshold = np.percentile(flat_intensities, percentile_low)
    embryo_threshold = max(embryo_threshold, background_threshold * 1.2)
    
    mask = (blur >= embryo_threshold).astype(np.uint8) * 255
    
    # Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return mask, embryo_threshold, background_threshold


def compare_on_file(tiff_path, frame_idx=0):
    """Compare both methods on a single file."""
    print(f"\n{'='*70}")
    print(f"Comparing thresholds: {Path(tiff_path).name}")
    print(f"{'='*70}")
    
    # Read TIFF
    with tiff.TiffFile(tiff_path) as tif:
        raw_data = tif.pages[frame_idx].asarray()
    
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
    
    # Current parser method
    mask_current, thresh_current, bg_current = current_parser_threshold(gray)
    area_current = np.sum(mask_current > 0)
    coverage_current = (area_current / total_pixels) * 100
    
    # Permissive method (5th percentile)
    mask_permissive, thresh_permissive, bg_permissive = permissive_threshold(gray, percentile_low=5)
    area_permissive = np.sum(mask_permissive > 0)
    coverage_permissive = (area_permissive / total_pixels) * 100
    
    # Difference
    diff_area = area_permissive - area_current
    diff_coverage = coverage_permissive - coverage_current
    
    print(f"\nCurrent Parser Method:")
    print(f"  Background threshold: {bg_current:.1f}")
    print(f"  Embryo threshold:      {thresh_current:.1f}")
    print(f"  Mask area:             {area_current:,} pixels")
    print(f"  Coverage:              {coverage_current:.2f}%")
    
    print(f"\nPermissive Method (5th percentile):")
    print(f"  Background threshold: {bg_permissive:.1f}")
    print(f"  Embryo threshold:      {thresh_permissive:.1f}")
    print(f"  Mask area:             {area_permissive:,} pixels")
    print(f"  Coverage:              {coverage_permissive:.2f}%")
    
    print(f"\nDifference:")
    print(f"  Additional pixels:    {diff_area:,} ({diff_coverage:+.2f}%)")
    
    # Intensity statistics
    flat_gray = gray.flatten()
    print(f"\nImage Statistics:")
    print(f"  Min intensity:        {flat_gray.min()}")
    print(f"  Max intensity:        {flat_gray.max()}")
    print(f"  Mean intensity:       {flat_gray.mean():.1f}")
    print(f"  Median intensity:     {np.median(flat_gray):.1f}")
    print(f"  5th percentile:       {np.percentile(flat_gray, 5):.1f}")
    print(f"  25th percentile:      {np.percentile(flat_gray, 25):.1f}")
    
    return {
        'current_coverage': coverage_current,
        'permissive_coverage': coverage_permissive,
        'difference': diff_coverage,
        'current_threshold': thresh_current,
        'permissive_threshold': thresh_permissive,
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python compare_thresholds.py <tiff_file> [frame_idx]")
        print("Example: python compare_thresholds.py /path/to/image.tif 0")
        sys.exit(1)
    
    tiff_path = sys.argv[1]
    frame_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    compare_on_file(tiff_path, frame_idx)
