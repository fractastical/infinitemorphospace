#!/usr/bin/env python3
"""
Convert TIFF image sequences to MP4 video files.

This script converts multi-page TIFF files (image sequences) into MP4 videos
that can be played on Mac or any video player.

Usage:
    python tiff_to_mp4.py <input_tiff> <output_mp4> [--fps 10] [--scale 1.0] [--contrast]
    python tiff_to_mp4.py <input_folder> <output_folder> [--fps 10] [--batch]
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

try:
    import tifffile as tiff
except ImportError:
    print("✗ Error: tifffile not installed. Install with: pip install tifffile")
    sys.exit(1)

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("⚠ Warning: OpenCV not available. Will try imageio instead.")

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    if not HAS_OPENCV:
        print("✗ Error: Neither OpenCV nor imageio available.")
        print("  Install with: pip install opencv-python OR pip install imageio imageio-ffmpeg")
        sys.exit(1)


def convert_16bit_to_8bit(img, method='percentile'):
    """Convert 16-bit image to 8-bit with contrast enhancement."""
    if img.dtype != np.uint16:
        if img.dtype != np.uint8:
            return img.astype(np.uint8)
        return img
    
    if method == 'percentile':
        # Use percentile-based stretching (clips outliers)
        p2 = np.percentile(img, 2)
        p98 = np.percentile(img, 98)
        img_scaled = np.clip((img.astype(np.float32) - p2) / (p98 - p2 + 1e-6) * 255, 0, 255)
        return img_scaled.astype(np.uint8)
    else:
        # Simple linear stretch
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img_scaled = ((img.astype(np.float32) - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_scaled = np.zeros_like(img, dtype=np.uint8)
        return img_scaled


def convert_tiff_to_mp4_opencv(tiff_path, output_path, fps=10, scale=1.0, enhance_contrast=True):
    """Convert TIFF to MP4 using OpenCV."""
    print(f"  Reading TIFF: {Path(tiff_path).name}")
    
    with tiff.TiffFile(tiff_path) as tif:
        num_pages = len(tif.pages)
        if num_pages == 0:
            print(f"  ✗ No pages found in TIFF")
            return False
        
        # Get dimensions from first page
        first_img = tif.asarray(key=0)
        h, w = first_img.shape[:2]
        
        # Scale dimensions if needed
        if scale != 1.0:
            w = int(w * scale)
            h = int(h * scale)
        
        # Always use color (BGR) for video writer - we'll convert grayscale to BGR
        is_color = True
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h), is_color)
        
        if not video_writer.isOpened():
            print(f"  ✗ Failed to open video writer")
            return False
        
        print(f"  Processing {num_pages} frames ({w}x{h}, {fps} fps)...")
        
        for page_idx in range(num_pages):
            img = tif.asarray(key=page_idx)
            
            # Convert to 8-bit if needed
            if img.dtype == np.uint16:
                if enhance_contrast:
                    img = convert_16bit_to_8bit(img, method='percentile')
                else:
                    img = (img >> 8).astype(np.uint8)  # Simple downscale
            
            # Convert grayscale to BGR if needed (OpenCV uses BGR)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 3:
                # Assume RGB, convert to BGR
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Resize if needed
            if scale != 1.0:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            
            video_writer.write(img)
            
            if (page_idx + 1) % 50 == 0:
                print(f"    Processed {page_idx + 1}/{num_pages} frames...")
        
        video_writer.release()
        print(f"  ✓ Saved: {Path(output_path).name}")
        return True


def convert_tiff_to_mp4_imageio(tiff_path, output_path, fps=10, scale=1.0, enhance_contrast=True):
    """Convert TIFF to MP4 using imageio."""
    print(f"  Reading TIFF: {Path(tiff_path).name}")
    
    try:
        # Read all frames
        with tiff.TiffFile(tiff_path) as tif:
            num_pages = len(tif.pages)
            if num_pages == 0:
                print(f"  ✗ No pages found in TIFF")
                return False
            
            print(f"  Processing {num_pages} frames ({fps} fps)...")
            
            frames = []
            for page_idx in range(num_pages):
                img = tif.asarray(key=page_idx)
                
                # Convert to 8-bit if needed
                if img.dtype == np.uint16:
                    if enhance_contrast:
                        img = convert_16bit_to_8bit(img, method='percentile')
                    else:
                        img = (img >> 8).astype(np.uint8)
                
                # Resize if needed
                if scale != 1.0:
                    h, w = img.shape[:2]
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    if HAS_OPENCV:
                        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    else:
                        from PIL import Image
                        pil_img = Image.fromarray(img)
                        img = np.array(pil_img.resize((new_w, new_h), Image.LANCZOS))
                
                frames.append(img)
                
                if (page_idx + 1) % 50 == 0:
                    print(f"    Processed {page_idx + 1}/{num_pages} frames...")
        
        # Write video
        print(f"  Writing MP4...")
        imageio.mimwrite(output_path, frames, fps=fps, codec='libx264', quality=8)
        print(f"  ✓ Saved: {Path(output_path).name}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def convert_tiff_to_mp4(tiff_path, output_path, fps=10, scale=1.0, enhance_contrast=True):
    """Convert TIFF to MP4 using available backend."""
    if HAS_OPENCV:
        return convert_tiff_to_mp4_opencv(tiff_path, output_path, fps, scale, enhance_contrast)
    elif HAS_IMAGEIO:
        return convert_tiff_to_mp4_imageio(tiff_path, output_path, fps, scale, enhance_contrast)
    else:
        print("✗ No video backend available")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert TIFF image sequences to MP4 video files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  python tiff_to_mp4.py input.tif output.mp4 --fps 10
  
  # Convert with contrast enhancement
  python tiff_to_mp4.py input.tif output.mp4 --fps 10 --contrast
  
  # Convert all TIFFs in folder (batch mode)
  python tiff_to_mp4.py /path/to/tiffs /path/to/output --batch --fps 10
  
  # Scale down video (0.5 = half size)
  python tiff_to_mp4.py input.tif output.mp4 --scale 0.5 --fps 10

Mac Viewing Options:
  - QuickTime Player (built-in): Can play MP4 files directly
  - VLC: Free, handles many formats
  - Preview: Can view individual TIFF frames
        """
    )
    
    parser.add_argument('input', help='Input TIFF file or folder')
    parser.add_argument('output', help='Output MP4 file or folder')
    parser.add_argument('--fps', type=float, default=10, help='Frames per second (default: 10)')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale factor (default: 1.0, use 0.5 for half size)')
    parser.add_argument('--contrast', action='store_true', help='Enhance contrast for 16-bit images')
    parser.add_argument('--batch', action='store_true', help='Batch mode: convert all TIFFs in input folder')
    parser.add_argument('--in-place', action='store_true', help='Place MP4 files next to TIFF files (same folder)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"✗ Error: Input path does not exist: {input_path}")
        return 1
    
    if args.batch or input_path.is_dir():
        # Batch mode
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        
        tiff_files = list(input_path.rglob('*.tif')) + list(input_path.rglob('*.tiff'))
        if not tiff_files:
            print(f"✗ No TIFF files found in {input_path}")
            return 1
        
        print(f"Found {len(tiff_files)} TIFF file(s)")
        print(f"Output folder: {output_path}")
        print(f"FPS: {args.fps}, Scale: {args.scale}, Contrast: {args.contrast}\n")
        
        success_count = 0
        for i, tiff_file in enumerate(sorted(tiff_files), 1):
            rel_path = tiff_file.relative_to(input_path)
            
            if args.in_place:
                # Place MP4 in same folder as TIFF
                output_mp4 = tiff_file.parent / f"{tiff_file.stem}.mp4"
            else:
                # Place MP4 in output folder, preserving subfolder structure
                output_mp4 = output_path / rel_path.with_suffix('.mp4')
                output_mp4.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"[{i}/{len(tiff_files)}] {rel_path}")
            if convert_tiff_to_mp4(str(tiff_file), str(output_mp4), args.fps, args.scale, args.contrast):
                success_count += 1
            print()
        
        print(f"✓ Batch conversion complete!")
        print(f"  Successfully converted: {success_count}/{len(tiff_files)}")
        
    else:
        # Single file mode
        if input_path.suffix.lower() not in ['.tif', '.tiff']:
            print(f"✗ Error: Not a TIFF file: {input_path}")
            return 1
        
        if output_path.suffix.lower() != '.mp4':
            # If output is a directory, create MP4 with same name
            if output_path.is_dir():
                output_mp4 = output_path / f"{input_path.stem}.mp4"
            else:
                output_mp4 = output_path.with_suffix('.mp4')
        else:
            output_mp4 = output_path
        
        # Create output directory if needed
        output_mp4.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Converting: {input_path.name}")
        print(f"Output: {output_mp4.name}")
        print(f"FPS: {args.fps}, Scale: {args.scale}, Contrast: {args.contrast}\n")
        
        if convert_tiff_to_mp4(str(input_path), str(output_mp4), args.fps, args.scale, args.contrast):
            print(f"\n✓ Conversion complete!")
            print(f"  You can now play {output_mp4.name} in QuickTime Player or VLC")
            return 0
        else:
            return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
