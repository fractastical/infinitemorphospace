# Embryo Mask Generation

This directory contains scripts and outputs for generating embryo masks from TIFF images.

## Overview

The embryo mask generation process creates binary masks that identify embryo regions in TIFF images. These masks are used for training machine learning models and excluding background pixels.

## Scripts

### `create_size_constrained_masks.py`

Main script for generating size-constrained embryo masks.

**Features:**
- Uses wider greyscale range (5th percentile) to capture dimmer embryo regions
- Filters by expected embryo size (0.5% - 15% of image) to exclude background
- Expands masks by configurable percentage (default: 10%)
- Skips blank masks automatically

**Usage:**
```bash
# Process single file
python create_size_constrained_masks.py path/to/image.tif

# Process all TIFF files in directory
python create_size_constrained_masks.py /path/to/tiffs --batch

# With custom parameters
python create_size_constrained_masks.py /path/to/tiffs --batch \
    --min-area 0.005 \
    --max-area 0.15 \
    --expand-percent 10.0 \
    --skip-blank \
    --output-dir ./masks
```

**Key Parameters:**
- `--min-area`: Minimum embryo area as fraction of image (default: 0.005 = 0.5%)
- `--max-area`: Maximum embryo area as fraction of image (default: 0.15 = 15%)
- `--percentile-low`: Lower percentile for thresholding (default: 5.0)
- `--expand-percent`: Expand mask size by this percentage (default: 10.0%)
- `--skip-blank`: Skip saving masks with 0% coverage
- `--no-visualize`: Skip visualization generation
- `--no-save`: Skip saving mask files

## Output Directories

### `embryo_masks_final/`
Final masks with 10% expansion, blank masks skipped.
- Contains: Binary PNG masks, CSV summary
- Coverage: ~10% average (2-40% range)
- Files: Only masks with detected embryos

### `embryo_masks_size_constrained/`
Initial size-constrained masks without expansion.
- Contains: Binary PNG masks, CSV summary
- Coverage: ~7% average (0-27% range)

### `embryo_masks/`
Permissive masks (for comparison only).
- Contains: Very permissive masks (~97% coverage)
- Note: Includes too much background, not recommended for training

## Mask Statistics

### Current Results (with 10% expansion):
- **Average coverage**: 10.41%
- **Median coverage**: 5.94%
- **Range**: 0% - 40.64%
- **Files with masks**: 64 out of 109 files
- **Blank masks skipped**: 45 files

### Comparison with Other Methods:

| Method | Average Coverage | Issue |
|--------|----------------|-------|
| Current Parser | 30.5% | Too restrictive, misses dimmer regions |
| Permissive Multi | 97.2% | Too permissive, includes background |
| **Size-Constrained** | **10.4%** | **Recommended - filters by size** |

## File Formats

### Mask PNG Files
- Format: Binary PNG (black = background, white = embryo)
- Naming: `{original_filename}_mask_frame{frame_idx}.png`
- Size: Varies by image dimensions

### Summary CSV
- Columns: File, Coverage_%, Mask_Area, Image_Width, Image_Height, Min_Area, Max_Area, Contours_Before_Filter, Contours_After_Filter, Embryo_Threshold, Background_Threshold
- Location: `size_constrained_coverage_summary.csv`

## Git Ignore

All PNG files are automatically ignored by git (see `.gitignore`). This includes:
- Mask PNG files
- Detection visualization PNGs
- Any other PNG outputs

To remove already-tracked PNG files from git:
```bash
git rm --cached wave-vector-analysis/**/*.png
```

## Workflow

1. **Generate masks**: Run `create_size_constrained_masks.py` with `--batch` flag
2. **Review results**: Check the CSV summary for coverage statistics
3. **Adjust parameters**: If needed, adjust `--min-area`, `--max-area`, or `--expand-percent`
4. **Use for training**: Masks are ready to use with the corresponding TIFF files

## Troubleshooting

### Many blank masks (0% coverage)
- Try more lenient size constraints: `--min-area 0.003 --max-area 0.20`
- Lower the percentile threshold: `--percentile-low 3.0`
- Check if images have unusual contrast or lighting

### Masks too large (include background)
- Reduce max area: `--max-area 0.10`
- Reduce expansion: `--expand-percent 5.0`

### Masks too small (miss embryo regions)
- Increase expansion: `--expand-percent 15.0`
- Increase max area: `--max-area 0.20`
- Lower percentile: `--percentile-low 3.0`

## Related Scripts

- `compare_thresholds.py`: Compare current parser vs. permissive thresholding
- `generate_coverage_report.py`: Generate detailed coverage comparison reports
