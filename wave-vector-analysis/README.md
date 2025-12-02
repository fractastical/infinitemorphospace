# Ca²⁺ Wave Vector Analysis

This directory contains tools for detecting, tracking, and analyzing Ca²⁺ signaling waves in time-lapse microscopy images of embryos.

## Overview

The pipeline processes multi-page TIFF images to:
- Detect bright Ca²⁺ "spark" events (GCaMP fluorescence signals)
- Track individual events across frames
- Segment embryos and compute anatomical coordinates (AP/DV axes)
- Generate per-frame and per-cluster summaries for downstream analysis

This follows the methodology described in:
- Shannon et al. 2017 (Biophys J): wound-induced Ca²⁺ signal dynamics
- Tung et al. 2024 (Nat Commun): inter-embryo injury waves in Xenopus laevis

## Files

### Core Scripts

- **`wave-vector-tiff-parser.py`**: Main processing script that:
  - Reads TIFF files (supports multi-page, 16-bit images)
  - Detects and tracks Ca²⁺ sparks frame-by-frame
  - Automatically segments embryos and estimates head-tail axes
  - Detects poke/injury sites (optional)
  - Outputs `spark_tracks.csv` with per-frame measurements

- **`spark_tracks_to_clusters.py`**: Aggregates per-frame data into per-cluster summaries
  - Converts `spark_tracks.csv` → `vector_clusters.csv`
  - Computes speeds, path lengths, durations, directionality metrics

- **`visualize_spark_tracks.py`**: Visualization and analysis tools
  - Trajectory plots
  - Speed/direction distributions
  - Time series analysis
  - Spatial heatmaps
  - Per-embryo comparisons

- **`generate_all_hypothesis_plots.py`**: Generate comprehensive plots for all testable hypotheses
  - Activity analysis with standard deviation
  - Inter-embryo comparisons
  - AP position analysis
  - Speed vs time analysis
  - All plots mapped to specific experimental hypotheses

### Documentation

- **`wave-vector.md`**: Detailed specification of output CSV formats
- **`SPARK_ANALYSIS_README.md`**: Usage guide for analysis scripts
- **`EXPERIMENTAL_HYPOTHESES.md`**: Mapping of experimental hypotheses to specific analyses
- **`PLOTTING_GUIDE.md`**: Complete guide to generating publication-quality plots
- **`TAIL_LABELING.md`**: Methodology for identifying and labeling tail regions

### Analysis Scripts

- **`analyze_experimental_hypotheses.py`**: Comprehensive analysis script for testing experimental claims
  - Calcium activity comparisons
  - Wave directionality analysis
  - Spatial matching
  - Tail response analysis
  - Condition comparisons (contact vs non-contact, etc.)

### Utilities

- **`diagnose_tiffs.py`**: Diagnostic tool to inspect TIFF file properties
- **`convert_tiffs_for_viewing.py`**: Convert 16-bit TIFFs to 8-bit for preview

### Output Data

Generated during processing (not included in repo if large):
- `spark_tracks.csv`: Per-frame Ca²⁺ event data
- `vector_clusters.csv`: Per-cluster summaries
- `plots/`: Visualization outputs

## Quick Start

### 1. Install Dependencies

```bash
pip install opencv-python numpy tifffile pandas matplotlib
```

### 2. Process TIFF Images

```bash
python wave-vector-tiff-parser.py /path/to/tiff/folder \
    --poke-frame 100 \
    --fps 10 \
    --output-video output.mp4
```

**Arguments:**
- `--poke-frame`: Frame index where injury/poke occurs (t=0 reference)
- `--fps`: Frame rate (frames per second)
- `--poke-x`, `--poke-y`: Optional poke coordinates (if auto-detection fails)
- `--output-video`: Optional output video with overlays

### 3. Generate Cluster Summaries

```bash
python spark_tracks_to_clusters.py spark_tracks.csv
```

### 4. Create Visualizations

```bash
python visualize_spark_tracks.py spark_tracks.csv \
    --clusters-csv vector_clusters.csv \
    --output-dir plots/
```

## Features

- **16-bit image support**: Works directly with scientific imaging data
- **Multi-page TIFF support**: Handles time-lapse stacks efficiently
- **Automatic embryo detection**: Dynamic edge-based segmentation
- **Anatomical coordinates**: AP/DV position mapping relative to head-tail axis
- **Spark tracking**: Links detections across frames with gap-filling
- **Memory efficient**: Processes one frame at a time to handle large datasets

## Output Format

See `wave-vector.md` for complete documentation of CSV formats.

### `spark_tracks.csv`
Per-frame measurements with columns:
- `track_id`, `frame_idx`, `time_s`
- `x`, `y`: Position in pixels
- `vx`, `vy`, `speed`, `angle_deg`: Velocity and direction
- `area`: Spatial extent
- `embryo_id`: Which embryo (A/B)
- `ap_norm`, `dv_px`: Anatomical coordinates
- `dist_from_poke_px`: Distance from injury site

### `vector_clusters.csv`
Per-cluster summaries with columns:
- `cluster_id`, `n_frames`, `duration_s`
- `start_x_px`, `end_x_px`, `net_displacement_px`, `path_length_px`
- `mean_speed_px_per_s`, `peak_speed_px_per_s`
- `mean_angle_deg`, `angle_dispersion_deg`
- `total_area_px2_frames`: Integrated signal "volume"

## Example Workflow

```bash
# 1. Process images
python wave-vector-tiff-parser.py ~/data/embryos --poke-frame 50 --fps 15

# 2. Generate summaries
python spark_tracks_to_clusters.py spark_tracks.csv

# 3. Visualize
python visualize_spark_tracks.py spark_tracks.csv \
    --clusters-csv vector_clusters.csv \
    --output-dir analysis_plots/

# 4. Analyze results
# Use pandas, R, or your preferred analysis tool on the CSV files
```

## Notes

- The script automatically detects embryos from the first frames where they're visible
- Poke detection can be automatic (pink arrow overlay) or manual (--poke-x/--poke-y)
- Supports recursive subdirectory traversal for organized data
- Memory-efficient processing handles large time-lapse series

