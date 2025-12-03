# Animated Signaling Pattern Visualization Guide

This guide describes how to create animated heatmaps showing calcium signaling patterns aggregated by poke location regions.

## Overview

The `animate_signaling_patterns.py` script creates animated visualizations that:
1. Group videos by poke location regions (spatial clusters, AP positions, or density-based)
2. Aggregate spark tracks from all videos with similar poke locations
3. Display animated heatmaps showing signal propagation over time
4. Export as animated GIF or MP4 video

## Usage

### Basic Usage

```bash
# Create animation with default settings (3 spatial regions, 60s window)
python animate_signaling_patterns.py spark_tracks.csv \
    --output analysis_results/signaling_animation.gif
```

### Advanced Options

```bash
# Group by AP position instead of spatial clustering
python animate_signaling_patterns.py spark_tracks.csv \
    --output analysis_results/signaling_animation_ap.gif \
    --region-method ap \
    --time-window 120 \
    --frame-interval 2 \
    --fps 5

# Use density-based clustering with custom distance threshold
python animate_signaling_patterns.py spark_tracks.csv \
    --output analysis_results/signaling_animation_density.gif \
    --region-method density \
    --time-window 90 \
    --frame-interval 1.5 \
    --fps 8

# Create MP4 video instead of GIF (requires ffmpeg)
python animate_signaling_patterns.py spark_tracks.csv \
    --output analysis_results/signaling_animation.mp4 \
    --time-window 60 \
    --fps 10
```

## Parameters

### Required Arguments

- `tracks_csv`: Path to `spark_tracks.csv` file

### Optional Arguments

- `--output`, `-o`: Output path for animation (default: `analysis_results/signaling_animation.gif`)
- `--poke-csv`: Path to `poke_locations.csv` (optional, will infer from spark tracks if not provided)
- `--region-method`: Method to group poke locations
  - `spatial`: K-means clustering by spatial coordinates (default)
  - `ap`: Group by AP position (anterior/mid/posterior)
  - `density`: Density-based clustering
- `--n-regions`: Number of regions for spatial clustering (default: 3)
- `--time-window`: Time window to animate in seconds (default: 60)
- `--frame-interval`: Time between frames in seconds (default: 1.0)
- `--fps`: Frames per second for animation (default: 5)
- `--bins`: Number of bins for heatmap resolution (default: 50)
- `--colormap`: Colormap name (default: 'hot', options: 'hot', 'viridis', 'plasma', etc.)

## Region Grouping Methods

### 1. Spatial Clustering (Default)

Groups poke locations by spatial proximity using K-means clustering:

```bash
python animate_signaling_patterns.py spark_tracks.csv \
    --region-method spatial \
    --n-regions 3
```

- Creates `n_regions` clusters based on X,Y coordinates
- Regions are automatically sorted by position
- Best for identifying spatial patterns in poke locations

### 2. AP Position Grouping

Groups by anatomical position (anterior/mid/posterior):

```bash
python animate_signaling_patterns.py spark_tracks.csv \
    --region-method ap
```

- Requires AP position data (`ap_norm` column)
- Creates 3 regions: anterior (0-0.33), mid (0.33-0.67), posterior (0.67-1.0)
- Best for analyzing position-dependent signaling patterns

### 3. Density-Based Clustering

Groups by spatial density with variable number of regions:

```bash
python animate_signaling_patterns.py spark_tracks.csv \
    --region-method density
```

- Groups poke locations within a distance threshold (default: 200 pixels)
- Creates variable number of regions based on data
- Best for identifying natural clusters in poke locations

## Output Formats

### Animated GIF

Default format, good for presentations and web:

```bash
python animate_signaling_patterns.py spark_tracks.csv \
    --output animation.gif \
    --fps 5
```

### MP4 Video

Requires `ffmpeg` to be installed:

```bash
python animate_signaling_patterns.py spark_tracks.csv \
    --output animation.mp4 \
    --fps 10
```

## Example Workflows

### Quick Preview (Fast)

```bash
# Short time window, lower resolution for quick testing
python animate_signaling_patterns.py spark_tracks.csv \
    --output preview.gif \
    --time-window 30 \
    --frame-interval 2 \
    --fps 5 \
    --bins 30
```

### High-Quality Animation (Slow)

```bash
# Longer time window, higher resolution
python animate_signaling_patterns.py spark_tracks.csv \
    --output high_quality.gif \
    --time-window 120 \
    --frame-interval 0.5 \
    --fps 10 \
    --bins 100
```

### AP-Based Analysis

```bash
# Analyze by anatomical position
python animate_signaling_patterns.py spark_tracks.csv \
    --output ap_signaling.gif \
    --region-method ap \
    --time-window 90 \
    --frame-interval 1 \
    --fps 6
```

## Tips

1. **Time Window**: Start with shorter windows (30-60s) to see immediate responses, then extend for long-term patterns
2. **Frame Interval**: Smaller intervals (0.5-1s) show smoother animations but take longer to render
3. **FPS**: 5-10 fps works well for most visualizations
4. **Bins**: 50 bins is a good balance; increase for finer detail, decrease for faster rendering
5. **Colormaps**: Try different colormaps ('hot', 'viridis', 'plasma', 'inferno') for different visual effects

## Troubleshooting

### No Events in Some Regions

Some regions may have few or no events, especially with short time windows. Try:
- Increasing `--time-window`
- Using `--region-method ap` to ensure more balanced grouping
- Checking which files are in each region with `plot_poke_locations.py`

### Animation Too Slow to Render

- Reduce `--bins` (e.g., 30 instead of 50)
- Increase `--frame-interval` (fewer frames)
- Reduce `--time-window` (fewer frames)
- Use lower `--fps`

### Large File Sizes

- Reduce `--fps` (fewer frames per second)
- Reduce `--bins` (lower resolution)
- Use MP4 format instead of GIF (better compression)

## Related Scripts

- `plot_poke_locations.py`: Visualize all poke locations to understand grouping
- `analyze_spatial_matching.py`: Analyze spatial matching patterns (static)
- `visualize_spark_tracks.py`: Other visualization options

