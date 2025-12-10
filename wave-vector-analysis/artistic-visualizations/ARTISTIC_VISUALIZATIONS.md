# Scientific Visualizations Guide

This guide describes how to create scientifically useful visualizations from your Ca²⁺ wave data that complement existing analysis tools.

## Overview

The `create_artistic_visualizations.py` script generates visualizations that provide new insights not available in standard analysis tools:
- **Spatial vector fields** (not just direction distributions)
- **Temporal-spatial dynamics** (not just separate 2D/1D views)
- **Spatial speed mapping** (not just speed histograms)

These visualizations are useful for:
- Understanding wave propagation patterns
- Identifying spatial flow patterns and vortices
- Analyzing temporal-spatial correlations
- Mapping speed gradients across space

## Available Visualizations

### 1. **Flow Field Paintings** (`flow`)
Spatial vector field visualizations showing wave propagation directions and speeds.

**Scientific Value:** HIGH - Shows spatial vector fields which existing tools (rose plots) don't provide.

**Styles:**
- `aurora` - Purple/pink colors (default)
- `fire` - Red/orange/yellow colors
- `ocean` - Blue/cyan colors
- `neon` - Green colors

**What it shows:** Wave propagation directions as flowing vectors overlaid on speed-based color gradients. Useful for identifying flow patterns, vortices, and directional changes across space.

### 2. **3D Time-Space Sculptures** (`3d`)
Three-dimensional visualizations with time as the z-axis.

**Scientific Value:** MEDIUM-HIGH - Combines temporal and spatial dimensions in a way existing 2D heatmaps and 1D time series don't.

**What it shows:** Spatial distribution of Ca²⁺ events evolving over time, color-coded by time point. Useful for understanding how spatial patterns evolve temporally.

### 3. **Speed Gradient Flow** (`gradient`)
Spatial mapping of propagation speeds with color and size encoding.

**Scientific Value:** MEDIUM - Shows speed mapped spatially, which existing speed histograms don't provide.

**What it shows:** Scatter plot with points colored and sized by propagation speed. Useful for identifying fast vs slow regions and speed gradients across space.

## Usage

### Basic Usage

Create all visualizations with default settings:

```bash
cd wave-vector-analysis/artistic-visualizations
python create_artistic_visualizations.py ../spark_tracks.csv \
    --clusters-csv ../vector_clusters.csv \
    --output-dir ../analysis_results/artistic
```

Or from the project root:

```bash
python wave-vector-analysis/artistic-visualizations/create_artistic_visualizations.py \
    wave-vector-analysis/spark_tracks.csv \
    --clusters-csv wave-vector-analysis/vector_clusters.csv \
    --output-dir wave-vector-analysis/analysis_results/artistic
```

### Create Specific Visualizations

Create only flow field and 3D visualizations:

```bash
python create_artistic_visualizations.py ../spark_tracks.csv \
    --visualizations flow 3d \
    --style aurora \
    --output-dir ../analysis_results/artistic
```

### Customize Color Schemes

Use different color styles for flow fields:

```bash
# Fire-themed flow field
python create_artistic_visualizations.py ../spark_tracks.csv \
    --visualizations flow \
    --style fire \
    --output-dir ../analysis_results/artistic

# Ocean-themed flow field
python create_artistic_visualizations.py ../spark_tracks.csv \
    --visualizations flow \
    --style ocean \
    --output-dir ../analysis_results/artistic
```

## Output Files

All visualizations are saved to the specified output directory:

- `flow_field_{style}.png` - Flow field painting (spatial vector fields)
- `3d_time_sculpture.png` - 3D time-space visualization (temporal-spatial dynamics)
- `speed_gradient_flow.png` - Speed gradient visualization (spatial speed mapping)

## Tips for Best Results

1. **High Resolution**: All images are saved at 300 DPI, perfect for printing or high-quality displays.

2. **Color Schemes**: Choose flow field colors based on your presentation needs:
   - `aurora` - Good contrast, works well for presentations
   - `fire` - Warm colors, good for highlighting intensity
   - `ocean` - Cool colors, good for calm/steady patterns
   - `neon` - High contrast, good for highlighting details

3. **Performance**: The script automatically samples large datasets for performance. For very large datasets (>100K points), consider filtering your data first.

4. **Time Windows**: For 3D visualizations, adjust the time window to focus on specific time periods of interest (e.g., immediate post-poke response).

## Example Workflows

### Create All Scientific Visualizations

```bash
# Generate all visualizations
python create_artistic_visualizations.py ../spark_tracks.csv \
    --visualizations all \
    --output-dir ../analysis_results/artistic
```

### Create Presentation Materials

```bash
# High-quality figures for presentations
python create_artistic_visualizations.py ../spark_tracks.csv \
    --visualizations flow 3d gradient \
    --style aurora \
    --output-dir ../presentation_figures
```

### Compare Different Color Schemes

```bash
# Generate flow fields with different color schemes
for style in aurora fire ocean neon; do
    python create_artistic_visualizations.py ../spark_tracks.csv \
        --visualizations flow \
        --style $style \
        --output-dir ../analysis_results/artistic/$style
done
```

## Technical Details

- **Image Format**: PNG for all visualizations
- **Resolution**: 300 DPI (high quality)
- **Color Space**: RGB with black backgrounds
- **Interpolation**: Bilinear for smooth gradients
- **Sampling**: Automatic sampling for datasets >50K points

## How These Complement Existing Tools

These visualizations provide insights not available in standard analysis tools:

| Visualization | What's New | Existing Tool Comparison |
|--------------|------------|-------------------------|
| **Flow Field** | Spatial vector fields | Rose plots show direction distributions, not spatial flow |
| **3D Time-Space** | Combined temporal-spatial view | Existing tools show 2D spatial OR 1D temporal, not combined |
| **Speed Gradient** | Spatial speed mapping | Speed histograms show distributions, not spatial patterns |

## Customization

To customize the visualizations further, edit `create_artistic_visualizations.py` in this directory:

- Adjust figure sizes in each function
- Modify color schemes in the `color_schemes` dictionary
- Change sampling rates for performance
- Adjust time windows for 3D visualizations
- Modify grid resolution for flow fields

## Scientific Applications

These visualizations are useful for:
- Identifying wave propagation patterns
- Detecting vortices or directional changes
- Understanding temporal-spatial correlations
- Mapping speed gradients across embryos
- Comparing pre-poke vs post-poke dynamics

