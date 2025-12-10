# Artistic Visualizations

This directory contains tools for creating visually striking and artistic visualizations from Ca²⁺ wave data.

## Files

- **`create_artistic_visualizations.py`** - Main script for generating artistic visualizations
- **`ARTISTIC_VISUALIZATIONS.md`** - Complete guide and documentation

## Quick Start

From this directory:

```bash
python create_artistic_visualizations.py ../spark_tracks.csv \
    --clusters-csv ../vector_clusters.csv \
    --output-dir ../analysis_results/artistic
```

From the project root:

```bash
python wave-vector-analysis/artistic-visualizations/create_artistic_visualizations.py \
    wave-vector-analysis/spark_tracks.csv \
    --clusters-csv wave-vector-analysis/vector_clusters.csv \
    --output-dir wave-vector-analysis/analysis_results/artistic
```

## Available Visualizations

1. **Flow Field Paintings** - Spatial vector fields showing wave propagation (scientifically useful)
2. **3D Time-Space Sculptures** - Temporal-spatial dynamics with time as z-axis (scientifically useful)
3. **Speed Gradient Flow** - Spatial mapping of propagation speeds (scientifically useful)
4. **Particle Trail Animations** - Animated flowing particle trails (useful for presentations)

These visualizations complement existing analysis tools by providing spatial vector fields, combined temporal-spatial views, spatial speed mapping, and animated temporal dynamics that aren't available in standard plots.

See `ARTISTIC_VISUALIZATIONS.md` for detailed documentation.

