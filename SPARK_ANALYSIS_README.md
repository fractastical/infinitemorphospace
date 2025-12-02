# Spark Tracks Analysis Tools

This directory contains tools for analyzing and visualizing Ca²⁺ wave data from the `wave-vector-tiff-parser.py` pipeline.

## Files

### 1. `spark_tracks_to_clusters.py`
Converts per-frame `spark_tracks.csv` into per-cluster summaries (`vector_clusters.csv`).

**Usage:**
```bash
python spark_tracks_to_clusters.py spark_tracks.csv [vector_clusters.csv]
```

**Output:** `vector_clusters.csv` with one row per track, containing:
- Duration and frame counts
- Start/end positions
- Net displacement and path length
- Speed statistics (mean, peak, net)
- Direction statistics (mean angle, dispersion)
- Area statistics
- Distance from poke site (if available)

### 2. `visualize_spark_tracks.py`
Creates various visualizations of the spark track data.

**Usage:**
```bash
# Show all plots interactively
python visualize_spark_tracks.py spark_tracks.csv

# Save all plots to a directory
python visualize_spark_tracks.py spark_tracks.csv --output-dir plots/

# Generate specific plots
python visualize_spark_tracks.py spark_tracks.csv --plot trajectories
python visualize_spark_tracks.py spark_tracks.csv --plot time
python visualize_spark_tracks.py spark_tracks.csv --plot heatmap
python visualize_spark_tracks.py spark_tracks.csv --plot embryo

# With clusters data for speed analysis
python visualize_spark_tracks.py spark_tracks.csv --clusters-csv vector_clusters.csv --plot speed
```

**Available plots:**
- `trajectories`: Overlay of all spark trajectories (start=green, end=red)
- `speed`: Distribution of propagation speeds (requires clusters CSV)
- `time`: Time series of active tracks and integrated signal
- `heatmap`: Spatial density map of Ca²⁺ events
- `embryo`: Comparison of dynamics between embryos (if embryo_id available)

## Workflow

1. **Generate tracks CSV** (if not already done):
   ```bash
   python wave-vector-tiff-parser.py /path/to/tiffs --poke-frame 100 --fps 10
   ```

2. **Generate clusters CSV**:
   ```bash
   python spark_tracks_to_clusters.py spark_tracks.csv
   ```

3. **Create visualizations**:
   ```bash
   python visualize_spark_tracks.py spark_tracks.csv --clusters-csv vector_clusters.csv --output-dir analysis_plots/
   ```

## Dependencies

- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `matplotlib` - Plotting

Install with:
```bash
pip install pandas numpy matplotlib
```

## Data Format

See `wave-vector.md` for detailed documentation of the CSV formats.

