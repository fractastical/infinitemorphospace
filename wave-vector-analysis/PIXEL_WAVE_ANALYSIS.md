# Pixel-Based Wave Analysis

This document describes the pixel-based wave analysis approach, where each spark is treated as a single activated pixel.

## Overview

The pixel-based analysis refactors wave calculations to:
1. **Count activated pixels**: Each spark detection is treated as 1 pixel, regardless of the original cluster area
2. **Measure wave propagation speed**: Track how rapidly waves spread between embryos
3. **Region-specific propagation**: Analyze propagation for specific anatomical regions (e.g., Heart)

## Key Changes

### From Area-Based to Pixel-Based

**Previous approach:**
- Used `area` field from spark detections
- Summed areas to get total signal intensity
- Area could vary significantly between detections

**New approach:**
- Each spark = 1 activated pixel
- Count unique `track_id` values per time point
- More consistent and interpretable metrics

### Updated Scripts

1. **`analyze_pixel_waves.py`**: New analysis script for pixel-based wave analysis
2. **`spark_tracks_to_clusters.py`**: Updated to include `n_pixels` and `n_unique_pixels` fields
3. **`visualize_spark_tracks.py`**: Updated `plot_time_series()` to support pixel counting

## Usage

### Basic Pixel Count Analysis

```bash
python wave-vector-analysis/analyze_pixel_waves.py spark_tracks.csv --output-dir results/
```

This will:
- Count activated pixels over time
- Analyze wave propagation between embryos
- Generate plots showing pixel activation and propagation delays

### Region-Specific Analysis

```bash
python wave-vector-analysis/analyze_pixel_waves.py spark_tracks.csv \
    --output-dir results/ \
    --region Heart
```

This analyzes propagation specifically for the Heart region.

### Custom Time Binning

```bash
python wave-vector-analysis/analyze_pixel_waves.py spark_tracks.csv \
    --output-dir results/ \
    --time-bin 0.2
```

Use 0.2 second time bins instead of the default 0.1 seconds.

## Output Files

### `pixel_counts.csv`
Time series of activated pixel counts:
- `time_s`: Time in seconds (relative to poke)
- `n_pixels`: Number of activated pixels in this time bin
- `n_detections`: Total number of detections (for reference)
- `embryo_id`: Embryo identifier (if available)

### `wave_propagation.csv`
Wave propagation metrics between embryos:
- `source_embryo`: Embryo where wave originates (A or B)
- `target_embryo`: Embryo where wave propagates to (A or B)
- `region`: Anatomical region analyzed (or 'all')
- `source_activation_time`: First time source embryo reaches activation threshold
- `target_activation_time`: First time target embryo reaches activation threshold
- `propagation_delay_s`: Time delay between source and target activation
- `propagation_detected`: Whether propagation was detected
- `source_peak_time`, `target_peak_time`: Peak activation times
- `source_peak_pixels`, `target_peak_pixels`: Peak pixel counts
- `source_activation_rate`, `target_activation_rate`: Pixels per second

### Plots

1. **`pixel_activation_timecourse.png`**: 
   - Top: Activated pixel count over time for each embryo
   - Bottom: Cumulative activated pixel count

2. **`wave_propagation_analysis.png`**:
   - Top-left: Propagation delays by region and direction
   - Top-right: First activation times comparison
   - Bottom-left: Activation rates
   - Bottom-right: Region-specific time series (e.g., Heart)

## Wave Propagation Analysis

The `measure_wave_propagation()` function tracks:

1. **First activation**: When does the source embryo first reach the activation threshold?
2. **Propagation delay**: How long until the target embryo also reaches the threshold?
3. **Activation rates**: How quickly does each embryo activate (pixels/second)?

### Example: Heart Region Propagation

If embryo A has activity in the Heart region at t=5s, and embryo B has activity in the Heart region at t=7s, then:
- `source_activation_time` = 5.0 seconds
- `target_activation_time` = 7.0 seconds  
- `propagation_delay_s` = 2.0 seconds

This indicates a 2-second delay for Heart activity to propagate from embryo A to embryo B.

## Integration with Existing Analysis

The pixel-based analysis complements existing area-based analysis:

- **Area-based**: Useful for understanding signal intensity and spatial extent
- **Pixel-based**: Useful for understanding activation patterns and propagation speed

Both approaches can be used together to get a complete picture of wave dynamics.

## Notes

- The activation threshold (`min_activation_threshold`) defaults to 3 pixels. This can be adjusted based on your data.
- Propagation analysis only considers post-poke time points (t >= 0).
- Region filtering requires the `region` column to be present in `spark_tracks.csv`.
