# Ca²⁺ Wave Simulation Framework

This directory contains tools for generating simulated Ca²⁺ wave data to test hypotheses and compare against real experimental data.

## Overview

The simulation framework allows you to:
- **Extract parameters from real experimental data** to calibrate simulations
- Generate synthetic spark tracking data with configurable parameters
- Test hypotheses with controlled conditions (multiple embryos, poke locations, etc.)
- Compare simulated results against real experimental data
- Validate analysis methods and identify potential artifacts

**Key Feature:** Simulations can be **calibrated from your real data**, ensuring realistic wave speeds, decay rates, and spatial patterns that match your experimental conditions.

## Files

- **`generate_simulated_data.py`**: Main script to generate simulated data
- **`calibrate_from_real_data.py`**: Extract parameters from real experimental data to calibrate simulations
- **`visualize_simulation.py`**: Visualize and compare simulated data
- **`run_all_scenarios.sh`**: Batch script to generate all predefined scenarios
- **`README.md`**: This file

## Quick Start

### Step 1: Calibrate from Real Data (Recommended)

Extract parameters from your real experimental data to make simulations realistic:

```bash
# Extract parameters from your real data
python3 calibrate_from_real_data.py \
    ../spark_tracks.csv \
    --clusters-csv ../vector_clusters.csv \
    --output calibrated_config.json

# This will extract:
# - Wave speeds from your data
# - Activity decay rates
# - Embryo positions and sizes
# - Poke locations
```

### Step 2: Generate Simulation Using Calibrated Parameters

```bash
# Use the calibrated configuration
python3 generate_simulated_data.py \
    --config calibrated_config.json \
    --output simulated_from_real_data.csv \
    --duration 30.0
```

### Generate a Basic Simulation (Default Parameters)

```bash
# Two embryos, head-head orientation
python3 generate_simulated_data.py \
    --scenario two_embryos_head_head \
    --output simulated_two_embryos.csv \
    --duration 30.0

# Three embryos in triangle formation
python3 generate_simulated_data.py \
    --scenario three_embryos_triangle \
    --output simulated_three_embryos.csv \
    --duration 45.0
```

### Visualize Simulation

```bash
# View simulation overview
python3 visualize_simulation.py simulated_two_embryos.csv \
    --output simulation_plot.png

# Compare with real data
python3 visualize_simulation.py simulated_two_embryos.csv \
    --real-csv ../spark_tracks.csv \
    --compare \
    --output comparison.png
```

## Available Scenarios

1. **`two_embryos_head_head`**: Two embryos oriented head-to-head, poke in embryo A
2. **`three_embryos_triangle`**: Three embryos in triangular formation
3. **`two_embryos_tail_tail`**: Two embryos tail-to-tail, poke at tail of A
4. **`multiple_pokes_same_embryo`**: Single embryo with multiple poke events at different times

## Custom Scenarios

To create custom scenarios, modify the `create_experiment_scenarios()` function in `generate_simulated_data.py` or create a JSON config file (coming soon).

## Output Format

The simulation generates data in the same format as `spark_tracks.csv`:
- `track_id`: Unique identifier for each spark cluster
- `frame_idx`: Frame index
- `time_s`: Time relative to poke (0 = poke time)
- `x`, `y`: Spatial coordinates
- `vx`, `vy`, `speed`, `angle_deg`: Velocity information
- `area`: Activity/spark size
- `embryo_id`: Which embryo the event belongs to
- `ap_norm`: AP position (0=head, 1=tail)
- `dist_from_poke_px`: Distance from poke location
- `filename`: Source file identifier

## Parameters

### Wave Configuration

- `speed_px_per_s`: Wave propagation speed (default: 5.0)
- `duration_s`: How long waves persist (default: 10.0)
- `decay_rate`: Activity decay over time (default: 0.1)
- `radial`: Radial vs directional propagation (default: True)
- `direction_deg`: Direction if not radial (optional)

### Embryo Configuration

- Position: `center_x`, `center_y`
- Size: `length`, `width` (pixels)
- Orientation: `angle` (degrees)
- Head/tail positions: `head_x`, `head_y`, `tail_x`, `tail_y`

### Poke Configuration

- Position: `x`, `y`
- Target embryo: `embryo_id`
- Timing: `time` (seconds)

## Use Cases

1. **Hypothesis Testing**: Test if analysis methods correctly identify patterns
2. **Parameter Sensitivity**: See how changes in wave speed, decay, etc. affect results
3. **Edge Cases**: Test with unusual configurations (many embryos, complex geometries)
4. **Method Validation**: Compare analysis results on simulated vs real data
5. **Training Data**: Generate data for machine learning or method development

## Comparison with Real Data

Use the comparison visualization to:
- Validate that simulation parameters match real wave dynamics
- Identify discrepancies between simulated and real data
- Adjust simulation parameters to better match observations
- Test whether hypotheses can be distinguished in simulated data

## Example Workflow

### Option 1: Data-Driven Simulation (Recommended)

```bash
# 1. Extract parameters from your real experimental data
python3 calibrate_from_real_data.py \
    ../spark_tracks.csv \
    --clusters-csv ../vector_clusters.csv \
    --output calibrated_config.json

# 2. Generate simulation using real data parameters
python3 generate_simulated_data.py \
    --config calibrated_config.json \
    --output sim_from_real_data.csv \
    --duration 30.0

# 3. Process with same analysis pipeline
python3 ../spark_tracks_to_clusters.py sim_from_real_data.csv sim_clusters.csv

# 4. Compare simulated vs real data
python3 visualize_simulation.py sim_from_real_data.csv \
    --real-csv ../spark_tracks.csv \
    --compare
```

### Option 2: Predefined Scenarios

```bash
# 1. Generate simulation with default parameters
python3 generate_simulated_data.py \
    --scenario two_embryos_head_head \
    --output sim_data.csv

# 2. Process and analyze
python3 ../spark_tracks_to_clusters.py sim_data.csv sim_clusters.csv
python3 ../analyze_experimental_hypotheses.py sim_data.csv \
    --clusters-csv sim_clusters.csv \
    --analysis activity
```

## Future Enhancements

- [ ] Custom scenario JSON config files
- [ ] More realistic wave propagation models
- [ ] Noise and artifact simulation
- [ ] Batch generation of multiple scenarios
- [ ] Statistical comparison tools
- [ ] Automated parameter fitting to real data

## Notes

- Simulated data uses simplified wave propagation models
- Real data may have artifacts, noise, and complex dynamics not captured
- Use simulations to understand expected patterns, but validate with real data
- Simulation parameters should be tuned to match observed experimental conditions

