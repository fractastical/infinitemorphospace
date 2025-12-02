# Calibration Guide: Using Real Data to Inform Simulations

## Overview

The simulation framework can extract parameters from your real experimental data to create **realistic, data-driven simulations**. This ensures that simulated waves match the actual dynamics observed in your experiments.

## Quick Start

```bash
# Step 1: Extract parameters from your real data
python3 calibrate_from_real_data.py \
    ../spark_tracks.csv \
    --clusters-csv ../vector_clusters.csv \
    --output my_experiment_config.json

# Step 2: Generate simulation using these parameters
python3 generate_simulated_data.py \
    --config my_experiment_config.json \
    --output simulated_data.csv \
    --duration 30.0

# Step 3: Compare simulated vs real data
python3 visualize_simulation.py simulated_data.csv \
    --real-csv ../spark_tracks.csv \
    --compare \
    --output comparison.png
```

## What Gets Extracted

The calibration script extracts:

1. **Wave Propagation Parameters**
   - Mean wave speed (px/s)
   - Speed distribution (std, min, max)
   - Activity decay rate over time

2. **Temporal Dynamics**
   - Mean track duration
   - Activity persistence time
   - Temporal decay patterns

3. **Spatial Characteristics**
   - Embryo positions and sizes
   - Spatial spread of waves
   - AP position distributions

4. **Poke Locations**
   - Positions of poke/injury sites
   - Which embryos were poked

## Example: Full Workflow

### 1. Calibrate from Real Data

```bash
python3 calibrate_from_real_data.py \
    ../spark_tracks.csv \
    --clusters-csv ../vector_clusters.csv \
    --output calibrated_config.json
```

**Output:**
```
Loading real data from ../spark_tracks.csv...
  → Loaded 15234 track states
Loading clusters from ../vector_clusters.csv...
  → Loaded 342 clusters

Extracting parameters from real data...
  ✓ Wave parameters: mean speed = 4.23 px/s
  ✓ Found 2 embryo geometry(ies)
  ✓ Found 85 poke location(s)

=== Extracted Parameters ===
Wave speed: 4.23 px/s (mean)
Wave duration: 8.45 s (mean)
Decay rate: 0.12
Embryos: 2
Poke locations: 85

✓ Saved simulation configuration to calibrated_config.json
```

### 2. Review Extracted Parameters

The JSON file contains:
- `wave_config`: Speed, duration, decay rate
- `embryos`: Position, size, orientation for each embryo
- `pokes`: Location and timing of each poke
- `extracted_parameters`: Full statistics

### 3. Generate Simulation

```bash
python3 generate_simulated_data.py \
    --config calibrated_config.json \
    --output simulated_realistic.csv \
    --duration 30.0
```

### 4. Validate Against Real Data

```bash
python3 visualize_simulation.py simulated_realistic.csv \
    --real-csv ../spark_tracks.csv \
    --compare \
    --output validation_comparison.png
```

This comparison plot shows:
- Speed distributions (simulated vs real)
- Activity over time
- Track durations
- Spatial spread

## Modifying Calibrated Parameters

You can edit the JSON config file to modify parameters:

```json
{
  "wave_config": {
    "speed_px_per_s": 4.23,     // <-- Change this to test different speeds
    "duration_s": 8.45,         // <-- Modify duration
    "decay_rate": 0.12,         // <-- Adjust decay
    "radial": true
  },
  "embryos": [...],             // <-- Add more embryos or modify positions
  "pokes": [...]                // <-- Change poke locations
}
```

Then regenerate:
```bash
python3 generate_simulated_data.py --config modified_config.json --output modified_sim.csv
```

## Use Cases

### 1. Test Hypotheses with Known Parameters
- Use your real data parameters to test if analysis methods can detect known patterns
- Ensure your analysis pipeline works correctly on data with known ground truth

### 2. Parameter Sensitivity Analysis
- Vary one parameter (e.g., speed) while keeping others constant
- See how changes affect wave propagation and analysis results

### 3. Test Edge Cases
- Add more embryos than in your real data
- Test unusual poke locations
- Try extreme parameter values

### 4. Validation
- Generate simulations matching your real data
- Compare analysis results: do they match expected patterns?

## Tips

1. **Use representative data**: Calibrate from a diverse subset of your experiments
2. **Check the config**: Review extracted parameters to ensure they're reasonable
3. **Validate**: Always compare simulated output with real data to verify realism
4. **Iterate**: Adjust parameters if simulations don't match real patterns

## Troubleshooting

**Problem:** Simulation doesn't match real data patterns

**Solution:** 
- Check if extracted parameters are reasonable
- Verify embryo positions are correct
- Adjust wave_config parameters manually in JSON

**Problem:** Not enough embryos or pokes extracted

**Solution:**
- Check that `embryo_id` and `filename` columns are present
- Verify data has proper spatial spread
- Manually add embryos/pokes to JSON config

**Problem:** Wave speeds seem unrealistic

**Solution:**
- Check units: speeds are in pixels/second
- Review your real data: are speeds actually this high?
- Adjust manually if needed

