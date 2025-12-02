# Simulation Quick Start Guide

## Generate Your First Simulation

```bash
cd wave-vector-analysis/simulations

# Generate a two-embryo scenario
python3 generate_simulated_data.py \
    --scenario two_embryos_head_head \
    --output sim_two_embryos.csv \
    --duration 30.0

# Visualize it
python3 visualize_simulation.py sim_two_embryos.csv \
    --output sim_plot.png
```

## Generate Multiple Scenarios

```bash
# Run all predefined scenarios
./run_all_scenarios.sh

# This creates:
# - simulation_outputs/simulated_two_embryos_head_head.csv
# - simulation_outputs/simulated_three_embryos_triangle.csv
# - simulation_outputs/simulated_two_embryos_tail_tail.csv
# - simulation_outputs/simulated_multiple_pokes_same_embryo.csv
```

## Compare with Real Data

```bash
# Compare simulated data with your real experimental data
python3 visualize_simulation.py simulation_outputs/simulated_two_embryos_head_head.csv \
    --real-csv ../../spark_tracks.csv \
    --compare \
    --output comparison.png
```

## Analyze Simulated Data

Simulated data can be analyzed with the same tools as real data:

```bash
# Generate cluster summaries
python3 ../spark_tracks_to_clusters.py \
    simulation_outputs/simulated_two_embryos_head_head.csv \
    simulation_outputs/simulated_two_embryos_head_head_clusters.csv

# Run hypothesis analysis
python3 ../analyze_experimental_hypotheses.py \
    simulation_outputs/simulated_two_embryos_head_head.csv \
    --clusters-csv simulation_outputs/simulated_two_embryos_head_head_clusters.csv \
    --analysis activity
```

## Create Custom Scenarios

Edit `generate_simulated_data.py` and modify the `create_experiment_scenarios()` function to add your own scenarios.

Example: Add a scenario with 4 embryos
```python
scenarios['four_embryos_square'] = {
    'embryos': [
        Embryo(id='A', center_x=500, center_y=500, length=300, width=120, 
              angle=0, head_x=350, head_y=500, tail_x=650, tail_y=500),
        Embryo(id='B', center_x=1500, center_y=500, length=300, width=120,
              angle=0, head_x=1350, head_y=500, tail_x=1650, tail_y=500),
        Embryo(id='C', center_x=500, center_y=1500, length=300, width=120,
              angle=0, head_x=350, head_y=1500, tail_x=650, tail_y=1500),
        Embryo(id='D', center_x=1500, center_y=1500, length=300, width=120,
              angle=0, head_x=1350, head_y=1500, tail_x=1650, tail_y=1500),
    ],
    'pokes': [PokeConfig(x=500, y=500, embryo_id='A', time=0.0)],
    'wave_config': WaveConfig(speed_px_per_s=5.0, duration_s=10.0, radial=True)
}
```

Then generate it:
```bash
python3 generate_simulated_data.py \
    --scenario four_embryos_square \
    --output sim_four_embryos.csv
```

