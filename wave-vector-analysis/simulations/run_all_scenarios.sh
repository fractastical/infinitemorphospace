#!/bin/bash
# Generate all simulation scenarios for comparison
# Optionally calibrates from real data if available

set -e  # Exit on error

OUTPUT_DIR="simulation_outputs"
mkdir -p "$OUTPUT_DIR"

# Paths to real data (check both parent directory and current directory)
REAL_TRACKS_CSV="../spark_tracks.csv"
REAL_CLUSTERS_CSV="../vector_clusters.csv"
CALIBRATED_CONFIG="$OUTPUT_DIR/calibrated_config.json"

# Check if real data exists and offer to calibrate
if [ -f "$REAL_TRACKS_CSV" ]; then
    echo "=========================================="
    echo "Real data detected: $REAL_TRACKS_CSV"
    echo "=========================================="
    echo ""
    
    # Check if clusters file exists
    if [ -f "$REAL_CLUSTERS_CSV" ]; then
        echo "Step 1/3: Calibrating from real data..."
        echo "------------------------------------------------------------"
        python3 calibrate_from_real_data.py \
            "$REAL_TRACKS_CSV" \
            --clusters-csv "$REAL_CLUSTERS_CSV" \
            --output "$CALIBRATED_CONFIG"
        echo ""
        
        CALIBRATED=1
    else
        echo "Step 1/3: Calibrating from real data (clusters file not found, using tracks only)..."
        echo "------------------------------------------------------------"
        python3 calibrate_from_real_data.py \
            "$REAL_TRACKS_CSV" \
            --output "$CALIBRATED_CONFIG"
        echo ""
        
        CALIBRATED=1
    fi
else
    echo "=========================================="
    echo "No real data found at $REAL_TRACKS_CSV"
    echo "Skipping calibration, using predefined scenarios only"
    echo "=========================================="
    echo ""
    CALIBRATED=0
fi

echo "Step 2/3: Generating predefined scenarios..."
echo "------------------------------------------------------------"

scenarios=(
    "two_embryos_head_head"
    "three_embryos_triangle"
    "two_embryos_tail_tail"
    "multiple_pokes_same_embryo"
)

for scenario in "${scenarios[@]}"; do
    echo "Generating: $scenario"
    python3 generate_simulated_data.py \
        --scenario "$scenario" \
        --output "$OUTPUT_DIR/simulated_${scenario}.csv" \
        --duration 30.0
    echo ""
done

# Generate calibrated simulation if we have real data
if [ "$CALIBRATED" -eq 1 ] && [ -f "$CALIBRATED_CONFIG" ]; then
    echo "Step 3/3: Generating calibrated simulation from real data..."
    echo "------------------------------------------------------------"
    python3 generate_simulated_data.py \
        --config "$CALIBRATED_CONFIG" \
        --output "$OUTPUT_DIR/simulated_calibrated_from_real_data.csv" \
        --duration 30.0
    echo ""
else
    echo "Step 3/3: Skipped (no calibration data available)"
    echo "------------------------------------------------------------"
    echo ""
fi

echo "=========================================="
echo "✓ All scenarios generated in $OUTPUT_DIR/"
echo "=========================================="
echo ""

# List generated files
echo "Generated files:"
ls -lh "$OUTPUT_DIR"/*.csv 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""

if [ "$CALIBRATED" -eq 1 ]; then
    echo "To compare calibrated simulation with real data:"
    echo "  python3 visualize_simulation.py $OUTPUT_DIR/simulated_calibrated_from_real_data.csv \\"
    echo "      --real-csv $REAL_TRACKS_CSV \\"
    echo "      --compare \\"
    echo "      --output $OUTPUT_DIR/calibration_comparison.png"
    echo ""
fi

echo "To visualize any scenario:"
echo "  python3 visualize_simulation.py $OUTPUT_DIR/simulated_two_embryos_head_head.csv"

