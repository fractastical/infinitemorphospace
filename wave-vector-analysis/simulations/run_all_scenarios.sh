#!/bin/bash
# Generate all simulation scenarios for comparison
# Optionally calibrates from real data if available

set -e  # Exit on error

# Get script directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Project root is two levels up from simulations/ folder
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

OUTPUT_DIR="simulation_outputs"
mkdir -p "$SCRIPT_DIR/$OUTPUT_DIR"

CALIBRATED_CONFIG="$OUTPUT_DIR/calibrated_config.json"

# Paths to real data (in project root, where CSV files are located)
REAL_TRACKS_CSV="$PROJECT_ROOT/spark_tracks.csv"
REAL_CLUSTERS_CSV="$PROJECT_ROOT/vector_clusters.csv"

# Check if real data exists and offer to calibrate
if [ -f "$REAL_TRACKS_CSV" ]; then
    echo "=========================================="
    echo "Real data detected: $REAL_TRACKS_CSV"
    echo "=========================================="
    echo ""
    
    # Determine base path for clusters file if not already set
    if [ -z "$REAL_CLUSTERS_CSV" ]; then
        BASE_DIR=$(dirname "$REAL_TRACKS_CSV")
        if [ -f "$BASE_DIR/vector_clusters.csv" ]; then
            REAL_CLUSTERS_CSV="$BASE_DIR/vector_clusters.csv"
        fi
    fi
    
    # Check if clusters file exists
    if [ -n "$REAL_CLUSTERS_CSV" ] && [ -f "$REAL_CLUSTERS_CSV" ]; then
        echo "Step 1/3: Calibrating from real data..."
        echo "------------------------------------------------------------"
        python3 "$SCRIPT_DIR/calibrate_from_real_data.py" \
            "$REAL_TRACKS_CSV" \
            --clusters-csv "$REAL_CLUSTERS_CSV" \
            --output "$SCRIPT_DIR/$CALIBRATED_CONFIG"
        echo ""
        
        CALIBRATED=1
    else
        echo "Step 1/3: Calibrating from real data (clusters file not found, using tracks only)..."
        echo "------------------------------------------------------------"
        python3 "$SCRIPT_DIR/calibrate_from_real_data.py" \
            "$REAL_TRACKS_CSV" \
            --output "$SCRIPT_DIR/$CALIBRATED_CONFIG"
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
    "three_embryos_line"
    "four_embryos_square"
    "four_embryos_line"
    "three_embryos_central_poke"
)

for scenario in "${scenarios[@]}"; do
    echo "Generating: $scenario"
    python3 "$SCRIPT_DIR/generate_simulated_data.py" \
        --scenario "$scenario" \
        --output "$SCRIPT_DIR/$OUTPUT_DIR/simulated_${scenario}.csv" \
        --duration 30.0
    echo ""
done

# Generate calibrated simulation if we have real data
if [ "$CALIBRATED" -eq 1 ] && [ -f "$SCRIPT_DIR/$CALIBRATED_CONFIG" ]; then
    echo "Step 3/3: Generating calibrated simulation from real data..."
    echo "------------------------------------------------------------"
    python3 "$SCRIPT_DIR/generate_simulated_data.py" \
        --config "$SCRIPT_DIR/$CALIBRATED_CONFIG" \
        --output "$SCRIPT_DIR/$OUTPUT_DIR/simulated_calibrated_from_real_data.csv" \
        --duration 30.0
    echo ""
else
    echo "Step 3/3: Skipped (no calibration data available)"
    echo "------------------------------------------------------------"
    echo ""
fi

echo "Step 4/4: Generating comparison plots..."
echo "------------------------------------------------------------"
python3 "$SCRIPT_DIR/compare_all_scenarios.py" \
    --input-dir "$SCRIPT_DIR/$OUTPUT_DIR" \
    --output-dir "$SCRIPT_DIR/$OUTPUT_DIR/comparisons"
echo ""

echo "=========================================="
echo "✓ All scenarios generated in $SCRIPT_DIR/$OUTPUT_DIR/"
echo "=========================================="
echo ""

# List generated files
echo "Generated files:"
ls -lh "$SCRIPT_DIR/$OUTPUT_DIR"/*.csv 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""

if [ "$CALIBRATED" -eq 1 ]; then
    echo "To compare calibrated simulation with real data:"
    echo "  python3 $SCRIPT_DIR/visualize_simulation.py $SCRIPT_DIR/$OUTPUT_DIR/simulated_calibrated_from_real_data.csv \\"
    echo "      --real-csv $REAL_TRACKS_CSV \\"
    echo "      --compare \\"
    echo "      --output $SCRIPT_DIR/$OUTPUT_DIR/calibration_comparison.png"
    echo ""
fi

echo "To visualize any scenario:"
echo "  python3 $SCRIPT_DIR/visualize_simulation.py $SCRIPT_DIR/$OUTPUT_DIR/simulated_two_embryos_head_head.csv"
echo ""
echo "To generate comparison plots for all scenarios:"
echo "  python3 $SCRIPT_DIR/compare_all_scenarios.py \\"
echo "      --input-dir $SCRIPT_DIR/$OUTPUT_DIR \\"
echo "      --output-dir $SCRIPT_DIR/$OUTPUT_DIR/comparisons"

