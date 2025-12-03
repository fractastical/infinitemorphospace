#!/bin/bash
# Generate animations for all simulation scenarios

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SIM_OUTPUT_DIR="$SCRIPT_DIR/simulation_outputs"
ANIM_DIR="$SIM_OUTPUT_DIR/animations"

# Activate virtual environment
if [[ -f "$PROJECT_ROOT/venv/bin/activate" ]]; then
    source "$PROJECT_ROOT/venv/bin/activate"
    echo "✓ Activated virtual environment"
elif [[ -f "../venv/bin/activate" ]]; then
    source "../venv/bin/activate"
    echo "✓ Activated virtual environment"
else
    echo "Warning: Virtual environment not found. Make sure dependencies are installed."
fi

# Create animations directory
mkdir -p "$ANIM_DIR"

echo "Generating animations for all simulation scenarios..."
echo ""

# Find all simulation CSV files
for sim_csv in "$SIM_OUTPUT_DIR"/*.csv; do
    if [[ -f "$sim_csv" ]]; then
        # Get base filename without extension
        base_name=$(basename "$sim_csv" .csv)
        output_gif="$ANIM_DIR/${base_name}_animation.gif"
        
        echo "Processing: $(basename "$sim_csv")"
        echo "  → Output: $(basename "$output_gif")"
        
        python3 "$SCRIPT_DIR/animate_simulation.py" \
            "$sim_csv" \
            --output "$output_gif" \
            --time-window 30 \
            --frame-interval 1 \
            --fps 5
        
        echo ""
    fi
done

echo "✓ All animations generated!"
echo ""
echo "Animations saved to: $ANIM_DIR"

