#!/bin/bash
# Complete pipeline to re-run analysis with new poke detection
# This script:
# 1. Re-runs the parser with spark-based poke detection
# 2. Regenerates clusters
# 3. Regenerates all plots
# 4. Checks data availability for all hypotheses

set -e  # Exit on error

# Configuration - UPDATE THESE VALUES:
TIFF_FOLDER="/Users/jdietz/Documents/Levin/Embryos"          # Path to folder containing TIFF files
POKE_FRAME=0                     # Frame index where poke occurs (0-based)
FPS=1.0                          # Frames per second
CSV_OUTPUT="spark_tracks.csv"    # Output CSV file name
CLUSTERS_OUTPUT="vector_clusters.csv"  # Clusters CSV file name

# Optional: If you know the poke coordinates, uncomment and set:
# POKE_X=123.5
# POKE_Y=456.7

echo "=========================================="
echo "Re-running Full Analysis Pipeline"
echo "=========================================="
echo ""

# Check if TIFF folder exists
if [ ! -d "$TIFF_FOLDER" ]; then
    echo "❌ Error: TIFF folder '$TIFF_FOLDER' not found"
    echo "Please update TIFF_FOLDER in this script with the correct path"
    exit 1
fi

# Step 1: Re-run parser with new poke detection
echo "Step 1/4: Re-running parser with updated poke detection..."
echo "------------------------------------------------------------"

CMD="python3 wave-vector-analysis/wave-vector-tiff-parser.py"
CMD="$CMD \"$TIFF_FOLDER\" $POKE_FRAME"
CMD="$CMD --fps $FPS"
CMD="$CMD --csv $CSV_OUTPUT"

# Add poke coordinates if provided
if [ ! -z "$POKE_X" ] && [ ! -z "$POKE_Y" ]; then
    CMD="$CMD --poke-x $POKE_X --poke-y $POKE_Y"
fi

echo "Command: $CMD"
echo ""

# Run the parser
eval $CMD

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Parser failed. Stopping."
    exit 1
fi

echo ""
echo "✓ Parser completed successfully!"
echo ""

# Step 2: Regenerate clusters
echo "Step 2/4: Regenerating cluster summaries..."
echo "------------------------------------------------------------"

if [ ! -f "$CSV_OUTPUT" ]; then
    echo "❌ Error: $CSV_OUTPUT not found"
    exit 1
fi

python3 wave-vector-analysis/spark_tracks_to_clusters.py "$CSV_OUTPUT" "$CLUSTERS_OUTPUT"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Cluster generation failed. Stopping."
    exit 1
fi

echo ""
echo "✓ Clusters generated successfully!"
echo ""

# Step 3: Check data availability
echo "Step 3/4: Checking data availability for all hypotheses..."
echo "------------------------------------------------------------"

python3 wave-vector-analysis/check_data_availability.py "$CSV_OUTPUT" \
    --clusters-csv "$CLUSTERS_OUTPUT"

echo ""
echo "✓ Data availability check completed!"
echo ""

# Step 4: Regenerate all plots
echo "Step 4/4: Regenerating all hypothesis plots..."
echo "------------------------------------------------------------"

OUTPUT_DIR="wave-vector-analysis/analysis_results"
mkdir -p "$OUTPUT_DIR"

python3 wave-vector-analysis/generate_all_hypothesis_plots.py "$CSV_OUTPUT" \
    --clusters-csv "$CLUSTERS_OUTPUT" \
    --output-dir "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo ""
    echo "⚠ Warning: Some plots may have failed to generate"
else
    echo ""
    echo "✓ All plots generated successfully!"
fi

# Step 5: Generate poke locations plot if we have poke data
echo ""
echo "Step 5/5: Generating poke locations plot..."
echo "------------------------------------------------------------"

if [ -f "wave-vector-analysis/plot_poke_locations.py" ]; then
    python3 wave-vector-analysis/plot_poke_locations.py "$CSV_OUTPUT" \
        --output "$OUTPUT_DIR/poke_locations.png"
    echo "✓ Poke locations plot generated!"
else
    echo "⚠ plot_poke_locations.py not found, skipping..."
fi

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Review the data availability report above"
echo "  2. Check which hypotheses can now be tested"
echo "  3. Review plots in $OUTPUT_DIR/"
echo "  4. Update EXPERIMENTAL_HYPOTHESES.md with new results"
echo ""
echo "Summary:"
echo "  • CSV files: $CSV_OUTPUT, $CLUSTERS_OUTPUT"
echo "  • Plots: $OUTPUT_DIR/"
echo "  • Total plots: $(ls -1 $OUTPUT_DIR/*.png 2>/dev/null | wc -l | tr -d ' ')"
echo ""

