#!/bin/bash
# Helper script to re-run the wave-vector-tiff-parser.py with correct parameters
# This will regenerate spark_tracks.csv with the updated morphological head/tail detection

# Configuration - UPDATE THESE VALUES:
TIFF_FOLDER="embryos/2"          # Path to folder containing TIFF files
POKE_FRAME=0                     # Frame index where poke occurs (0-based)
FPS=1.0                          # Frames per second
CSV_OUTPUT="spark_tracks.csv"    # Output CSV file name

# Optional: If you know the poke coordinates, uncomment and set:
# POKE_X=123.5
# POKE_Y=456.7

# Check if TIFF folder exists
if [ ! -d "$TIFF_FOLDER" ]; then
    echo "❌ Error: TIFF folder '$TIFF_FOLDER' not found"
    echo "Please update TIFF_FOLDER in this script with the correct path"
    exit 1
fi

# Build command
CMD="python3 wave-vector-analysis/wave-vector-tiff-parser.py"
CMD="$CMD \"$TIFF_FOLDER\" $POKE_FRAME"
CMD="$CMD --fps $FPS"
CMD="$CMD --csv $CSV_OUTPUT"

# Add poke coordinates if provided
if [ ! -z "$POKE_X" ] && [ ! -z "$POKE_Y" ]; then
    CMD="$CMD --poke-x $POKE_X --poke-y $POKE_Y"
fi

echo "Re-running parser with updated morphological head/tail detection..."
echo "Command: $CMD"
echo ""

# Run the parser
eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Parser completed successfully!"
    echo "Next steps:"
    echo "  1. Regenerate clusters: python3 wave-vector-analysis/spark_tracks_to_clusters.py"
    echo "  2. Regenerate plots: python3 wave-vector-analysis/generate_all_hypothesis_plots.py"
else
    echo ""
    echo "❌ Parser failed. Check error messages above."
    exit 1
fi

