#!/bin/bash
# Batch process all folders to regenerate spark_tracks.csv with region data

set -e

TIFF_BASE="/Users/jdietz/Documents/Levin/Embryos"
OUTPUT_CSV="spark_tracks_with_regions.csv"
FPS=1.0
POKE_FRAME=0

cd /Users/jdietz/Documents/GitHub/infinitemorphospace
source venv/bin/activate

echo "Batch processing all folders to generate spark_tracks.csv with region data..."
echo "TIFF base: $TIFF_BASE"
echo "Output: $OUTPUT_CSV"
echo ""

# Get list of folders (1-33 based on existing data)
FOLDERS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33)

# Remove output CSV if it exists (we'll create a new one)
rm -f "$OUTPUT_CSV"

# Process each folder
for folder in "${FOLDERS[@]}"; do
    folder_path="$TIFF_BASE/$folder"
    
    if [ ! -d "$folder_path" ]; then
        echo "⚠ Skipping folder $folder (not found: $folder_path)"
        continue
    fi
    
    echo "Processing folder $folder..."
    
    # Run parser on this folder, append to CSV
    python3 wave-vector-analysis/wave-vector-tiff-parser.py \
        "$folder_path" \
        $POKE_FRAME \
        --fps $FPS \
        --csv "temp_${folder}.csv" || {
        echo "  ⚠ Error processing folder $folder, skipping..."
        continue
    }
    
    # Append to main CSV (skip header after first file)
    if [ ! -f "$OUTPUT_CSV" ]; then
        # First file: include header
        cp "temp_${folder}.csv" "$OUTPUT_CSV"
    else
        # Subsequent files: skip header
        tail -n +2 "temp_${folder}.csv" >> "$OUTPUT_CSV"
    fi
    
    # Clean up temp file
    rm -f "temp_${folder}.csv"
    
    echo "  ✓ Completed folder $folder"
done

echo ""
echo "✓ Batch processing complete!"
echo "Output saved to: $OUTPUT_CSV"
echo ""
echo "Next steps:"
echo "  1. Backup old CSV: mv spark_tracks.csv spark_tracks_backup.csv"
echo "  2. Use new CSV: mv $OUTPUT_CSV spark_tracks.csv"
echo "  3. Regenerate detection summary: python3 wave-vector-analysis/generate_detection_summary.py spark_tracks.csv"
