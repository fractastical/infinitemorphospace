# Re-running Analysis with New Poke Detection

This guide explains how to re-run the full analysis pipeline with the updated poke detection features.

## What's New

The parser now includes:
1. **Pink arrow detection** - Automatically detects poke location from pink arrow overlays in TIFF files
2. **Spark-based poke detection** - Falls back to inferring poke location from the centroid of the first spark clusters
3. **Improved head/tail detection** - Uses morphological features (width analysis) instead of orientation assumptions

## Quick Start

### Option 1: Use the Automated Script

```bash
# Edit the configuration in rerun_full_analysis.sh first!
cd /Users/jdietz/Documents/GitHub/infinitemorphospace

# Update these variables in the script:
# - TIFF_FOLDER="embryos/2"
# - POKE_FRAME=0
# - FPS=1.0

# Then run:
./wave-vector-analysis/rerun_full_analysis.sh
```

This script will:
1. ✅ Re-run the parser with new poke detection
2. ✅ Regenerate cluster summaries
3. ✅ Check data availability for all hypotheses
4. ✅ Regenerate all plots
5. ✅ Generate poke locations plot

### Option 2: Manual Steps

```bash
# Step 1: Re-run parser
python3 wave-vector-analysis/wave-vector-tiff-parser.py \
    embryos/2 0 \
    --fps 1.0 \
    --csv spark_tracks.csv

# Step 2: Regenerate clusters
python3 wave-vector-analysis/spark_tracks_to_clusters.py \
    spark_tracks.csv \
    vector_clusters.csv

# Step 3: Check data availability
python3 wave-vector-analysis/check_data_availability.py \
    spark_tracks.csv \
    --clusters-csv vector_clusters.csv

# Step 4: Regenerate all plots
python3 wave-vector-analysis/generate_all_hypothesis_plots.py \
    spark_tracks.csv \
    --clusters-csv vector_clusters.csv \
    --output-dir wave-vector-analysis/analysis_results/

# Step 5: Generate poke locations plot
python3 wave-vector-analysis/plot_poke_locations.py \
    spark_tracks.csv \
    --output wave-vector-analysis/analysis_results/poke_locations.png
```

## Expected Improvements

After re-running with new poke detection, you should see:

1. **`dist_from_poke_px` column populated** - Currently 0%, should improve significantly
   - This enables testing of:
     - **Spatial Matching** hypothesis
     - Radial wave propagation analysis
     - Distance-dependent responses

2. **Better head/tail detection** - More accurate AP position assignments
   - Improves:
     - **Local Tail Response** analysis
     - **Posterior Damage Effect** hypothesis
     - Anatomical position analysis

3. **Poke locations plot** - Visualize where pokes occurred across all files
   - New plot: `analysis_results/poke_locations.png`

## Which Hypotheses Can Now Be Tested?

### Previously Testable (Still Testable):
- ✅ Hypothesis 1: Presence of calcium activity
- ✅ Hypothesis 3: Wave directionality within embryo
- ✅ Hypothesis 4: Wave directionality between embryos
- ✅ Hypothesis 6: Spatial patterning
- ✅ Hypothesis 7: Local tail response

### Now Testable (with poke detection):
- ✅ **Spatial Matching** - Can now compare XY coordinates of poke vs response
- ⚠️ **Spatial Matching (Posterior)** - Can test absence of response for posterior pokes
- ⚠️ **Posterior Damage Effect** - Better AP position data improves testing

### Still Cannot Test (require additional data):
- ❌ Hypothesis 2: Distance effect (contact vs non-contact) - Needs multiple conditions
- ❌ Hypothesis 8: Age-dependent localization - Needs age metadata
- ❌ Contraction - Needs separate contraction detection pipeline
- ⚠️ Wound Memory - Needs healed wound location coordinates

## After Re-running

1. **Check the data availability report** - See which hypotheses can now be tested
2. **Review poke locations plot** - Verify poke detection worked correctly
3. **Update EXPERIMENTAL_HYPOTHESES.md** - Add new results and update testability status
4. **Generate new analyses** - Run specific hypothesis tests that are now possible

## Troubleshooting

### Poke detection failed

If `dist_from_poke_px` is still empty:
1. Check if pink arrows are visible in TIFF files:
   ```bash
   python3 wave-vector-analysis/check_pink_arrows.py embryos/2 10
   ```
2. If no pink arrows, the spark-based detection should still work
3. If both fail, provide manual coordinates:
   ```bash
   python3 wave-vector-analysis/wave-vector-tiff-parser.py \
       embryos/2 0 \
       --poke-x 123.5 \
       --poke-y 456.7 \
       --csv spark_tracks.csv
   ```

### Embryo detection issues

If embryo detection fails:
- Check that TIFF files aren't all black
- Verify files are readable with:
  ```bash
  python3 wave-vector-analysis/diagnose_tiffs.py embryos/2
  ```

## Next Steps

After successfully re-running:
1. Update `EXPERIMENTAL_HYPOTHESES.md` with:
   - New testability status
   - Updated data availability
   - New results from spatial matching analyses
2. Generate spatial matching plots
3. Analyze distance-dependent responses
4. Compare results with previous analysis

