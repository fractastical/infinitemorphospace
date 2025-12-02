# Plotting Guide for Experimental Hypotheses

This guide shows how to generate plots for each experimental hypothesis using `analyze_experimental_hypotheses.py`.

## Quick Reference

All plots can be generated with:
```bash
python analyze_experimental_hypotheses.py spark_tracks.csv \
    --clusters-csv vector_clusters.csv \
    --analysis <analysis_type> \
    --output <output_file.png>
```

## Hypothesis-Specific Plotting Commands

### 1. Presence of Calcium Activity (Hypothesis 1)

**Plot:** Activity comparison between embryos and pre/post poke

```bash
python analyze_experimental_hypotheses.py spark_tracks.csv \
    --analysis activity \
    --output plots/activity_comparison.png
```

**What it shows:**
- Activity over time for each embryo
- Pre-poke vs post-poke comparison
- Peak activity times
- Activity ratios

---

### 2. Distance Effect - Contact vs Non-contact (Hypothesis 2)

**Plot:** Condition comparison between two experimental conditions

```bash
# First, ensure you have data from both conditions as separate CSV files
python analyze_experimental_hypotheses.py spark_tracks_contact.csv \
    --analysis compare \
    --compare-csv2 spark_tracks_noncontact.csv \
    --label1 "Contact" \
    --label2 "Non-contact" \
    --output plots/contact_vs_noncontact.png
```

**What it shows:**
- Activity over time comparison (embryo B)
- Total activity comparison
- Time to peak activity
- Activity ratio

---

### 3. Wave Directionality Within Embryo (Hypothesis 3)

**Plot:** Wave direction and speed analysis

```bash
python analyze_experimental_hypotheses.py spark_tracks.csv \
    --clusters-csv vector_clusters.csv \
    --analysis directionality \
    --embryo A \
    --output plots/wave_directionality_embryoA.png
```

**What it shows:**
- Rose plot (polar histogram) of wave directions
- Speed distribution
- Direction vs speed scatter plot
- Activity over time by direction quadrant

---

### 4. Wave Directionality Between Embryos (Hypothesis 4)

**Plot:** Same as Hypothesis 3, but filter to embryo B

```bash
python analyze_experimental_hypotheses.py spark_tracks.csv \
    --clusters-csv vector_clusters.csv \
    --analysis directionality \
    --embryo B \
    --output plots/wave_directionality_embryoB.png
```

---

### 5. Spatial Matching

**Plot:** Spatial overlay of poke site and response locations

```bash
python analyze_experimental_hypotheses.py spark_tracks.csv \
    --analysis spatial \
    --poke-x <x_coordinate> \
    --poke-y <y_coordinate> \
    --output plots/spatial_matching.png
```

**What it shows:**
- Spatial overlay with poke site (red star) and responses
- Closest response marked (green circle)
- Distance distribution histogram

**Note:** Poke coordinates can be extracted from video frames or provided manually.

---

### 6. Local Tail Response (Hypotheses 7 & 8)

**Plot:** Tail region activity and speed analysis

```bash
python analyze_experimental_hypotheses.py spark_tracks.csv \
    --clusters-csv vector_clusters.csv \
    --analysis tail \
    --output plots/tail_response.png
```

**What it shows:**
- Tail activity over time
- Spatial distribution in tail region
- Speed distribution of tail responses
- AP position distribution

**Customize tail threshold:**
Edit the script to change `tail_threshold` (default 0.7, where 0=head, 1=tail).

---

## Batch Plotting

Generate all plots at once:

```bash
#!/bin/bash

# Create output directory
mkdir -p plots/hypothesis_analysis

# 1. Activity comparison
python analyze_experimental_hypotheses.py spark_tracks.csv \
    --analysis activity \
    --output plots/hypothesis_analysis/activity.png

# 2. Directionality - Embryo A
python analyze_experimental_hypotheses.py spark_tracks.csv \
    --clusters-csv vector_clusters.csv \
    --analysis directionality \
    --embryo A \
    --output plots/hypothesis_analysis/directionality_A.png

# 3. Directionality - Embryo B
python analyze_experimental_hypotheses.py spark_tracks.csv \
    --clusters-csv vector_clusters.csv \
    --analysis directionality \
    --embryo B \
    --output plots/hypothesis_analysis/directionality_B.png

# 4. Tail response
python analyze_experimental_hypotheses.py spark_tracks.csv \
    --clusters-csv vector_clusters.csv \
    --analysis tail \
    --output plots/hypothesis_analysis/tail_response.png

# 5. Spatial matching (if you have poke coordinates)
python analyze_experimental_hypotheses.py spark_tracks.csv \
    --analysis spatial \
    --poke-x <X> \
    --poke-y <Y> \
    --output plots/hypothesis_analysis/spatial_matching.png
```

## Plot Customization

All plots are saved as high-resolution PNG files (150 DPI) suitable for publications. To modify:

- **Figure sizes:** Edit the `figsize` parameter in each plotting function
- **Colors:** Modify the `color` and `cmap` parameters
- **Font sizes:** Adjust `fontsize` parameters
- **DPI:** Change `dpi=150` to higher values (e.g., 300 for print)

## Tips

1. **For publications:** Use DPI 300 and larger figure sizes
2. **For presentations:** Default settings (150 DPI) work well
3. **Colorblind-friendly:** The plots use distinguishable colors, but you can modify colormaps if needed
4. **Time windows:** Add time filtering by modifying the data before plotting
5. **Multiple conditions:** Compare by running separate analyses and overlaying in post-processing

## Example Workflow

```bash
# 1. Generate all basic plots
python analyze_experimental_hypotheses.py spark_tracks.csv \
    --clusters-csv vector_clusters.csv \
    --analysis activity \
    --output plots/all_activity.png

# 2. Analyze directionality for both embryos
for embryo in A B; do
    python analyze_experimental_hypotheses.py spark_tracks.csv \
        --clusters-csv vector_clusters.csv \
        --analysis directionality \
        --embryo $embryo \
        --output plots/directionality_${embryo}.png
done

# 3. Analyze tail response
python analyze_experimental_hypotheses.py spark_tracks.csv \
    --clusters-csv vector_clusters.csv \
    --analysis tail \
    --output plots/tail_response.png
```

