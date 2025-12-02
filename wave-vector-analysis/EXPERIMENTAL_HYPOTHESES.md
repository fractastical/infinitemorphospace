# Experimental Hypotheses Analysis Guide

This document maps experimental hypotheses to specific analyses that can be performed with the `spark_tracks.csv` and `vector_clusters.csv` data.

## Table 1: Calcium Activity Experiments

### 1. Presence of Calcium Activity

**Claim:** Damaging embryo A increases the calcium activity in embryo A and B

**Analysis:**
- **Metric:** Total activity over time (ΔF/F₀ proxy: `total_area_px2_frames` or sum of `area` per frame)
- **Comparison:** 
  - Pre-poke baseline (time < 0) vs post-poke (time > 0)
  - Embryo A vs Embryo B
  - Control (no poke) vs experimental
- **Script:** Use `analyze_calcium_activity.py` with `--compare-embryos`

**Outcome to Measure:**
- Standard deviation of calcium activity over time
- Peak activity time
- Integrated activity (area under curve)
- Activity ratio: post-poke / pre-poke

---

### 2. Distance Effect (Contact vs Non-contact)

**Claim:** Damaging embryo A at a distance increases calcium activity in A and B, but response in B is lower than direct-contact condition

**Analysis:**
- **Metric:** Total activity in embryo B
- **Comparison:** Contact condition vs non-contact condition
- **Script:** Use `analyze_calcium_activity.py` with `--condition contact` vs `--condition non-contact`

**Outcome to Measure:**
- Compare ΔF/F₀ (or area sum) in embryo B between conditions
- Time to peak activity
- Duration of response

---

### 3. Wave Directionality Within Embryo

**Claim:** Damaging embryo A (mid region) causes a bidirectional calcium wave in embryo A

**Analysis:**
- **Metric:** Wave propagation direction (`angle_deg`, `mean_angle_deg`)
- **Filter:** Events in embryo A only
- **Script:** Use `analyze_wave_directionality.py` with `--embryo A`

**Outcome to Measure:**
- Wave directionality: Distribution of angles
- Intensity over time: Activity vs time from poke
- Speed: `mean_speed_px_per_s`, `peak_speed_px_per_s`

**Notes:** Response differs between cut vs. poked conditions

---

### 4. Wave Directionality Between Embryos

**Claim:** Damaging embryo A (anterior and mid region) triggers a calcium wave in embryo B when oriented head-head, head-tail, tail-tail

**Analysis:**
- **Metric:** Wave presence, intensity, speed
- **Filter:** Events in embryo B
- **Script:** Use `analyze_inter_embryo_waves.py`

**Outcome to Measure:**
- Presence of calcium wave: Binary (yes/no)
- Intensity over time: `total_area_px2_frames` per time bin
- Speed: `mean_speed_px_per_s`
- Direction: `mean_angle_deg` (should point from A to B)

---

### 5. Posterior Damage Effect

**Claim:** Damaging embryo A (posterior region) does NOT trigger a calcium wave in embryo B when oriented tail-tail

**Analysis:**
- **Metric:** Wave presence, intensity, speed
- **Filter:** Events in embryo B, when poke is in posterior region of A
- **Script:** Use `analyze_inter_embryo_waves.py` with `--poke-region posterior`

**Outcome to Measure:**
- Presence: Compare to anterior/mid results (should be absent/weak)
- Activity level: Should be much lower than anterior/mid pokes

**Notes:** A slight increase may be observed (n=2), but should be significantly lower

---

### 6. Spatial Patterning

**Claim:** The calcium wave in embryo A and B can be spatially patterned

**Analysis:**
- **Metric:** Spatial distribution of events
- **Visualization:** Heatmap of event density over space
- **Script:** Use `visualize_spark_tracks.py --plot heatmap`

**Outcome to Measure:**
- Signal intensity over time at different spatial locations
- Pattern structure (radial, directional, etc.)

**Notes:** High variability across embryos expected

---

### 7. Local Tail Response

**Claim:** Damaging embryo A causes a (fast) localized posterior response in embryo A and B

**Analysis:**
- **Metric:** Activity and speed within tail ROI
- **Filter:** Events where `ap_norm > threshold` (e.g., > 0.7 for tail region)
- **Script:** Use `analyze_tail_response.py`

**Outcome to Measure:**
- Activity within defined ROI in tail region
- Speed: Should be faster than calcium wave
- Time to response: Should be faster than full wave

**Notes:** This may be a distinct mechanism from the calcium wave

---

### 8. Age-Dependent Localization

**Claim:** The posterior response gets more localized with age

**Analysis:**
- **Metric:** Spatial extent of tail response
- **Comparison:** Early stage vs late stage embryos
- **Script:** Use `analyze_tail_response.py` with age/stage grouping

**Outcome to Measure:**
- Spatial spread: Standard deviation of event positions in tail
- ROI size: Number of events within tail region
- Duration: How long the response is localized

---

## Table 2: Spatial Matching, Contraction, and Wound Memory

### 1. Spatial Matching

**Claim:** Embryo A/B shows a local calcium response in a similar region as the wound site of embryo A/B

**Analysis:**
- **Metric:** XY coordinates of wound vs response location
- **Filter:** Local responses (high intensity, short duration)
- **Script:** Use `analyze_spatial_matching.py`

**Outcome to Measure:**
- Distance between wound coordinates and response coordinates
- AP position matching: `ap_norm` of wound vs `ap_norm` of response
- Correlation: Does poking at distance X from head result in flash at distance X in neighbor?

**Notes:** 
- Local response is faster than calcium wave
- Sometimes replaces the wave
- More testing needed, orientations differ

---

### 2. Contraction

**Claim:** Damaging embryo A causes a contraction in embryo B in a similar region as the wound site of embryo A

**Analysis:**
- **Note:** Contraction is not directly measured in current pipeline (requires separate analysis)
- **Metric:** XY coordinates of wound vs contraction location
- **Script:** Requires additional image analysis for contraction detection

**Outcome to Measure:**
- Spatial matching: Distance between wound and contraction coordinates
- AP position matching

**Questions:** F-actin from A to B?

---

### 3. Wound Memory (Increased Activity)

**Claim:** Presence of embryo A increases calcium activity in embryo B at a previously 'healed' wound location

**Analysis:**
- **Metric:** ΔF/F₀ at healed wound site
- **Comparison:** Before vs after embryo A is present
- **Time:** Response measured 24h after wound regeneration
- **Script:** Use `analyze_wound_memory.py`

**Outcome to Measure:**
- Activity at healed wound site: Before vs after A is present
- Baseline activity: Should be low before A arrives

**Notes:** Not sure yet, more testing needed

---

### 4. Wound Memory (Local Response)

**Claim:** Damaging embryo A causes a local response at the location of a previously 'healed' wound in embryo B

**Analysis:**
- **Metric:** ΔF/F₀ in embryo B before vs after poking embryo A
- **Filter:** Events at healed wound coordinates
- **Script:** Use `analyze_wound_memory.py` with `--healed-wound-coords`

**Outcome to Measure:**
- Activity at healed wound location: Before poke vs after poke
- Spatial matching: Does response match anatomical region of poke?

**Notes:** N=1, but response matched anatomical region; testing additional locations

---

## Analysis Scripts Available

1. **`analyze_calcium_activity.py`**: Compare activity levels between conditions
2. **`analyze_wave_directionality.py`**: Measure wave directions and speeds
3. **`analyze_inter_embryo_waves.py`**: Analyze waves between embryos
4. **`analyze_tail_response.py`**: Measure local tail responses
5. **`analyze_spatial_matching.py`**: Compare wound and response locations
6. **`analyze_wound_memory.py`**: Analyze healed wound responses

See individual script documentation for usage details.

