# Tail Region Labeling Methodology

## Current Approach

The tail region is identified using the **anterior-posterior normalized position** (`ap_norm`) coordinate system.

### How `ap_norm` is Calculated

1. **Embryo axis estimation:**
   - Uses PCA (Principal Component Analysis) on the embryo contour to find the longest axis
   - Identifies two endpoints along this axis
   - **Head/tail determination:** Uses morphological features (width) rather than orientation
   - Head = wider/rounder end (more bulbous)
   - Tail = narrower/pointed end
   - **Does NOT assume any specific orientation** - works regardless of how embryo is rotated

2. **Normalized position:**
   - Projects each pixel/event onto the head-to-tail axis
   - Normalizes by the axis length
   - **`ap_norm = 0`** → head (wider/more bulbous end)
   - **`ap_norm = 1`** → tail (narrower/more pointed end)
   - Values in between represent normalized position along the axis
   - Head/tail determined by morphological features, not image orientation

3. **Tail region definition:**
   - Currently uses **`ap_norm >= 0.7`** as the threshold
   - This means the **posterior 30%** of the embryo is considered "tail"
   - Can be adjusted via `tail_threshold` parameter

### Current Data Distribution

From the current dataset:
- **AP position range:** 0.000 to 1.001
- **Events with `ap_norm >= 0.7`:** 51,828 / 142,770 (36.3%)
- **Events with `ap_norm >= 0.8`:** 34,998 / 142,770 (24.5%)
- **Events with `ap_norm >= 0.9`:** 13,386 / 142,770 (9.4%)

### Code Location

The labeling is implemented in:
- **Calculation:** `wave-vector-tiff-parser.py` lines 420-478
- **Usage:** `analyze_experimental_hypotheses.py` function `analyze_tail_response()`

## Limitations and Considerations

### 1. Head/Tail Identification

**Current method:** Uses morphological features (width profile)
- Head = wider/rounder end (more bulbous)
- Tail = narrower/pointed end
- Calculates average width in first/last 20% of embryo length
- Wider end is designated as head
- **No assumption about orientation** - works regardless of embryo rotation

**Potential issues:**
- May fail if head and tail have similar widths
- Requires sufficient points in each region for accurate width calculation
- Edge cases where morphology is ambiguous may produce warnings

**Current improvements made:**
- ✅ Removed orientation assumption
- ✅ Uses morphological width analysis
- ✅ Provides warnings when detection is uncertain
- ⚠️ Could add: Manual override option for cases where detection fails

### 2. Threshold Selection

**Current threshold:** `ap_norm >= 0.7` (posterior 30%)

**Considerations:**
- Should tail be defined more narrowly (e.g., ≥ 0.8 or 0.9)?
- Should it be based on anatomical landmarks rather than percentage?
- May need different thresholds for different embryo stages

**Options to explore:**
- **Stricter:** `>= 0.8` (posterior 20%) - 24.5% of events
- **Stricter:** `>= 0.9` (posterior 10%) - 9.4% of events  
- **Anatomical:** Use specific distance from tail endpoint
- **Dynamic:** Adjust based on embryo size/stage

### 3. Coverage

**Current coverage:** Only 46.1% of events have `ap_norm` values

**Why some events are missing:**
- Events detected outside embryo boundaries
- Embryo detection failed for some frames
- Multi-embryo scenarios where assignment is ambiguous

**Impact:** Analysis only includes events that could be assigned to an embryo

## Recommendations

### For Immediate Use

1. **Keep current threshold (0.7)** but document it clearly
2. **Verify head/tail detection** is correct by checking a sample of frames
3. **If detection is wrong:** The code will print warnings. You may need to:
   - Manually verify which end is actually head
   - Add option to flip the axis (can be added as feature)
   - Provide manual annotation for reference frames

### For Improved Accuracy

1. **Validate head/tail identification:**
   - Manually verify a sample of frames
   - Check if "head = leftmost" assumption holds

2. **Consider stricter thresholds:**
   - Test with 0.8 or 0.9 for more localized tail responses
   - Compare results across threshold values

3. **Add anatomical landmarks:**
   - If possible, use specific morphological features
   - Consider using distance from tail endpoint in pixels

### Usage

To change the tail threshold when analyzing:

```python
# In analyze_experimental_hypotheses.py
results = analyze_tail_response(tracks_df, clusters_df, tail_threshold=0.8)  # More strict
results = analyze_tail_response(tracks_df, clusters_df, tail_threshold=0.9)  # Very strict
```

Or from command line (would need to add parameter):
```bash
# Currently hardcoded at 0.7, but can be modified in code
```

## Questions to Address

1. **Is the morphological detection accurate?**
   - Check a sample of images to verify head/tail assignment
   - Look for warnings in the output indicating uncertain detection
   - Consider manual annotation for validation if detection fails

2. **What defines "tail" anatomically?**
   - Last X% of embryo length?
   - Specific distance from tail tip?
   - Region with specific morphology?

3. **Should threshold be adjustable per experiment?**
   - Different stages may need different thresholds
   - Consider making it a command-line parameter

