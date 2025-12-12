# Head/Tail Assignment Logic Explanation

## Current Logic Flow

### 1. **Initial Contour Detection** (lines 200-473)
- Detects embryo contours from TIFF files using thresholding and morphological operations
- Returns 3 contour sets: `contour_old` (restrictive), `contour_intermediate` (middle), `contour` (inclusive)
- **Initial A/B Assignment**: Contours are sorted by centroid x-position (line 466-473)
  ```python
  # Leftmost centroid → A, Rightmost centroid → B
  emb_contours.sort(key=contour_cx)
  ```

### 2. **Head/Tail Determination** (lines 516-577)
For each contour (A or B), the algorithm:
- Uses **PCA** to find the principal axis (longest dimension)
- Projects all contour points onto this axis
- Finds the two extreme points (min/max projection)
- Uses **width analysis** to determine which end is wider:
  - Compares average width of the 20% region at each end
  - Wider end = **head** (embryo heads are typically wider)
  - Narrower end = **tail**

**Problem**: This width-based logic can fail if:
- The embryo is rotated or the head isn't clearly wider
- Two embryos are connected and the PCA axis spans both
- The contour detection captures both embryos as one

### 3. **A/B Validation & Reassignment** (lines 615-663)
After initial assignment, the code checks:
- If `centroid_A[0] > centroid_B[0]` → swap A and B
- If `x_A_max > x_B_max` → swap A and B
- If significant overlap → swap A and B

**Problem**: This only checks centroids and boundaries, not actual head positions.

### 4. **Separation & Clipping** (lines 665-776)
If embryos overlap:
- Calculates `split_x` (midpoint of overlap or between boundaries)
- Clips contour A to `x <= split_x` (left side)
- Clips contour B to `x >= split_x` (right side)

### 5. **Head/Tail Recalculation After Clipping** (lines 778-822)
After clipping, uses **simple position-based logic**:
- **Embryo A**: 
  - Head = **leftmost** point of clipped contour
  - Tail = **rightmost** point of clipped contour
- **Embryo B**:
  - Head = **rightmost** point of clipped contour
  - Tail = **leftmost** point of clipped contour

**This is the critical step** - it should ensure B's head is always on the right.

### 6. **Post-Validation** (lines 835-919)
Multiple rounds of validation:
- Checks if A and B still overlap → re-clips
- Forces A head to absolute leftmost point
- Forces B head to absolute rightmost point
- Emergency re-clipping if overlap persists

### 7. **B Head Position Check** (lines 955-988)
Final validation:
- Checks if B's head is in the right half of image
- If B head is < `center_x - 0.2*w`, forces correction to rightmost point

## Why B Can Still End Up on the Left

Despite all this logic, B can still be assigned on the left because:

1. **Initial Contour Detection Issue**: If the TIFF detection finds a large contour that spans both embryos, and the centroid happens to be on the left, B gets assigned to the left side.

2. **Clipping Failure**: If the clipping doesn't properly separate the contours (e.g., if `split_x` is calculated incorrectly or the clipping function fails), B's contour might still extend into the left side.

3. **Reassignment Logic Gap**: The reassignment only checks centroids, not head positions. So even if A and B are swapped, the head/tail positions might still be wrong.

4. **Timing Issue**: The head/tail recalculation happens AFTER clipping, but if the clipping didn't work properly, the recalculation uses the wrong contour.

## How Vector Wave Data Can Help

We now have velocity vectors (`vx`, `vy`) from the spark data. These can help because:

1. **Wave Propagation Direction**: Ca²⁺ waves typically propagate from the poke location (usually near the head) outward. The dominant vector direction can indicate head vs tail.

2. **Vector Field Validation**: 
   - If we see strong vectors pointing from left to right in the left half → suggests head is on left (A)
   - If we see strong vectors pointing from right to left in the right half → suggests head is on right (B)

3. **Poke Location**: If we can detect the poke location (early spark clusters), that's usually near the head. This can validate head/tail assignments.

4. **Spatial Validation**: 
   - Check if vectors in the left half point predominantly leftward → A head should be on left
   - Check if vectors in the right half point predominantly rightward → B head should be on right

## Proposed Improvements

1. **Add Vector-Based Head/Tail Validation**:
   - After initial head/tail assignment, check vector directions
   - If vectors in the left half point predominantly rightward → A head might be on right (wrong!)
   - If vectors in the right half point predominantly leftward → B head might be on left (wrong!)

2. **Use Vector Field to Detect Split Point**:
   - Instead of just using contour overlap, use vector field discontinuities
   - Where vectors change direction abruptly → likely boundary between embryos

3. **Vazlidate A/B Assignment with Vectors**:
   - Calculate average vector direction in left half vs right half
   - If left half vectors point right → suggests head is on left (correct for A)
   - If right half vectors point left → suggests head is on right (correct for B)
   - If this doesn't match the assignment → swap A and B

4. **Use Poke Location**:
   - Detect poke location from early spark clusters
   - Poke is usually near head → validate head position matches poke location

5. **Force B Head to Right Using Vectors**:
   - After all other logic, check: if B's head is on left but vectors in right half point rightward → force B head to rightmost point

