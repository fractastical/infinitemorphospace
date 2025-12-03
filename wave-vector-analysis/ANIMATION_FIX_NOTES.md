# Animation Visualization Issues and Fixes

## Current Issues (from user feedback)

1. **One region is completely orange and doesn't change** - suggests normalization or data aggregation issue
2. **Other regions show black background with red X marks that don't change** - poke locations are static (which is correct), but sparks aren't showing as a heatmap

## What the Visualization Should Show

1. **Poke Locations** (static reference points):
   - Shown once as cyan X markers
   - Indicate where the poke occurred
   - Do NOT animate (static)

2. **Spark Accumulation** (animated heatmap):
   - Shows cumulative sparks from t=0 to current time
   - Each frame shows ALL sparks that have occurred up to that time
   - Heatmap intensity builds up over time
   - This is what should animate

3. **Time Progression**:
   - Each frame represents a time point
   - Sparks accumulate cumulatively (frame 1 shows sparks 0-2s, frame 2 shows 0-4s, etc.)
   - The heatmap should visibly build up over time

## Fixes Needed

1. Make poke locations clearly static (drawn once, not in animation loop)
2. Ensure heatmap shows actual spark data (not just black)
3. Fix normalization so regions are comparable
4. Make sure animation actually progresses (frames change)
5. Add better labels and explanation

## Testing

To verify the fix works:
- Poke locations should appear once and stay visible
- Heatmap should start empty/weak and build up over time
- Different regions should show different patterns
- Time counter should increment
- Spark count should increase

