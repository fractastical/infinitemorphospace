# Paper Comparison Images

This folder contains side-by-side comparisons of our implementation vs. original Raup (1966) paper figures.

## Generated Images

### Variable Comparisons
- `variable_W_comparison.png` - Effect of Whorl Expansion Rate (W)
- `variable_D_comparison.png` - Effect of Distance from Axis (D)
- `variable_T_comparison.png` - Effect of Translation Rate (T)
- `variable_S_comparison.png` - Effect of Generating Curve Shape (S)

### Empirical Shell Comparisons
- `empirical_planispiral_comparison.png` - Planispiral (Nautilus) vs. Figure 5
- `empirical_turreted_comparison.png` - Turreted (Turritella) vs. Figure 6
- `empirical_disc_like_comparison.png` - Disc-like (Ammonite) vs. Figure 7
- `empirical_conical_comparison.png` - Conical (Conus) vs. Figure 8
- `empirical_high_spired_comparison.png` - High-spired (Cerithium) vs. Figure 9
- `empirical_wide_comparison.png` - Wide (Helix) vs. Figure 10

### Paper Figure Reconstructions
- `paper_figure_1_comparison.png` - Our reconstruction of Figure 1 (Effect of W)
- `paper_figure_2_comparison.png` - Our reconstruction of Figure 2 (Effect of D)
- `paper_figure_3_comparison.png` - Our reconstruction of Figure 3 (Effect of T)

## Adding Scanned Paper Images

To complete the comparisons with actual paper figures:

1. **Scan figures from Raup (1966) paper**
   - Use high-resolution scanning (300+ DPI recommended)
   - Save as PNG format

2. **Name files according to figure numbers:**
   - `paper_figure_1.png` - Figure 1 from paper
   - `paper_figure_2.png` - Figure 2 from paper
   - `paper_figure_5.png` - Figure 5 from paper (Planispiral)
   - `paper_figure_6.png` - Figure 6 from paper (Turreted)
   - etc.

3. **Place scanned images in this folder** (`raup_dataset/comparisons/`)

4. **Regenerate comparisons:**
   ```bash
   python3 datasets/raup/generate_paper_comparisons.py
   ```

The script will automatically detect scanned images and display them side-by-side with our implementations.

## Current Status

**Generated:** 2025-01-05  
**Scanned Images:** Not yet added (placeholders shown)  
**Next Step:** Scan original figures from Raup (1966) paper and add to this folder
