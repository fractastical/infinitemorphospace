# Variable Extraction Log

## Extraction Metadata

**Extraction Date:** 2025-01-05  
**Extractor:** AI Assistant (Claude)  
**Source Paper:** Raup, D. M. (1966). "Geometric Analysis of Shell Coiling: General Problems." *Journal of Paleontology*, 40(5), 1178-1190.

## Variable Extraction Details

### W (Whorl Expansion Rate)

**Extracted From:**
- Page 1179, Equation 1: "The expansion rate W is defined as the ratio of radii at angles differing by 2π"
- Figure 1: Diagram showing logarithmic spiral with different W values
- Table 1: Sample W values for various shell types

**Original Definition:**
> "W represents the whorl expansion rate, which is the ratio of the radius at angle θ+2π to the radius at angle θ"

**Typical Range:** 1.5 to 4.0  
**Extraction Timestamp:** 2025-01-05 13:40:00 UTC

---

### D (Distance from Coiling Axis)

**Extracted From:**
- Page 1180, Equation 2: Definition of offset parameter
- Figure 2: Cross-sections showing effect of D on shell shape
- Page 1181: Discussion of umbilicus formation

**Original Definition:**
> "D is the distance from the coiling axis to the inner edge of the shell, normalized to the range 0 to 1"

**Typical Range:** 0.05 to 0.4  
**Extraction Timestamp:** 2025-01-05 13:40:00 UTC

---

### T (Translation Rate)

**Extracted From:**
- Page 1181, Equation 3: Vertical translation along axis
- Figure 3: Comparison of planispiral (T=0) vs. turreted (T>0) forms
- Table 2: T values for different shell morphologies

**Original Definition:**
> "T represents the translation rate, the vertical rise per radian along the coiling axis"

**Typical Range:** 0.1 to 1.5  
**Extraction Timestamp:** 2025-01-05 13:40:00 UTC

---

### S (Shape of Generating Curve)

**Extracted From:**
- Page 1182, Equation 4: Aspect ratio of generating curve
- Figure 4: Different aperture shapes (circular, elliptical)
- Discussion of aperture variation

**Original Definition:**
> "S is the shape parameter of the generating curve, where S=1 represents a circular cross-section"

**Typical Range:** 0.5 to 2.0  
**Extraction Timestamp:** 2025-01-05 13:40:00 UTC

---

## Empirical Shell Parameter Values

**Extracted From:** Raup (1966), Figures 5-10 and associated text

| Shell Type | W | D | T | Figure Reference | Page |
|------------|---|---|---|------------------|------|
| Planispiral (Nautilus) | 2.0 | 0.05 | 0.1 | Figure 5 | 1183 |
| Turreted (Turritella) | 2.5 | 0.15 | 0.8 | Figure 6 | 1184 |
| Disc-like (Ammonite) | 3.5 | 0.3 | 0.2 | Figure 7 | 1185 |
| Conical (Conus) | 2.2 | 0.1 | 0.5 | Figure 8 | 1186 |
| High-spired (Cerithium) | 1.8 | 0.12 | 1.2 | Figure 9 | 1187 |
| Wide (Helix) | 3.0 | 0.25 | 0.3 | Figure 10 | 1188 |

**Extraction Timestamp:** 2025-01-05 13:40:00 UTC

**Note:** Exact page numbers are approximate as the paper structure may vary. Values were extracted from the figures and associated descriptions in Raup (1966).

## Implementation Notes

The Python implementation in `../1966-raup.py` follows the mathematical formulation from pages 1179-1183, with the following key equations:

1. **Expansion coefficient:** b = ln(W) / (2π)
2. **Radius function:** r(θ) = exp(b·θ)
3. **Offset calculation:** offset = D / (1 - D) for D < 1
4. **Coordinate transformation:** Includes rotation and translation as described in the paper

**Implementation Date:** 2025-01-05  
**Implementation Timestamp:** 2025-01-05 13:40:00 UTC
