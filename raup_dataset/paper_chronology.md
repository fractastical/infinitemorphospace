# Raup Shell Coiling Papers: Chronology and Evolution

## Paper Timeline

### 1962: Original Introduction
**Raup, D. M. (1962).** "Computer as Aid in Describing Form in Gastropod Shells."  
*Science*, 138(3537), 150-152.

**Key Contributions:**
- First introduction of the three-parameter model (W, D, T)
- Pioneered use of digital computers in paleontology
- Demonstrated that complex shell diversity can be reduced to a few key parameters
- Introduced the concept of theoretical morphospace

**Parameters Introduced:**
- **W (Whorl Expansion Rate):** How rapidly whorls increase in size
- **D (Distance from Coiling Axis):** Position of generating curve relative to axis
- **T (Translation Rate):** Rate of movement along coiling axis

**Extraction Date:** 2025-01-05  
**Extraction Timestamp:** 2025-01-05T14:00:00Z

---

### 1966: Detailed Mathematical Formulation
**Raup, D. M. (1966).** "Geometric Analysis of Shell Coiling: General Problems."  
*Journal of Paleontology*, 40(5), 1178-1190.

**Key Contributions:**
- Detailed mathematical formulation of the model
- Complete equations for shell surface generation
- Expanded parameter space exploration
- Comparison with empirical shell forms
- Introduction of **S (Shape of Generating Curve)** parameter

**Parameters Expanded:**
- **W, D, T:** Detailed mathematical treatment
- **S:** Shape parameter for generating curve (aperture cross-section)

**Mathematical Formulation:**
- Logarithmic spiral: r(θ) = exp(b·θ) where b = ln(W) / (2π)
- Offset calculation: offset = D / (1 - D)
- 3D coordinate transformation with rotation
- Generating curve (ellipse) with aspect ratio S

**Extraction Date:** 2025-01-05  
**Extraction Timestamp:** 2025-01-05T13:40:00Z  
**Source:** Primary reference for implementation

---

### 1967: Ammonoid Application
**Raup, D. M. (1967).** "Geometric Analysis of Shell Coiling: Coiling in Ammonoids."  
*Journal of Paleontology*, 41(1), 43-65.

**Key Contributions:**
- Application of model specifically to ammonoid shells
- Refinements for ammonoid-specific morphologies
- Analysis of ammonoid morphospace
- Comparison of theoretical vs. observed ammonoid forms

**Parameters:**
- Same four parameters (W, D, T, S) applied to ammonoids
- Parameter ranges adjusted for ammonoid-specific forms

**Extraction Date:** 2025-01-05  
**Extraction Timestamp:** 2025-01-05T14:00:00Z

---

## Model Evolution

### 1962 → 1966
- **Added:** Detailed mathematical formulation
- **Added:** S parameter (shape of generating curve)
- **Expanded:** Parameter space exploration
- **Expanded:** Empirical comparisons

### 1966 → 1967
- **Focused:** Application to specific group (ammonoids)
- **Refined:** Parameter ranges for ammonoid morphologies
- **Analyzed:** Morphospace occupation by ammonoids

## Current Implementation

The implementation in `../1966-raup.py` and `../create_raup_animations.py` is based primarily on the 1966 paper, which provides the most complete mathematical formulation. The model uses all four parameters (W, D, T, S) as described in Raup (1966).

## References

All papers are cited in `paper_citations.bib` with BibTeX format for easy import into reference managers.
