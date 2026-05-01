# Design: Reduce Mask-Induced Mode Coupling in SBI Inference

**Date:** 2026-05-01
**Status:** Approved

---

## Problem

When applying galactic masks (20°, 30°) to the SBI proper-motion inference:

- Posterior credible intervals for gx, gy, gz blow up far beyond what pixel-count reduction alone explains
- At 30° mask, gx blows up more than gy and gz (asymmetric)
- Full model posteriors show gx median shifted away from det model posteriors
- TARP coverage tests confirm posteriors are well-calibrated — the uncertainty is real

**Root cause: mask-induced ℓ-leakage**

Full sky: Y_lm are orthogonal → ℓ=1 (dipole) decoupled from ℓ=4–10.
Masked sky: orthogonality broken → coupling matrix W_{1m,ℓ'm'} ≠ 0.
Full model includes stochastic ℓ=4–10 power (Cl_E, Cl_B). This power leaks through W into ℓ=1 estimates → posterior variance inflates proportional to higher-ℓ power, not pixel count.

Det model (higher-ℓ = 0) shows small blowup → confirms leakage is the cause.

gx-specific blowup at 30°: galactic mask is geometrically asymmetric in equatorial coordinates. The 20°→30° galactic strip specifically removes (RA≈0°, Dec≈+30°) and (RA≈180°, Dec≈-30°), which are both at galactic latitude |b|≈31.5° and have maximum gx sensitivity (cos(Dec)cos(RA) ≈ 0.87). The gy-sensitive equatorial regions (RA≈90°, 270°) were already masked at 20°.

---

## Solution

Two complementary changes:

### 1. Apodized Mask

Replace the hard galactic cut (boolean mask) with a smooth cosine taper. Hard edges contribute high-frequency power to the mask's spherical harmonic expansion, which broadens the coupling matrix W. A smooth taper concentrates mask power near m=0 and reduces off-diagonal coupling.

**Taper definition:**

```
w(b) = 0                                            if |b| < b_cut
       0.5 * (1 - cos(π * (|b| - b_cut) / apo_deg))  if b_cut ≤ |b| < b_cut + apo_deg
       1                                            if |b| ≥ b_cut + apo_deg
```

Alternatively: use `nmt.mask_apodization(binary_mask, apo_size=apo_deg, apotype='C2')` from NaMaster (already a project dependency).

Recommended default: `apo_deg=5°`. Provides meaningful coupling reduction without removing significant additional sky area.

**Pixel weighting:** In `WeightedSHT.__init__()`, multiply per-pixel inverse covariance by w²:

```python
self.Cinv_rr[mask] = w[mask]**2 * f / sigma_pmra[mask]**2
self.Cinv_rd[mask] = w[mask]**2 * (...)
self.Cinv_dd[mask] = w[mask]**2 * f / sigma_pmdec[mask]**2
```

Boolean inclusion mask: `w > 0`. Interface of `solve_fast()` unchanged.
Backward compatible: `apo_deg=0` reproduces original hard mask exactly.

**Files changed:** `data.py`, `forward_model.py`

---

### 2. Galactic Frame alm Extraction

The galactic mask in equatorial coordinates has azimuthal asymmetry (m_eq ≠ 0 components), which creates asymmetric coupling between different m modes — explaining why gx (m=1, Re) blows up more than gy (m=1, Im) or gz (m=0) at 30°.

In galactic coordinates the mask is a latitude band (b = const) → azimuthally symmetric → coupling matrix satisfies Δm_gal = 0 → each m block decouples → gx_gal and gy_gal are symmetrically constrained.

**Changes:**

#### 2a. Data pipeline — rotate proper motions to galactic frame

In `compute_pixel_stats()`, after catalog loading, transform (pmra, pmdec) → (pm_l_cosb, pm_b) using astropy's built-in frame transform (handles position-angle rotation per source automatically):

```python
c = SkyCoord(ra=ra*u.deg, dec=dec*u.deg,
             pm_ra_cosdec=pmra*u.mas/u.yr,
             pm_dec=pmdec*u.mas/u.yr, frame='icrs')
c_gal = c.galactic
pm_l = c_gal.pm_l_cosb.to(u.mas/u.yr).value
pm_b = c_gal.pm_b.to(u.mas/u.yr).value
```

**Critical: noise covariance must also be rotated.** The per-source error ellipse (sigma_pmra, sigma_pmdec, pm_corr) is in equatorial frame. The 2×2 rotation matrix R (position-angle of galactic North at each source) transforms it as Σ_gal = R · Σ_eq · R^T. This gives new (sigma_l, sigma_b, corr_lb) per source in galactic frame. These rotated errors must be passed to `pixelize_full_covariance()`.

The position-angle C can be computed per source as:
```python
# R = [[cos C, sin C], [-sin C, cos C]]
# C = position angle of galactic North in equatorial frame at each (ra, dec)
# Derivable from the SkyCoord transformation Jacobian, or via finite differencing
```

Pixelize using galactic (l, b) coordinates → HEALPix theta = π/2 - b_rad, phi = l_rad. WeightedSHT receives galactic-frame maps → solves for alm in galactic frame.

#### 2b. Alm rotation back to equatorial frame

After `solve_fast()`, rotate alm from galactic to equatorial using `healpy.Rotator`:

```python
rot = hp.Rotator(coord=['G', 'C'])  # galactic → equatorial
alm_eq = rot.rotate_alm(alm_gal, lmax=lmax)
```

This applies the Wigner-D matrices for all ℓ simultaneously. For ℓ=1 this is a 3×3 rotation; higher ℓ handled automatically.

#### 2c. Forward simulator update

The forward simulator (`simulator_full`, `simulator_det` in `sbi_setup.py`) must generate synthetic maps in galactic frame so WeightedSHT training simulations match inference:

1. Build alm in equatorial frame (existing logic)
2. Rotate equatorial alm → galactic using `hp.Rotator(coord=['C', 'G']).rotate_alm()`
3. Synthesize galactic-frame spin-1 maps via `hp.alm2map_spin()`
4. Add noise in galactic frame (noise covariance already rotated — see 2a)
5. `solve_fast()` returns galactic alm → rotate back to equatorial → return summary x

The returned summary x remains equatorial alm — no change to SBI prior or posterior extraction.

**Files changed:** `data.py`, `forward_model.py`, `sbi_setup.py`

---

## Data Flow (After Changes)

```
Catalog (RA/Dec, pmra/pmdec)
    │
    ├─ [NEW] rotate pm vectors to galactic frame (astropy)
    │
    ▼
pixelize_full_covariance() in galactic (l,b) coords
    │
    ├─ [NEW] make_apodized_galactic_mask() → float weights w
    │
    ▼
WeightedSHT (galactic frame, Cinv scaled by w²)
    │
    ▼
solve_fast() → alm in galactic frame
    │
    ├─ [NEW] hp.Rotator(['G','C']).rotate_alm() → alm in equatorial
    │
    ▼
s2g() → gx, gy, gz  (unchanged)
```

---

## Trade-offs

| | Apodized mask | Galactic frame |
|---|---|---|
| Mode coupling reduction | Moderate (smooth edges) | Stronger (Δm=0 block structure) |
| Implementation effort | Low | Medium |
| Backward compatible | Yes (apo_deg=0) | Requires simulator rewrite |
| SBI retraining needed | No (minor change) | Yes (new summary statistics) |
| Risk | Very low | Medium (rotation bugs) |

---

## Out of Scope

- Increasing lmax (separate change, expensive)
- Tighter higher-ℓ priors (science decision)
- Two-step inference (future work)
- Diagnostic computation of mode coupling matrix W (separate analysis task)

---

## Files Affected

| File | Change |
|---|---|
| `sbi_hpm/data.py` | `make_galactic_mask()` → `make_apodized_galactic_mask()`; galactic pm rotation |
| `sbi_hpm/forward_model.py` | `WeightedSHT.__init__()` accepts float weights; scales Cinv by w² |
| `sbi_hpm/sbi_setup.py` | Simulators generate galactic-frame maps; rotate alm before/after solve |
