# Quaia Z-Bin SBI Acceleration Analysis

**Date:** 2026-05-03
**Status:** Approved

## Goal

Single notebook that runs the full SBI pipeline for each of 5 redshift bins of the Quaia quasar catalog, saves posterior samples per bin, and produces `gxyz_vs_z_quaia.pdf` and `gmag_vs_z_quaia.pdf`.

## File

`notebooks/quaia_sbi_zbins.ipynb`

## Parameters

| Parameter | Value |
|-----------|-------|
| nside | 64 |
| lmax_sig | 3 |
| lmax_fit | 10 |
| galcut | 30 deg |
| N_BINS | 5 (quantile) |
| n_simulations | 100,000 |
| n_posterior | 10,000 |
| model | full (alm_l≤3 + Cl_4..10) |
| alm_bound | 50e-3 mas/yr |
| Cl_bound | 50e-6 (mas/yr)² |

## Data

- Source: `quaia_G20.0_with_pmra_pmdec_corr.csv`
- Columns used: `ra, dec, pmra, pmdec, pmra_error, pmdec_error, pmra_pmdec_corr, redshift_quaia`
- Binning: `pd.qcut(redshift_quaia, q=5)` → equal-count bins
- No cross-match with Gaia needed (Quaia file has all required columns)

## Notebook Structure

### Section 1 — Imports & Config
Standard imports from `sbi_hpm.*`. Define all parameters. `os.makedirs('output/zbins', exist_ok=True)`.

### Section 2 — Load Data & Define Bins
```python
quaia = load_gaia_qso(path)
quaia['z_bin'], z_bin_edges = pd.qcut(quaia['redshift_quaia'], q=N_BINS, labels=False, retbins=True)
z_centers = 0.5 * (z_bin_edges[:-1] + z_bin_edges[1:])
```

### Section 3 — Main Loop (bins 0–4)

For each bin `i`:

1. **Resume check**: if `output/zbins/zbin_{i}.npz` exists, skip (print message).
2. **Filter**: `mask_z = quaia['z_bin'] == i` → extract ra, dec, pmra, etc.
3. **Pixelize**: `compute_pixel_stats(..., nside=64, galcut=30)` → `pmra_wm`, `pmdec_wm`, `sig_*`, `mask`
4. **Solver**: `WeightedSHT(nside, mask, ...)` + `build_lu(lmax_fit)` + `build_lu(lmax_sig)`
5. **Noise Cholesky**: `precompute_noise_cholesky(...)`
6. **Prior**: `BoxUniform` over `[alm_l≤3 (30), Cl_E_4..10 (7), Cl_B_4..10 (7)]` → 44-dim
7. **Simulate**: `simulate_for_sbi(sim_full_wrapped, prior, num_simulations=100_000)`
8. **Train**: `SNPE(...).train()` → `build_posterior()`
9. **Sample**: `posterior.sample((10_000,), x=x_obs_bin)` where `x_obs_bin = solve_fast(pmra_wm, pmdec_wm, lmax_fit)`
10. **Extract g**: `extract_g(samples_np)` → `g_full` (10000, 3)
11. **MAP g**: `skyfunc.s2g(best_fit_l3[dipole_idx] * 1e3)` from `solve_fast(..., lmax_sig)`
12. **Save**: `np.savez('output/zbins/zbin_{i}.npz', g_full=g_full, g_obs=g_obs, z_center=z_centers[i], z_edges=z_bin_edges[i:i+2])`

### Section 4 — Load Results
```python
results = [np.load(f'output/zbins/zbin_{i}.npz') for i in range(N_BINS)]
z_centers_loaded = [r['z_center'] for r in results]
g_fulls = [r['g_full'] for r in results]
```
Compute per-bin: `q16, q50, q84 = np.percentile(g_full, [16, 50, 84], axis=0)`

### Section 5 — Plot: gxyz_vs_z_quaia.pdf
- 3 components (gx, gy, gz) as error bars vs z_centers
- Asymmetric errors from [q16, q50, q84]
- Save to `output/zbins/gxyz_vs_z_quaia.pdf`

### Section 6 — Plot: gmag_vs_z_quaia.pdf
- |g| = sqrt(gx² + gy² + gz²) per sample → median + 16/84th percentile
- Single panel vs z_centers
- Save to `output/zbins/gmag_vs_z_quaia.pdf`

## Output Files

```
output/zbins/
  zbin_0.npz  …  zbin_4.npz   # posteriors (g_full, g_obs, z_center, z_edges)
  gxyz_vs_z_quaia.pdf
  gmag_vs_z_quaia.pdf
```

## Resume Safety

Before running each bin: check if `.npz` exists. If yes, load and skip simulation+training. This means a partial run can be resumed without re-running completed bins.

## Key Constraints

- `WeightedSHT.build_lu` is the expensive one-time cost per bin (~seconds, not minutes)
- Sparser bins (fewer sources) → sparser mask → smaller valid_pix count → faster noise generation
- Each bin is independent; no state leaks between bins
