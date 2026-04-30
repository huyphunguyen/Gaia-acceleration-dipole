# tests/test_sbi_setup.py
import numpy as np
import healpy as hp
import pytest
from sbi_hpm.sbi_setup import (
    n_real_params,
    build_full_alm_complex,
    build_det_alm_complex,
    precompute_noise_cholesky,
    generate_pixel_noise,
)

LMAX_SIG = 3
LMAX_FIT = 10
N_ALM_L3 = n_real_params(LMAX_SIG)   # 30
N_CL = 2 * (LMAX_FIT - LMAX_SIG)     # 14

def test_n_real_params():
    assert n_real_params(1) == 6
    assert n_real_params(3) == 30
    assert n_real_params(10) == 240

def test_build_full_alm_complex_shape():
    theta_l3 = np.zeros(N_ALM_L3)
    Cl_E = np.ones(LMAX_FIT - LMAX_SIG) * 1e-6
    Cl_B = np.ones(LMAX_FIT - LMAX_SIG) * 1e-6
    E, B = build_full_alm_complex(theta_l3, Cl_E, Cl_B, LMAX_SIG, LMAX_FIT)
    expected = hp.Alm.getsize(LMAX_FIT)
    assert E.shape == (expected,)
    assert B.shape == (expected,)

def test_build_det_alm_deterministic():
    # With Cl=0, det and full should give same result
    theta_l3 = np.random.randn(N_ALM_L3) * 1e-3
    E_det, B_det = build_det_alm_complex(theta_l3, LMAX_SIG, LMAX_FIT)
    E_full, B_full = build_full_alm_complex(
        theta_l3, np.zeros(LMAX_FIT - LMAX_SIG), np.zeros(LMAX_FIT - LMAX_SIG),
        LMAX_SIG, LMAX_FIT,
    )
    np.testing.assert_array_equal(E_det, E_full)

def test_generate_pixel_noise_shape():
    nside = 16
    npix = hp.nside2npix(nside)
    mask = np.ones(npix, dtype=bool)
    sig = np.ones(npix) * 0.01
    corr = np.zeros(npix)
    L11, L21, L22, valid_pix = precompute_noise_cholesky(sig, sig, corr, mask, npix)
    noise_ra, noise_dec = generate_pixel_noise(L11, L21, L22, valid_pix, npix)
    assert noise_ra.shape == (npix,)
    assert noise_dec.shape == (npix,)
