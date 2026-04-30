# tests/test_geometry.py
import numpy as np
import pytest
from sbi_hpm.geometry import SkyFunction

skyfunc = SkyFunction(nside=16)

def test_dir2vec_vec2dir_roundtrip():
    ra, dec = 45.0, 30.0
    vec = skyfunc.dir2vec(ra, dec)
    ra_out, dec_out = skyfunc.vec2dir(vec)
    assert abs(ra_out - ra) < 1e-10
    assert abs(dec_out - dec) < 1e-10

def test_s2g_g2s_roundtrip():
    g_in = (1.5, -2.3, 0.7)
    s = skyfunc.g2s(g_in)
    g_out = skyfunc.s2g(s)
    np.testing.assert_allclose(g_out, g_in, atol=1e-12)

def test_dir2vec_unit_length():
    ra, dec = 120.0, -45.0
    vec = skyfunc.dir2vec(ra, dec)
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-12
