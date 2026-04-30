import numpy as np
import pandas as pd
import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u

from .pixelization import Pixelize, pixelize_full_covariance


def load_gaia_qso(path):
    """Load Gaia QSO catalog and apply basic selection cuts.

    Returns DataFrame with columns: ra, dec, pmra, pmdec,
    pmra_error, pmdec_error, pmra_pmdec_corr.
    """
    gaia = pd.read_csv(path)
    sel = (
        (np.abs(gaia["ra"]) > 0.0) &
        (np.sqrt(gaia["pmra"]**2 + gaia["pmdec"]**2) < 100)
    )
    return gaia[sel].reset_index(drop=True)


def compute_pixel_stats(ra, dec, pmra, pmdec, sig_pmra, sig_pmdec, pm_corr, nside, galcut = 0):
    """Bin catalog sources into HEALPix pixels using full 2x2 covariance weighting.

    Uses pixelize_full_covariance for jointly optimal pixel means.
    Returns pixel maps suitable for feeding directly into WeightedSHT.

    Parameters
    ----------
    ra, dec, pmra, pmdec, sig_pmra, sig_pmdec, pm_corr : ndarray (N,)
    nside : int

    Returns
    -------
    dict with keys:
        pmra_wm, pmdec_wm         : covariance-weighted proper motion maps
        sig_pmra_wm, sig_pmdec_wm : effective pixel uncertainty maps
        pm_corr_wm                : effective pixel correlation map
        n_per_pixel               : source count map
        mask                      : boolean validity mask
    """
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra) 
    
    mask_galactic = make_galactic_mask(nside, b_cut_deg=galcut)

    pix_obj = Pixelize(nside)
    n_pix = pix_obj.sum_in_pixels(theta, phi)

    pmra_wm, pmdec_wm, sig_pmra_wm, sig_pmdec_wm, pm_corr_wm = \
        pixelize_full_covariance(theta, phi, pmra, pmdec,
                                 sig_pmra, sig_pmdec, pm_corr, nside)

    mask = (
        (pmra_wm != 0) & (pmdec_wm != 0) &
        (sig_pmra_wm != 0) & (sig_pmdec_wm != 0) & mask_galactic
    )

    return dict(
        pmra_wm=pmra_wm,
        pmdec_wm=pmdec_wm,
        sig_pmra_wm=sig_pmra_wm,
        sig_pmdec_wm=sig_pmdec_wm,
        pm_corr_wm=pm_corr_wm,
        n_per_pixel=n_pix,
        mask=mask,
    )


def make_galactic_mask(nside, b_cut_deg=20.0):
    """Return boolean array: True for pixels with |b| > b_cut_deg."""
    npix = hp.nside2npix(nside)
    pix_idx = np.arange(npix)
    theta_g, phi_g = hp.pix2ang(nside, pix_idx, lonlat=False)
    ra_g = np.degrees(phi_g)
    dec_g = 90.0 - np.degrees(theta_g)
    coords = SkyCoord(ra_g * u.deg, dec_g * u.deg, frame="icrs")
    return np.abs(coords.galactic.b.deg) > b_cut_deg
