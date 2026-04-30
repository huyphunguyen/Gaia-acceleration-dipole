import numpy as np


class SkyFunction:
    def __init__(self, nside):
        self.nside = nside  # reserved for future HEALPix operations; unused by current methods

    def vec2dir(self, vec):
        gx, gy, gz = vec
        hyp = np.sqrt(gx**2 + gy**2)
        dec = np.arctan2(gz, hyp)
        ra = np.arctan2(gy, gx)
        return np.degrees(ra) % 360, np.degrees(dec)

    def dir2vec(self, ra, dec):
        deg2rad = np.pi / 180.0
        return np.array([
            np.cos(deg2rad * ra) * np.cos(deg2rad * dec),
            np.sin(deg2rad * ra) * np.cos(deg2rad * dec),
            np.sin(deg2rad * dec),
        ])

    def s2g(self, s):
        # Real SH normalization: Y_1^0 ~ sqrt(3/4pi)*cos(theta), so gz gets sqrt(3/8pi)
        # while Y_1^{+/-1} ~ sqrt(3/4pi)*sin(theta), so gx/gy get sqrt(3/4pi).
        s10, s11_r, s11_i = s
        gz =  s10  * np.sqrt(3 / (8 * np.pi))
        gx = -s11_r * np.sqrt(3 / (4 * np.pi))
        gy =  s11_i * np.sqrt(3 / (4 * np.pi))
        return gx, gy, gz

    def g2s(self, g):
        gx, gy, gz = g
        s10  =  np.sqrt((8 * np.pi) / 3) * gz
        s11_r = -np.sqrt((4 * np.pi) / 3) * gx
        s11_i =  np.sqrt((4 * np.pi) / 3) * gy
        return s10, s11_r, s11_i
