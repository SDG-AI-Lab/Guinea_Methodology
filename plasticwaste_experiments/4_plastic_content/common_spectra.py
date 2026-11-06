"""Shared by the plastic-content tests: how deep a C-H absorption is in a spectrum."""
import numpy as np

# Centre and the two shoulders of each C-H feature, in nm. 1730 stays clear of the
# water-vapour window that starts at 1800.
FEATURES = {1215: (1170, 1260), 1730: (1660, 1780)}
HALF_WIDTH_NM = 10


def band_mean(spectra, wl, centre):
    """Mean reflectance within 10 nm of a wavelength, for each spectrum."""
    use = np.abs(wl - centre) <= HALF_WIDTH_NM
    if not use.any():
        use = np.abs(wl - centre) == np.abs(wl - centre).min()
    return np.nanmean(spectra[:, use], axis=1)


def depth(spectra, wl, feature):
    """Continuum-removed depth: 0 means no dip, 0.05 means 5 % below the straight
    line between the shoulders."""
    left, right = FEATURES[feature]
    a, b = band_mean(spectra, wl, left), band_mean(spectra, wl, right)
    weight = (feature - left) / (right - left)
    line = a + (b - a) * weight
    return 1.0 - band_mean(spectra, wl, feature) / np.where(np.abs(line) < 1e-6, np.nan, line)
