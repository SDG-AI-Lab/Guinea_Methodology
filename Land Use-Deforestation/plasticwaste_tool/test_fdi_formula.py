"""
Offline tests for the Floating Debris Index definition -- no network, no keys.

Pins `fdi_tool.spectral.compute_fdi` to Biermann et al. (2020, Sci. Rep. 10,
5364, eq. 1-2): NIR = B8, RE2 = B6, SWIR1 = B11, with the RED (B4) central
wavelength in the interpolation ratio. Run in the api image like
test_toolchain_006.py:

    docker compose exec -T api sh -c \\
        'cd /landuse_tool && pytest -q -p no:cacheprovider -p no:django plasticwaste_tool/test_fdi_formula.py'
"""

import numpy as np
import pytest

from plasticwaste_tool import config as pwcfg
from plasticwaste_tool.fdi_tool import spectral as SP


def _hand_fdi(b6, b8, b11):
    # Typed out from the paper, with the Sentinel-2A central wavelengths,
    # independently of the config the implementation reads.
    lam_red, lam_nir, lam_swir1 = 664.6, 832.8, 1613.7
    r_prime = b6 + (b11 - b6) * (lam_nir - lam_red) / (lam_swir1 - lam_red) * 10
    return b8 - r_prime


def test_matches_published_equation():
    b6, b8, b11 = 0.05, 0.10, 0.02
    got = SP.compute_fdi({"B4": np.array([0.03]), "B6": np.array([b6]),
                          "B8": np.array([b8]), "B11": np.array([b11])})
    assert got[0] == pytest.approx(_hand_fdi(b6, b8, b11), abs=1e-6)
    # 0.10 - [0.05 + (0.02 - 0.05) * 1.77220] = 0.103166
    assert got[0] == pytest.approx(0.103166, abs=1e-5)


def test_does_not_read_b8a():
    base = {"B4": np.array([0.03]), "B6": np.array([0.05]),
            "B8": np.array([0.10]), "B11": np.array([0.02])}
    with_b8a = dict(base, B8A=np.array([0.9]))
    assert SP.compute_fdi(base)[0] == SP.compute_fdi(with_b8a)[0]


def test_requires_b6():
    with pytest.raises(KeyError):
        SP.compute_fdi({"B4": np.array([0.03]), "B8": np.array([0.1]),
                        "B8A": np.array([0.1]), "B11": np.array([0.02])})


def test_floating_material_raises_fdi_over_water():
    # NIR lifted above the red-edge/SWIR baseline -> positive anomaly.
    water = _hand_fdi(0.025, 0.020, 0.015)
    debris = _hand_fdi(0.085, 0.120, 0.025)
    got = SP.compute_fdi({"B6": np.array([0.025, 0.085]),
                          "B8": np.array([0.020, 0.120]),
                          "B11": np.array([0.015, 0.025])})
    assert got[0] == pytest.approx(water, abs=1e-6)
    assert got[1] == pytest.approx(debris, abs=1e-6)
    assert got[1] > got[0]


def test_formula_id_is_in_the_cache_key(monkeypatch):
    from plasticwaste_tool import api_connector as api
    before = api.aoi_cache_token("conakry")
    monkeypatch.setitem(pwcfg.S2_CONFIG, "fdi_formula", "some-other-definition")
    assert api.aoi_cache_token("conakry") != before
