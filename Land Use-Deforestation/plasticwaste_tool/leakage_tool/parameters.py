"""
Every number the leakage model uses, with its source and its uncertainty.

WHY THIS IS A SEPARATE MODULE, AND THE FIRST ONE

The mass-balance arithmetic in `massbalance.py` is trivial -- six multiplications.
Nothing about this component is difficult except deciding what to multiply by,
and every one of those decisions is a literature judgement that a reviewer is
entitled to disagree with. Burying them as float literals inside the calculation
would make the model look more authoritative than it is and would make the
disagreement impossible to express.

So each parameter is an object carrying a central value, a plausible range, a
distribution, and the citation it came from. A reviewer who rejects one can
change it in one place and re-run; a reader can see the whole evidential basis
of the estimate in one screen.

THE HONEST SUMMARY OF WHAT IS KNOWN

Guinea-specific measurements exist for the FRONT of the chain and essentially
nothing exists for the BACK of it:

  generation rate   Guinea-specific, modelled       (What a Waste, 2022)
  collection rate   Guinea-specific, SURVEYED       (JICA 2022, urban/rural)
  plastic share     Guinea-specific but CONTESTED   (see PLASTIC_SHARE below)
  open burning      regional/global only            (Cottom et al. 2024)
  leakage to water  global model only               (Meijer et al. 2021)

The uncertainty therefore grows as the chain proceeds, and the output is a
range spanning roughly an order of magnitude. That is the honest answer and the
model is built to report it rather than to hide it behind a point estimate.

UNITS: mass is tonnes per year throughout. Rates are dimensionless fractions
unless the name says otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Parameter:
    """
    One quantity, its uncertainty, and where it came from.

    `dist` is how a Monte Carlo draw is taken:
      "lognormal"  positive quantities whose error is multiplicative -- the
                   default for rates derived from modelled national statistics,
                   because such estimates are wrong by FACTORS, not by
                   additive amounts, and a lognormal cannot go negative.
      "uniform"    used where the literature gives a range with no basis for
                   preferring its centre -- an admission of ignorance, not a
                   modelling choice.
      "beta"       fractions bounded to [0, 1] that must not be allowed to
                   escape that interval under a wide draw.
      "fixed"      no uncertainty propagated (scenario parameters, which are
                   varied explicitly instead).
    """

    name: str
    value: float
    low: float
    high: float
    units: str
    source: str
    dist: str = "lognormal"
    note: str = ""

    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """n independent samples. Correlation between parameters is handled by
        the CALLER drawing once per iteration and reusing -- see uncertainty.py."""
        if self.dist == "fixed":
            return np.full(n, self.value, dtype=float)
        if self.dist == "uniform":
            return rng.uniform(self.low, self.high, n)
        if self.dist == "lognormal":
            # Map [low, high] onto a central 90% interval, so the quoted range
            # means "90% credible", not "absolute bound".
            if self.value <= 0 or self.low <= 0:
                return rng.uniform(self.low, self.high, n)
            mu = np.log(self.value)
            sigma = (np.log(self.high) - np.log(self.low)) / (2 * 1.645)
            return rng.lognormal(mu, sigma, n)
        if self.dist == "beta":
            # Method of moments on a [0,1] fraction, treating the range as
            # +/- 1.645 sd, then clipped. Keeps fractions physical.
            sd = (self.high - self.low) / (2 * 1.645)
            m = np.clip(self.value, 1e-6, 1 - 1e-6)
            v = min(sd ** 2, m * (1 - m) * 0.999)
            if v <= 0:
                return np.full(n, m)
            k = m * (1 - m) / v - 1
            return rng.beta(max(m * k, 1e-3), max((1 - m) * k, 1e-3), n)
        raise ValueError(f"unknown distribution {self.dist!r}")


# ---------------------------------------------------------------------------
# 1. WASTE GENERATION
# ---------------------------------------------------------------------------

MSW_RATE = Parameter(
    name="msw_rate",
    value=0.3095, low=0.232, high=0.413,
    units="kg/person/day",
    source=("World Bank, What a Waste Global Database (WB_WAW_WIDEF), "
            "indicator WM_MSW_GEN, Guinea, 2022. Underlying method: Kaza, S., "
            "Yao, L., Bhada-Tata, P. & Van Woerden, F. (2018) What a Waste 2.0: "
            "A Global Snapshot of Solid Waste Management to 2050. World Bank."),
    dist="lognormal",
    note=("The 2022 value is MODELLED, not measured -- Guinea has no national "
          "waste-generation survey in the database. The range is +/-33%, which "
          "is the spread Kaza et al. report between reported and modelled "
          "per-capita rates for low-income countries. The database's own "
          "projections (0.3188 in 2030, 0.3382 in 2050) sit well inside it, so "
          "the range also absorbs the choice of reference year."),
)

# ---------------------------------------------------------------------------
# 2. PLASTIC SHARE -- THE DOMINANT UNCERTAINTY, HANDLED AS A SCENARIO
# ---------------------------------------------------------------------------
#
# This single parameter moves the answer by a factor of four to five, more than
# every other parameter combined. It is NOT propagated as a distribution,
# because averaging over it would produce a number belonging to neither world
# and would conceal that the disagreement is about evidence, not noise.
# Every result is reported under each scenario separately.

PLASTIC_SHARE_SCENARIOS: Dict[str, Parameter] = {
    "waw_2007": Parameter(
        name="plastic_share", value=0.04, low=0.03, high=0.05,
        units="fraction of MSW by mass",
        source=("United Nations Statistics Division (2010), Country Files from "
                "the UNSD/UNEP data collection on environment statistics, "
                "Guinea 2007; distributed via the World Bank What a Waste "
                "database, indicator WM_MSW_COMP, WASTE_PRODUCT = Plastic."),
        dist="beta",
        note=("The only Guinea-wide composition figure in any international "
              "database. It is from 2007 and its companion entries (58% food, "
              "23% other, 9% paper) are typical of a pre-sachet-economy waste "
              "stream. Using it unmodified in 2024 assumes eighteen years of "
              "no change in packaging -- implausible, but it is what the "
              "authoritative source says, so it is carried as the LOW "
              "scenario rather than silently discarded."),
    ),
    "conakry_recent": Parameter(
        name="plastic_share", value=0.185, low=0.15, high=0.22,
        units="fraction of MSW by mass",
        source=("Range reported for Conakry municipal waste characterisation "
                "in recent national and municipal assessments; consistent with "
                "the 13-20% plastic fractions reported for West African "
                "coastal cities in Kaza et al. (2018), What a Waste 2.0, "
                "Table 2.4 (Sub-Saharan Africa urban composition)."),
        dist="beta",
        note=("Reflects the sachet-water and single-use packaging economy that "
              "the 2024 presidential decree banning single-use plastic "
              "packaging (recorded in the same database under WM_LEG_PLS) "
              "was a response to. Applies to URBAN Conakry; applying it "
              "nationally, including to rural prefectures, likely "
              "over-estimates. This is the HIGH scenario."),
    ),
}

DEFAULT_SCENARIO = "waw_2007"

# ---------------------------------------------------------------------------
# 3. COLLECTION -- THE ONE SURVEYED LINK IN THE CHAIN
# ---------------------------------------------------------------------------
#
# These are the strongest numbers in the model: an actual field survey, split
# urban/rural, which is why the model bothers to classify population by
# settlement density at all (see population.py). Collapsing them to the 9.7%
# national average would discard the structure that drives the whole spatial
# pattern.

COLLECTION_URBAN = Parameter(
    name="collection_urban",
    value=0.237, low=0.19, high=0.285,
    units="fraction of population served",
    source=("Japan International Cooperation Agency (JICA) (2022), Data "
            "Collection Survey on Municipal Solid Waste Management in Africa, "
            "Guinea country sheet, reference year 2018; distributed via the "
            "World Bank What a Waste database, indicator WM_COL_COV, "
            "URBANIZATION = Urban area."),
    dist="beta",
    note=("+/-20% relative, reflecting that a single survey year is being "
          "applied to a different year and that collection coverage in "
          "Conakry is known to fluctuate with contractor arrangements."),
)

COLLECTION_RURAL = Parameter(
    name="collection_rural",
    value=0.021, low=0.010, high=0.042,
    units="fraction of population served",
    source=("JICA (2022), as above, URBANIZATION = Rural area, 2018."),
    dist="beta",
    note=("2.1% is close enough to zero that the multiplicative range is wide "
          "(half to double) without materially changing any result: rural "
          "waste is essentially entirely uncollected under any reading."),
)

# National cross-check, not used in the calculation: WM_COL_COV Total = 9.7%.
# The urban/rural pair must reproduce roughly this when weighted by the
# population split. `massbalance.check_collection_consistency` tests it.
COLLECTION_NATIONAL_REPORTED = 0.097

# ---------------------------------------------------------------------------
# 4. FATE OF UNCOLLECTED WASTE
# ---------------------------------------------------------------------------
#
# Uncollected does NOT mean leaked. In low-income settings most uncollected
# household waste is burned in the open or buried in a household pit, and
# burned plastic is destroyed as macroplastic -- it becomes an air-quality
# problem instead, which this model does not attempt to quantify.
#
# Omitting this step is the single most common way national leakage estimates
# come out too high, so it is included even though the evidence is regional
# rather than Guinean.

OPEN_BURN_FRACTION = Parameter(
    name="open_burn_fraction",
    value=0.55, low=0.40, high=0.70,
    units="fraction of uncollected plastic waste",
    source=("Cottom, J.W., Cook, E. & Velis, C.A. (2024), A local-to-global "
            "emissions inventory of macroplastic pollution, Nature 633, "
            "101-108 -- which finds open burning to be the dominant fate of "
            "uncollected plastic across the Global South, exceeding "
            "unburned debris emissions. Consistent with Wiedinmyer, C., "
            "Yokelson, R.J. & Gullett, B.K. (2014), Global emissions of trace "
            "gases, particulate matter, and hazardous air pollutants from open "
            "burning of domestic waste, Environmental Science & Technology "
            "48(16), 9523-9530."),
    dist="beta",
    note=("No Guinea-specific measurement exists. The range is deliberately "
          "wide. This parameter and LEAK_TO_ENV below are the two places where "
          "the model stops being about Guinea and starts being about the "
          "Global South in general; both are flagged as such in the output."),
)

LEAK_TO_ENV = Parameter(
    name="leak_to_env",
    value=0.75, low=0.55, high=0.90,
    units="fraction of unburned uncollected plastic",
    source=("Jambeck, J.R. et al. (2015), Plastic waste inputs from land into "
            "the ocean, Science 347(6223), 768-771, which treats mismanaged "
            "waste as that which is 'littered or inadequately disposed'; and "
            "UNEP/IUCN (2020), National Guidance for Plastic Pollution "
            "Hotspotting and Shaping Action."),
    dist="beta",
    note=("The remainder is buried on-plot or otherwise contained. The "
          "distinction between 'inadequately disposed at an unmanaged dump' "
          "and 'leaked to the environment' is genuinely blurred in the "
          "literature -- an unmanaged dumpsite IS the environment in "
          "hydrological terms, which is precisely why the mobilisation model "
          "in characterisation_tool treats dumpsites as sources rather than "
          "as sinks."),
)

# Collected waste is not perfectly contained either.
LITTER_RATE_COLLECTED = Parameter(
    name="litter_rate_collected",
    value=0.02, low=0.01, high=0.04,
    units="fraction of collected plastic waste",
    source=("Jambeck, J.R. et al. (2015), Science 347, 768-771, which applies "
            "a 2% littering rate to waste within collection systems."),
    dist="beta",
    note=("Small in relative terms and, at Guinea's 9.7% national collection "
          "coverage, negligible in absolute terms. Retained so the mass "
          "balance is complete rather than because it changes the answer."),
)

# ---------------------------------------------------------------------------
# 5. TRANSPORT TO WATER
# ---------------------------------------------------------------------------

P_TO_WATER = Parameter(
    name="p_to_water",
    value=0.03, low=0.005, high=0.10,
    units="fraction of environmental plastic reaching a watercourse per year",
    source=("Meijer, L.J.J., van Emmerik, T., van der Ent, R., Schmidt, C. & "
            "Lebreton, L. (2021), More than 1000 rivers account for 80% of "
            "global riverine plastic emissions into the ocean, Science "
            "Advances 7(18), eaaz5803 -- whose probabilistic transport model "
            "makes river entry a function of distance to water, terrain slope "
            "and runoff."),
    dist="lognormal",
    note=("A NATIONAL AVERAGE placeholder, and the weakest number in the "
          "model: it spans a factor of twenty. It exists so the chain can be "
          "closed end to end. For any SITE where terrain and hydrology are "
          "known, characterisation_tool/mobilisation.py computes a "
          "site-specific transport probability from slope, distance to "
          "drainage and flow accumulation, and that should be preferred over "
          "this constant. The two are combined in sites.py."),
)

# ---------------------------------------------------------------------------
# 6. URBAN / RURAL CLASSIFICATION
# ---------------------------------------------------------------------------
#
# Needed only because the collection survey is split that way. The threshold
# follows the UN-endorsed Degree of Urbanisation, applied to a 1 km population
# grid, which is exactly the input DEGURBA is defined on.

URBAN_DENSITY_THRESHOLD = Parameter(
    name="urban_density_threshold",
    value=300.0, low=300.0, high=1500.0,
    units="persons per km2",
    source=("Dijkstra, L., Florczyk, A.J., Freire, S., Kemper, T., Melchiorri, "
            "M., Pesaresi, M. & Schiavina, M. (2021), Applying the Degree of "
            "Urbanisation to the globe: A new harmonised definition reveals a "
            "different picture of global urbanisation, Journal of Urban "
            "Economics 125, 103312. Endorsed by the UN Statistical Commission "
            "(51st session, 2020)."),
    dist="uniform",
    note=("DEGURBA sets 300/km2 for an urban CLUSTER and 1500/km2 for an urban "
          "CENTRE. The JICA survey does not say which definition its "
          "'urban area' matches, so the choice is itself an uncertainty and is "
          "sampled between the two rather than fixed. Sensitivity to it is "
          "reported explicitly -- see `massbalance.urban_split_sensitivity`."),
)

# ---------------------------------------------------------------------------

DAYS_PER_YEAR = 365.25
KG_PER_TONNE = 1000.0


def all_parameters(scenario: str = DEFAULT_SCENARIO) -> Dict[str, Parameter]:
    """The complete parameter set under one plastic-share scenario."""
    if scenario not in PLASTIC_SHARE_SCENARIOS:
        raise KeyError(
            f"unknown scenario {scenario!r}; "
            f"choose from {sorted(PLASTIC_SHARE_SCENARIOS)}")
    return {
        "msw_rate": MSW_RATE,
        "plastic_share": PLASTIC_SHARE_SCENARIOS[scenario],
        "collection_urban": COLLECTION_URBAN,
        "collection_rural": COLLECTION_RURAL,
        "open_burn_fraction": OPEN_BURN_FRACTION,
        "leak_to_env": LEAK_TO_ENV,
        "litter_rate_collected": LITTER_RATE_COLLECTED,
        "p_to_water": P_TO_WATER,
        "urban_density_threshold": URBAN_DENSITY_THRESHOLD,
    }


def provenance_table(scenario: str = DEFAULT_SCENARIO):
    """Every parameter as a row, for the methodology appendix."""
    import pandas as pd
    rows = []
    for key, p in all_parameters(scenario).items():
        rows.append({
            "parameter": p.name,
            "value": p.value,
            "low": p.low,
            "high": p.high,
            "units": p.units,
            "distribution": p.dist,
            "guinea_specific": key in ("msw_rate", "plastic_share",
                                       "collection_urban", "collection_rural"),
            "source": p.source,
            "note": p.note,
        })
    return pd.DataFrame(rows)
