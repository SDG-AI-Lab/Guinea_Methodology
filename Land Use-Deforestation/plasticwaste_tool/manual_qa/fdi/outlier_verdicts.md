# Outlier-patch verdicts — Conakry AOI

Photo-interpretation of the largest published patch in each month the dominance
check flags (one patch above 10% of the month's mapped area), against the
true-colour Sentinel-2 composite for the same window. These are the checks
`plasticwaste_analysis.ipynb` section 8 asks for before the monthly trend figure
may be used.

Detector: FDI per Biermann et al. (2020) with B6, mean+3σ FDI-and-NDVI rule,
then the MARIDA debris classifier (P(debris) ≥ 0.5). Verdicts recorded for the
detector in use before October 2026 describe different patches; they remain in
git history and are not carried over.

Interpreter: project GIS analyst (single interpreter — no inter-rater estimate).
Recorded: 2026-10-02.

| Month | Patches | Largest patch | Share of month's area | Dominance flag | Verdict | Interpreter confidence | Note |
|---|---|---|---|---|---|---|---|
| 2023-11 | 44 | 1.04 ha (score 0.676) | 9.8% | no | — | — | Below the 10% flag; chip saved as `2023-11_outlier_qa.png` for reference |
| 2023-12 | 0 | — | — | — | — | — | No published patches |
| 2024-01 | 26 | 2.04 ha (score 0.749) | 22.6% | **yes** | **ambiguous** | medium | `patch_HIGH_8` in the photo review: near-shore water with land in view |

## What follows

**January 2024 cannot be adjudicated by eye.** Its largest patch is the same
class as most of the January review: near-shore water in which the outlined
patch is not distinguishable from the water around it. The January total
(9.0 ha) should be quoted with that caveat, and with the review's precision
(36.8% of decided patches, 26.9% worst case).

**November and December need no outlier verdict.** November's largest patch is
below the dominance threshold, and December has nothing published.
