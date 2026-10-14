# Calculation methodology improvement

Workspace for improving the calculation methodology of the three Guinea
modules: Land Use, Deforestation and Plastic Waste.

Branch `feat/calc-methodology-improvement` was cut from
`fix/fdi-bands-and-esri-confirm` at `45d7b77` (MARIDA debris classifier after
the FDI rule; Biermann et al. FDI bands; free Esri stage 2). That is the most
recent calculation state. The 15 commits `main` has gained since then touch
only `frontend/`, `backend/` and `specs/`, never the calculation code.

## Layout

```
methodology_improvement/
├── README.md        this file: rules and workflow
├── BASELINES.md     the numbers every change is compared against
├── BACKLOG.md       candidate improvements, owner, status
├── DECISIONS.md     what was tried, what was kept, what was rejected and why
└── experiments/
    ├── _template/   copy this to start an experiment
    ├── landuse/
    ├── deforestation/
    └── plasticwaste/
```

The production code does not move. `backend/` imports `landuse_tool` and
`plasticwaste_tool` from `Land Use-Deforestation/`, and docker compose mounts
that folder at `/landuse_tool`. Renaming or moving anything there breaks the
platform.

| Module        | Production code                                         | Methodology doc                                                   |
|---------------|---------------------------------------------------------|-------------------------------------------------------------------|
| Land Use      | `landuse_tool/`                                         | `Docs/Methodology_LandUse_Guinea.docx`                            |
| Deforestation | `landuse_tool/deforestation_tool/`                      | `Docs/Methodology_Deforestation_Guinea.docx`, `DEFORESTATION_METHODOLOGY.md` |
| Plastic Waste | `plasticwaste_tool/` (`fdi_tool`, `dumpsite_tool`, `marida_tool`, `leakage_tool`, `characterisation_tool`) | `Docs/Methodology_PlasticWasteAnalysis_Guinea_v2.docx` |

## Workflow

1. **Pick or add an item** in `BACKLOG.md` and put your name on it.
2. **Start an experiment.** Copy `experiments/_template/` to
   `experiments/<module>/<YYYY-MM-DD>_<initials>_<slug>/`. The date and
   initials prefix means the two of us never collide on a folder name.
3. **Write the hypothesis and the baseline number before running anything.**
   The baseline comes from `BASELINES.md`. If the metric you need is not there,
   measure the current method first and add it.
4. **Run it.** Scripts and notebooks for the experiment live in its folder and
   import the production packages read-only. Do not edit the production
   modules while you are still experimenting.
5. **Record the verdict** in the experiment's `README.md`, then add one line to
   `DECISIONS.md`, whether the result was kept or rejected. Rejected results are
   worth recording: they stop the next person repeating the test.
6. **Promote** only an accepted change. Move it into the production module as
   its own commit, together with a test, the updated baseline row in
   `BASELINES.md` and the matching methodology doc change.

## Working together on the branch

- Each person works on a short-lived branch off this one, for example
  `calc/<initials>/<slug>`, and opens a PR back into
  `feat/calc-methodology-improvement`. The other person reviews it.
- Pull this branch before starting work: `git pull --rebase`.
- When `main` gains calculation changes, or before the final PR to `main`,
  merge `main` into this branch.
- Commit summaries, metric JSONs and small CSVs. Do not commit rasters, tile
  caches or model checkpoints over a few MB. `experiments/**/outputs/` is
  ignored for that reason.

## Acceptance bar

A change is accepted only when:

- it beats the baseline on the same held-out data, with the same metric;
- the gain holds on data the change was not tuned on (an unseen tile, region
  or year);
- the cost is stated, in runtime, Earth Engine quota or licensed imagery.

A change that does not meet all three is recorded as rejected.
