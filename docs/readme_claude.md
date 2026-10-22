# Guidance for agents: the `docs/` Quarto site

This folder holds the public-facing report on forest and forest loss in Guinea. It is a Quarto website: a set of
`.qmd` pages that mix prose with Python code. The code runs when the site is rendered, and the results (tables,
plots, maps) are written into the HTML. Read this file before changing anything here.

## 0. First time: setting up on a new machine or a fork

Use these steps when you have just cloned or forked the repository, or when `docs/` is not on your machine yet.

1. **Get the repository.** Clone your fork and open a terminal in its root:

   ```bash
   git clone https://github.com/<your-account>/Guinea_Methodology.git
   cd Guinea_Methodology
   ```

2. **Install the two tools.**
   - [uv](https://docs.astral.sh/uv/getting-started/installation/) manages Python and the project's packages. It
     installs Python 3.13 itself if you do not have it (the repository pins 3.13 in `.python-version`).
   - [Quarto](https://quarto.org/docs/get-started/) renders the site. Check the install with `quarto --version`.

3. **Create the Python environment.** From the repository root:

   ```bash
   uv sync
   ```

   This creates `.venv/` with every dependency in `pyproject.toml`, including `nbclient` and `pyyaml`, which Quarto
   needs to run Python chunks. `.venv/` is git-ignored.

4. **Check that the data is there.** The report reads tables from `studies/output_datasets/`. The render stops with an
   error if one is missing. The one most likely to be absent in a fresh clone is
   `hansen_forest_loss_h3res8.parquet` (17 MB). It was produced by section 0.3 of `studies/report_1.ipynb` and must be
   committed to the repository, not only kept on the original machine.

5. **Preview the site.** Go into `docs/` and run:

   ```bash
   cd docs
   QUARTO_PYTHON=../.venv/bin/python quarto preview
   ```

   On Windows the interpreter is `..\.venv\Scripts\python.exe`, and the same variable is set with `set` in the
   Command Prompt or `$env:QUARTO_PYTHON = ...` in PowerShell.

The first render executes every Python chunk, so it takes noticeably longer than a later preview of an unchanged
page. The report does not need `gee-account.json` or any Earth Engine access: everything it reads is already saved in
the repository. Do not copy that key onto a fork or into the repository; it is git-ignored for that reason.

## 1. How Quarto works, briefly

- **Source files** are `.qmd` pages. Text is Markdown. A fenced block starting with ```` ```{python} ```` is a code
  *chunk*, and its output is inserted in the page. Inline values are written as `` `{python} expression` `` and are
  replaced by the result of that expression.
- **Chunk options** go at the top of the chunk as `#|` lines: `#| label:` gives an ID, `#| fig-cap:` and
  `#| tbl-cap:` give captions that become numbered figures and tables, and `#| echo: false` hides the code. Cross-
  references are written `@fig-name`, `@tbl-name` or `@sec-name`, and the label must match.
- **Config** is `_quarto.yml`: the site title, the navigation bar (which pages appear in it), the theme and `styles.css`.
- **Execution.** Quarto runs the Python code with the interpreter in `../.venv` (set via `QUARTO_PYTHON`). The
  `execute: freeze: auto` setting stores the last results in `_freeze/`, so a page is only re-executed when its source
  changes.
- **Output** goes to `_site/`. That folder is generated; never edit it by hand.

## 2. Running the site locally

Open a terminal in this folder and run:

```bash
quarto preview
```

Quarto renders the site, starts a local server and opens it in the browser; it reloads the page when a source file or
the stylesheet changes. To preview only the report, run `quarto preview forest_loss_assessment.qmd`. For a full
render without preview, use `quarto render`.

If Quarto cannot find Python, point it at the project environment first:

```bash
QUARTO_PYTHON=../.venv/bin/python quarto preview
```

The project environment must contain `nbclient` and `pyyaml`; they are in `pyproject.toml`. Install the dependencies
with `uv sync` from the repository root if the environment is missing.

## 3. Layout

| File | Role |
| --- | --- |
| `index.qmd` | Landing page and link to the report |
| `forest_loss_assessment.qmd` | The report: all text, tables, plots and maps |
| `report_utils.py` | Shared code: data loading, statistics, map helpers. Imported by the report |
| `references.bib` | Bibliography, cited as `[@key]` |
| `styles.css` | Visual styling (table headers, captions, map width) |
| `_quarto.yml` | Site configuration |
| `_freeze/`, `_site/`, `.quarto/`, `__pycache__/` | Generated. All four are git-ignored |

## 4. Where the numbers come from

Nothing in `docs/` calls Google Earth Engine. Every figure is read from tables saved by the notebooks in `studies/`,
under `studies/output_datasets/`:

- `hansen_loss_by_polygon_cell_year.parquet`: wall-to-wall Hansen by ecoregion, 1° cell and loss year
  (`deforestation_hansen.ipynb`).
- `hansen_loss_by_release_year.parquet`: loss per year for each Hansen release (`hansen_annual_maps.ipynb`).
- `hansen_forest_loss_h3res8.parquet`: forest and loss per H3 resolution-8 hexagon (`report_1.ipynb`, section 0.3).
  This is the only table that cost Earth Engine compute, and it is cached.
- `lulc_benchmark/*_h3res7.parquet` and `sample_points_h3res7.parquet`: the seven products read at 56,712 points
  (`lulc_benchmark.ipynb`).

`report_utils.py` loads these with `functools.lru_cache`, so each file is read once per render. If a table is missing,
`fine_hexagons()` raises an error that says which notebook makes it. Do not add a call to Earth Engine to the report.
If new numbers are needed, compute them in a notebook, save the result, and read it here.

## 5. Conventions the report follows

- **Units.** Areas are in hectares (ha) in text and tables. Plotly charts are interactive; maps are static Matplotlib
  images or Leaflet maps through `folium`.
- **Forest** means canopy cover above 30 % in 2000 for Hansen, unless the text says otherwise. The 10 % and 50 %
  thresholds are shown for sensitivity.
- **Analysis period** is 2013–2025 (after Hansen's method change). 2020–2025 is the second focus period. Loss before
  2013 is used only to compute the forest at the start of 2013.
- **Hexagons.** H3 resolution 8 (about 62 ha) is the analysis unit. Resolution 6 (about 3,000 ha) is used for the
  hover layers.
- **Numbers in the text** come from the hidden setup chunk (`N` dictionary, `fmt()` for formatting). Write new
  numbers the same way, so the text cannot drift from the data. Do not type a number that the data produce.
- **Writing style.** The audience is senior officials and technical staff in the forestry departments. Keep
  sentences direct and specific, avoid generic phrasing, and do not add emphasis that the data do not support.

## 6. Known limitations you must keep in mind

Some choices in this report are not the right ones for a robust, production monitoring pipeline. They were made to keep
the interactive maps light enough to load in a browser:

- **Maps are thinned for size, not for accuracy.** Hover values are shown on 3,000 ha hexagons (resolution 6) rather
  than the 62 ha hexagons the analysis uses. Hexagons with no forest and no loss are dropped from the hover layer. The
  forest-cover map has no hover layer at all. Coordinates are rounded to about 100 m. A pipeline should serve the
  analysis resolution, with full precision, and load it from a tile service or a database instead of embedding it.
- **The page is large.** The report is about 17 MB because the interactive maps are embedded in the HTML as images and
  inline GeoJSON. A production site should keep the data in separate files or a backend and load them on demand.
- **Images are embedded as overlays.** The hexagon layers are raster images (`ImageOverlay`) made on a 0.0025° display
  grid. That is fast to draw but cannot be queried or styled per hexagon. Vector hexagons with a proper tile server
  would be the robust option.
- **Frozen results.** `_freeze/` keeps the last computed output. If a notebook table changes, delete the matching
  folder in `_freeze/` (for the report, `_freeze/forest_loss_assessment/`) and render again, or the old numbers will
  remain. `quarto render --cache-refresh` refreshes the execution cache instead.
- **Single-machine dependencies.** Rendering depends on the `../.venv` environment and on the `studies/output_datasets`
  tables being present. A reproducible pipeline would pin these and record their versions.

These trade-offs are deliberate for the current stage. They are not a template for the operational system.

## 7. Making changes

- **Editing text:** change the Markdown in the `.qmd` file. Keep numbers in inline `{python}` expressions.
- **Adding a figure:** put a chunk with `#| label: fig-…` and `#| fig-cap:` in the report, and refer to it with
  `@fig-…`. Use the existing helpers in `report_utils.py` for maps.
- **Adding a citation:** add a BibTeX entry to `references.bib`, then cite it as `[@key]`.
- **Adding a page:** create the `.qmd`, add it to the navigation in `_quarto.yml`, and link it from `index.qmd`.
- **Checking the result:** run `quarto render` and read the terminal for `Error` or `Warning` lines. Open the HTML and
  check that no `{python}` expression appears as raw text and that cross-references show numbers, not `?@`.
- **Do not commit** `_site/`, `.quarto/` or `__pycache__/`. `docs/.gitignore` covers them.

## 8. How heavy the site is, and what that means for git

Measured on the last render of the report:

| What | Size | Committed? |
| --- | --- | --- |
| `_site/forest_loss_assessment.html` (the report page) | about 17 MB | No, ignored |
| `_site/` in total | about 21 MB | No, ignored |
| `_freeze/` (stored results of the code chunks) | about 20 MB | No, ignored |
| `studies/output_datasets/hansen_forest_loss_h3res8.parquet` (report input) | 17 MB | Must be committed |

**Why the page is heavy.** The three interactive maps are embedded in the HTML: the hexagon layers are PNG images
(about 2 MB each) and the outlines are inline GeoJSON. Each map is a self-contained file, so no tile server or data
folder is needed, which is why the page works on any host. The cost is that the whole page downloads on first view.

**Why this matters for the repository.** GitHub rejects any single file over 100 MB, and warns above 50 MB. The page
is well under both, so the limit is not reached today. The problem is growth: every commit of a changed `_site/` or
`_freeze/` adds its full size to the git history permanently, even if the next commit deletes it. A few re-renders
would add 100 MB or more to the clone that every collaborator and fork downloads. The rule is therefore:

- Commit the source (`.qmd`, `report_utils.py`, `_quarto.yml`, `references.bib`, `styles.css`) and the input tables.
- Do not commit `_site/`. Publish it instead (section 9).
- `_freeze/` is git-ignored. Committing it would let a fresh clone preview without running the code, but every change
  to the report would add about 20 MB to the history. The deploy workflow renders everything itself, so it does not
  need `_freeze/`.

GitHub Pages sites are limited to 1 GB of published files, and GitHub documents a soft bandwidth limit of 100 GB per
month. A page of 17 MB is far from both, but a site with many reports, or a report that loads the full-resolution
hexagons, would need the data split into separate files (section 6).

## 9. Deploying to GitHub Pages

The repository's remote is `https://github.com/SDG-AI-Lab/Guinea_Methodology.git`, so a published site would normally
be at `https://sdg-ai-lab.github.io/Guinea_Methodology/`. Check the exact address in the repository's Pages settings.

**Visibility.** The repository is public, so the site is readable by anyone with the link. The report is written for
internal decision-makers and contains hotspot locations, so confirm with the team that publication is intended before
the first deploy.

### How it is published now: GitHub Actions

`.github/workflows/publish-report.yml` renders and publishes the site. It runs on every push to `main` that changes
`docs/`, `studies/output_datasets/`, `pyproject.toml`, `uv.lock` or `.python-version`, and it can be started by hand
from the **Actions** tab (*Publish report → Run workflow*).

What the workflow does:

1. Checks out the repository and installs uv.
2. Runs `uv sync --frozen`, which builds `.venv` from `uv.lock`. Python 3.13 is taken from `.python-version`.
3. Installs Quarto at the pinned version (1.10.18).
4. Runs `quarto render` in `docs/` with `QUARTO_PYTHON` pointing at the created `.venv`. Every chunk executes, since
   `_freeze/` is not in the repository. No Earth Engine access and no secret is needed.
5. Uploads `docs/_site` and deploys it to GitHub Pages.

Requirements that must hold in the repository:

- **Pages source is GitHub Actions.** Set once in **Settings → Pages → Build and deployment → Source**. If it is set to
  a branch, the workflow's deploy step fails.
- **`pyproject.toml` and `uv.lock` are committed together.** `uv sync --frozen` fails if the lock file does not match
  the project, so a dependency change must include both files.
- **The inputs are committed**, including `studies/output_datasets/hansen_forest_loss_h3res8.parquet` (section 0, step 4).

Checking a run: open the **Actions** tab, select *Publish report*, and read the *Render the site* step for `Error` or
`Warning` lines. The live address appears on the workflow run's `github-pages` environment link.

To render and check the page before pushing, run `quarto render` in `docs/` locally, as in section 7. The workflow
does not check the rendered output for you; a failed chunk makes the run fail, but a visual problem is only caught by
looking at the page.

### Alternative: `quarto publish gh-pages` from a laptop

`quarto publish gh-pages` renders the site on your machine and pushes it to a `gh-pages` branch. It is not used for
the normal deploy, because the published version would then depend on whoever ran it. If it is used, switch the
Pages source to **Deploy from a branch** (`gh-pages`, `/ (root)`), and disable the workflow, or the two methods will
overwrite each other.
