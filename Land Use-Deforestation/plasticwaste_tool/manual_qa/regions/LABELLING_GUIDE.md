# Labelling guide — national random sample

For `national_random_sample.geojson` / `national_random_sample_labels.csv`
(600 cells, 250 m each, 75 per region, shuffled).

**Label in file order and stop whenever you need to.** The rows are shuffled
across all 8 regions on purpose, so any prefix of the file is still an unbiased,
geographically spread sample. Working region-by-region and stopping early is not.

---

## The four columns

Only `label` is consumed by the analysis today. The other three are cheap
insurance — fill them if they don't slow you down, skip them if they do.

### `label` — required

| value | meaning |
|---|---|
| `1` | a dumpsite is visible **anywhere** inside the 250 m cell |
| `0` | no dumpsite visible anywhere in the cell |
| *(blank)* + a note | you looked but genuinely cannot tell |
| *(blank)*, no note | not yet labelled |

A blank with a note is dropped from training automatically, which is the right
outcome — a coin-flip guess is worse than no label. But **use it sparingly**. If
you'd bet 70/30, commit and set `confidence` to `low` instead.

The cell is positive if *any part* of a dumpsite falls inside it, even if most
of the dump lies in the neighbouring cell. This matches how your earlier labels
were aggregated onto the 250 m grid.

### `confidence` — optional but useful

`high` / `medium` / `low`. Free text is fine, keep it to these three.

Worth filling because it lets us separate model error from label noise, and
down-weight shaky labels rather than discarding them.

### `notes` — only when something is odd

Leave blank for ordinary cells. Write something when:

- you couldn't tell (**required** here — it's what distinguishes "reviewed but
  unclear" from "not yet done")
- the imagery is cloudy, blurred, or looks older/newer than neighbours
- it's a confuser you had to think about: `quarry`, `mining spoil`,
  `construction`, `market`, `burn scar`, `sand stockpile`
- it's a large or notable site worth revisiting

### For every `1`, also record where the dump actually is

**This is the one addition worth the extra few seconds.** When you mark a cell
positive, put the dump's approximate centre in `notes` as `@lat,lon` — e.g.
`@9.5412,-13.6701`. In QGIS, hover the dump and read the coordinate off the
status bar, or drop a temporary point and copy it.

Why: a `1` on a 250 m cell says "a dump is somewhere in these 6.25 hectares".
If the tile size ever changes — and it might, see the grid-size discussion —
that label cannot be transferred, because nothing records *where* inside the
cell the dump was. A coordinate makes every positive reusable at any grid size,
forever. Negatives need no coordinate; the whole cell is the evidence.

There are few positives in a random sample, so this costs you very little in
total and protects the most expensive labels you will produce.

### `annotator` — fill once, copy down

Your name or initials, same string every row. Only matters if someone else
labels too, so we can measure agreement rather than assume it.

---

## What counts as a dumpsite

**Positive: an accumulation of solid waste on open ground.** Formal landfills,
informal/illegal dumps, and roadside or drainage-channel accumulations all count.
There is no minimum size — if you can see it is waste, mark `1`.

### What waste looks like from above

- **Heterogeneous, speckled texture.** Mixed colours at fine scale — plastic
  reads as bright, light-toned flecks against darker organic material. This is
  the single most reliable cue.
- **Irregular, amorphous outline.** Waste spreads; it does not have straight
  edges or right angles.
- **Vehicle tracks converging** on it, or a spur track off a road — waste has to
  be brought in.
- **Burn scars or ash patches** within or beside the waste body. Very common at
  informal sites.
- **Typical siting**: settlement edge, roadside, ravines, drainage channels,
  old borrow pits, riverbanks, low ground.

### Confusers — mark these `0`

These matter in Guinea specifically and are where most errors will come from.

| looks like | tell it apart by |
|---|---|
| **Bauxite mining spoil** (Boké) | geometric terraced benches, haul roads, uniform red/ochre, huge scale |
| **Gold mining** (Siguiri, Haute-Guinée) | pits, turbid ponds, spoil heaps in rows, river-adjacent |
| **Quarry / borrow pit** | straight cut faces, benches, a single access ramp |
| **Bare laterite / soil** | uniform reddish tone, *smooth* texture, no speckle |
| **Construction site** | regular outline, materials in neat stacks, new structures nearby |
| **Sand / gravel stockpile** | conical heaps, uniform colour, near a road or river |
| **Market** | clutter under regular rows of roofs/awnings, inside the built area |
| **Burnt or cleared field** | field geometry, straight boundaries, no speckled debris |
| **Cemetery** | regular rows, uniform spacing |

If it is uniform in colour and smooth in texture, it is almost certainly ground,
not waste. **Speckle and irregularity are the signal.**

---

## Workflow in QGIS

1. Add `national_random_sample.geojson` as a layer over your usual Google
   satellite basemap.
2. Style it with no fill and a bright outline so you can see through it, and
   switch on labels showing `tile_id`.
3. Open the attribute table, toggle editing, and **type straight into the
   `label` / `confidence` / `notes` / `annotator` columns**. The fields are
   already there.
4. Sort the table by nothing — leave it in file order and work top to bottom.
5. Save edits (Ctrl+S) regularly.

Editing the GeoJSON in place avoids matching rows between QGIS and a spreadsheet,
which is where labelling errors usually come from.

If you prefer a spreadsheet, use `national_random_sample_labels.csv` instead —
same `tile_id` values, same order. Use one or the other, not both.

**Send back whichever file you edited.** Don't merge it into
`manual_labels_template.csv` — this sample must stay identifiable as a random
sample, because that is what makes it usable for calibrating the operating
threshold. The hand-picked labels and the random labels do different jobs and
have to stay separable.

---

## What to expect

**Most cells will be negative, and many will be dull.** That is the point.

The existing 242 labels are 63% positive because they were collected by looking
for dumpsites. The model that comes out of that expects dumpsites to be common,
so it over-predicts. This sample is drawn at random from the ground the deployed
model will actually scan, where true sites are a fraction of a percent — so the
negatives here are doing the real work. They teach the model what ordinary
Guinean ground looks like and they are what makes a precision figure honest.

Do not skip boring cells, and do not go hunting for dumps outside the sampled
cells. A tempting-looking site just outside a cell boundary is a `0` for that
cell. Hunting reintroduces exactly the bias this sample exists to remove.

If you label 200 of these and find only 3 positives, that is a useful and
correct result, not a wasted session.
