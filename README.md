# IMD Radar Updates — automated nowcast map

Polls IMD's NIOT/Pallikaranai (X-band) and Karaikal (S-band) radar feeds
every 15 minutes, tracks storm cells across both, and publishes a fused
storm-movement forecast map (reflectivity raster + projected motion cones)
to `plots.chennairains.com`.

`nowcast_bot.py` is extracted from the interactive Colab notebook
(`NIOT_Radar_Nowcast_Bot.ipynb`) that this project was originally
developed in — that notebook stays the place to develop and test new
detection/calibration logic live. This script is only the part that needs
to run unattended on a schedule: one polling cycle per invocation, with
all tracking state read from and written back to the `state/` and
`archive/` folders (a fresh GitHub Actions runner has no memory between
runs, so nothing here relies on an in-process cache surviving between
cycles).

## How it runs

`.github/workflows/nowcast.yml` fires every 15 minutes (cron) or on demand
(the **Run workflow** button under the repo's Actions tab). Each run:

1. Installs Tesseract OCR + the Python dependencies
2. Runs `python nowcast_bot.py` — one polling cycle: fetch both radars,
   decode reflectivity, track storm cells, build `output/storm_forecast_map.html`,
   capture this cycle's radar-loop frame, and (once an hour) rebuild the
   3h/6h/12h radar-loop GIFs — see "Past 3h/6h/12h radar loop" below
3. Commits the updated `state/`, `archive/`, `mosaic_frames/` and
   `radar_loop/` folders back to this repo (so the next scheduled run
   picks up where this one left off)
4. Uploads everything in `output/` — the live map plus the radar-loop
   page and its GIFs — to your cPanel hosting over FTP

## One-time setup

Add these under **Settings → Secrets and variables → Actions** on this
repo (never commit them to a file):

| Secret | Value |
|---|---|
| `FTP_HOST` | Your cPanel FTP hostname (e.g. `ftp.chennairains.com`) |
| `FTP_USERNAME` | The FTP account username |
| `FTP_PASSWORD` | The FTP account password |
| `FTP_REMOTE_DIR` | The remote path to upload into — the document root for `plots.chennairains.com`, **with a trailing slash** (e.g. `/public_html/plots.chennairains.com/`) |

Find the exact remote path and FTP account details under cPanel →
**FTP Accounts** (or **File Manager**, right-click the `plots.chennairains.com`
folder → check its full path).

Once those four secrets exist, trigger a manual run from the **Actions**
tab (**Update nowcast radar map** → **Run workflow**) to confirm it works
end to end before waiting for the first scheduled tick.

## Why state is committed to the repo

There's no separate database here — `state/poll_state.json`,
`state/prev_cells.json`, and `state/prev_obs_time.json` hold everything
needed to compute storm velocity between cycles (which requires knowing
where each tracked cell was last time), and `archive/` keeps a short
rolling window of raw radar frames per product as a fallback for any
cycle where the live fetch fails. Committing them back each run is the
simplest way to give a stateless runner memory across runs, appropriate
for a project at this scale. A run's own commit does **not** re-trigger
the workflow (it only runs on `schedule`/`workflow_dispatch`, not `push`),
so there's no risk of a commit loop.

This also means GitHub won't auto-disable the schedule for repo
inactivity (it does that after 60 days with zero commits) — the workflow
keeps the repo active on its own.

## Past 3h/6h/12h radar loop

Alongside the live storm map, every run also builds
`output/radar_loop.html` — a static page embedding three looping GIFs
(past 3, 6 and 12 hours) of the fused NIOT+Karaikal+Kochi reflectivity
picture, on a fixed map extent sized to keep all three radars' range
circles in frame together (roughly Mangaluru/Shivamogga in the NW to
Kanyakumari/northern Sri Lanka in the south to Chennai/the Bay of Bengal
in the NE).

- A frame is captured every ordinary 15-minute cycle (reusing that
  cycle's already-fetched/decoded reflectivity — no extra network calls),
  archived into `mosaic_frames/`, kept for ~13 hours (12h window + an
  hour's slack) and committed back to the repo so there's always a full
  history to animate from.
- The GIFs themselves are only **rebuilt once an hour** (the first cycle
  of each hour) — see `nowcast_bot.py`'s "Past 3h/6h/12h radar loop"
  module docstring for the reasoning. The built files live in a
  committed `radar_loop/` folder and are re-copied into `output/` on
  *every* cycle regardless, so the page never flickers in and out of
  existence on the hours it isn't being regenerated (FTP-Deploy-Action
  deletes any remote file missing from a run's local `output/`).
- `mosaic_basemap.png` (the stitched CARTO Voyager tile background) is
  fetched once ever and committed, so this never hits the tile server
  again after the first run.

This is production-only — the 500km test repo doesn't build a radar loop.

## Running locally

```bash
pip install -r requirements.txt
# tesseract-ocr must also be installed as a system package
python nowcast_bot.py
```

Produces `output/storm_forecast_map.html` — open it directly in a browser
to check it before it ever reaches the live site.
