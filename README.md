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

## Minimizing server load from idle/inactive viewers

Three things keep a browser tab left open on this site from costing
server resources for no reason, on top of each other:

1. **The auto-refresh pauses when the tab isn't visible.**
   `storm_forecast_map.html`'s reload timer (`build_autorefresh_script()`
   in `nowcast_bot.py`) now uses the Page Visibility API: a backgrounded
   or minimized tab makes zero requests while hidden, and reloads once
   immediately on becoming visible again if it's gone stale, or arms a
   timer for whatever time is left otherwise. There's no persistent
   per-visitor connection to a static file host to "close" the way there
   would be for a chat or streaming server — not making the request in
   the first place while nobody's looking is the real equivalent of that
   for a page like this one.
2. **It also stops for a visible-but-abandoned tab.** If there's been no
   mouse/touch/keyboard/scroll/click activity for
   `AUTOREFRESH_IDLE_TIMEOUT_MINUTES` (60), the tab stops reloading even
   though it's technically still on-screen (e.g. a monitor left on) —
   and picks back up the moment there's any activity again.
3. **`.htaccess` sets `Cache-Control` on every published file**, matched
   to how often it actually changes (5 min for the live map, 15 min for
   the radar loop) — see that file's own comments. This both lets
   browsers skip a request entirely when their own cached copy is still
   fresh, and (see below) is what lets Cloudflare cache the GIFs at the
   edge.

## Cloudflare caching

`plots.chennairains.com` is proxied through Cloudflare, which is a much
bigger lever on origin load than anything client-side above: if 50
people load the live map in the same few minutes, a properly cached edge
can serve 49 of those without this cPanel server seeing a request at
all. Two things make that work:

**Images are already covered.** Cloudflare caches common static file
types (including `.gif`) at the edge by default, honoring the origin's
`Cache-Control` header for how long — which `.htaccess` now sets. No
dashboard change needed for the radar-loop GIFs.

**HTML needs an explicit Cache Rule.** Cloudflare does NOT cache `.html`
by default, regardless of `Cache-Control` — so `storm_forecast_map.html`
and `radar_loop.html` currently bypass Cloudflare's cache entirely on
every single visit, `Cache-Control` header or not. One-time setup, in
the Cloudflare dashboard for this zone:

1. **Caching → Cache Rules** (or **Rules → Page Rules** on an older
   dashboard) → create a rule.
2. Match: `hostname equals plots.chennairains.com` AND
   `URI Path equals /storm_forecast_map.html` — add a second rule the
   same way for `/radar_loop.html` (Cache Rules match one condition set
   per rule; two simple rules are easier to reason about than one
   combined one here).
3. Cache eligibility: **Eligible for cache** (this is the setting that
   overrides HTML's default bypass).
4. Edge TTL: **Respect origin headers** — since `.htaccess` now sends a
   real `Cache-Control`, Cloudflare will use exactly that (5 min / 15 min)
   rather than needing a separately-maintained TTL in two places.

**Keeping it fresh after every deploy.** Once that Cache Rule exists, a
viewer could be served an edge-cached copy for up to its TTL after a
fresh upload. `nowcast.yml`'s "Purge Cloudflare cache" step (right after
the FTP upload) purges exactly those 5 URLs on every run, so Cloudflare
still serves the newest build within seconds, not minutes — it only
needs two more repo secrets to activate (it's a safe no-op without
them):

| Secret | Where to get it |
|---|---|
| `CLOUDFLARE_API_TOKEN` | Cloudflare dashboard → **My Profile → API Tokens → Create Token** → use the **Edit zone DNS** template as a base but you only need the **Zone → Cache Purge → Purge** permission, scoped to this one zone (`chennairains.com`) |
| `CLOUDFLARE_ZONE_ID` | Cloudflare dashboard → select the `chennairains.com` zone → right sidebar, **Zone ID** (just a value to copy, not a secret you create) |

Add both under this repo's **Settings → Secrets and variables → Actions**,
same as the FTP secrets above — the purge step picks them up on the very
next run with no other change needed.

## Running locally

```bash
pip install -r requirements.txt
# tesseract-ocr must also be installed as a system package
python nowcast_bot.py
```

Produces `output/storm_forecast_map.html` — open it directly in a browser
to check it before it ever reaches the live site.
