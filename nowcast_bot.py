# === CELL 2 (imports) ===
import base64
import hashlib
import io
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests
from PIL import Image, UnidentifiedImageError
from scipy import ndimage
import cv2

try:
    import pytesseract
except ImportError:
    pytesseract = None  # OCR is optional; falls back to poll-receipt time


# === CELL 4 (config) ===
RADAR_META = {
    "site_lat": 12.9451,
    "site_lon": 80.2115,
    "site_elev_m": 24.0,
    "band": "X",
}

# --- Second radar: IMD's Karaikal DWR, added to extend coverage down the
# TN coast (Nagapattinam/Thanjavur/Puducherry belt) that NIOT's 85-150km
# radius doesn't reach. Two independent physical radars now feed this bot;
# RADAR_SITES holds each one's real-world anchor, PRODUCT_RADAR says which
# site a given product key belongs to, and pixel_to_latlon/latlon_to_pixel/
# range_from_site_km below resolve the right site through that lookup
# instead of assuming the single global RADAR_META.
#
# CALIBRATION STATUS (Karaikal) — unlike NIOT's constants below (measured
# off real full-resolution frames with an automated crosshair/ring-label
# detector), these were derived from three ~880x720 sample frames pasted
# into chat, using the same technique at lower resolution:
#   - site_px / km_per_px: solid. Fit from the "XX km" range-ring labels
#     (multiple rings per product, cross-validated against each other —
#     e.g. PPI's 100km and 150km rings independently agree on the same
#     px-per-km to within 0.1%).
#   - site_lat / site_lon / site_elev_m / band: NOT independently verified
#     — no public source gave Karaikal's exact site coordinates, so
#     site_lat/site_lon are estimated as Karaikal town's coordinates
#     (the radar is presumably sited at/near the town it's named for, but
#     "presumably" is doing real work in that sentence). This is exactly
#     the kind of thing that caused NIOT's original SE-offset bug — check
#     the merged map against known towns (Karaikal, Nagapattinam,
#     Puducherry) before trusting absolute storm positions from this radar.

# --- Third radar: IMD's Kochi DWR, added to extend coverage up the Kerala
# coast (west coast, unlike NIOT/Karaikal's east coast). Unlike Karaikal,
# Kochi's own frame prints an explicit lat/lon axis (tick labels "74"-"78"
# along the bottom, "8.0"-"12.0" along the left) directly on the plot, so
# site_lat/site_lon/km_per_px here were derived from THAT axis rather than
# guessed from the town's public coordinates:
#   1. Degree-gridline pixel positions were measured two independent ways —
#      OCR'd tick label bounding boxes, and a plain brightness-peak scan
#      for the actual gridlines over a pure-ocean strip (no text/coastline
#      to confuse it) — and agreed to within ~1px on both axes.
#   2. That gives an affine pixel<->lat/lon mapping directly from the
#      image itself (pixel_per_deg_lat=134, pixel_per_deg_lon=131, whose
#      ratio matches cos(10.43 deg)=0.983 almost exactly -- confirms this
#      is a properly isotropic-km projection, not just a plain square
#      degree grid, so the existing single-km_per_px pixel_to_latlon
#      formula applies here too).
#   3. The radar's own site pixel was found from the concentric dashed
#      range-ring geometry itself (RANSAC circle fit over the dashed-only
#      mask, after stripping the solid gridlines): two independent rings
#      agreed on the same center to within 0.3px (and were in an exact
#      2:1 radius ratio, a strong internal-consistency check). That pixel,
#      run back through the same axis mapping, gives site_lat/site_lon —
#      so unlike Karaikal, these are MEASURED from the image's own
#      geometry, not a guessed town center. Still worth a sanity check
#      against the merged map once real frames start flowing (the fitted
#      site sits just offshore right next to the "KOC" station-code label,
#      which matches expectations for a coastal DWR).
RADAR_SITES = {
    "niot": RADAR_META,
    "karaikal": {
        "site_lat": 10.9254,      # UNVERIFIED — Karaikal town center, not a
        "site_lon": 79.8380,      # confirmed radar-tower coordinate
        "site_elev_m": 5.0,       # UNVERIFIED — coastal-town guess
        "band": "S",              # UNVERIFIED — IMD's ~2015-era coastal
                                   # DWRs are typically S-band, not confirmed
    },
    "kochi": {
        # CORRECTED — the image-based fit below (site_px + the axis
        # mapping) originally landed on 10.43N, off from the true site by
        # almost exactly one 0.5-degree gridline (~55km) north. Root
        # cause: the gridline-peak scan used to build the lat axis mapping
        # only covered y=100:900 of the frame and its first detected peak
        # (y=455) was mislabeled as the 11.5N line -- it's actually the
        # 11.0N line, since 12.0N (y=322) and 11.5N (y=389) both sit in
        # the cross-section-strip region above the plan-view panel
        # (y<300) and were never in that peak list to begin with. Fixed
        # values below are the site's real public coordinates (Palluruthy,
        # West Kochi) -- re-deriving them from the corrected axis mapping
        # against the ORIGINAL (unchanged) site_px lands within ~0.3km of
        # these, confirming site_px itself was fine all along and only
        # this lat conversion was off.
        "site_lat": 9.9264,
        "site_lon": 76.2621,
        "site_elev_m": 5.0,       # UNVERIFIED — coastal-site guess, same
                                   # basis as Karaikal's
        "band": "S",               # UNVERIFIED — same basis as Karaikal's
    },
}

PRODUCT_RADAR = {
    "ppi": "niot", "maxz": "niot", "ppz": "niot",
    "kkl_ppi": "karaikal", "kkl_ppz": "karaikal", "kkl_maxz": "karaikal",
    "koc_maxz": "kochi",
}

EFFECTIVE_EARTH_RADIUS_KM = 8494.0  # standard 4/3-Earth-radius model


def compute_beam_height_km(range_km: float, elevation_deg: float,
                            site_elev_m: float = 0.0) -> float:
    """Height of the beam center above ground at a given range, using the
    standard 4/3-earth-radius approximation. This is the number that makes
    "not sure about accuracy at longer distances" concrete rather than a
    vague worry."""
    elev_rad = np.radians(elevation_deg)
    h_km = (range_km * np.sin(elev_rad)
            + (range_km ** 2) / (2 * EFFECTIVE_EARTH_RADIUS_KM)
            + site_elev_m / 1000.0)
    return h_km


# Three products, same physical radar, each a genuinely different scan.
#
# site_px / km_per_px replace the old lat/lon-gridline guesswork. These were
# measured directly from real full-resolution sample frames (not compressed
# chat screenshots) using:
#   1. cv2.HoughLinesP to find the long dashed crosshair through the radar
#      site -> gives the exact site pixel (px, py).
#   2. cv2.connectedComponentsWithStats to find the white "XX km" range-ring
#      label boxes along the vertical strip through the site -> gives
#      pixel-distance-from-site for each labeled ring.
#   3. numpy.polyfit of (known ring km) vs (measured pixel distance) -> an
#      isotropic px-per-km scale (sub-pixel residuals in every case).
# Do not assume these transfer between products even where panels look
# visually similar — PPI/PPZ/MAXZ each have a different pixel scale because
# of differing range and (for MAXZ) the extra height-ladder inset panel.
PRODUCTS = {
    "ppi": {
        "url": "https://mausam.imd.gov.in/Radar/ppi_plk.gif",
        "role": "surface_confirmation",
        "range_km": 85.0,
        "elevation_deg": 0.2,
        "image_size": (3114, 2490),        # (width, height), measured
        "site_px": (1230.0, 1233.0),       # measured via crosshair detection
        "km_per_px": 14.107,               # measured via ring-label fit
        "colorbar_bbox": (2671, 646, 2728, 2432),
        "value_at_top": 73.0,
        "value_at_bottom": 0.0,
        "timestamp_bbox": (2440, 390, 3114, 436),
        "elevation_bbox": (2440, 215, 3114, 250),
        "plot_bbox": (29, 37, 2430, 2435),
        "cell_dbz_threshold": 30,
        "label_exclude_boxes": [
            (1129, 30, 1205, 108), (1187, 40, 1269, 82), (1187, 392, 1269, 436),
            (1187, 675, 1269, 717), (1187, 957, 1269, 999), (1187, 1230, 1273, 1301),
        ],
    },
    "maxz": {
        "url": "https://mausam.imd.gov.in/Radar/caz_plk.gif",
        "role": "aloft_early_warning",
        "range_km": 85.0,
        "elevation_deg": None,   # column-max across 0-18km, not a single tilt
        "image_size": (3045, 2490),
        "site_px": (928.0, 1528.0),
        "km_per_px": 10.557,
        "colorbar_bbox": (2641, 641, 2698, 2427),
        "value_at_top": 60.7,
        "value_at_bottom": 20.0,
        "timestamp_bbox": (2440, 445, 3045, 490),
        "elevation_bbox": None,
        # MAXZ has an extra full-width "height ladder" inset panel (1-17km
        # labels) running across the top of the frame; the geographic panel
        # starts below it, at y=629, not y=0.
        "plot_bbox": (30, 629, 1830, 2429),
        "cell_dbz_threshold": 35,
        "label_exclude_boxes": [
            (895, 624, 976, 692), (895, 888, 978, 934), (895, 1103, 977, 1145),
            (895, 1314, 977, 1357), (895, 1524, 965, 1592),
        ],
    },
    "ppz": {
        "url": "https://mausam.imd.gov.in/Radar/ppz_plk.gif",
        "role": "regional_early_warning",
        "range_km": 150.0,
        "elevation_deg": 1.1,
        "image_size": (3114, 2485),
        "site_px": (1230.0, 1228.0),
        "km_per_px": 7.9865,
        "colorbar_bbox": (2671, 641, 2728, 2427),
        "value_at_top": 61.7,
        "value_at_bottom": 20.0,
        "timestamp_bbox": (2440, 385, 3114, 432),
        "elevation_bbox": (2440, 330, 3114, 365),
        "plot_bbox": (30, 30, 2430, 2430),
        "cell_dbz_threshold": 30,
        "label_exclude_boxes": [
            (1200, 25, 1295, 65),   # 150 km label — sits right at the top
                                    # border, easy to miss with a generic
                                    # detector; added after spotting a
                                    # residual false cell here
            (1222, 260, 1294, 305), (1207, 419, 1294, 467), (1201, 735, 1282, 783),
            (1201, 900, 1282, 942), (1201, 1060, 1282, 1102), (1201, 1219, 1275, 1278),
        ],
    },

    # --- Karaikal DWR (see calibration-status comment above RADAR_SITES) ---
    "kkl_ppi": {
        "url": "https://mausam.imd.gov.in/Radar/ppi_kkl.gif",
        "role": "surface_confirmation",
        "range_km": 150.0,
        "elevation_deg": 0.5,
        "image_size": (880, 720),
        "site_px": (359.4, 359.3),   # fit from 100km + 150km range-ring labels
        "km_per_px": 2.398,
        "colorbar_bbox": (780, 375, 800, 615),
        # PPI's dBZ scale is NOT evenly spaced per band (measured tick
        # values: 72,66,60,55,53,50,44,39,37,34,28,23,21,18,12,7,2 — pixel
        # spacing between ticks IS uniform at 15px, only the dBZ width of
        # each band varies), so it needs the piecewise breakpoint list
        # below rather than a single value_at_top/value_at_bottom pair.
        "colorbar_breakpoints": [72, 66, 60, 55, 53, 50, 44, 39, 37, 34,
                                  28, 23, 21, 18, 12, 7, 2],
        "timestamp_bbox": (715, 218, 880, 262),
        "elevation_bbox": (715, 188, 880, 206),
        "plot_bbox": (0, 0, 720, 720),
        "cell_dbz_threshold": 30,
        "label_exclude_boxes": [
            (167, 40, 193, 57), (527, 40, 553, 57), (347, 112, 373, 129),
            (139, 231, 165, 248), (555, 231, 581, 248), (139, 471, 165, 488),
            (555, 471, 581, 488), (347, 591, 373, 608), (167, 663, 193, 680),
            (527, 663, 553, 680),
        ],
    },
    "kkl_ppz": {
        "url": "https://mausam.imd.gov.in/Radar/ppz_kkl.gif",
        "role": "regional_early_warning",
        "range_km": 500.0,
        "elevation_deg": 0.2,
        "image_size": (880, 720),
        "site_px": (359.5, 359.5),   # fit from 200/300/400/500km range-ring labels
        "km_per_px": 0.7174,
        "colorbar_bbox": (780, 375, 800, 615),
        # PPZ's scale IS evenly spaced (~2.7 dBZ/band) — plain top/bottom
        # values are enough here, same as every NIOT product.
        "value_at_top": 60.0,
        "value_at_bottom": 20.0,
        "timestamp_bbox": (715, 218, 880, 262),
        "elevation_bbox": (715, 188, 880, 206),
        "plot_bbox": (0, 0, 720, 720),
        "cell_dbz_threshold": 30,
        "label_exclude_boxes": [
            (168, 40, 193, 57), (527, 40, 553, 57), (347, 64, 373, 81),
            (239, 164, 265, 181), (455, 164, 481, 181), (98, 208, 124, 225),
            (347, 208, 373, 225), (596, 208, 622, 225), (222, 280, 248, 297),
            (472, 280, 498, 297), (563, 351, 589, 368), (131, 352, 157, 369),
            (222, 423, 248, 440), (472, 423, 498, 440), (98, 495, 124, 512),
            (347, 495, 373, 512), (596, 495, 622, 512), (239, 539, 265, 556),
            (455, 539, 481, 556), (347, 639, 373, 656), (169, 663, 193, 680),
            (527, 663, 552, 680),
        ],
    },
    "kkl_maxz": {
        "url": "https://mausam.imd.gov.in/Radar/caz_kkl.gif",
        "role": "aloft_early_warning",
        "range_km": 250.0,
        "elevation_deg": None,   # column-max across 0-15km, not a single tilt
        "image_size": (880, 720),
        # This product's frame isn't one plan-view panel like PPI/PPZ — it's
        # "Max with panels": a vertical N-S cross-section strip across the
        # top and per-azimuth cross-section strips down the right side,
        # with the actual plan-view map confined to the bottom-left
        # quadrant. plot_bbox below crops to JUST that quadrant; site_px
        # is measured within it (fit from six "200 km" range-ring labels,
        # all agreeing on the same radius to within 0.3px).
        "site_px": (259.5, 459.5),
        "km_per_px": 1.0411,
        "colorbar_bbox": (780, 375, 800, 615),
        "value_at_top": 60.0,    # same uniform ~2.7dB/band scale as kkl_ppz
        "value_at_bottom": 20.0,
        "timestamp_bbox": (715, 218, 880, 262),
        "elevation_bbox": None,  # no elevation line — column-max product
        "plot_bbox": (0, 198, 519, 720),
        "cell_dbz_threshold": 35,   # matches NIOT MAXZ's higher threshold
        "label_exclude_boxes": [
            (247, 243, 273, 260), (67, 347, 93, 364), (427, 347, 453, 364),
            (67, 556, 93, 573), (427, 556, 453, 573), (247, 660, 273, 677),
        ],
    },

    # --- Kochi DWR (see calibration-status comment above RADAR_SITES) ---
    "koc_maxz": {
        "url": "https://mausam.imd.gov.in/Radar/caz_koc.gif",
        "role": "aloft_early_warning",
        "range_km": 250.0,
        "elevation_deg": None,   # column-max, not a single tilt -- same as
                                  # every other MAXZ product here
        "image_size": (1000, 1000),
        # Same "Max with panels" layout as kkl_maxz (cross-section strips
        # across the top and down the right, plan-view confined to the
        # bottom-left), but at yet another resolution/crop -- measured
        # directly from a real caz_koc.gif frame (see calibration-status
        # comment above RADAR_SITES for how site_px/km_per_px were derived
        # from this radar's own printed lat/lon axis + range-ring geometry,
        # rather than guessed).
        "site_px": (397.5, 598.5),
        "km_per_px": 1.2072,
        "colorbar_bbox": (921, 301, 940, 900),
        "value_at_top": 60.0,     # same uniform scale as kkl_maxz/kkl_ppz
        "value_at_bottom": 20.0,
        # "Date:DD/MM/YYYY" on one line, "Time:HH:MM:SS UTC" on the next --
        # a third timestamp layout, see extract_observation_time's new
        # "Date:"/"Time:" branch. Plain OCR reads this cleanly (a normal
        # digital font, not Karaikal's bold small font that needed template
        # matching), so no special per-glyph path is needed here.
        "timestamp_bbox": (700, 155, 915, 215),
        "elevation_bbox": None,  # no elevation line — column-max product
        # Plan-view quadrant only, excluding the top cross-section strip,
        # the right per-azimuth strip, and the colorbar -- bounds measured
        # from the frame's own panel border pixels.
        "plot_bbox": (100, 300, 700, 900),
        "cell_dbz_threshold": 35,   # matches every other MAXZ product here
        # This layout has no printed "XX km" range-ring label boxes to
        # exclude the way NIOT/Karaikal need -- but it draws full-panel
        # solid white lat/lon GRIDLINES instead, and this product's LUT
        # happens to have a pale yellow-white band (~35-42 dBZ) that a
        # near-white gridline pixel lands within DIST_THRESHOLD of. Live
        # frames confirmed this: over half of one cycle's detected
        # "cells" had their centroid sitting exactly on a gridline
        # row/column, all in that same pale dBZ band -- a real storm
        # doesn't line up with the degree grid. These boxes mask an ~11px
        # strip along every gridline (measured from the frame's own
        # gridline pixel positions, same source as site_px/km_per_px
        # above, then padded a few px past the exact line to also catch
        # the anti-aliased halo of near-white pixels immediately next to
        # it -- a first pass using a tighter ~5px strip still let a
        # handful of these halo pixels through on real frames) so those
        # pixels can't register as fake weak echo. Real echo is a filled
        # area, not a thin line, so this costs at most a sliver of a
        # genuine storm that happens to sit right under a gridline -- not
        # the storm itself.
        "label_exclude_boxes": [
            # lat gridlines (12.0N down to 8.0N, every 0.5 deg) -- full
            # plot width, y from the frame's own gridline pixel rows
            (100, 317, 700, 328), (100, 384, 700, 395), (100, 450, 700, 461),
            (100, 517, 700, 528), (100, 584, 700, 595), (100, 651, 700, 662),
            (100, 718, 700, 729), (100, 785, 700, 796), (100, 852, 700, 863),
            # lon gridlines (75E-78E, every 1 deg; 74E coincides with the
            # crop's own left border and needs no separate box) -- full
            # plot height
            (226, 300, 237, 900), (358, 300, 369, 900),
            (489, 300, 500, 900), (621, 300, 632, 900),
        ],
        # Beyond the gridlines, this layout draws a shaded terrain/coastline
        # basemap UNDER the plan-view panel (unlike NIOT/Karaikal's plain
        # background) — and a handful of that basemap's flat fill colors
        # (land shading, coastline outline, city-name text) turn out to
        # EXACTLY equal colorbar swatch colors (distance 0 in the KD-tree
        # match, same as genuine echo -- no DIST_THRESHOLD tweak can tell
        # these apart from real echo by colour alone). Confirmed by
        # intersecting the raw threshold mask across 6 independent archived
        # frames spanning ~2 hours: ~1650 of ~1850 matched pixels per frame
        # (about 90%) sat at the literal same pixel in every single frame,
        # traced to only 10 distinct RGB values, all clearly land/text
        # colors (dark near-black text, tan/brown terrain shades) rather
        # than the blue/green tones real echo renders in. A real storm
        # would not occupy the exact same pixels, unchanging, for 2 hours
        # straight while everything else on the map moves. Excluding by
        # exact colour would also suppress genuine echo anywhere else in
        # the frame that happens to render at one of these same dBZ levels
        # (38-42, right at this product's low end) -- excluding by fixed
        # PIXEL POSITION instead (this mask) only ever blocks these known
        # static basemap pixels, and a genuine storm forming anywhere else,
        # even at the same intensity, is untouched. This mask only covers
        # what these 6 sample frames happened to show as static; if a
        # future frame reveals another persistently-colliding basemap
        # pixel, extend masks/koc_maxz_static_exclude.png the same way
        # (regenerate the intersection over a fresh batch of frames).
        "static_exclude_mask": "masks/koc_maxz_static_exclude.png",
    },
}

POLL_SECONDS = 120
STALE_AFTER_SECONDS = 40 * 60
ARCHIVE_DIR = Path("./niot_frames")
STATE_FILE = Path("./niot_bot_state.json")

# Pixel-count minimums don't transfer across radars with very different
# resolutions (NIOT's ~10-14 px/km vs Karaikal's ~0.7-2.4 px/km — the same
# 15-pixel floor would be an absurdly tiny real-world area for NIOT and an
# absurdly large one for Karaikal PPZ). MIN_CELL_AREA_KM2 is the real
# constant now; the per-product pixel floor is derived from it in
# extract_cells() using that product's own km_per_px, calibrated so NIOT
# MAXZ keeps behaving exactly as before (15px at its 10.557 km_per_px).
MIN_CELL_AREA_KM2 = 15 / (10.557 ** 2)

# The area-based floor above degenerates to well under 1 pixel at coarse
# resolutions (Karaikal's km_per_px=1.0411 -> ~0.15px, Kochi's 1.2072 ->
# ~0.2px), meaning literally any single stray pixel — a colour-LUT
# near-match against anti-aliasing, a text edge, a coastline, or basemap
# noise — trivially "passes" the area filter with room to spare. A real
# echo, even a small/weak one, is a filled patch several pixels wide, not
# an isolated speck. Measured on real archived Kochi frames: of ~1250 raw
# connected components per frame surviving the area floor alone, the
# pixel-count histogram was overwhelmingly 1-4px, tapering off sharply,
# with essentially nothing resembling a genuine storm's footprint above
# ~15px in any of 6 consecutive frames spanning ~2 hours with no reported
# severe weather. A floor of 5px cuts that noise by roughly 80% (down to
# the 40s per frame) while barely touching Karaikal, which already runs
# 1-6 raw components per frame at this same near-zero area floor (i.e. it
# wasn't relying on tiny blobs to detect real storms either). This is a
# hard floor on raw pixel count, applied on TOP of (never instead of) the
# area-based floor, so it only ever discards components the area-based
# floor would have let through at coarse resolutions — it changes nothing
# for NIOT MAXZ, whose area floor (15px) already exceeds this.
MIN_CELL_ABSOLUTE_PIXELS = 5
MATCH_ACROSS_PRODUCTS_KM = 8.0

# Nearby individual blobs within this radius of each other get merged
# into one storm-complex cluster before tracking/forecasting — see
# cluster_cells() below for why.
CLUSTER_RADIUS_KM = 15.0

# Beam height above which a detection is flagged "elevated" — i.e. likely
# not representative of near-surface rain, only of a tall/mature system.
ELEVATED_BEAM_THRESHOLD_KM = 1.5

# How often an already-open browser tab reloads itself to pick up a newer
# map -- see build_autorefresh_script(). Was 20; dropped to 10 because the
# 20-minute refresh compounded with IMD's own radar-data lag (frames can
# already be ~10 min stale by the time they're polled) to sometimes show a
# map ~30 min out of date by the time a viewer's tab reloaded. 10 min means
# an open tab will occasionally reload onto a cycle that hasn't produced a
# newer frame yet (re-showing the same map), which is the accepted
# trade-off -- a same-frame reload is a much smaller downside than sitting
# on a stale one for up to 30 minutes. Deliberately not exactly 15 (the
# upload cadence, see nowcast.yml/cron-job.org) so a reload rarely lands
# on exactly the same moment as an in-progress upload every single cycle.
AUTOREFRESH_MINUTES = 10

# How old a product's displayed observation time has to be before
# build_info_banner_html() flags it as stale instead of showing it with
# plain styling -- see that function for why this matters (a reader can't
# otherwise tell "this is live" from "this radar's feed has been stuck for
# hours" just by glancing at the banner). Comfortably above the ~15 min
# poll cadence to allow for IMD's own ordinary publishing lag without
# false-flagging a normal reading.
STALE_OBS_MINUTES = 60

# === CELL 6 (georeferencing) ===
def site_for(product: str) -> dict:
    """Which physical radar a product belongs to (see RADAR_SITES /
    PRODUCT_RADAR above) — every georeferencing function below goes
    through this instead of assuming the single NIOT RADAR_META, which is
    what makes this notebook multi-radar-aware without every function
    signature needing its own new `radar` parameter."""
    return RADAR_SITES[PRODUCT_RADAR[product]]


def pixel_to_latlon(px: float, py: float, product: str) -> tuple[float, float]:
    """Convert an image pixel to (lat, lon) using the measured site pixel
    and isotropic km-per-pixel scale for this product (see PRODUCTS above).
    This replaces the old lat/lon-gridline linear fit, which was only as
    good as the eyeballed gridline pixel positions — the site+scale method
    is anchored to the radar's own crosshair and range rings, which are
    exact by construction in every frame."""
    site = site_for(product)
    site_x, site_y = PRODUCTS[product]["site_px"]
    km_per_px = PRODUCTS[product]["km_per_px"]
    km_north = (site_y - py) / km_per_px
    km_east = (px - site_x) / km_per_px
    lat = site["site_lat"] + km_north / 111.0
    lon = site["site_lon"] + km_east / (111.0 * np.cos(np.radians(site["site_lat"])))
    return lat, lon


def latlon_to_pixel(lat: float, lon: float, product: str) -> tuple[float, float]:
    """Inverse of pixel_to_latlon — used for drawing the OSM overlay bounds."""
    site = site_for(product)
    site_x, site_y = PRODUCTS[product]["site_px"]
    km_per_px = PRODUCTS[product]["km_per_px"]
    km_north = (lat - site["site_lat"]) * 111.0
    km_east = (lon - site["site_lon"]) * 111.0 * np.cos(np.radians(site["site_lat"]))
    px = site_x + km_east * km_per_px
    py = site_y - km_north * km_per_px
    return px, py


def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, [*a, *b])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(h))


def range_from_site_km(latlon: tuple[float, float], product: str = "maxz") -> float:
    """Range from the radar site. `product` picks WHICH radar via
    site_for() — defaults to "maxz" (NIOT) only for backward compatibility
    with existing call sites that don't pass a product; anything computing
    a Karaikal cell's range must pass its actual product key."""
    site = site_for(product)
    return _haversine_km((site["site_lat"], site["site_lon"]), latlon)

# === CELL 8a (Karaikal bold-timestamp template matcher) ===
# Tesseract's LSTM engine (the only one available in this sandbox/CI image)
# was confirmed, via direct testing against real archived Karaikal frames,
# to consistently misread this specific bold "digital display" font
# regardless of upscale/threshold/psm/oem tuning (e.g. "28" -> "o8", "2026"
# -> "2006") -- not a preprocessing problem, the model just doesn't know
# this font. Karaikal's time line ("HH:MM:SSZ") is otherwise very
# favourable for a much simpler, much more reliable technique: it's a
# fixed-position, monospaced, high-contrast glyph run, so connected-
# component segmentation + nearest-template matching against a small
# labelled glyph library (built once from known-good archived frames,
# shipped as assets/kkl_timestamp_templates.npz so it survives archive/
# pruning) reads it correctly. Falls back to per-character Tesseract (more
# reliable than whole-string OCR since there's no multi-char context to
# confuse it) for any glyph the template library hasn't seen (notably: the
# library was built from a handful of real frames and happens to have no
# examples of digits 7/8/9 yet), and if that also can't confidently name a
# glyph, extraction is abandoned for that frame -- same safe fall-back to
# poll-receipt time as every other failure mode here.
_KKL_TIMESTAMP_TEMPLATES: dict[str, list[np.ndarray]] | None = None
_KKL_TEMPLATE_SIZE = (40, 60)  # (w, h), matches assets/kkl_timestamp_templates.npz


def _load_kkl_timestamp_templates() -> dict[str, list[np.ndarray]]:
    global _KKL_TIMESTAMP_TEMPLATES
    if _KKL_TIMESTAMP_TEMPLATES is None:
        templates: dict[str, list[np.ndarray]] = {}
        path = Path(__file__).parent / "assets" / "kkl_timestamp_templates.npz"
        data = np.load(path)
        for img, ch in zip(data["images"], data["chars"]):
            templates.setdefault(str(ch), []).append(img)
        _KKL_TIMESTAMP_TEMPLATES = templates
    return _KKL_TIMESTAMP_TEMPLATES


def _kkl_char_groups(bw: np.ndarray, min_area: int = 15,
                      left_margin: int = 40, right_margin: int = 780,
                      merge_gap: int = 6) -> list[list[int]]:
    """Connected-component segmentation for one line of Karaikal's bold
    timestamp text. left_margin/right_margin drop a static artifact (a
    border/edge sliver that shows up at the same x-position in every frame
    regardless of what digits are printed -- confirmed by comparing frames
    with different leading digits). Adjacent components are merged when
    the gap between them is small (merge_gap): several digits in this font
    render as 2-3 disconnected strokes (e.g. "5", and "0"'s hollow centre
    can fully separate into two ink blobs at this threshold/resolution),
    while genuine gaps between different characters are consistently
    wider -- validated against every archived calibration frame."""
    inv = (bw < 128).astype(np.uint8)
    labeled, n = ndimage.label(inv, structure=np.ones((3, 3)))
    boxes = []
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if len(xs) < min_area:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        if x1 < left_margin or x0 > right_margin:
            continue
        boxes.append([x0, x1, y0, y1])
    boxes.sort(key=lambda b: b[0])
    merged: list[list[int]] = []
    for b in boxes:
        if merged and b[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], b[1])
            merged[-1][2] = min(merged[-1][2], b[2])
            merged[-1][3] = max(merged[-1][3], b[3])
        else:
            merged.append(list(b))
    return merged


def _kkl_crop_glyph(bw: np.ndarray, box: list[int]) -> np.ndarray:
    x0, x1, y0, y1 = box
    sub = bw[y0:y1 + 1, x0:x1 + 1]
    return cv2.resize(sub, _KKL_TEMPLATE_SIZE, interpolation=cv2.INTER_NEAREST)


def _kkl_classify_glyph(glyph: np.ndarray, templates: dict[str, list[np.ndarray]],
                         threshold: float = 0.15) -> str | None:
    best_ch, best_score = None, 1e9
    for ch, temps in templates.items():
        for t in temps:
            diff = np.abs(glyph.astype(int) - t.astype(int)).mean() / 255.0
            if diff < best_score:
                best_score, best_ch = diff, ch
    return best_ch if best_score <= threshold else None


def _kkl_ocr_single_glyph(bw: np.ndarray, box: list[int]) -> str | None:
    """Fallback for a glyph the template library doesn't recognise --
    isolating a single character and constraining Tesseract to a tight
    whitelist is far more reliable than whole-string OCR, which is what
    the primary pipeline (extract_observation_time) already does and
    already struggles with on this font."""
    if pytesseract is None:
        return None
    x0, x1, y0, y1 = box
    crop = bw[y0:y1 + 1, x0:x1 + 1]
    padded = cv2.copyMakeBorder(crop, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    text = pytesseract.image_to_string(
        padded, config="--psm 10 -c tessedit_char_whitelist=0123456789"
    ).strip()
    return text if len(text) == 1 and text.isdigit() else None


def extract_kkl_time_via_templates(img: Image.Image) -> tuple[int, int, int] | None:
    """Reads just the (H, M, S) from Karaikal's bold 'HH:MM:SSZ' line via
    template matching (see the block comment above). Deliberately doesn't
    attempt the date line too -- poll_and_decode already has a perfectly
    good, much simpler source of the date (today's UTC date, corrected for
    the rare case where local midnight fell between the frame being
    printed and this poll picking it up), and decoding this font's date
    line reliably would need letter templates for the month abbreviation,
    which is a lot more segmentation work for something we don't need.
    Returns None (safe fall-back to poll-receipt time) on any ambiguity.

    Called for every Karaikal product (see extract_observation_time), not
    just kkl_maxz -- hardcoding the crop to kkl_maxz's own timestamp_bbox
    here is safe because all Karaikal products render this same bold font
    at this same pixel position (verified identical across kkl_maxz/
    kkl_ppz/kkl_ppi); there's no cfg[product] equivalent to fall back to
    if that ever stops being true for some future Karaikal product."""
    crop = img.convert("RGB").crop(PRODUCTS["kkl_maxz"]["timestamp_bbox"])
    arr = np.array(crop)
    big = cv2.resize(arr, (arr.shape[1] * 5, arr.shape[0] * 5), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_RGB2GRAY)
    _, bw_full = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
    bw = bw_full[: int(bw_full.shape[0] * 0.5), :]  # top line only ("HH:MM:SSZ")

    groups = _kkl_char_groups(bw)
    if len(groups) != 9:  # "HH:MM:SSZ" is always exactly 9 glyphs
        return None

    templates = _load_kkl_timestamp_templates()
    chars = []
    for box in groups:
        glyph = _kkl_crop_glyph(bw, box)
        ch = _kkl_classify_glyph(glyph, templates)
        if ch is None:
            ch = _kkl_ocr_single_glyph(bw, box)
        if ch is None:
            return None
        chars.append(ch)

    expected_shape = "dd:dd:ddZ"
    text = "".join(chars)
    for c, kind in zip(text, expected_shape):
        if kind == "d" and not c.isdigit():
            return None
        if kind == "Z" and c != "Z":
            return None
        if kind == ":" and c != ":":
            return None
    try:
        h, m, s = int(text[0:2]), int(text[3:5]), int(text[6:8])
        datetime(2000, 1, 1, h, m, s)  # validates ranges, raises ValueError if bogus
    except ValueError:
        return None
    return h, m, s


# === CELL 8 (OCR) ===
def extract_observation_time(img: Image.Image, product: str) -> datetime | None:
    """Two IMD timestamp layouts are in play here: NIOT prints
    'HH:MM:SS UTC / DD Mon YYYY' on one line; Karaikal prints 'HH:MM:SSZ'
    and 'DD Mon YYYY UTC' on two separate lines, in a bold font that OCRs
    noticeably worse at Karaikal's ~880x720 resolution than NIOT's
    full-resolution frames do (spot check: a stray digit or two gets
    misread often enough to matter). Karaikal is tried first via
    extract_kkl_time_via_templates (far more reliable on this font, see
    that function's comment); this OCR path remains the primary route for
    NIOT and the fallback for Karaikal if template matching can't
    confidently read a frame. A miss falls back to poll-receipt time same
    as always, so this never blocks the pipeline, just loses timestamp
    precision for that frame.

    All Karaikal products share this same bold font AND the same
    timestamp_bbox pixel position (verified: kkl_maxz/kkl_ppz/kkl_ppi are
    all identical, (715, 218, 880, 262) on an 880x720 frame -- it's shared
    UI chrome around the plot, not something that moves per-product), so
    the template-matching path applies to any of them, not just kkl_maxz.
    This was first caught as a real bug, not just a theoretical gap: when
    kkl_ppz was polled for the first time in a test build, its date came
    back 10 days stale ("20 Sep" instead of "30 Sep") while the time-of-
    day was read correctly -- a single 3-misread-as-2 digit, exactly the
    known failure mode this template path exists to avoid, silently
    accepted because the generic OCR fallback below only sanity-checks the
    YEAR, not the day. Routing every Karaikal product through the
    template path sidesteps the date line's OCR entirely (see that
    function's docstring for why), which is the actual fix; the loosened
    generic-path sanity check further down is a second line of defense
    for whatever other product hits a similar single-digit miss."""
    if PRODUCT_RADAR.get(product) == "karaikal":
        hms = extract_kkl_time_via_templates(img)
        if hms is not None:
            h, m, s = hms
            now_utc = datetime.now(timezone.utc)
            dt = now_utc.replace(hour=h, minute=m, second=s, microsecond=0)
            # Day-rollover guard: the frame's printed time can be on the
            # other side of UTC midnight from "now" if this poll landed
            # just after midnight for a frame stamped just before it (or
            # vice versa) -- pick whichever of {yesterday, today,
            # tomorrow}'s calendar date puts the frame time closest to now,
            # capped well under 24h so this never drifts onto the wrong day.
            candidates = [dt + timedelta(days=d) for d in (-1, 0, 1)]
            dt = min(candidates, key=lambda c: abs((c - now_utc).total_seconds()))
            return dt

    if pytesseract is None:
        return None
    crop = img.convert("RGB").crop(PRODUCTS[product]["timestamp_bbox"])
    # Binarize + upscale (cv2, not PIL resize — tested meaningfully more
    # reliable on Karaikal's small bold text) — helps Karaikal's OCR
    # accuracy noticeably; harmless on NIOT's already-good crops.
    arr = np.array(crop)
    big = cv2.resize(arr, (arr.shape[1] * 5, arr.shape[0] * 5), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(bw, config="--psm 6")

    # Kochi's layout is a third one: "Date:DD/MM/YYYY" and "Time:HH:MM:SS
    # UTC" on two separate lines (numeric month, unlike NIOT/Karaikal's
    # abbreviated-month text) -- tried first since its "Date:"/"Time:"
    # keywords are distinctive enough not to collide with the other two
    # patterns.
    m = re.search(r"Date:\s*(\d{1,2}/\d{1,2}/\d{4}).*?Time:\s*(\d{2}:\d{2}:\d{2})\s*UTC",
                   text, re.DOTALL)
    if m:
        date_str, time_str, date_fmt = m.group(1), m.group(2), "%d/%m/%Y"
    else:
        m = re.search(r"(\d{2}:\d{2}:\d{2})\s*UTC\s*/\s*(\d{1,2}\s+\w{3}\s+\d{4})", text)
        if m:
            date_str, time_str, date_fmt = m.group(2), m.group(1), "%d %b %Y"
        else:
            m = re.search(r"(\d{2}:\d{2}:\d{2})\s*Z?.*?(\d{1,2}\s+\w{3}\s+\d{4})\s*UTC",
                           text, re.DOTALL)
            if not m:
                return None
            date_str, time_str, date_fmt = m.group(2), m.group(1), "%d %b %Y"

    try:
        dt = datetime.strptime(f"{date_str} {time_str}", f"{date_fmt} %H:%M:%S")
    except ValueError:
        return None
    dt = dt.replace(tzinfo=timezone.utc)
    # OCR sanity check: a single misread digit can still parse as a valid
    # (wrong) date/time (e.g. "2096" instead of "2026", or "20 Sep" instead
    # of "30 Sep" — both observed on real Karaikal-font frames) without
    # ever looking obviously malformed. These are near-real-time radar
    # scans; a genuine frame is always within minutes of "now", so any
    # multi-day gap means OCR misread a digit, not that IMD served a stale
    # frame from a different day — falling back to poll-receipt time is
    # safer than trusting it. Wide enough (days, not hours) to never
    # false-positive on ordinary network/poll lag, tight enough to catch
    # a wrong day-of-month the same way the old year-only version caught a
    # wrong year (that check only ever fired on a wrong-decade misread —
    # this generalizes it down to the much more common single-digit-in-
    # the-date-field case, like the one that motivated this comment).
    if abs((dt - datetime.now(timezone.utc)).total_seconds()) > 3 * 86400:
        return None
    return dt


def extract_elevation(img: Image.Image, product: str) -> float | None:
    """Parse the printed 'Elevation: X.X' line so live elevation can
    override the configured default if the feed ever cycles tilts.
    MAXZ has no elevation line at all (it's a column-max product across
    0-18km, not a single tilt), so elevation_bbox is None there and this
    always returns None, leaving cfg["elevation_deg"] (already None) as-is."""
    if pytesseract is None:
        return None
    box = PRODUCTS[product].get("elevation_bbox")
    if box is None:
        return None
    crop = img.convert("RGB").crop(box)
    text = pytesseract.image_to_string(crop)
    m = re.search(r"Elevation:\s*([\d.]+)", text)
    return float(m.group(1)) if m else None


# === CELL 10 (colorbar/LUT) ===
def build_lut_from_colorbar(img: Image.Image, product: str) -> list[tuple[tuple[int, int, int], float]]:
    cfg = PRODUCTS[product]
    l, t, r, b = cfg["colorbar_bbox"]
    bar = np.array(img.convert("RGB").crop((l, t, r, b)))
    h = bar.shape[0]

    if "colorbar_breakpoints" in cfg:
        # Non-uniform scale (Karaikal PPI): the printed tick VALUES aren't
        # evenly spaced even though their PIXEL positions are (measured:
        # 15px/tick, but tick deltas go 6,6,5,2,3,6,5,2,3,... dBZ) — so a
        # single top/bottom pair interpolated linearly would misassign
        # every value in between. Piecewise-linear across all the real
        # tick breakpoints instead of just the two endpoints.
        breakpoints = cfg["colorbar_breakpoints"]
        bp_y = np.linspace(0, h - 1, len(breakpoints))
        values = np.interp(np.arange(h), bp_y, breakpoints)
    else:
        top_val, bot_val = cfg["value_at_top"], cfg["value_at_bottom"]
        values = top_val + (bot_val - top_val) * (np.arange(h) / (h - 1))

    lut = []
    for y in range(h):
        rgb = tuple(int(v) for v in bar[y, bar.shape[1] // 2])
        lut.append((rgb, round(float(values[y]), 1)))
    return lut


def product_value_range(product: str) -> tuple[float, float]:
    """(vmin, vmax) for a product's color scale, regardless of whether it's
    configured as a plain linear range (value_at_top/value_at_bottom — every
    NIOT product, Karaikal PPZ) or a piecewise banded scale
    (colorbar_breakpoints — Karaikal PPI, non-uniform dBZ steps at uniform
    pixel spacing, see build_lut_from_colorbar above). Used everywhere a
    color scale needs to be drawn (verification plots, map overlays, the
    animation) so none of them have to know which representation a given
    product uses."""
    cfg = PRODUCTS[product]
    if "colorbar_breakpoints" in cfg:
        bp = cfg["colorbar_breakpoints"]
        return min(bp), max(bp)
    return cfg["value_at_bottom"], cfg["value_at_top"]


from scipy.spatial import cKDTree

# KD-tree per product, built once from that product's color LUT. This is
# what makes decode_reflectivity fast: instead of comparing every pixel
# against every LUT entry in a nested Python loop (which is what made the
# first version of this take tens of minutes on a full-resolution image),
# the whole image is matched against the LUT in one vectorized query.
_kdtree_cache: dict[str, tuple[cKDTree, np.ndarray]] = {}


def get_kdtree(product: str, lut: list) -> tuple[cKDTree, np.ndarray]:
    if product not in _kdtree_cache:
        colors = np.array([c for c, _ in lut], dtype=float)
        values = np.array([v for _, v in lut], dtype=float)
        _kdtree_cache[product] = (cKDTree(colors), values)
    return _kdtree_cache[product]


# Per-product static exclude mask, loaded once and cached — see
# "static_exclude_mask" in PRODUCTS[...] below for why this exists (koc_maxz
# specifically). Boolean array, True = permanently excluded pixel, same
# shape as that product's plot_bbox crop (h, w).
_static_mask_cache: dict[str, np.ndarray] = {}


def get_static_exclude_mask(product: str) -> np.ndarray | None:
    cfg = PRODUCTS[product]
    path = cfg.get("static_exclude_mask")
    if not path:
        return None
    if product not in _static_mask_cache:
        arr = np.array(Image.open(path).convert("L"))
        _static_mask_cache[product] = arr > 0
    return _static_mask_cache[product]


def nearest_dbz(pixel_rgb: tuple[int, int, int],
                 lut: list[tuple[tuple[int, int, int], float]]) -> float | None:
    """Single-pixel lookup — kept for occasional ad-hoc use, but
    decode_reflectivity below no longer calls this in a loop."""
    best, best_dist = None, 1e9
    for (r, g, b), dbz in lut:
        dist = (pixel_rgb[0] - r) ** 2 + (pixel_rgb[1] - g) ** 2 + (pixel_rgb[2] - b) ** 2
        if dist < best_dist:
            best_dist, best = dist, dbz
    return best if best_dist < 900 else None


# Two lessons from calibrating against real full-resolution frames (not
# compressed screenshots):
#
# 1. The old rejection distance (Euclidean 30) was far too loose. The ocean
#    basemap fill color (e.g. (203,218,226) on PPI) sits only ~17 units from
#    a pale colorbar entry, so at threshold=30 nearly half the image's ocean
#    area matched some "reflectivity" value and connected-component labeling
#    fused it all into one enormous fake cell. Real echo pixels match their
#    LUT entry at distance 0 (these renders use flat, unblended fill colors,
#    no anti-aliasing gradient) — so distance 10 is a very comfortable
#    margin above true matches and well below the closest known basemap
#    collision (17).
# 2. Even at distance 10, the white "20 km"/"40 km"/etc. range-ring label
#    boxes still match the palest colorbar band(s) almost exactly, because
#    those boxes are rendered in the same near-white the colorbar uses for
#    its lightest values. A blanket "exclude near-white pixels" rule isn't
#    safe either — some products (MAXZ) render real light echo in that same
#    near-white shade. So instead we exclude the label boxes by their known,
#    fixed pixel position (label_exclude_boxes, measured once per product —
#    they don't move frame-to-frame since the crosshair and range rings are
#    static), not by color.
# 3. Distance 10 was still too loose for PPI/PPZ specifically: this radar's
#    own ocean-fill uses several close variants of pale blue-gray (a subtle
#    shading/antialiasing gradient), some of which land within ~9-17 units
#    of a pale reflectivity color — close enough to register as fake weak
#    echo strung along the whole coastline. Checking the actual distance
#    distribution showed real echo pixels match their LUT entry EXACTLY
#    (distance 0 — these renders use flat, unblended fills, no gradient),
#    with a hard gap before the next cluster around distance 9+. So
#    DIST_THRESHOLD=2 sits safely in that gap: comfortably above any real
#    match, comfortably below every known false one.
DIST_THRESHOLD = 2


def decode_reflectivity(img: Image.Image, product: str, lut: list) -> np.ndarray:
    cfg = PRODUCTS[product]
    l, t, r, b = cfg["plot_bbox"]
    arr = np.array(img.convert("RGB").crop((l, t, r, b)), dtype=float)
    h, w, _ = arr.shape
    flat = arr.reshape(-1, 3)

    tree, values = get_kdtree(product, lut)
    dist, idx = tree.query(flat, k=1)   # vectorized — one call for the whole image

    dbz_flat = values[idx]
    dbz_flat[dist > DIST_THRESHOLD] = np.nan
    dbz = dbz_flat.reshape(h, w)

    for (bl, bt, br, bb) in cfg.get("label_exclude_boxes", []):
        # convert from full-image coords to plot_bbox-relative coords
        y0, y1 = max(bt - t, 0), min(bb - t, h)
        x0, x1 = max(bl - l, 0), min(br - l, w)
        if y1 > y0 and x1 > x0:
            dbz[y0:y1, x0:x1] = np.nan

    static_mask = get_static_exclude_mask(product)
    if static_mask is not None:
        dbz[static_mask] = np.nan

    return dbz


# === CELL 12 (Cell/extract/cluster/track/fuse) ===
@dataclass
class Cell:
    id: int
    product: str
    centroid_latlon: tuple[float, float]
    max_dbz: float
    pixel_count: int
    px: float = 0.0
    py: float = 0.0
    range_km: float = 0.0
    beam_height_km: float = 0.0
    elevated: bool = False   # True if beam_height_km exceeds threshold —
                              # likely not representative of surface rain
    velocity_kmh: tuple[float, float] | None = None
    trend: str = "unknown"
    confirmed_at_surface: bool = False


def extract_cells(dbz: np.ndarray, product: str, elevation_deg: float | None,
                   next_id_start: int = 0) -> list[Cell]:
    cfg = PRODUCTS[product]
    threshold = cfg["cell_dbz_threshold"]
    # Real minimum pixel floor for THIS product's resolution — see
    # MIN_CELL_AREA_KM2 above for why a flat pixel count doesn't transfer
    # across radars with very different km_per_px. Also enforce a small
    # absolute pixel-count floor on top (see MIN_CELL_ABSOLUTE_PIXELS) —
    # at coarse resolutions the area-derived floor alone is sub-pixel and
    # provides no real protection against single-pixel noise.
    min_cell_pixels = max(MIN_CELL_AREA_KM2 * cfg["km_per_px"] ** 2,
                           MIN_CELL_ABSOLUTE_PIXELS)
    mask = dbz >= threshold
    # Binary closing merges pixel-scale gaps within a single storm so it
    # reads as one coherent cell instead of a dozen adjacent fragments —
    # mainly a display/count cleanliness improvement, not a correctness fix.
    # IMPORTANT: closing (dilate-then-erode) can pull in a thin halo of
    # pixels that were NOT part of the original threshold mask — including
    # NaN background. Labeling on the closed mask (for grouping) is fine,
    # but computing stats (max_dbz, centroid, pixel_count) from that closed
    # region is not: `.max()` on an array containing even one NaN silently
    # returns NaN for the whole cell, which is exactly the NaN max_dbz rows
    # you saw in summarize_cycle(). So: label on the closed mask, but always
    # intersect back with the original (pre-closing) mask before touching
    # dbz values.
    closed = ndimage.binary_closing(mask, structure=np.ones((3, 3)))
    labeled, n = ndimage.label(closed)
    ox, oy = cfg["plot_bbox"][0], cfg["plot_bbox"][1]
    cells = []
    for i in range(1, n + 1):
        region = (labeled == i) & mask   # <- intersect with ORIGINAL mask
        ys, xs = np.where(region)
        if len(xs) < min_cell_pixels:
            continue
        cy, cx = ys.mean(), xs.mean()
        latlon = pixel_to_latlon(cx + ox, cy + oy, product)
        rng_km = range_from_site_km(latlon, product)
        # A real radar never plots real echo beyond its own stated maximum
        # range -- a blob whose centroid falls past that (a little slack
        # for rounding/pixel noise near the edge) isn't a storm, it's a
        # stray color match against some non-echo graphic element (a
        # gridline, a boundary anti-alias edge, label text) landing within
        # DIST_THRESHOLD of a LUT color by coincidence. First caught on
        # koc_maxz, whose plan-view crop is a square that reaches well
        # past its 250km range circle at the corners (its plot draws full-
        # panel lat/lon gridlines rather than the range-ring style NIOT/
        # Karaikal use, and this radar's LUT happens to have a pale
        # yellow-white band a near-white gridline pixel can land within 2
        # units of) -- but the check is generic and applies to every
        # product, not just Kochi, as cheap extra insurance everywhere.
        if rng_km > cfg["range_km"] * 1.05:
            continue
        if elevation_deg is not None:
            beam_h = compute_beam_height_km(rng_km, elevation_deg,
                                             site_for(product)["site_elev_m"])
        else:
            beam_h = 0.0  # MAXZ is column-max — "beam height" isn't meaningful
        cells.append(Cell(
            id=next_id_start + i,
            product=product,
            centroid_latlon=latlon,
            px=float(cx),
            py=float(cy),
            max_dbz=float(dbz[ys, xs].max()),
            pixel_count=len(xs),
            range_km=round(rng_km, 1),
            beam_height_km=round(beam_h, 2),
            elevated=beam_h > ELEVATED_BEAM_THRESHOLD_KM,
        ))
    return cells


def cluster_cells(cells: list[Cell], cluster_radius_km: float = CLUSTER_RADIUS_KM) -> list[Cell]:
    """Merge individual connected-component blobs sitting within
    cluster_radius_km of each other into one storm-complex cell, using a
    density-weighted centroid (weighted by pixel_count) — same clustering
    principle as the lightning-strike methodology note's spatio-temporal
    clustering step. Without this, a single storm that segments into
    several adjacent blobs (very common right at the detection threshold)
    gets tracked as several separate, independently-noisy cells, each with
    its own centroid jitter and therefore its own spurious bearing — which
    is exactly what produced the fan of forecast cones pointing in many
    directions instead of one coherent one. Union-find on pairwise
    haversine distance: cheap, and fine at these cell counts (tens, not
    thousands)."""
    if not cells:
        return []
    n = len(cells)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if _haversine_km(cells[i].centroid_latlon, cells[j].centroid_latlon) <= cluster_radius_km:
                union(i, j)

    groups: dict[int, list[Cell]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(cells[i])

    merged = []
    for group in groups.values():
        total_px = sum(c.pixel_count for c in group)
        wlat = sum(c.centroid_latlon[0] * c.pixel_count for c in group) / total_px
        wlon = sum(c.centroid_latlon[1] * c.pixel_count for c in group) / total_px
        wpx = sum(c.px * c.pixel_count for c in group) / total_px
        wpy = sum(c.py * c.pixel_count for c in group) / total_px
        latlon = (wlat, wlon)
        # Velocity/trend can't be density-weighted the way position can (a
        # weighted-average bearing across sources isn't meaningful) -- take
        # both from whichever group member has the largest pixel_count, on
        # the assumption its track is the best-established one. Without
        # this, a merged cell's velocity_kmh silently reverts to the
        # dataclass default (None), which build_forecast_map's fuse=True
        # path depends on for cone projection -- a fused storm would never
        # get a forecast cone even when its source cells clearly had one.
        representative = max(group, key=lambda c: c.pixel_count)
        merged.append(Cell(
            id=min(c.id for c in group),
            product=group[0].product,
            centroid_latlon=latlon,
            px=wpx,
            py=wpy,
            max_dbz=max(c.max_dbz for c in group),
            pixel_count=total_px,
            range_km=round(range_from_site_km(latlon, group[0].product), 1),
            beam_height_km=max(c.beam_height_km for c in group),
            elevated=any(c.elevated for c in group),
            velocity_kmh=representative.velocity_kmh,
            trend=representative.trend,
            confirmed_at_surface=any(c.confirmed_at_surface for c in group),
        ))
    return merged


# Even a genuinely fast-moving convective cell essentially never exceeds
# this -- a computed speed above it is a red flag that dt_minutes was
# spuriously tiny (e.g. a re-served/unchanged frame mistaken for a new
# one, or two polls landing close together) or that track_cells matched
# two DIFFERENT nearby cells across cycles rather than the same cell
# having moved, not that a storm is real. See the reported "1564 km/h"
# cone: root-caused to poll_and_decode's old is_new_frame check treating
# every Karaikal OCR failure as a new frame regardless of whether the
# underlying image had actually changed, fixed at the source now
# (content-hash based) -- but even after that fix, a since-reported
# 101 km/h was still well above anything realistic for monsoon-season
# convective cell motion (typically 15-40 km/h even for a fast, well-
# organized squall line -- per on-the-ground tracking experience, not
# just a guess), so the bound here is set from that domain knowledge
# rather than "some large number that's merely not impossible". Kept as
# a second line of defense against any other way a bad dt or a bad
# nearest-neighbour match sneaks through track_cells -- speeds above
# this get dropped (no cone) rather than shown as if trustworthy.
MAX_PLAUSIBLE_CELL_SPEED_KMH = 60.0


def track_cells(prev_cells: list[Cell], new_cells: list[Cell], dt_minutes: float,
                 max_match_km: float = 15.0) -> list[Cell]:
    """Matches each new cell to its most likely predecessor (nearest
    centroid, same product) purely to label intensity trend
    (intensifying/weakening/steady). Motion (velocity_kmh) is NOT computed
    here any more -- centroid-to-centroid displacement is a single-point
    summary of a whole storm's shape, and gets actively worse as more
    radars/cells are added (a segmented blob splitting, merging or being
    matched to the wrong neighbour between polls all produce a spurious
    "jump" and therefore a spurious bearing/speed). velocity_kmh is instead
    set by poll_and_decode from dense optical flow over the whole
    reflectivity field (see compute_optical_flow / sample_cell_velocity_from_flow)
    before this function runs, and is left untouched here."""
    if not prev_cells:
        return new_cells
    used_prev = set()
    for nc in new_cells:
        best, best_dist = None, 1e9
        for pc in prev_cells:
            if pc.id in used_prev or pc.product != nc.product:
                continue
            d = _haversine_km(pc.centroid_latlon, nc.centroid_latlon)
            if d < best_dist:
                best_dist, best = d, pc
        if best and best_dist <= max_match_km:
            used_prev.add(best.id)
            nc.trend = ("intensifying" if nc.max_dbz > best.max_dbz + 3
                        else "weakening" if nc.max_dbz < best.max_dbz - 3
                        else "steady")
    return new_cells


def fuse_products(ppi_cells: list[Cell], maxz_cells: list[Cell],
                   ppz_cells: list[Cell]) -> None:
    """PPI<->MAXZ: mark MAXZ cells confirmed at the surface once a nearby
    PPI cell exists (mutates maxz_cells). PPZ cells are reported
    separately as regional awareness — they're a different lead-time
    tier, not merged into the near-field confirmation logic."""
    for mc in maxz_cells:
        for pc in ppi_cells:
            if _haversine_km(mc.centroid_latlon, pc.centroid_latlon) <= MATCH_ACROSS_PRODUCTS_KM:
                mc.confirmed_at_surface = True
                break


# === CELL 14 (fetch/state/health) ===
def fetch_frame(url: str) -> bytes | None:
    # Cache-busting: IMD's server (or a CDN/proxy in front of it) may serve
    # a cached copy of the GIF on a plain GET, which would look identical
    # to "you forgot to re-run this cell" from the outside — a fresh query
    # string plus explicit no-cache headers rules that out.
    bust_url = f"{url}{'&' if '?' in url else '?'}_={int(time.time())}"
    try:
        r = requests.get(bust_url, timeout=20,
                          headers={"Cache-Control": "no-cache", "Pragma": "no-cache"})
        r.raise_for_status()
        return r.content
    except requests.RequestException as e:
        print(f"[fetch] {url} failed: {e}")
        return None


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {p: {"last_obs_time": None, "last_new_frame_ts": 0} for p in PRODUCTS}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def health_check(state: dict, product: str) -> str:
    pstate = state.get(product, {})
    if not pstate.get("last_new_frame_ts"):
        return "unknown"
    age = time.time() - pstate["last_new_frame_ts"]
    return "stale" if age > STALE_AFTER_SECONDS else "live"


# === CELL 22 shared display helpers ===
import folium
from folium.raster_layers import ImageOverlay
from PIL import Image as PILImage
import matplotlib
import matplotlib.colors as mcolors

# Discrete/banded reflectivity color scale, used for BOTH the map overlay
# (dbz_array_to_png) and the legend (build_dbz_legend_html) -- a single
# shared definition so they can never drift out of sync with each other.
#
# Previously both used a continuous colormap (matplotlib "turbo") sampled
# through mcolors.Normalize -- smooth gradient blending between colors, the
# same technique a satellite temperature map or elevation shading would
# use. That reads as muted for reflectivity specifically: turbo only
# reaches a saturated "hot" red right near its very top, so a genuinely
# strong 45-50 dBZ cell -- solidly in "heavy rain" territory -- still
# rendered as yellow/yellow-orange, visually undramatic next to something
# like a rain-rate nowcast app using hard, fully-saturated color bands per
# tier (each band maxed out on its own, not blended toward neighbors).
# That's a rendering-style gap, not a true difference in the measured
# storm intensity (see the archived-frame check this was diagnosed from --
# raw peak dBZ values were physically reasonable, matching IMD's own
# printed colorbar).
#
# Fix: classic banded weather-radar reflectivity colors (the scheme used
# by NWS/most radar viewers -- cyan/blue for light echo, green, yellow,
# orange, red, then into magenta/purple for the most intense/rare readings)
# with matplotlib's BoundaryNorm + ListedColormap, which assigns each pixel
# the FLAT color of whichever band it falls in rather than blending -- so a
# 46 dBZ cell snaps straight to solid orange instead of a yellow-orange
# in-between, and the legend's swatches are then literally the same bins
# the map uses, not samples of a separate gradient.
DBZ_BAND_COLORS = [
    "#5ce1e6",  # cyan       -- light/weak echo
    "#2f80ed",  # blue
    "#27ae60",  # green
    "#a0d911",  # yellow-green
    "#f2c811",  # yellow
    "#f2994a",  # orange
    "#eb4034",  # red
    "#a3123a",  # dark red / crimson -- strongest in-range band
    "#c724b1",  # magenta -- anything at/above the scale's own top (rare, e.g. hail cores)
]


def dbz_colormap_and_norm(product: str) -> tuple[mcolors.ListedColormap, mcolors.BoundaryNorm, list]:
    """The shared banded scale for `product`: n_bands evenly spaced dBZ
    boundaries across its native (vmin, vmax) range (from its own printed
    colorbar -- see product_value_range), plus one extra top boundary so
    the last color is reserved for values at/above vmax rather than being
    just another mid-range band. Returns (cmap, norm, boundaries) --
    callers use boundaries to place tick labels/swatches at the same edges
    the map itself renders."""
    vmin, vmax = product_value_range(product)
    n_bands = len(DBZ_BAND_COLORS)
    # n_bands boundaries spanning vmin..vmax (n_bands-1 "normal" bands),
    # plus a final boundary above vmax reserved for the top/overflow band.
    boundaries = list(np.linspace(vmin, vmax, n_bands)) + [vmax + (vmax - vmin) * 0.15]
    cmap = mcolors.ListedColormap(DBZ_BAND_COLORS)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)
    return cmap, norm, boundaries


PRODUCT_STYLE = {
    "ppi":  {"color": "#d62728", "label": "NIOT PPI - surface confirmed"},
    "maxz": {"color": "#ff7f0e", "label": "NIOT MAXZ - aloft / building"},
    "ppz":  {"color": "#1f77b4", "label": "NIOT PPZ - regional awareness"},
    "kkl_ppi": {"color": "#9467bd", "label": "Karaikal PPI - surface confirmed"},
    "kkl_ppz": {"color": "#17becf", "label": "Karaikal PPZ - regional awareness"},
    "kkl_maxz": {"color": "#8c564b", "label": "Karaikal MAXZ - aloft / building"},
    "koc_maxz": {"color": "#2ca02c", "label": "Kochi MAXZ - aloft / building"},
}

RADAR_MARKER_LABEL = {"niot": "NIOT X-DWR Chennai", "karaikal": "Karaikal DWR",
                       "kochi": "Kochi DWR"}

# CARTO now requires an API key even for the free Voyager basemap tiles.
# NOTE: this key travels in the tile URL itself, so it's visible to anyone
# who opens this notebook's source or the saved HTML map — that's inherent
# to how slippy-map tile keys work (it's not a backend secret), but worth
# knowing before this project goes public: if the saved HTML map gets
# heavy traffic, CARTO's usage limits apply against this key regardless of
# who's viewing it. Rotate it if that becomes a concern.
CARTO_VOYAGER_URL = 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png?key=cb1_2eki_1_8725e9310ea39758b1e8f374'
CARTO_ATTR = '&copy; <a href="https://carto.com/attributions">CARTO</a> &copy; OpenStreetMap contributors'


def _smooth_nan_aware(dbz: np.ndarray, sigma_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian-blur `dbz` while ignoring NaN (no-echo) pixels, using
    normalized convolution: blur the data and a 0/1 coverage mask
    separately, then divide -- this avoids NaNs poisoning neighboring
    pixels and avoids the false "fade to zero" you'd get from naively
    filling NaN with 0 before blurring. Returns (smoothed_values,
    coverage_fraction); coverage tells the caller how much real data
    backed each output pixel, so thin/speckled edges can be dropped
    instead of smeared out into fake weak echo."""
    mask = ~np.isnan(dbz)
    filled = np.where(mask, dbz, 0.0)
    coverage = ndimage.gaussian_filter(mask.astype(float), sigma_px)
    blurred = ndimage.gaussian_filter(filled, sigma_px)
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = blurred / coverage
    return smoothed, coverage


def dbz_array_to_png(dbz: np.ndarray, product: str, path: str,
                      smooth_sigma_px: float = 2.5) -> None:
    """Render the decoded reflectivity as a transparent-background PNG,
    ready to drape over a map at the product's true geographic bounds.

    Rendered smoothed (normalized-convolution Gaussian blur) rather than
    pixel-for-pixel -- raw per-pixel dBZ reads as speckled noise once
    draped over a basemap; a smoothed fill reads as a single coherent
    storm the way rain-radar apps (e.g. IMD's own Merged Rainfall viewer)
    render it. Pass smooth_sigma_px=0 to disable and see the raw pixels.
    Detection (decode_reflectivity/extract_cells) always uses the raw,
    unsmoothed array -- this function only affects what's drawn."""
    vmin, vmax = product_value_range(product)
    cmap, norm, _ = dbz_colormap_and_norm(product)

    if smooth_sigma_px > 0:
        values, coverage = _smooth_nan_aware(dbz, smooth_sigma_px)
        alpha = np.clip(coverage * 2.0, 0.0, 1.0)  # fade thin/edge coverage instead of a hard cutoff
        alpha[coverage < 0.1] = 0.0                # drop pixels with almost no real data behind them
    else:
        values = dbz
        alpha = np.where(np.isnan(dbz), 0.0, 1.0)

    filled = np.nan_to_num(values, nan=vmin - 100)
    rgba = cmap(norm(filled))
    rgba[..., 3] = alpha * 0.85  # 0.85 cap -- bumped for contrast vs. Voyager
    img = PILImage.fromarray((rgba * 255).astype(np.uint8))  # dtype+4 channels -> RGBA inferred
    img.save(path)


def plot_bounds_latlon(product: str) -> tuple[float, float, float, float]:
    """(south, west, north, east) for the product's plot_bbox, using the
    SAME calibration as everything else — if this map looks wrong, the
    calibration constants are wrong, not just this overlay."""
    l, t, r, b = PRODUCTS[product]["plot_bbox"]
    lat1, lon1 = pixel_to_latlon(l, t, product)
    lat2, lon2 = pixel_to_latlon(r, b, product)
    south, north = sorted([lat1, lat2])
    west, east = sorted([lon1, lon2])
    return south, west, north, east


def _effective_obs_time(product: str) -> datetime:
    """For freshness COMPARISON only (never shown to a viewer, never used
    for the info banner) -- a product whose timestamp couldn't be read
    this cycle (last_obs_time_seen is None) is treated as the oldest
    possible frame, so it always loses an overlap to any product with a
    genuinely known timestamp. Two unknown-timestamp products compare
    equal, which _mask_stale_overlap treats as a tie (mask neither) --
    there's no basis to prefer one over the other."""
    t = _last_obs_time_seen.get(product)
    return t if t is not None else datetime.min.replace(tzinfo=timezone.utc)


def _mask_stale_overlap(dbz_by_product: dict[str, np.ndarray], products: tuple) -> dict[str, np.ndarray]:
    """When more than one radar's raster is drawn over the same map (two+
    distinct radars in `products`), simply stacking both semi-transparent
    rasters wherever their coverage circles overlap produces a muddy
    double-exposure, and can leave a storm blob visibly lingering in the
    staler radar's frame well after the fresher radar already shows it
    having moved off -- confusing for a reader trying to judge where a
    storm actually is right now.

    Fix: in any patch of ground covered by more than one radar currently
    being drawn, where the FRESHER radar actually has real reflectivity
    data at that exact spot, drop (NaN out -> fully transparent, same
    convention dbz_array_to_png already uses for "no data") the staler
    radar's pixel there. Each radar's raster is left completely untouched
    outside an overlap -- its own unique-coverage area always renders
    exactly as before -- so this only ever removes a pixel a viewer could
    otherwise see fresher data for at the same spot, never data unique to
    that radar.

    Checking the fresher radar's ACTUAL per-pixel data (not just whether
    the spot falls within its range CIRCLE) matters: a range circle covers
    ground the radar has a clear view of, not ground it's currently
    reporting rain over -- most of any radar's range circle is normally
    empty/NaN (no echo). The original version dropped every staler pixel
    anywhere inside the fresher radar's full range circle regardless of
    whether the fresher radar had any data there, which silently erased
    genuine storms sitting in a range-circle overlap the fresher radar
    simply hadn't detected anything at (e.g. NIOT and Kochi's circles
    overlap out past Erode, and a real storm only Karaikal or NIOT was
    picking up there would vanish under Kochi's empty circle) -- reported
    directly as storms visible in one radar's own map but missing from the
    combined one, in exactly this kind of overlap zone.

    Purely a display fix: doesn't touch cell extraction, tracking, or the
    forecast-cone fusion (`fuse=`) logic above, and runs regardless of
    whether `fuse` is on for this call."""
    radars_present = {PRODUCT_RADAR[p] for p in products if p in dbz_by_product}
    if len(radars_present) < 2:
        return dbz_by_product

    masked = {p: arr.copy() for p, arr in dbz_by_product.items()}
    for product in products:
        if product not in masked:
            continue
        arr = masked[product]
        ys, xs = np.where(~np.isnan(arr))
        if len(xs) == 0:
            continue
        ox, oy = PRODUCTS[product]["plot_bbox"][0], PRODUCTS[product]["plot_bbox"][1]
        lat, lon = pixel_to_latlon(xs + ox, ys + oy, product)
        this_time = _effective_obs_time(product)

        drop = np.zeros(len(xs), dtype=bool)
        for other_product in products:
            # Compared against the ORIGINAL, pre-masking rasters
            # (dbz_by_product, not `masked`) so this doesn't depend on
            # what order `products` happens to process in -- with 3
            # radars shown together, an earlier product in this loop may
            # already have had some of ITS pixels dropped by a third,
            # even-fresher radar, and that shouldn't change what counts
            # as "real data" when checking a later product here.
            if other_product not in dbz_by_product or other_product == product:
                continue
            if PRODUCT_RADAR[other_product] == PRODUCT_RADAR[product]:
                continue  # same radar, e.g. two products off one site -- not an overlap case
            if _effective_obs_time(other_product) <= this_time:
                continue  # other isn't strictly fresher -- doesn't get to mask this one

            other_arr = dbz_by_product[other_product]
            other_ox = PRODUCTS[other_product]["plot_bbox"][0]
            other_oy = PRODUCTS[other_product]["plot_bbox"][1]
            other_px, other_py = latlon_to_pixel(lat, lon, other_product)
            other_x = np.round(other_px - other_ox).astype(int)
            other_y = np.round(other_py - other_oy).astype(int)
            in_bounds = ((other_x >= 0) & (other_x < other_arr.shape[1]) &
                         (other_y >= 0) & (other_y < other_arr.shape[0]))
            has_data = np.zeros(len(xs), dtype=bool)
            idx = np.where(in_bounds)[0]
            has_data[idx] = ~np.isnan(other_arr[other_y[idx], other_x[idx]])
            drop |= has_data

        if drop.any():
            arr[ys[drop], xs[drop]] = np.nan
    return masked


def build_dbz_legend_html(products: tuple) -> str:
    """A floating color-scale legend (swatches + value ticks), the same
    idea as the mm/h bar on rain-radar apps -- so a viewer can read a
    color on the map back into a dBZ value without opening a popup.
    Each swatch is one flat band from DBZ_BAND_COLORS via
    dbz_colormap_and_norm, the SAME shared definition dbz_array_to_png
    uses, so the legend is never just an approximation of what's drawn --
    it's literally the same bins."""
    vmin, vmax = product_value_range(products[0])  # legend reflects the primary displayed product's scale
    cmap, norm, boundaries = dbz_colormap_and_norm(products[0])

    swatches = ""
    for i, hexcolor in enumerate(DBZ_BAND_COLORS):
        # Label each band by its lower edge, except the last (open-ended
        # top band, e.g. "60+") since it has no real upper bound.
        label_val = boundaries[i]
        label = f"{label_val:.0f}+" if i == len(DBZ_BAND_COLORS) - 1 else f"{label_val:.0f}"
        swatches += (
            f'<div style="flex:1; text-align:center;">'
            f'<div style="background:{hexcolor}; height:14px; width:100%; '
            f'border:1px solid rgba(0,0,0,0.15);"></div>'
            f'<div style="font-size:11px; color:#333; margin-top:2px;">{label}</div>'
            f'</div>'
        )

    label = " / ".join(PRODUCT_STYLE[p]["label"].split(" - ")[0] for p in products)
    return f"""
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999;
                background: rgba(255,255,255,0.92); padding: 10px 14px;
                border-radius: 8px; box-shadow: 0 1px 6px rgba(0,0,0,0.3);
                font-family: -apple-system, Arial, sans-serif;">
        <div style="font-size: 12px; font-weight: 600; color: #222; margin-bottom: 6px;">
            {label} — reflectivity (dBZ)
        </div>
        <div style="display:flex; width: 260px;">{swatches}</div>
    </div>
    """


from zoneinfo import ZoneInfo

def project_forward(lat: float, lon: float, speed_kmh: float, bearing_deg: float,
                     lead_minutes: float) -> tuple[float, float]:
    """Straight-line kinematic projection of a point, same flat-earth
    approximation used everywhere else in this notebook."""
    dist_km = speed_kmh * (lead_minutes / 60.0)
    bearing_rad = np.radians(bearing_deg)
    dlat = dist_km * np.cos(bearing_rad) / 111.0
    dlon = dist_km * np.sin(bearing_rad) / (111.0 * np.cos(np.radians(lat)))
    return lat + dlat, lon + dlon


def build_bearing_arrow_icon(bearing_deg: float, color: str) -> folium.DivIcon:
    """A small filled triangle, rotated to point along `bearing_deg` (a
    standard compass bearing -- 0=N, 90=E, same convention project_forward
    uses, so a position computed via project_forward's bearing lines up
    with this icon's rotation with no separate conversion needed).

    This used to sit inside a widening translucent "uncertainty cone"
    polygon per lead time (one wedge each for +30/+60/+90 min), which is
    where the "own centerline" framing below comes from. The cones were
    dropped -- on a busy multi-cell frame they covered enough of the
    reflectivity raster underneath to make the actual storms hard to see,
    reported directly against real screenshots -- so the arrow is now the
    only way direction is shown on the map, not just the disambiguating
    detail inside a bigger shape. It's kept precisely because, cone or no
    cone, nothing about a plain colored dot says "moving this way" to
    someone glancing at the map -- and a tooltip nobody hovers on a public
    read-only page doesn't help either.

    Plain inline SVG in a DivIcon rather than a Leaflet plugin (e.g.
    leaflet-polylinedecorator, the "proper" way to put arrowheads on a
    line) deliberately -- an extra plugin means another third-party CDN
    script this page depends on to render at all, and depending on one is
    exactly what caused the AwesomeMarkers blank-map incident earlier.
    This has no dependency beyond Leaflet itself, which the map can't
    render without anyway."""
    svg = f"""
    <div style="transform: rotate({bearing_deg}deg); width:22px; height:22px;
                pointer-events:none;">
        <svg width="22" height="22" viewBox="0 0 22 22">
            <polygon points="11,1 3,19 11,14 19,19" fill="{color}"
                     stroke="white" stroke-width="1.5" stroke-linejoin="round"/>
        </svg>
    </div>
    """
    return folium.DivIcon(html=svg, icon_size=(22, 22), icon_anchor=(11, 11))


LOGO_PATH = Path(__file__).parent / "assets" / "chennairains_logo.jpg"


def _logo_data_uri() -> str | None:
    """Base64-embeds the ChennaiRains logo directly into the HTML so the
    map stays a single self-contained file (same reasoning as the raster
    overlay PNGs -- no second asset to host/keep in sync). Returns None
    if the logo asset is missing so a broken path never breaks the map
    build itself, just silently skips the branding."""
    if not LOGO_PATH.exists():
        print(f"Logo asset not found at {LOGO_PATH} -- skipping branding.")
        return None
    encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_logo_tag(height_px: int = 26) -> str:
    """Just the <a><img></a> fragment (no fixed positioning of its own) so
    it can be dropped inline next to other UI, e.g. inside the info banner's
    header row. Empty string if the logo asset is missing."""
    logo_uri = _logo_data_uri()
    if not logo_uri:
        return ""
    return (
        f'<a href="https://www.chennairains.com" target="_blank" rel="noopener" '
        f'style="text-decoration:none; flex-shrink:0;">'
        f'<img src="{logo_uri}" alt="ChennaiRains" '
        f'style="height:{height_px}px; width:{height_px}px; display:block; '
        f'border-radius:6px;"></a>'
    )


def build_info_banner_html(products: tuple, last_obs_time_seen: dict) -> str:
    """A small fixed banner (top-right, where the layer toggle used to sit)
    showing each displayed product's most recent radar observation time --
    converted to IST, since the audience is chennairains.com's readers, not
    UTC-native -- plus a standing disclaimer. Without a visible timestamp a
    viewer has no way to tell whether they're looking at a live storm or a
    stale page that failed to refresh; the disclaimer matters because this
    is a derived/experimental product, not an official IMD or ChennaiRains
    forecast, and that needs to be obvious on the page itself, not just in
    a README nobody visiting the site will ever read.

    The ChennaiRains logo sits inline in this same box, next to the title --
    previously it floated on its own over the top-left corner, where it sat
    on top of Leaflet's zoom control. Anchoring it here instead keeps it
    away from any map control regardless of screen size."""
    ist = ZoneInfo("Asia/Kolkata")
    now_utc = datetime.now(timezone.utc)
    lines = []
    for product in products:
        obs_time = last_obs_time_seen.get(product)
        label = PRODUCT_STYLE.get(product, {"label": product.upper()})["label"].split(" - ")[0]
        if obs_time:
            local = obs_time.astimezone(ist)
            age_min = (now_utc - obs_time).total_seconds() / 60.0
            # A genuinely odd/old time here (reported directly -- e.g.
            # Karaikal reading several hours behind NIOT/Kochi in the same
            # banner) has two real causes that look identical at a glance:
            # IMD's own feed for that one radar has stalled/gone down for a
            # while and keeps re-serving the same old frame (this happens;
            # our poll correctly reads whatever timestamp is actually
            # printed on the image), or a rarer OCR/template misread.
            # Either way, presenting it with the same plain styling as a
            # normal fresh reading is the actual problem -- a reader can't
            # tell "this is live" from "this radar's been down since
            # lunch" just by glancing at the banner. STALE_OBS_MINUTES
            # (comfortably above the ~15 min poll cadence, to allow for
            # IMD's own ordinary publishing lag) flags it instead of
            # silently blending in.
            if age_min > STALE_OBS_MINUTES:
                lines.append(
                    f'{label}: <span style="color:#c0392b; font-weight:600;">'
                    f'{local.strftime("%d %b, %H:%M")} IST ⚠ stale'
                    f'</span>'
                )
            else:
                lines.append(f"{label}: {local.strftime('%d %b, %H:%M')} IST")
        else:
            lines.append(f"{label}: time unavailable")
    times_html = "<br>".join(lines)
    logo_tag = build_logo_tag()
    return f"""
    <div style="position: fixed; top: 12px; right: 12px; z-index: 9999;
                background: rgba(255,255,255,0.92); padding: 8px 12px;
                border-radius: 8px; box-shadow: 0 1px 6px rgba(0,0,0,0.3);
                font-family: -apple-system, Arial, sans-serif; font-size: 11px;
                color: #333; max-width: 230px; line-height: 1.5;">
        <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 4px;">
            {logo_tag}
            <div style="font-weight: 700; color: #b45309;">
                &#9888; Experimental nowcast
            </div>
        </div>
        <div style="margin-bottom: 4px;">{times_html}</div>
        <div style="font-size: 10px; color: #666;">
            Based on IMD radar imagery. Not an official forecast.<br>
            Auto-refreshes every {AUTOREFRESH_MINUTES} min.
        </div>
    </div>
    """


def build_autorefresh_script(interval_minutes: int = AUTOREFRESH_MINUTES) -> str:
    """This is a static HTML file re-uploaded on a schedule -- a browser
    tab left open on it otherwise just sits on whatever was live at the
    moment it was opened, with no way to know a newer map has since been
    published. A plain timed reload fixes that for anyone who leaves the
    page open; the schedule (nowcast.yml -> cron-job.org, see that
    workflow's comments) already keeps the live file itself fresh every
    ~15 min, this just makes sure an already-open tab actually picks that
    up instead of going stale in the background.

    Deliberately a plain JS reload, not a <meta http-equiv="refresh">:
    the meta-refresh timer is anchored to when the browser PARSED the
    page, which can lag noticeably behind when the tab actually became
    visible to the reader (e.g. a page pre-loaded in a background tab);
    this script's timer starts from the same moment, but since it's easy
    to swap for a visibility-aware version later (only start counting
    once the tab is actually visible) if that ever turns out to matter,
    plain JS is kept here as the more extensible starting point.

    interval_minutes is deliberately NOT the same 15 as the upload cron --
    a little offset (see AUTOREFRESH_MINUTES) means a reload landing
    exactly mid-upload (this repo's FTP step isn't atomic -- see
    nowcast.yml) is rare rather than a fixed date with every single
    upload; if it ever does land mid-transfer, the NEXT reload a few
    minutes later self-heals it, same as any other transient fetch
    hiccup on this site.

    Returns BARE JavaScript, deliberately with no <script> tags of its
    own -- this is added via m.get_root().script.add_child(...), and
    branca's Figure template already wraps everything under .script in
    ONE shared <script>...</script> for the whole page (same convention
    every folium element's own script macro follows, e.g. Marker's).
    Wrapping this snippet in its own nested <script>/</script> (as an
    earlier version of this function did) inserts a literal '</script>'
    into the middle of that already-open tag -- browsers don't parse
    nested script elements, so that closes the shared script block right
    there, silently killing every statement after it INCLUDING Leaflet's
    own map/tile-layer/marker init code that folium adds the same way.
    That's a real regression this function caused once already: it broke
    the whole map (blank page, only the plain-HTML overlays like the
    banner/legend/watermark still rendered, since those go through
    .html, not .script) while looking completely fine in isolation."""
    interval_ms = interval_minutes * 60 * 1000
    return f"setTimeout(function() {{ window.location.reload(); }}, {interval_ms});"


def build_fallback_logo_html() -> str:
    """Standalone logo box for the (rare) case there's no reflectivity data
    yet to show the info banner at all -- keeps the branding present on
    every map, not just ones with a storm to draw. Anchored top-right, same
    corner as the info banner would occupy, so it never overlaps Leaflet's
    top-left zoom control."""
    logo_uri = _logo_data_uri()
    if not logo_uri:
        return ""
    return f"""
    <a href="https://www.chennairains.com" target="_blank" rel="noopener"
       style="position: fixed; top: 12px; right: 12px; z-index: 9999;
              text-decoration: none;">
        <img src="{logo_uri}" alt="ChennaiRains"
             style="height: 40px; width: 40px; display: block;
                    border-radius: 8px; box-shadow: 0 1px 6px rgba(0,0,0,0.35);
                    background: rgba(255,255,255,0.85); padding: 2px;">
    </a>
    """


def build_watermark_html() -> str:
    """A light, tiled diagonal watermark across the whole map. It exists
    specifically so a screenshot of this page still carries attribution --
    faint enough not to interfere with reading the radar/cones, but present
    everywhere so it can't just be cropped out of a corner. pointer-events:
    none so it never blocks clicking the map underneath."""
    # Smaller and fainter than the original -- still a real watermark (a
    # screenshot of the map still carries attribution, still tiled so it
    # can't be cropped out of one corner), but the goal shifted from
    # "clearly branded" to "present without getting in the way": readable
    # on close inspection, easy to look past at a glance.
    watermark_text = "Radar Nowcast by https://plots.chennairains.com/"
    watermark_svg = f"""
    <svg xmlns='http://www.w3.org/2000/svg' width='460' height='260'>
        <text x='230' y='135' transform='rotate(-28 230 135)'
              font-family='Arial, sans-serif' font-size='9'
              fill='rgba(0,0,0,0.10)' text-anchor='middle'
              font-weight='600'>{watermark_text}</text>
    </svg>
    """
    watermark_data_uri = "data:image/svg+xml;base64," + base64.b64encode(
        watermark_svg.encode("utf-8")
    ).decode("ascii")

    return f"""
    <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                z-index: 9997; pointer-events: none;
                background-image: url('{watermark_data_uri}');
                background-repeat: repeat;"></div>
    """


def build_home_button_html() -> str:
    """Small floating pill, bottom-center, back to the forecast-maps hub
    (plots.chennairains.com/index.html -- the "All forecasts" index other
    ChennaiRains map pages already link back to) so a reader who lands
    straight on the live radar page isn't stuck here with no way back
    except the browser's back button, which won't work if they arrived via
    a bookmark/shared link. Deliberately separate from the small logo icon
    (build_logo_tag/build_fallback_logo_html), which points at the main
    chennairains.com site instead -- this is "back to the forecasts I was
    just browsing", not "back to the blog homepage". Bottom-center, clear
    of the legend (bottom-left) and info banner/logo (top-right)."""
    return """
    <a href="https://plots.chennairains.com/index.html"
       style="position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
              z-index: 9999; background: rgba(30,30,30,0.85); color: #fff;
              font-family: -apple-system, Arial, sans-serif; font-size: 13px;
              font-weight: 600; padding: 9px 16px; border-radius: 999px;
              text-decoration: none; box-shadow: 0 1px 6px rgba(0,0,0,0.35);
              white-space: nowrap;">
        &#8592; All forecasts
    </a>
    """


def build_forecast_map(products: tuple = ("maxz",), fuse: bool = False,
                        lead_times_min: tuple = (30, 60, 90),
                        min_speed_kmh: float = 5.0,
                        out_html: str | None = "storm_forecast_map.html"):
    """Draws the current reflectivity raster, each tracked cell's current
    position, and a direction arrow for its projected motion -- on the
    CARTO basemap + per-radar range-boundary ring(s), same multi-radar
    pattern as build_osm_verification_map(). The raster overlay is what
    makes the storm's actual shape/extent visible (not just a dot at its
    centroid), same reasoning as build_osm_verification_map's overlay.
    (An earlier version also drew a widening translucent "uncertainty
    cone" polygon per lead time behind the arrow -- dropped because it
    covered too much of that raster on a busy multi-cell frame; see
    build_bearing_arrow_icon's docstring.)

    products: tuple of product keys, e.g. ("maxz",) for NIOT only, or
        ("maxz", "kkl_maxz") for NIOT + Karaikal together. Unlike
        build_osm_verification_map(), this always draws every shown
        product's raster directly on the map with no per-radar toggle --
        live comparison between NIOT and Karaikal showed the two agree
        closely enough on storm location that a separate on/off control
        per radar wasn't adding anything, just one more thing to misclick.

    fuse: with more than one radar in `products`, cross-radar cluster the
        cells first (cluster_cells over every shown product's cells) so a
        storm sitting in both radars' coverage gets ONE projected arrow
        instead of two overlapping, possibly-disagreeing ones. Velocity for
        a fused cell comes from whichever source cell cluster_cells picked
        as representative -- same as build_osm_verification_map's fused
        markers, no separate velocity-fusion logic here. The raster
        overlay itself is always drawn per-product either way (it's a
        colored fill, not a confusable point marker, so there's no
        duplication problem from drawing both radars' rasters together).

    Cells slower than `min_speed_kmh` are skipped -- a near-stationary
    cell's bearing is mostly noise, and projecting noise forward just draws
    a confident-looking cone around nothing (same "suppress low-confidence
    tracks" principle as the methodology note's sparse-strike caveat,
    applied here to sparse-motion cells instead)."""
    radars_shown = sorted({PRODUCT_RADAR[p] for p in products})
    center_lat = np.mean([RADAR_SITES[r]["site_lat"] for r in radars_shown])
    center_lon = np.mean([RADAR_SITES[r]["site_lon"] for r in radars_shown])
    zoom = 9 if len(radars_shown) == 1 else 7
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles=None)
    folium.TileLayer(tiles=CARTO_VOYAGER_URL, attr=CARTO_ATTR, name="CARTO Voyager").add_to(m)
    # Every shape with a tooltip/popup (every cell marker, every forecast
    # cone) gets a tabindex from Leaflet for keyboard-accessibility, and
    # browsers draw their own default focus outline on click/hover -- a
    # plain black rectangle around that SHAPE'S BOUNDING BOX, not its
    # actual outline -- which is exactly the "black box on every cone"
    # reported: it's a browser default, unrelated to FORECAST_STYLE's
    # actual (orange/red) polygon colors. Suppressing just the outline,
    # not the tabindex itself, keeps the shapes keyboard-focusable (so a
    # screen reader / keyboard user can still tab to each tooltip) while
    # dropping only the visual artifact.
    m.get_root().header.add_child(folium.Element(
        "<style>.leaflet-interactive:focus { outline: none; }</style>"
    ))

    for r in radars_shown:
        site = RADAR_SITES[r]
        # Plain default marker, deliberately NOT icon=folium.Icon(icon=...,
        # prefix="fa") -- that routes through Leaflet's AwesomeMarkers
        # plugin, which pulls in two extra third-party CDN resources
        # (leaflet.awesome-markers.js + Font Awesome's CSS) just to draw
        # this one decorative pin. If EITHER fails to load -- blocked by
        # an ad-blocker or a corporate/school network, or just a flaky
        # CDN moment -- the resulting uncaught JS exception
        # ("Cannot read properties of undefined (reading 'icon')") halts
        # the REST of this script, including every storm cell/cone/
        # tile-layer call still queued after it -- producing exactly the
        # "banner and legend show, but the map itself is blank" symptom
        # reported and reproduced (locally, by blocking that one CDN
        # request and watching the whole render die on this line). Not
        # worth risking the entire map over a pin's color/glyph -- the
        # tooltip already names which radar this is.
        folium.Marker(
            [site["site_lat"], site["site_lon"]],
            tooltip=RADAR_MARKER_LABEL.get(r, r),
        ).add_to(m)

    # One range-boundary ring per (radar, range) pair actually shown --
    # same dedup logic as build_osm_verification_map, so passing e.g.
    # ("maxz", "kkl_maxz") draws two rings, one centered on each site.
    drawn_ranges = set()
    for product in products:
        range_km = PRODUCTS[product]["range_km"]
        site = site_for(product)
        key = (PRODUCT_RADAR[product], range_km)
        if key in drawn_ranges:
            continue
        drawn_ranges.add(key)
        folium.Circle(
            location=[site["site_lat"], site["site_lon"]],
            radius=range_km * 1000.0, color="#555555", weight=1.5, opacity=0.6,
            dash_array="6,6", fill=False,
            tooltip=f"{range_km:.0f} km range — {product.upper()} scan boundary",
        ).add_to(m)

    # Reflectivity raster overlay -- same dbz_array_to_png()/plot_bounds_latlon()
    # pipeline as build_osm_verification_map(), so the storm's real shape is
    # visible under the markers/cones instead of just a dot at its centroid.
    # Drawn directly on the map (no per-product FeatureGroup/toggle -- see
    # the products docstring above for why). Where two+ radars' coverage
    # overlaps, _mask_stale_overlap keeps only the freshest radar's pixels
    # in that shared patch so overlapping storms don't render as a muddy
    # double-exposure or show a stale blob the fresher radar has already
    # moved past -- see that function's docstring.
    rasters_to_draw = _mask_stale_overlap(_last_dbz, products)
    for product in products:
        if product not in rasters_to_draw:
            continue
        style = PRODUCT_STYLE.get(product, {"label": product.upper()})
        png_path = f"_forecast_overlay_{product}.png"
        dbz_array_to_png(rasters_to_draw[product], product, png_path)
        south, west, north, east = plot_bounds_latlon(product)
        ImageOverlay(image=png_path, bounds=[[south, west], [north, east]],
                     opacity=0.7, name=f"{style['label']} reflectivity").add_to(m)

    do_fuse = fuse and len({PRODUCT_RADAR[p] for p in products}) > 1
    if do_fuse:
        cells_to_project = cluster_cells([c for p in products for c in _prev_cells.get(p, [])])
    else:
        cells_to_project = [c for p in products for c in _prev_cells.get(p, [])]

    n_projected = 0
    for c in cells_to_project:
        radius = 7 + min(c.pixel_count / 40, 8)
        fill_color = "#2ca02c" if do_fuse else PRODUCT_STYLE.get(c.product, {}).get("color", "#08306b")
        label_prefix = "Fused cell" if do_fuse else f"{c.product.upper()} cell"
        folium.CircleMarker(
            location=c.centroid_latlon, radius=radius,
            color="white", weight=3, opacity=1.0,
            fill=True, fill_color=fill_color, fill_opacity=1.0,
            popup=f"{label_prefix} #{c.id} — {c.max_dbz:.0f} dBZ now, trend: {c.trend}",
        ).add_to(m)

        if not c.velocity_kmh or c.velocity_kmh[0] < min_speed_kmh:
            continue  # no reliable motion yet, or effectively stationary
        speed_kmh, bearing_deg = c.velocity_kmh

        # Direction arrow only -- no uncertainty cone. The cones (one
        # widening translucent wedge per lead time) were covering enough of
        # the raster underneath, on a busy multi-cell frame, that the
        # actual reflectivity imagery they're meant to sit on top of became
        # hard to read -- reported directly, and visible in side-by-side
        # screenshots where the storms were barely visible under a stack of
        # overlapping orange/red wedges. The arrow alone still answers the
        # question the cones existed for ("which way is this cell
        # heading"), just without the added canvas coverage a full
        # confidence-envelope shape brings. Lead-time-specific reach/speed
        # detail is still in the tooltip below; it's just not drawn as
        # shapes on the map any more.
        #
        # Placed a bit past the halfway point of the SHORTEST configured
        # lead time: close enough to the storm marker to clearly belong to
        # it, far enough out to read as "this way" rather than sit on top
        # of the marker.
        nearest_lead = min(lead_times_min)
        arrow_lat, arrow_lon = project_forward(*c.centroid_latlon, speed_kmh,
                                                bearing_deg, nearest_lead * 0.55)
        folium.Marker(
            location=(arrow_lat, arrow_lon),
            icon=build_bearing_arrow_icon(bearing_deg, "#333333"),
            tooltip=f"{label_prefix} #{c.id} heading {bearing_deg:.0f}° "
                    f"at {speed_kmh:.0f} km/h",
        ).add_to(m)
        n_projected += 1

    if n_projected == 0:
        print("No cells with usable velocity yet — need at least two consecutive "
              "run_once() calls on a moving storm before there's anything to project.")

    if any(p in _last_dbz for p in products):
        m.get_root().html.add_child(folium.Element(build_dbz_legend_html(products)))
        m.get_root().html.add_child(folium.Element(build_info_banner_html(products, _last_obs_time_seen)))
    else:
        # No reflectivity yet to anchor the info banner to -- show the logo
        # on its own so branding is still present on every map.
        m.get_root().html.add_child(folium.Element(build_fallback_logo_html()))

    m.get_root().html.add_child(folium.Element(build_watermark_html()))
    m.get_root().html.add_child(folium.Element(build_home_button_html()))
    m.get_root().script.add_child(folium.Element(build_autorefresh_script()))

    if out_html:
        m.save(out_html)
        print(f"Saved {out_html}  (self-contained — forkable/hostable on its own)")

    return m


# Same MAXZ-only, cross-radar-fused default as build_osm_verification_map,
# and for the same reason (PPI/PPZ's near-white/near-ocean-blue bands
# collide with each radar's own pale basemap fill often enough to be
# visually distracting). Drop back to build_forecast_map() (NIOT MAXZ only)
# if Karaikal's calibration hasn't been visually verified yet -- see the
# CALIBRATION STATUS comment above RADAR_SITES.

# === AUTOMATION GLUE (not from the notebook) ===
#
# The Colab notebook keeps tracking state (_prev_cells, _prev_obs_time,
# _last_dbz, _lut_cache) as plain Python globals that live for as long as
# the Colab session stays open -- fine there, since one cell run builds on
# the last. A GitHub Actions run is a brand-new machine every single time
# (nothing survives between runs except what's committed back to the repo),
# so that state has to be serialized to disk and reloaded each run instead.
#
# Everything above this point (cells 2/4/6/8/10/12/14 + the shared display
# helpers from cell 22 + build_forecast_map from cell 26) is copied
# unmodified from the notebook. Only what follows is new, and its whole job
# is: load state -> poll both radars once -> save state -> build the map.

# Point the notebook's ARCHIVE_DIR/STATE_FILE constants (originally
# relative paths meant for a Colab working directory) at this repo's
# dedicated folders instead, and make sure they exist.
ARCHIVE_DIR = Path("archive")
STATE_FILE = Path("state/poll_state.json")
CELLS_STATE_FILE = Path("state/prev_cells.json")
OBS_TIME_STATE_FILE = Path("state/prev_obs_time.json")
OUTPUT_HTML = Path("output/storm_forecast_map.html")
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)

# Only archive frames for the products this pipeline actually polls, so the
# repo doesn't accumulate PPI/PPZ frames for products we never fetch here.
POLLED_PRODUCTS = ("maxz", "kkl_maxz", "koc_maxz")

# How many archived frames to keep per product. Kept small on purpose --
# this repo is committing binary GIFs on every run, and unlike the
# Colab-only workflow there's no forecast-animation cell reading from this
# archive yet, so there's no reason to let it grow unbounded. Bump this if
# a multi-radar build_storm_animation_mp4 run gets wired into the
# automation later and needs a longer history to animate from.
KEEP_FRAMES_PER_PRODUCT = 6


# build_forecast_map() (from cell 26) reads these three module-level globals
# directly, same as it does in the notebook -- initialized here so the name
# exists even before run_pipeline() first assigns into it.
_prev_cells: dict[str, list[Cell]] = {p: [] for p in POLLED_PRODUCTS}
_last_dbz: dict[str, np.ndarray] = {}
# Feeds build_info_banner_html()'s per-radar timestamp readout -- set fresh
# each run in run_pipeline() from whatever observation time was actually
# used to build this cycle's map (either a freshly-polled frame's own
# obs_time, or -- on a fetch failure -- the timestamp parsed back out of
# the archived frame we fell back to).
_last_obs_time_seen: dict[str, "datetime | None"] = {p: None for p in POLLED_PRODUCTS}


def _cell_to_dict(c: Cell) -> dict:
    return {
        "id": c.id, "product": c.product, "centroid_latlon": list(c.centroid_latlon),
        "max_dbz": c.max_dbz, "pixel_count": c.pixel_count, "px": c.px, "py": c.py,
        "range_km": c.range_km, "beam_height_km": c.beam_height_km, "elevated": c.elevated,
        "velocity_kmh": list(c.velocity_kmh) if c.velocity_kmh else None,
        "trend": c.trend, "confirmed_at_surface": c.confirmed_at_surface,
    }


def _cell_from_dict(d: dict) -> Cell:
    return Cell(
        id=d["id"], product=d["product"], centroid_latlon=tuple(d["centroid_latlon"]),
        max_dbz=d["max_dbz"], pixel_count=d["pixel_count"], px=d["px"], py=d["py"],
        range_km=d["range_km"], beam_height_km=d["beam_height_km"], elevated=d["elevated"],
        velocity_kmh=tuple(d["velocity_kmh"]) if d["velocity_kmh"] else None,
        trend=d["trend"], confirmed_at_surface=d["confirmed_at_surface"],
    )


def load_prev_cells() -> dict[str, list[Cell]]:
    if not CELLS_STATE_FILE.exists():
        return {p: [] for p in POLLED_PRODUCTS}
    raw = json.loads(CELLS_STATE_FILE.read_text())
    return {p: [_cell_from_dict(d) for d in raw.get(p, [])] for p in POLLED_PRODUCTS}


def save_prev_cells(prev_cells: dict[str, list[Cell]]) -> None:
    raw = {p: [_cell_to_dict(c) for c in cells] for p, cells in prev_cells.items()}
    CELLS_STATE_FILE.write_text(json.dumps(raw, indent=2))


def load_prev_obs_time() -> dict[str, datetime | None]:
    if not OBS_TIME_STATE_FILE.exists():
        return {p: None for p in POLLED_PRODUCTS}
    raw = json.loads(OBS_TIME_STATE_FILE.read_text())
    return {p: (datetime.fromisoformat(raw[p]) if raw.get(p) else None) for p in POLLED_PRODUCTS}


def save_prev_obs_time(prev_obs_time: dict[str, datetime | None]) -> None:
    raw = {p: (t.isoformat() if t else None) for p, t in prev_obs_time.items()}
    OBS_TIME_STATE_FILE.write_text(json.dumps(raw, indent=2))


def prune_archive(product: str) -> None:
    """Keep only the most recent KEEP_FRAMES_PER_PRODUCT archived frames for
    `product` -- this repo commits every new frame as a binary file, so an
    unbounded archive would make the repo grow forever for no benefit this
    pipeline currently uses (nothing here builds an animation from it yet)."""
    frames = sorted(ARCHIVE_DIR.glob(f"{product}_*.gif"))
    for old in frames[:-KEEP_FRAMES_PER_PRODUCT]:
        old.unlink()


def _obs_time_from_archive_filename(path: Path) -> "datetime | None":
    """Archived frames are named `{product}_{tag}.gif` where tag is either
    an obs_time formatted as %Y%m%dT%H%M%SZ, or -- when a frame had no
    readable timestamp at capture time -- a raw epoch-seconds int instead
    (see poll_and_decode's `tag = ... else int(now)` fallback). Used when a
    live fetch fails and we fall back to the last archived frame, so the
    banner can still show a real time instead of just "time unavailable"."""
    stem = path.stem  # e.g. "maxz_20260928T125640Z" or "kkl_maxz_1790602773"
    tag = stem.rsplit("_", 1)[-1]
    try:
        return datetime.strptime(tag, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None  # epoch-int fallback name -- no reliable obs_time to recover


def _load_previous_dbz_for_flow(product: str) -> np.ndarray | None:
    """Load and decode the most recently archived frame for `product`
    (BEFORE this cycle's new frame gets saved into the same archive --
    caller must call this first) so compute_optical_flow has something to
    diff the new frame against. Returns None if there's no prior frame yet
    (first-ever run for this product) or if the archived file fails to
    open/decode for any reason -- optical flow is simply skipped for this
    cycle in that case, same as "no prior track" already does elsewhere."""
    existing = sorted(ARCHIVE_DIR.glob(f"{product}_*.gif"))
    if not existing:
        return None
    try:
        img = Image.open(existing[-1])
        img.load()
        lut = build_lut_from_colorbar(img, product)
        return decode_reflectivity(img, product, lut)
    except Exception as e:
        print(f"[flow] {product}: couldn't decode previous archived frame "
              f"{existing[-1]} for optical flow, skipping this cycle: {e}")
        return None


def compute_optical_flow(prev_dbz: np.ndarray, curr_dbz: np.ndarray,
                          product: str) -> np.ndarray | None:
    """Dense optical flow (Farneback) between two consecutive dBZ arrays of
    the SAME radar/product, as a more robust alternative to matching
    discrete segmented cells' centroids across frames (see track_cells'
    docstring for why centroid-matching gets noisier as more radars/cells
    are added). Operating on the whole reflectivity field instead of a
    handful of blob centroids means a storm splitting, merging, or
    reforming between polls no longer produces a spurious "jump" -- the
    flow field just tracks how the pattern as a whole shifted.

    Returns a (H, W, 2) array of per-pixel (dx, dy) pixel displacement, or
    None if the two frames don't even have matching shapes (e.g. a
    resolution/crop change) -- that's treated the same as "no prior frame"
    upstream: flow-based velocity is simply skipped for this cycle."""
    if prev_dbz.shape != curr_dbz.shape:
        print(f"[flow] {product}: shape mismatch {prev_dbz.shape} vs "
              f"{curr_dbz.shape}, skipping optical flow this cycle")
        return None

    vmin, vmax = product_value_range(product)
    span = (vmax - vmin) or 1.0

    def to_gray(dbz: np.ndarray) -> np.ndarray:
        # NaN (no-echo/background) pixels read as the scale's floor value --
        # they still carry real information for flow (clear-air boundaries
        # move too), and this keeps them numerically well-behaved instead of
        # propagating NaN into calcOpticalFlowFarneback.
        filled = np.nan_to_num(dbz, nan=vmin)
        clipped = np.clip(filled, vmin, vmax)
        return (((clipped - vmin) / span) * 255.0).astype(np.uint8)

    prev_gray = to_gray(prev_dbz)
    curr_gray = to_gray(curr_dbz)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=25, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0,
    )
    return flow


def sample_cell_velocity_from_flow(flow: np.ndarray, cell: Cell, dt_minutes: float,
                                    product: str,
                                    sample_radius_px: int = 3) -> tuple[float, float] | None:
    """Read this cell's motion off the optical-flow field at its own pixel
    location, rather than off a centroid difference against a (possibly
    mismatched) cell from the previous cycle. Median over a small
    neighbourhood (not just the single nearest pixel) for robustness against
    per-pixel flow noise -- same spirit as cluster_cells' density-weighted
    centroid, just applied to a vector field instead of a point.

    Returns (speed_kmh, bearing_deg) rounded the same way the old centroid
    based calculation did, or None if dt_minutes isn't usable or the
    resulting speed still fails the MAX_PLAUSIBLE_CELL_SPEED_KMH sanity
    check (kept as a second line of defense here too, same reasoning as
    before -- a bad flow estimate should mean "no cone", never "a cone
    that's obviously wrong")."""
    if dt_minutes <= 0:
        return None
    h, w = flow.shape[:2]
    cy, cx = int(round(cell.py)), int(round(cell.px))
    y0, y1 = max(0, cy - sample_radius_px), min(h, cy + sample_radius_px + 1)
    x0, x1 = max(0, cx - sample_radius_px), min(w, cx + sample_radius_px + 1)
    if y0 >= y1 or x0 >= x1:
        return None
    patch = flow[y0:y1, x0:x1]
    dx = float(np.median(patch[..., 0]))
    dy = float(np.median(patch[..., 1]))

    km_per_px = PRODUCTS[product]["km_per_px"]
    # Same sign convention as pixel_to_latlon: increasing py = moving south,
    # increasing px = moving east.
    km_east = dx / km_per_px
    km_north = -dy / km_per_px
    dist_km = (km_east ** 2 + km_north ** 2) ** 0.5
    speed_kmh = dist_km / (dt_minutes / 60.0)

    if speed_kmh > MAX_PLAUSIBLE_CELL_SPEED_KMH:
        print(f"[flow] {product}: dropping implausible flow-based speed "
              f"{speed_kmh:.0f} km/h for cell {cell.id} -- treating as no "
              f"reliable velocity yet")
        return None

    bearing = np.degrees(np.arctan2(km_east, km_north)) % 360
    return (round(speed_kmh, 1), round(bearing, 0))


def poll_and_decode(product: str, state: dict, prev_cells: dict, prev_obs_time: dict,
                     last_obs_time_seen: dict) -> np.ndarray | None:
    """One polling cycle for `product`, adapted from the notebook's
    poll_one_product() for a stateless-process world: tracking state
    (prev_cells/prev_obs_time) is passed in and mutated in place instead of
    read from module globals, since nothing here persists between runs
    except what's explicitly saved to state/ afterward.

    `last_obs_time_seen` is mutated in place with this product's observation
    time as soon as it's known (whether or not the frame turns out to be
    new) -- it's how build_info_banner_html() finds out what to display,
    via run_pipeline() copying this dict into the module-level global.

    Returns the decoded reflectivity array for the map to draw, or None if
    the fetch itself failed outright (radar feed unreachable this cycle) --
    in that case the CALLER decides whether to fall back to the last
    archived frame (see run_pipeline()) rather than skip the product on the
    map entirely just because one poll had a bad network moment."""
    cfg = PRODUCTS[product]
    raw = fetch_frame(cfg["url"])
    print(f"[health] {product}: {health_check(state, product)}")
    if raw is None:
        return None

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        print(f"[poll] {product}: fetched data isn't a valid image - skipping this cycle: {e}")
        return None

    obs_time = extract_observation_time(img, product)
    last_obs_time_seen[product] = obs_time
    lut = build_lut_from_colorbar(img, product)
    dbz = decode_reflectivity(img, product, lut)

    pstate = state.setdefault(product, {"last_obs_time": None, "last_new_frame_ts": 0,
                                         "last_frame_hash": None})
    # Whether raw BYTES changed since the last poll -- not whether obs_time
    # changed. obs_time can be None (Karaikal's OCR/template read fails
    # noticeably more often than NIOT's, see extract_observation_time) on a
    # perfectly ordinary re-served, UNCHANGED frame just as easily as on a
    # genuinely new one, so it's not a safe signal for "is this new" on its
    # own. The previous check (`obs_time is None` counted as automatically
    # "new") fed track_cells a near-zero dt_minutes whenever two polls
    # landed close together in wall-clock time with a Karaikal OCR miss in
    # between -- e.g. the external cron-job.org pinger and GitHub's own
    # (less reliable, kept as backup) schedule trigger occasionally firing
    # only a couple of minutes apart -- which is exactly what produced a
    # reported "1564 km/h" storm cell: not a real storm, a same-frame
    # centroid nudge divided by an almost-zero dt. Content hash sidesteps
    # needing obs_time for this decision at all.
    frame_hash = hashlib.sha256(raw).hexdigest()
    is_new_frame = frame_hash != pstate.get("last_frame_hash")

    if not is_new_frame:
        print(f"[poll] {product}: no new frame yet (obs_time={obs_time})")
        return dbz  # still return it -- same content as last time, fine to (re)draw

    # Must happen BEFORE this cycle's new frame is archived below -- this is
    # looking for whatever was already the most recent archived frame going
    # into this cycle, i.e. the "previous" half of the optical-flow pair.
    prev_dbz_for_flow = _load_previous_dbz_for_flow(product)

    now = time.time()
    pstate["last_new_frame_ts"] = now
    pstate["last_obs_time"] = obs_time.isoformat() if obs_time else None
    pstate["last_frame_hash"] = frame_hash
    tag = obs_time.strftime("%Y%m%dT%H%M%SZ") if obs_time else int(now)
    fname = ARCHIVE_DIR / f"{product}_{tag}.gif"
    fname.write_bytes(raw)
    print(f"[poll] {product}: new frame saved: {fname} (obs_time={obs_time})")
    prune_archive(product)

    live_elev = extract_elevation(img, product)
    elevation_deg = live_elev if live_elev is not None else cfg["elevation_deg"]
    new_cells = extract_cells(dbz, product, elevation_deg)
    new_cells = cluster_cells(new_cells)

    # Karaikal's OCR-based obs_time read fails noticeably more often than
    # NIOT's (see extract_observation_time's docstring) -- when it does,
    # obs_time is None here. Requiring a *real* obs_time to compute
    # dt_minutes silently forced dt_minutes to 0 on every OCR miss, which
    # made track_cells report every matched cell's speed as exactly 0 km/h
    # (velocity_kmh = dist / (dt/60), and the "dt <= 0 -> speed 0" branch
    # in track_cells), which in turn made build_forecast_map's
    # min_speed_kmh filter drop every single Karaikal cell's forecast cone
    # -- even for storms that were genuinely moving. Falling back to
    # poll-receipt time (now) here, same as the archive-filename tag
    # already does a few lines up, keeps dt_minutes close to the real ~15
    # min cron cadence instead of collapsing to zero. This is only the
    # basis for the VELOCITY math; last_obs_time_seen (the banner) still
    # reports the honest OCR result, None included, so the displayed
    # timestamp is never silently fudged.
    obs_time_effective = obs_time or datetime.now(timezone.utc)
    dt_minutes = ((obs_time_effective - prev_obs_time[product]).total_seconds() / 60.0
                  if prev_obs_time[product] else 0)

    # Motion first, via optical flow over the whole reflectivity field
    # (falls back to leaving velocity_kmh at its dataclass default of None
    # when there's no usable previous frame yet -- e.g. the very first run
    # for this product, or a shape mismatch) -- then matching/trend-only
    # tracking. See track_cells' docstring for why these are now split.
    flow = (compute_optical_flow(prev_dbz_for_flow, dbz, product)
            if prev_dbz_for_flow is not None and dt_minutes > 0 else None)
    if flow is not None:
        for c in new_cells:
            c.velocity_kmh = sample_cell_velocity_from_flow(flow, c, dt_minutes, product)

    tracked = track_cells(prev_cells[product], new_cells, dt_minutes)

    prev_cells[product] = tracked
    prev_obs_time[product] = obs_time_effective
    return dbz


def run_pipeline() -> None:
    """One full cycle: poll NIOT MAXZ + Karaikal MAXZ, update on-disk
    tracking state, build storm_forecast_map.html. Meant to be invoked
    once per GitHub Actions run (see .github/workflows/nowcast.yml) -- the
    15-minute cadence comes from the workflow's cron schedule, not from
    anything in this script looping."""
    global _last_dbz, _prev_cells, _last_obs_time_seen

    state = load_state()
    prev_cells = load_prev_cells()
    prev_obs_time = load_prev_obs_time()

    _last_dbz = {}
    last_obs_time_seen: dict[str, "datetime | None"] = {p: None for p in POLLED_PRODUCTS}
    for product in POLLED_PRODUCTS:
        dbz = poll_and_decode(product, state, prev_cells, prev_obs_time, last_obs_time_seen)
        if dbz is not None:
            _last_dbz[product] = dbz
        else:
            # Fetch failed outright this cycle -- fall back to the most
            # recently archived frame so the map isn't just missing a
            # radar because of one bad network moment. Re-decodes it
            # fresh rather than trusting a cached array from a run that no
            # longer exists.
            existing = sorted(ARCHIVE_DIR.glob(f"{product}_*.gif"))
            if existing:
                print(f"[poll] {product}: fetch failed, falling back to last archived frame {existing[-1]}")
                img = Image.open(existing[-1])
                lut = build_lut_from_colorbar(img, product)
                _last_dbz[product] = decode_reflectivity(img, product, lut)
                last_obs_time_seen[product] = _obs_time_from_archive_filename(existing[-1])
            else:
                print(f"[poll] {product}: fetch failed and no archived frame to fall back to -- "
                      f"this product will be missing from this cycle's map")

    _prev_cells = prev_cells
    _last_obs_time_seen = last_obs_time_seen

    save_state(state)
    save_prev_cells(prev_cells)
    save_prev_obs_time(prev_obs_time)

    for product, cells in prev_cells.items():
        for c in cells:
            tag = "SURFACE-CONFIRMED" if c.confirmed_at_surface else ""
            print(f"  {product} cell {c.id}: {c.centroid_latlon}, {c.max_dbz:.0f} dBZ, "
                  f"trend={c.trend}, velocity={c.velocity_kmh} {tag}")

    build_forecast_map(products=POLLED_PRODUCTS, fuse=True, out_html=str(OUTPUT_HTML))
    print(f"Wrote {OUTPUT_HTML}")


if __name__ == "__main__":
    run_pipeline()
