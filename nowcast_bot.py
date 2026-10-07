# === CELL 2 (imports) ===
import base64
import hashlib
import io
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
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

    # --- Fourth radar: IMD's own Chennai DWR (S-band), caz_cni.gif -- came
    # back online 2026-09-30 after being off-air; same "Max with panels"
    # layout + printed lat/lon axis as Kochi's, so calibrated the same
    # measured-not-guessed way: vertical gridline columns at the printed
    # 78/79/80/81/82 degE labels and horizontal gridline rows at the
    # printed 12/13/14/15 degN labels were found via a brightness-peak scan
    # over the panel (clear of land/text clutter), giving an affine
    # pixel<->degree fit on each axis. That fit's degree-per-pixel spacing
    # converts to 1.0023 km/px (lon axis) and 1.0000 km/px (lat axis) --
    # both independently matching the frame's own printed "Hor Res: 1.000
    # km/pixel" almost exactly, a strong cross-check that this fit is
    # right. The radar's own site marker (a small bright crosshair dot
    # immediately left of the printed "CHN" label) was then found directly
    # in pixel space and run back through that same fit to get
    # site_lat/site_lon below -- same method as Kochi's, not a guessed town
    # center. Lands about 7km west of central Chennai, inland from the
    # coast -- plausible for a DWR siting, but still worth a sanity check
    # against the merged map once a few real cycles have run.
    "chennai": {
        "site_lat": 13.0901,
        "site_lon": 80.2067,
        "site_elev_m": 6.0,        # UNVERIFIED — Chennai-area average-elevation guess
        "band": "S",               # per direct report -- S-band, unlike NIOT/Karaikal/Kochi
    },
}

PRODUCT_RADAR = {
    "ppi": "niot", "maxz": "niot", "ppz": "niot",
    "kkl_ppi": "karaikal", "kkl_ppz": "karaikal", "kkl_maxz": "karaikal",
    "koc_maxz": "kochi",
    "cni_maxz": "chennai", "cni_ppz": "chennai",
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
        # When shown alongside kkl_maxz (the higher-resolution 250km
        # Karaikal product -- see test_500km_pipeline.py), this product's
        # own pixels out to 250km are redundant AND lower-resolution than
        # kkl_maxz's -- ceding that inner disc to kkl_maxz and only
        # showing this product's genuinely EXTRA coverage (250-500km, the
        # ring kkl_maxz's own image simply doesn't reach) gives the best
        # of both: full resolution close in, real extended range further
        # out, instead of the coarser PPZ image overwriting/competing with
        # the sharper MAXZ one in the region they both cover. See
        # decode_reflectivity's use of this field. No effect when kkl_ppz
        # is shown on its own (e.g. without kkl_maxz alongside it).
        "mask_within_km": 250.0,
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

    # --- Chennai DWR (see calibration-status comment above RADAR_SITES) ---
    "cni_maxz": {
        "url": "https://mausam.imd.gov.in/Radar/caz_cni.gif",
        # Drop colour-bar swatches only 1 row tall: on some frames the bar's
        # anti-aliasing leaves a stray khaki row that matches terrain shading
        # (Eastern Ghats) and was decoded as ~40 dBZ (false alert 7 Oct 01:51 IST).
        "min_swatch_rows": 2,
        "role": "aloft_early_warning",
        "range_km": 250.0,         # printed directly in the frame's own metadata panel ("Range: 250 km")
        "elevation_deg": None,     # column-max, not a single tilt -- same as every other MAXZ product here
        "image_size": (919, 700),
        # Same "Max with panels" layout as Kochi (cross-section strips top
        # + right, plan-view confined to bottom-left, full lat/lon
        # gridlines printed on the panel) -- site_px/km_per_px measured
        # from THIS frame's own gridlines + site marker, see the
        # calibration-status comment above RADAR_SITES.
        "site_px": (240.0, 447.0),
        "km_per_px": 1.000,        # frame's own metadata prints "Hor Res: 1.000 km/pixel" -- confirmed independently via the gridline fit on both axes
        "colorbar_bbox": (712, 74, 785, 424),
        "value_at_top": 60.0,      # same uniform 2.5dB/band MAX(dBZ) scale as every other MAXZ/PPZ product
        "value_at_bottom": 20.0,
        "timestamp_bbox": (630, 28, 919, 50),
        "elevation_bbox": None,    # no elevation line -- column-max product
        "plot_bbox": (0, 198, 499, 700),
        "cell_dbz_threshold": 35,  # matches every other MAXZ product here
        # Deliberately NOT hand-boxing the printed "NN.N km" range-ring
        # labels or the lat/lon gridlines the way kkl_maxz/koc_maxz do --
        # unlike Kochi's WHITE gridlines (which land within DIST_THRESHOLD
        # of this scale's pale near-white 35-42dBZ band, see koc_maxz's own
        # comment), this frame's gridlines/text render in near-BLACK, and
        # DIST_THRESHOLD=2 is tight enough that near-black isn't within
        # range of any of this LUT's saturated red/orange/yellow/blue
        # swatches (checked directly against this frame's own colorbar
        # samples). extract_cells' existing out-of-range check (rng_km >
        # range_km*1.05) is also still live as backstop insurance either
        # way. If real cycles turn up a text/gridline false "cell" anyway,
        # build a static_exclude_mask the same way koc_maxz's was (diff a
        # batch of real archived frames for persistently-colliding pixels)
        # -- not possible yet from a single still frame.
        # Three small static collisions found once the open-water texture
        # fix (exclude_colors, below) uncovered what was left: the
        # national-emblem logo every frame draws in the same bottom-right
        # corner (its own colors include reds/blues that land on real LUT
        # swatches), plus two small station-code text labels whose
        # anti-aliased strokes happened to do the same. Found by direct
        # inspection of this one real frame -- unlike the gridlines/km
        # labels (left unboxed, see below), these ARE fixed-position
        # static graphics, so a position box is the right tool here.
        "label_exclude_boxes": [
            (430, 618, 484, 686),   # national emblem, bottom-right corner
            (0, 460, 14, 490),      # station-code label text, left edge
            (78, 288, 88, 298),     # station-code label text, near "RCT"
        ],
        # This radar came back online right before this was added, and the
        # very first live frame showed a lot of scattered single/few-pixel
        # false echo across open water -- isolated points/dashes rather
        # than the coherent filled patches real reflectivity forms,
        # consistent with residual ground/sea clutter or AP the printed
        # "Clutter Filter: IIRDoppler 7" hasn't fully suppressed yet. These
        # specks were already too small to ever pass extract_cells' own
        # MIN_CELL_ABSOLUTE_PIXELS floor for cell DETECTION -- but that
        # floor does nothing for the raster OVERLAY image, which draws
        # every surviving pixel directly (see decode_reflectivity's
        # despeckle_min_px handling). 8px is intentionally a bit above
        # MIN_CELL_ABSOLUTE_PIXELS' own 5px floor given how dense this
        # radar's speckle was on first look -- worth tightening or loosening
        # once a few real cycles are in.
        "despeckle_min_px": 8,
        # The REAL driver of "a lot of fake echoes" turned out to be much
        # bigger than scattered speckle: this radar's open-water basemap
        # fill is itself dithered across several light-blue shades (a
        # watercolor-ish texture, not a flat fill), and one of those
        # shades -- (153,204,255) -- lands close enough to this scale's
        # real 35.0dBZ swatch to read as moderate rain across almost the
        # ENTIRE visible sea surface (confirmed directly: a 35+dBZ mask
        # over a verified-clean open-water patch covered the whole
        # patch, concentrated overwhelmingly in this one exact RGB
        # value). Despeckling alone can't fix this -- it's not isolated
        # points, it's a near-full-coverage texture, indistinguishable
        # from real echo by component size alone. Measured the 8 distinct
        # shades actually in use across several verified-clean open-water
        # patches (collectively >93% of all pixels sampled there) and
        # exclude exactly those -- not a position-based mask (koc_maxz's
        # style), since this texture tiles the whole water surface rather
        # than sitting at fixed coordinates. Deliberately leaving the
        # darkest two near-matches ((0,0,153)/(0,0,204), which are this
        # scale's own real 20.0/22.5dBZ swatches) OUT of this list -- at
        # low sample counts in the clean patches, those read more like
        # gridline anti-aliasing over water than general texture, and
        # excluding a product's own real lowest-band colors outright
        # risks silently zeroing out genuine weak echo there instead.
        # Worth revisiting with more real frames; this is a first pass.
        "exclude_colors": [
            (102, 204, 255), (153, 204, 255), (102, 153, 255), (153, 153, 255),
            (102, 153, 204), (102, 204, 204), (153, 153, 204), (153, 204, 204),
        ],
    },

    # Chennai's own 500km-class extended-range product (printed "Range: 600
    # km", actually better reach than Karaikal's kkl_ppz) -- a single-tilt
    # PPI (elevation -0.2deg, printed directly), not a column-max product,
    # unlike cni_maxz. Same hybrid intent as kkl_maxz+kkl_ppz: this product
    # cedes its own inner 250km disc to cni_maxz's higher resolution via
    # mask_within_km, contributing only the 250-600km ring cni_maxz's own
    # image can't reach.
    "cni_ppz": {
        "url": "https://mausam.imd.gov.in/Radar/ppz_cni.gif",
        # Same single-row colour-bar swatch fix as cni_maxz (see there): stray
        # khaki rows matched terrain shading and gave ~200 false 35-40 dBZ px
        # on about 20 archived frames since 1 Oct.
        "min_swatch_rows": 2,
        "role": "regional_early_warning",
        "range_km": 600.0,         # printed directly ("Range: 600 km") -- genuinely more reach than Karaikal's 500km kkl_ppz
        "elevation_deg": -0.2,     # printed directly ("Elevation: -0.2 deg") -- a real single low-angle tilt, not a column-max product
        "image_size": (1014, 800),
        # Full lat/lon-gridded single-panel layout (like cni_maxz/koc_maxz,
        # but no cross-section strips -- this is one plan-view panel filling
        # most of the frame, legend down the right side). site_px measured
        # the same way as cni_maxz: fit this frame's OWN lat/lon gridlines
        # (75-83 degE across the top, 8-18 degN down the left), then used
        # that fit to locate RADAR_SITES["chennai"]'s already-established
        # site_lat/site_lon in THIS image's pixel space -- same physical
        # radar as cni_maxz, so the real-world coordinate has to be
        # identical; cross-checked against the frame's own (noisy, overlaid
        # by coastline/labels right at the site) crosshair marker and landed
        # within a few px, consistent with each other.
        "site_px": (392.6, 395.3),
        "km_per_px": 1.500,        # frame's own metadata prints "Resolution: 1.500 km/pixel" -- confirmed independently via the gridline fit on both axes (within ~1.5%)
        "colorbar_bbox": (820, 74, 884, 525),
        "value_at_top": 60.0,      # same uniform 2.5dB/band MAX(dBZ) scale as every other product here
        "value_at_bottom": 20.0,
        "timestamp_bbox": (780, 27, 1014, 50),
        "elevation_bbox": (780, 690, 1014, 702),
        "plot_bbox": (0, 0, 799, 800),
        "cell_dbz_threshold": 30,  # matches kkl_ppz -- the other long-range PPZ-style product
        # Ceding its own inner 250km disc to cni_maxz, exactly like
        # kkl_ppz ceding to kkl_maxz -- see that field's comment on
        # kkl_ppz above for the full reasoning. Labeled "Chennai Extended
        # Radar" on the page (PRODUCT_STYLE), same naming pattern as
        # Karaikal's masked product.
        "mask_within_km": 250.0,
        # Same open-water texture-collision risk as cni_maxz (same vendor
        # software/palette, same symptom expected) -- starting from
        # cni_maxz's own measured exclude_colors/despeckle_min_px as a
        # first pass. Verified directly against this product's own real
        # frame: with these in place, decode_reflectivity() dropped from
        # a sea-wide false-echo field down to 207 finite px out of 639,200
        # (and 10 spurious "cells"), ALL of them clustered at
        # x:733-779, y:723-783 in plot_bbox-relative coords -- sampling
        # those exact pixels' RGB (pale yellow/white/pale green/pale pink)
        # and visually inspecting that crop confirmed it's the IMD
        # national emblem (Ashoka lion seal + orange/green ribbon),
        # printed over open water just south-east of the site, same
        # confound cni_maxz needed its own label_exclude_boxes entry for.
        # No genuine echo was lost to this -- the whole affected region is
        # a static graphic, not sea surface.
        "despeckle_min_px": 8,
        "exclude_colors": [
            (102, 204, 255), (153, 204, 255), (102, 153, 255), (153, 153, 255),
            (102, 153, 204), (102, 204, 204), (153, 153, 204), (153, 204, 204),
        ],
        "label_exclude_boxes": [
            (705, 685, 799, 800),   # national emblem/seal, bottom-right of the plot panel
        ],
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

# A tab left open and VISIBLE but genuinely unattended (monitor left on,
# browser forgotten in a window nobody's looked at) still shouldn't poll
# the server forever -- see build_autorefresh_script()'s visibility+idle
# handling below. 60 min is deliberately generous: this only stops an
# abandoned tab from refreshing, never a tab someone's actually glanced
# at within the last hour.
AUTOREFRESH_IDLE_TIMEOUT_MINUTES = 60

# How old a product's displayed observation time has to be before
# build_info_banner_html() flags it as stale instead of showing it with
# plain styling -- see that function for why this matters (a reader can't
# otherwise tell "this is live" from "this radar's feed has been stuck for
# hours" just by glancing at the banner). Comfortably above the ~15 min
# poll cadence to allow for IMD's own ordinary publishing lag without
# false-flagging a normal reading.
STALE_OBS_MINUTES = 60

# Direction arrows are only reliable where the underlying imagery is full
# resolution. Karaikal cells detected out on the 250-500km extended ring
# (kkl_ppz, coarser pixels — see mask_within_km on that PRODUCTS entry)
# still get a storm marker and tooltip, just no arrow: a bearing computed
# off a coarse-pixel centroid track is noisier than it looks once drawn as
# a confident-looking arrow. Keyed by radar name (RADAR_SITES/PRODUCT_RADAR),
# not product, since a cell's product can be either of Karaikal's two.
# Radars not listed here (niot, kochi) are unrestricted — this is currently
# a Karaikal-only concern.
ARROW_MAX_RANGE_KM = {"karaikal": 250.0, "chennai": 250.0}

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
# confuse it) for any glyph the template library doesn't recognise, and if
# that also can't confidently name a glyph, extraction is abandoned for
# that frame -- same safe fall-back to poll-receipt time as every other
# failure mode here.
#
# The library was rebuilt on 2026-10-02 from all 199 timestamped Karaikal
# frames in this repo's history (kkl_maxz + kkl_ppz, 30 Sep - 2 Oct). The
# first one (54 glyphs from a handful of frames) had no 7/8/9 at all, and
# its only "0" -- like several other entries -- carried a stray mark
# picked up from the text line above (see _kkl_char_groups), so it only
# matched a 0 in the minutes position: every frame with an hour of 00-09
# UTC, or any 7/8/9, failed here and depended on whole-string Tesseract
# reading the DATE line correctly too. Measured over those 199 frames:
# 65 read by templates before, 199 after. With the stray mark removed the
# font turns out to be fully deterministic -- 13 distinct bitmaps cover
# every character ever seen (one each for 1-9, ":" and "Z", two for "0").
_KKL_TIMESTAMP_TEMPLATES: dict[str, list[np.ndarray]] | None = None
_KKL_TEMPLATE_SIZE = (40, 60)  # (w, h), matches assets/kkl_timestamp_templates.npz
# Geometry of the 5x-upscaled time line _kkl_char_groups works on: digit
# ink always spans rows 25-109, and the widest single character is 85 px.
_KKL_TOP_STRIP_PX = 20    # ink ending above this row is not part of any character
_KKL_MAX_CHAR_W_PX = 100  # two characters side by side are 160+ px wide


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
    wider -- validated against every archived calibration frame.

    Two exceptions to that last sentence, both found once a few days of
    real frames had accumulated, and both fixed here:

    - Two small marks from the text line above poke into the top of this
      crop at fixed x-positions, directly over the two MINUTES digits.
      Merging purely by x-gap glued each mark onto the digit beneath it,
      which stretched that glyph's box up to row 0 -- so the same digit
      looked different in the minutes position than anywhere else, and a
      template taken from one position didn't match the other. Ink lying
      wholly inside the top strip is now dropped before merging.
    - "4" is wide enough that the gap to the next character is exactly
      merge_gap, so "4Z" (any time ending in 4 seconds) merged into one
      box and the line came out as 8 characters instead of 9. A merge is
      now refused if the result would be wider than one character."""
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
        if y1 < _KKL_TOP_STRIP_PX:
            continue
        boxes.append([x0, x1, y0, y1])
    boxes.sort(key=lambda b: b[0])
    merged: list[list[int]] = []
    for b in boxes:
        if (merged and b[0] - merged[-1][1] <= merge_gap
                and b[1] - merged[-1][0] < _KKL_MAX_CHAR_W_PX):
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
        if hms is None:
            # REPORTED AGAIN, this time for kkl_ppz specifically (production,
            # right after it was promoted): its banner showed the Extended
            # Radar as stale/24h-old while the live IMD frame was clearly
            # current. Traced to exactly the failure mode this whole
            # template-matching path was built to avoid (see this
            # function's main docstring, "10 days stale" incident): when
            # extract_kkl_time_via_templates can't confidently read a
            # frame, falling through to the generic OCR path below means
            # reading this same unreliable bold font's DATE line via plain
            # OCR, and that path's sanity check only rejects a result more
            # than 3 DAYS from "now" -- far too loose to catch a single
            # misread digit costing exactly one day (e.g. "03"->"02"),
            # which is what happened here: the time-of-day kept reading
            # correctly every cycle (proving frames WERE arriving and
            # updating), but the date stuck one day behind for hours,
            # making a perfectly live frame look stale on the map. Template
            # matching apparently fails on kkl_ppz's frames often enough
            # for this to matter in practice, not just in theory. Fix:
            # never fall through to that generic date-line OCR for ANY
            # Karaikal product -- return None here instead (same as any
            # other extraction miss), which shows as an honest "time
            # unavailable" in the banner and is treated as "no data yet"
            # rather than stale, instead of a confident, silently wrong
            # date that reads as real staleness.
            return None
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
        # \D{0,3} before the day: Tesseract occasionally inserts a stray
        # character there -- "03:17:44 UTC / O02 Oct 2026" on a perfectly
        # clean NIOT frame (2026-10-02) -- which made the whole timestamp
        # unreadable even though every real character was read correctly.
        # A stray that REPLACES a digit instead still fails to parse, or
        # lands days away and is rejected by the sanity check below.
        m = re.search(r"(\d{2}:\d{2}:\d{2})\s*UTC\s*/\D{0,3}(\d{1,2}\s+\w{3}\s+\d{4})", text)
        if m:
            date_str, time_str, date_fmt = m.group(2), m.group(1), "%d %b %Y"
        else:
            m = re.search(r"(\d{2}:\d{2}:\d{2})\s*Z?.*?(\d{1,2}\s+\w{3}\s+\d{4})\s*UTC",
                           text, re.DOTALL)
            if m:
                date_str, time_str, date_fmt = m.group(2), m.group(1), "%d %b %Y"
            else:
                # Chennai's layout: a fourth one, "HH:MM / DD-Mon-YYYY" --
                # no seconds field and no explicit "UTC" suffix anywhere
                # near it (unlike every other layout here, which all print
                # UTC explicitly). Assuming UTC anyway for consistency with
                # every other IMD radar product polled here -- worth a
                # direct sanity check against the merged map's banner once
                # real frames are flowing, since this is the one layout
                # where that assumption isn't confirmed by the frame's own
                # printed text.
                m = re.search(r"(\d{1,2}:\d{2})\s*/\s*(\d{1,2}-\w{3}-\d{4})", text)
                if not m:
                    return None
                time_str, date_str, date_fmt, time_fmt = m.group(1), m.group(2), "%d-%b-%Y", "%H:%M"
                try:
                    dt = datetime.strptime(f"{date_str} {time_str}", f"{date_fmt} {time_fmt}")
                except ValueError:
                    return None
                dt = dt.replace(tzinfo=timezone.utc)
                if abs((dt - datetime.now(timezone.utc)).total_seconds()) > 3 * 86400:
                    return None
                return dt

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
    min_rows = cfg.get("min_swatch_rows")
    if min_rows:
        from collections import Counter
        cnt = Counter(c for c, _ in lut)
        kept = [(c, v) for c, v in lut if cnt[c] >= min_rows]
        if kept:
            lut = kept
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

    exclude_colors = cfg.get("exclude_colors")
    if exclude_colors:
        # Known-background colors that must NEVER register as echo,
        # regardless of what the nearest-neighbour LUT match says --
        # distinct from DIST_THRESHOLD's generic "too far from any swatch
        # to trust" cutoff, this is for colors that are themselves WITHIN
        # DIST_THRESHOLD of a real swatch (so the generic cutoff can't
        # catch them) but are known, by direct inspection of a real clean
        # frame, to be the basemap's own fill/texture rather than echo.
        # First needed for cni_maxz: its open-water texture dithers
        # between several light-blue shades to render a plain, echo-free
        # sea surface, and one of those shades (153,204,255) sits close
        # enough to this scale's real 35.0dBZ swatch to get matched as
        # moderate rain across almost the entire visible ocean (see that
        # PRODUCTS entry's comment) -- a scale/overlap problem the generic
        # distance cutoff has no way to distinguish from genuine echo,
        # since by construction it search for the CLOSEST real swatch.
        # Encode each flat pixel as one int (like a packed RGB888 value)
        # for a fast vectorized membership test instead of a per-pixel loop.
        packed = (flat[:, 0].astype(np.int64) * 65536
                  + flat[:, 1].astype(np.int64) * 256
                  + flat[:, 2].astype(np.int64))
        exclude_packed = np.array([r * 65536 + g * 256 + b for (r, g, b) in exclude_colors])
        dbz_flat[np.isin(packed, exclude_packed)] = np.nan

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

    mask_within_km = cfg.get("mask_within_km")
    if mask_within_km is not None:
        # See PRODUCTS[...]["mask_within_km"]'s comment -- cedes this
        # product's inner disc to a higher-resolution product covering the
        # same ground (e.g. kkl_ppz ceding 0-250km to kkl_maxz). site_px is
        # full-image pixel coordinates (same frame pixel_to_latlon uses),
        # so it's offset back by plot_bbox's own origin (l, t) to land in
        # THIS array's local (row, col) space before measuring distance in
        # pixels and converting to km via km_per_px -- exactly the same
        # site-relative geometry pixel_to_latlon uses, just without the
        # extra round trip through lat/lon since this is a plain circular
        # cutoff, not a shape that needs the real map projection.
        site_x, site_y = cfg["site_px"]
        km_per_px = cfg["km_per_px"]
        yy, xx = np.mgrid[0:h, 0:w]
        dist_km = np.hypot(xx - (site_x - l), yy - (site_y - t)) / km_per_px
        dbz[dist_km < mask_within_km] = np.nan

    despeckle_min_px = cfg.get("despeckle_min_px")
    if despeckle_min_px is not None:
        # Isolated-pixel/few-pixel noise -- distinct from extract_cells'
        # own MIN_CELL_ABSOLUTE_PIXELS floor, which only ever protects the
        # CELL-DETECTION path (storm markers/arrows/forecasts); this array
        # is ALSO drawn directly as the raster overlay image via
        # dbz_array_to_png(), completely independent of extract_cells, so a
        # product whose clutter rejection leaves a lot of scattered single-
        # pixel false echo (reported directly for cni_maxz right after that
        # radar came back online -- see its PRODUCTS entry) still painted
        # that speckle all over the visible map even though every one of
        # those specks was already too small to ever become a tracked
        # cell. Opt-in (unset means exactly today's behavior) and generic,
        # same pattern as mask_within_km -- a no-op for every product that
        # doesn't set it.
        noise_mask = ~np.isnan(dbz)
        labeled, n = ndimage.label(noise_mask, structure=np.ones((3, 3)))
        if n > 0:
            sizes = ndimage.sum(noise_mask, labeled, index=np.arange(1, n + 1))
            small_labels = np.flatnonzero(sizes < despeckle_min_px) + 1
            if len(small_labels):
                dbz[np.isin(labeled, small_labels)] = np.nan

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
# Same palette and 5-dBZ bands as the Telegram/social alert maps
# (social_alerts.render_image), so the radar page and the alerts read
# alike and the echoes stand out on the basemap. Bands start at 20 dBZ;
# anything weaker is not drawn (see dbz_array_to_png).
DBZ_BAND_COLORS = [
    "#a8e6a1",  # 20-25  light green
    "#5fcf6a",  # 25-30  green
    "#f4e04d",  # 30-35  yellow
    "#f7a936",  # 35-40  orange
    "#ee6a2e",  # 40-45  deep orange
    "#d62828",  # 45-50  red
    "#a4133c",  # 50-55  crimson
    "#7b2cbf",  # 55+    purple
]
DBZ_BAND_BOUNDARIES = [20, 25, 30, 35, 40, 45, 50, 55, 80]
DBZ_DRAW_MIN = 20.0


def dbz_colormap_and_norm(product: str) -> tuple[mcolors.ListedColormap, mcolors.BoundaryNorm, list]:
    """The shared banded scale (identical for every product): fixed 5 dBZ
    bands from 20 dBZ up, matching the alert maps. Returns (cmap, norm,
    boundaries); the legend uses boundaries for its swatch labels."""
    cmap = mcolors.ListedColormap(DBZ_BAND_COLORS)
    norm = mcolors.BoundaryNorm(DBZ_BAND_BOUNDARIES, cmap.N, clip=True)
    return cmap, norm, list(DBZ_BAND_BOUNDARIES)


PRODUCT_STYLE = {
    "ppi":  {"color": "#d62728", "label": "NIOT PPI - surface confirmed"},
    "maxz": {"color": "#ff7f0e", "label": "NIOT MAXZ - aloft / building"},
    "ppz":  {"color": "#1f77b4", "label": "NIOT PPZ - regional awareness"},
    "kkl_ppi": {"color": "#9467bd", "label": "Karaikal PPI - surface confirmed"},
    # "Karaikal Extended Radar" -- named for what a viewer actually sees:
    # with mask_within_km set on this product (see PRODUCTS[...]), it only
    # ever renders the 250-500km ring kkl_maxz's own image can't reach,
    # never the inner disc, so "extended" describes the visible result
    # whenever it's shown alongside kkl_maxz (the normal case).
    "kkl_ppz": {"color": "#17becf", "label": "Karaikal Extended Radar"},
    "kkl_maxz": {"color": "#8c564b", "label": "Karaikal MAXZ - aloft / building"},
    "koc_maxz": {"color": "#2ca02c", "label": "Kochi MAXZ - aloft / building"},
    "cni_maxz": {"color": "#e377c2", "label": "Chennai DWR MAXZ - aloft / building"},
    "cni_ppz": {"color": "#7f7f7f", "label": "Chennai Extended Radar"},
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

    alpha = np.where(np.nan_to_num(values, nan=-999.0) < DBZ_DRAW_MIN, 0.0, alpha)  # weaker than the first band: not drawn
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

    Fix (v1, now superseded for the drop-entirely case -- see v2 below):
    in any patch of ground covered by more than one radar currently being
    drawn, where the FRESHER radar actually has real reflectivity data at
    that exact spot, drop (NaN out -> fully transparent, same convention
    dbz_array_to_png already uses for "no data") the staler radar's pixel
    there. Each radar's raster is left completely untouched outside an
    overlap -- its own unique-coverage area always renders exactly as
    before.

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

    v2 -- MAX-BLEND instead of drop, reported directly from a real overlap
    zone (near Madurai, between Kochi/koc_maxz and Karaikal's extended
    kkl_ppz): dropping the staler pixel outright assumed the fresher
    radar's OWN separately-rendered raster would visibly cover that same
    ground instead, but the two products don't share a pixel grid (very
    different resolution/projection -- kkl_ppz's coarse long-range pixels
    vs. koc_maxz's finer ones), so "fresher has SOME data nearby" and
    "fresher's own image is actually opaque at this exact geographic
    point" are not the same thing. A staler pixel got dropped to fully
    transparent purely because a fresher radar had even a faint, barely-
    above-threshold reading somewhere in the same reprojected cell, while
    that fresher radar's own independently-smoothed image could still be
    faded/transparent at that precise spot -- net result, a hole where a
    real, often STRONGER echo had been visible a moment ago. Confirmed
    directly (in the 500km test repo): a widespread storm straddling the
    overlap boundary showed a checkerboard of real data and blank squares
    along exactly that seam. Ported here once that fix had run clean in
    the test repo for several real days -- this repo had been left on the
    v1 drop behavior since the cell-marker-styling carryover, which was
    explicitly noted as out of scope for that change at the time. Now
    takes the ELEMENTWISE MAX of this pixel's own value and the fresher
    radar's value at the same reprojected spot instead of discarding this
    one -- a location keeps whichever radar's reading is stronger rather
    than ever going fully blank over a disagreement, at the cost of
    occasionally keeping a slightly-staler but stronger reading instead of
    showing "freshest, no matter how much weaker". Still only ever touches
    pixels where THIS product already detected real echo (the
    np.where(~np.isnan(arr)) scan below) -- doesn't paint in new data at
    spots this radar saw nothing at all.

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

        # Tracks the strongest fresher-radar value seen at each of this
        # product's own pixels (NaN = no fresher radar had data there) --
        # blended in with np.fmax against this pixel's own value below,
        # rather than dropped, so an overlap never wipes a pixel to fully
        # transparent (see v2 comment above).
        fresher_value = np.full(len(xs), np.nan)
        for other_product in products:
            # Compared against the ORIGINAL, pre-masking rasters
            # (dbz_by_product, not `masked`) so this doesn't depend on
            # what order `products` happens to process in -- with 3
            # radars shown together, an earlier product in this loop may
            # already have had some of ITS pixels already blended with a
            # third, even-fresher radar, and that shouldn't change what
            # counts as "real data" when checking a later product here.
            if other_product not in dbz_by_product or other_product == product:
                continue
            if PRODUCT_RADAR[other_product] == PRODUCT_RADAR[product]:
                continue  # same radar, e.g. two products off one site -- not an overlap case
            if _effective_obs_time(other_product) <= this_time:
                continue  # other isn't strictly fresher -- doesn't get to blend into this one

            other_arr = dbz_by_product[other_product]
            other_ox = PRODUCTS[other_product]["plot_bbox"][0]
            other_oy = PRODUCTS[other_product]["plot_bbox"][1]
            other_px, other_py = latlon_to_pixel(lat, lon, other_product)
            other_x = np.round(other_px - other_ox).astype(int)
            other_y = np.round(other_py - other_oy).astype(int)
            in_bounds = ((other_x >= 0) & (other_x < other_arr.shape[1]) &
                         (other_y >= 0) & (other_y < other_arr.shape[0]))
            idx = np.where(in_bounds)[0]
            other_vals = other_arr[other_y[idx], other_x[idx]]
            # fmax (not max) treats NaN as "no opinion" rather than
            # propagating it -- a pixel this product already blended a
            # value in for from one fresher radar must survive untouched
            # when a second fresher radar simply doesn't reach that spot.
            fresher_value[idx] = np.fmax(fresher_value[idx], other_vals)

        have_fresher = ~np.isnan(fresher_value)
        if have_fresher.any():
            arr[ys[have_fresher], xs[have_fresher]] = np.fmax(
                arr[ys[have_fresher], xs[have_fresher]], fresher_value[have_fresher])
    return masked


def _collapsible_panel_toggle_js(body_id: str, chevron_id: str) -> str:
    """Inline onclick handler (NOT a <script> tag -- see
    build_autorefresh_script's docstring for why a literal nested
    <script>...</script> inside folium's one shared script block is
    dangerous; an onclick attribute carries no such risk) shared by every
    floating panel that collapses to a compact header on narrow screens.

    Reads the body's CURRENT rendered state via getComputedStyle rather
    than tracking open/closed in a separate JS variable -- that way one
    piece of logic correctly handles both starting points a panel can be
    in: expanded (desktop/tablet, nothing overrides the browser default
    block display) or collapsed (phones, via each panel's own @media rule
    hiding the body by id). Toggling by directly setting .style.display
    afterwards always wins over that stylesheet rule regardless of
    viewport width, which is exactly "hide and show" rather than a fixed
    mobile/desktop split: a reader on a phone can still tap a panel open
    to read it, same as a reader on a laptop can collapse one out of the
    way."""
    return (
        f"var b=document.getElementById('{body_id}');"
        f"var open=getComputedStyle(b).display!=='none';"
        f"b.style.display=open?'none':'block';"
        f"var c=document.getElementById('{chevron_id}');"
        f"if(c)c.innerHTML=open?'&#9656;':'&#9662;';"
    )


def build_dbz_legend_html(products: tuple) -> str:
    """A floating color-scale legend (swatches + value ticks), the same
    idea as the mm/h bar on rain-radar apps -- so a viewer can read a
    color on the map back into a dBZ value without opening a popup.
    Each swatch is one flat band from DBZ_BAND_COLORS via
    dbz_colormap_and_norm, the SAME shared definition dbz_array_to_png
    uses, so the legend is never just an approximation of what's drawn --
    it's literally the same bins.

    The swatch row collapses to just its header on narrow screens (see
    the @media rule below) -- on a phone this (plus the info banner,
    collapsed the same way by build_info_banner_html) was covering enough
    of the map that the reflectivity underneath wasn't usable, reported
    directly against a real screenshot. The header stays tappable to
    expand it back -- this is "hide and show", not "remove on mobile"."""
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
    toggle_js = _collapsible_panel_toggle_js("nowcast-legend-body", "nowcast-legend-chevron")
    return f"""
    <style>
      @media (max-width: 600px) {{
        #nowcast-legend-body {{ display: none; }}
      }}
    </style>
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999;
                max-width: min(300px, calc(100vw - 48px));
                background: rgba(255,255,255,0.92); border-radius: 8px;
                box-shadow: 0 1px 6px rgba(0,0,0,0.3);
                font-family: -apple-system, Arial, sans-serif; overflow: hidden;">
        <div onclick="{toggle_js}"
             style="display: flex; align-items: center; gap: 10px; padding: 10px 14px;
                    cursor: pointer; user-select: none;">
            <div style="font-size: 12px; font-weight: 600; color: #222; flex: 1; min-width: 0;
                        overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                {label} — reflectivity (dBZ)
            </div>
            <div id="nowcast-legend-chevron" style="color: #888; font-size: 10px; flex-shrink: 0;">&#9662;</div>
        </div>
        <div id="nowcast-legend-body" style="padding: 0 14px 10px 14px;">
            <div style="display:flex; width: 260px; max-width: calc(100vw - 76px);">{swatches}</div>
        </div>
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
    away from any map control regardless of screen size.

    The time list + disclaimer collapse to just the header row (logo +
    title) on narrow screens -- with 5-6 radars/products listed (NIOT,
    Karaikal, Karaikal Extended, Kochi, Chennai DWR, Chennai Extended) this
    banner got tall enough on a phone to cover a real chunk of the map
    itself, reported directly against a screenshot. Tapping the header
    still expands it -- same "hide and show" toggle build_dbz_legend_html
    uses for the same reason, via the shared _collapsible_panel_toggle_js."""
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
    toggle_js = _collapsible_panel_toggle_js("nowcast-info-body", "nowcast-info-chevron")
    return f"""
    <style>
      @media (max-width: 600px) {{
        #nowcast-info-body {{ display: none; }}
      }}
    </style>
    <div style="position: fixed; top: 12px; right: 12px; z-index: 9999;
                max-width: min(230px, calc(100vw - 24px));
                background: rgba(255,255,255,0.92); border-radius: 8px;
                box-shadow: 0 1px 6px rgba(0,0,0,0.3);
                font-family: -apple-system, Arial, sans-serif; font-size: 11px;
                color: #333; line-height: 1.5; overflow: hidden;">
        <div onclick="{toggle_js}"
             style="display: flex; align-items: center; gap: 6px; padding: 8px 12px;
                    cursor: pointer; user-select: none;">
            {logo_tag}
            <div style="font-weight: 700; color: #b45309; flex: 1;">
                &#9888; Experimental nowcast
            </div>
            <div id="nowcast-info-chevron" style="color: #888; font-size: 10px; flex-shrink: 0;">&#9662;</div>
        </div>
        <div id="nowcast-info-body" style="padding: 0 12px 8px 12px;">
            <div style="margin-bottom: 4px;">{times_html}</div>
            <div style="font-size: 10px; color: #666;">
                Based on IMD radar imagery. Not an official forecast.<br>
                Auto-refreshes every {AUTOREFRESH_MINUTES} min.
            </div>
        </div>
    </div>
    """


def build_autorefresh_script(interval_minutes: int = AUTOREFRESH_MINUTES,
                              idle_timeout_minutes: int = AUTOREFRESH_IDLE_TIMEOUT_MINUTES) -> str:
    """This is a static HTML file re-uploaded on a schedule -- a browser
    tab left open on it otherwise just sits on whatever was live at the
    moment it was opened, with no way to know a newer map has since been
    published. A plain timed reload fixes that for anyone who leaves the
    page open; the schedule (nowcast.yml -> cron-job.org, see that
    workflow's comments) already keeps the live file itself fresh every
    ~15 min, this just makes sure an already-open tab actually picks that
    up instead of going stale in the background.

    Visibility- and idle-aware (an earlier version was plain
    setTimeout(reload, interval) regardless of whether the tab was even
    being looked at -- this docstring used to note that swapping in a
    visibility-aware version later would be easy "if that ever turns out
    to matter"; it did, see the server-load discussion that prompted
    this): a backgrounded/minimized tab doesn't poll the server AT ALL
    while hidden -- there's no persistent per-viewer connection to a
    static file host to "close", so not making the request in the first
    place is the real equivalent. The moment the tab becomes visible
    again, it reloads immediately if a full interval has already elapsed
    while hidden (so the reader never sees stale data), or otherwise
    arms a timer for whatever time is left. On top of that, a tab that's
    visible but genuinely unattended (no mouse/touch/key/scroll/click
    for idle_timeout_minutes) also stops reloading until it sees
    activity again -- catches "left open on a monitor nobody's at",
    which plain visibility alone wouldn't.

    Deliberately still a plain JS timer, not a <meta http-equiv="refresh">:
    a meta-refresh can't be paused/resumed based on visibility or
    activity at all, which is the entire point here.

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
    .html, not .script) while looking completely fine in isolation. The
    IIFE below is one statement, same single-nested-function shape as
    before, for the same reason."""
    interval_ms = interval_minutes * 60 * 1000
    idle_ms = idle_timeout_minutes * 60 * 1000
    return f"""(function() {{
  var INTERVAL_MS = {interval_ms};
  var IDLE_MS = {idle_ms};
  var loadTime = Date.now();
  var lastActivity = Date.now();
  var timer = null;

  function markActive() {{ lastActivity = Date.now(); }}
  ['mousemove', 'keydown', 'touchstart', 'scroll', 'click'].forEach(function(evt) {{
    document.addEventListener(evt, markActive, {{passive: true}});
  }});

  function isIdle() {{ return (Date.now() - lastActivity) > IDLE_MS; }}

  function clearTimer() {{
    if (timer) {{ clearTimeout(timer); timer = null; }}
  }}

  function armTimer() {{
    clearTimer();
    if (document.visibilityState !== 'visible' || isIdle()) return;
    var remaining = Math.max(INTERVAL_MS - (Date.now() - loadTime), 0);
    timer = setTimeout(function() {{ window.location.reload(); }}, remaining);
  }}

  document.addEventListener('visibilitychange', function() {{
    if (document.visibilityState !== 'visible') {{ clearTimer(); return; }}
    if (Date.now() - loadTime >= INTERVAL_MS) {{ window.location.reload(); }}
    else {{ armTimer(); }}
  }});

  // No single event fires purely from time passing without interaction,
  // so poll once a minute to notice "just went idle" (stop the timer)
  // or "just became active again after being idle" (re-arm it).
  setInterval(function() {{
    if (isIdle()) {{ clearTimer(); }}
    else if (!timer && document.visibilityState === 'visible') {{ armTimer(); }}
  }}, 60000);

  armTimer();
}})();"""


def build_cell_zoom_declutter_script(map_var: str, registry: list[tuple[str, float, str]]) -> str:
    """Thins out cell markers AND their direction arrows by ZOOM LEVEL the
    same way OSM/Google Maps declutters place labels -- fewer, only the
    most significant ones visible zoomed out over a wide area,
    progressively more revealed as you zoom into a smaller area --
    requested directly after a wide-area view (multiple radars, many
    small cells) read as too busy/cluttered, then again to extend the
    same treatment to the arrows. "Significant" here is the cell's own
    real detected area in km^2 (the same area_km2 already driving its
    marker radius -- see that comment a few lines up), a reasonable
    stand-in for "how much this matters to a reader scanning the whole
    map" the same way a city's population decides whether it's labeled at
    a given map zoom. An arrow entry carries its OWN cell's area_km2, not
    a separately-computed score, specifically so it declutters in lock
    step with its own circle -- never one visible without the other,
    which would just read as a rendering bug.

    Deliberately a flat lookup table of (max zoom, min area_km2 to show)
    tiers rather than a continuous formula -- easier to reason about and
    retune by eye than a smooth curve, and label-declutter systems
    elsewhere (OSM included) are themselves tiered, not continuous, for
    the same reason. Tuned against this map's own default zoom levels
    (build_forecast_map uses 7 for a multi-radar view, 9 for single-radar)
    and will likely want retuning once this has been watched through a
    real widespread-storm day where there are enough cells for decluttering
    to matter at all -- at low cell counts every tier shows everything
    anyway. Every tier's threshold is inclusive of the zoom below it (the
    last matching row wins), so zoom 12 and up always shows every cell
    regardless of size.

    Hides rather than removes a layer outside its tier -- cheaper than
    re-adding/removing layers on every zoom change, and a hidden layer's
    popup/tooltip still technically works if a reader somehow clicks
    through an invisible one, a fine trade for how rarely that'll happen.
    A circle (Leaflet Path) and a direction-arrow Marker don't share a
    visibility API, so each entry's "kind" picks the right one: a
    circle's opacity/fillOpacity via setStyle, a marker's single opacity
    via setOpacity (markers have no separate fill).

    Returns BARE JavaScript (no <script> tags) for the same reason
    build_autorefresh_script's docstring explains -- added the same way,
    via m.get_root().script.add_child(...).

    CRITICAL ORDERING NOTE, learned the hard way (reported directly as a
    completely blank map, banner/legend still visible -- the exact
    "everything after this line in the shared script block silently
    stops running" symptom build_autorefresh_script's own docstring warns
    about, caused a different way this time): m.get_root().script.add_child
    appends to the ROOT figure's script list, which branca renders BEFORE
    the map/circle/marker elements' own auto-generated init code later in
    that same shared <script> block -- so at the moment this function's
    code would normally run, `{{map_var}}` and every `circle_*`/`marker_*`
    variable it references don't exist yet. Referencing an undefined var
    throws, and since it's all one synchronous <script> tag, that
    exception kills every statement after it -- including Leaflet's own
    map/tile-layer init -- leaving a blank page with only the plain-HTML
    overlays (which don't depend on any JS running) still showing.
    Wrapping the whole body in setTimeout(fn, 0) defers it to the next
    event-loop tick, by which point the REST of this same script block
    (map/circle/marker declarations included) has already finished
    executing synchronously -- cheap and sufficient, no need for a
    DOMContentLoaded/load listener since nothing here waits on external
    resources, just on later lines of the same script having run."""
    if not registry:
        return ""
    entries = ",".join(f'{{m:{name},a:{area:.3f},k:"{kind}"}}' for name, area, kind in registry)
    # (max_zoom_for_this_tier, min_area_km2_to_show) -- first row whose
    # max_zoom is >= the current zoom wins; last matching row wins ties,
    # see tiers.length-1 fallback below for "zoom higher than every listed
    # tier -> show everything".
    tiers = [(6, 60.0), (7, 25.0), (8, 12.0), (9, 6.0), (10, 2.0), (11, 0.5)]
    tiers_js = ",".join(f"[{z},{a}]" for z, a in tiers)
    return f"""
setTimeout(function() {{
    var map = {map_var};
    var cells = [{entries}];
    var tiers = [{tiers_js}];
    function minAreaForZoom(z) {{
        for (var i = 0; i < tiers.length; i++) {{
            if (z <= tiers[i][0]) return tiers[i][1];
        }}
        return 0;  // past the last tier -- show every cell, however small
    }}
    function updateCellVisibility() {{
        var minArea = minAreaForZoom(map.getZoom());
        cells.forEach(function(c) {{
            var show = c.a >= minArea;
            if (c.k === 'marker') {{
                c.m.setOpacity(show ? 1 : 0);
            }} else {{
                c.m.setStyle({{opacity: show ? 0.6 : 0, fillOpacity: show ? 0.3 : 0}});
            }}
        }});
    }}
    map.on('zoomend', updateCellVisibility);
    updateCellVisibility();
}}, 0);
"""


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
    just browsing", not "back to the blog homepage". Bottom-right,
    mirroring the legend's bottom-left position -- this used to sit
    bottom-center instead, which put it directly on top of the legend's
    header text on narrow screens (the legend collapsing to a slim header
    bar made this visibly overlap rather than just partially cover the
    swatches underneath, reported directly against a screenshot). Bottom-
    right keeps it clear of both the legend (bottom-left) and the info
    banner/logo (top-right) at any screen width, without the two needing
    to know about each other's height."""
    return """
    <a href="https://plots.chennairains.com/index.html"
       style="position: fixed; bottom: 24px; right: 24px;
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
    # Reported: Chrome was popping up "Translate this page from
    # Malagasy?" on load. branca's own Figure template (the thing that
    # actually emits the <html> tag) hardcodes a bare <html> with no lang
    # attribute at all, so Chrome's language detector has nothing to go
    # on and falls back to guessing from the visible text -- station
    # names, dBZ, IST, short all-caps place labels -- which is exactly
    # the kind of sparse, abbreviation-heavy text that detector is known
    # to misread as all sorts of things, Malagasy here. Two independent
    # fixes, both standard practice for this: an explicit
    # Content-Language header so a detector that does look has a real
    # answer, and Google's own documented <meta name="google"
    # content="notranslate"> tag, which tells Chrome/Google Translate
    # outright not to offer translation for this page regardless of what
    # the detector guesses.
    m.get_root().header.add_child(folium.Element(
        '<meta http-equiv="Content-Language" content="en">'
        '<meta name="google" content="notranslate">'
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
            # Thin/faint on purpose -- this ring is reference context (how
            # far out a radar's scan nominally reaches), not data, and with
            # 2+ radars' rings now often overlapping (see _mask_stale_overlap
            # and, with the 500km Karaikal extension, a second ring on top
            # of that), the original weight/opacity stacked into a visually
            # heavy crosshatch right where readers most need to see the
            # actual reflectivity underneath. Reported directly as too
            # intrusive; still visible, just recedes behind the data now.
            location=[site["site_lat"], site["site_lon"]],
            radius=range_km * 1000.0, color="#555555", weight=0.75, opacity=0.4,
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
    # Drop any product whose last-seen observation is older than
    # STALE_OBS_MINUTES from the RENDERED map entirely -- both the raster
    # and its storm markers/arrows below -- not just the banner's red-text
    # flag. A flagged-but-still-drawn stale frame still puts a confident-
    # looking colored blob and cell marker on the map for a storm that may
    # have already dissipated, or simply isn't there any more by the time
    # a viewer looks; the banner text is easy to miss next to a vivid map.
    # This is the same STALE_OBS_MINUTES threshold build_info_banner_html()
    # already uses for its own red-text flag, so "flagged" and "dropped"
    # always agree -- a product never shows live-looking data on the map
    # while its own banner entry calls it stale, or vice versa. A product
    # that's never been seen at all this run (obs_time None) is treated as
    # fresh here, not dropped -- that's "no data yet", a materially
    # different situation from "had data, it's gone stale", and dropping it
    # here would just silently blank a product that was never polled in
    # this call to begin with (e.g. one of `products` genuinely absent from
    # POLLED_PRODUCTS this run).
    now_utc = datetime.now(timezone.utc)
    fresh_products = {
        p for p in products
        if _last_obs_time_seen.get(p) is None
        or (now_utc - _last_obs_time_seen[p]).total_seconds() / 60.0 <= STALE_OBS_MINUTES
    }
    stale_dropped = [p for p in products if p not in fresh_products]
    if stale_dropped:
        print(f"[render] dropping stale product(s) from the map (>{STALE_OBS_MINUTES}min old): "
              f"{', '.join(stale_dropped)}")

    rasters_to_draw = _mask_stale_overlap(
        {p: a for p, a in _last_dbz.items() if p in fresh_products}, products)
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
        cells_to_project = cluster_cells([c for p in fresh_products for c in _prev_cells.get(p, [])])
    else:
        cells_to_project = [c for p in fresh_products for c in _prev_cells.get(p, [])]

    # Registry feeding build_cell_zoom_declutter_script() below -- each
    # entry is (this layer's Leaflet JS variable name, its parent cell's
    # real area in km^2, "circle" or "marker"), collected as markers are
    # created so the post-loop script can reference every one of them by
    # name. A cell's direction arrow shares its own entry's area_km2 (not
    # a separate importance score) specifically so it declutters in lock
    # step with its own circle -- an arrow with no circle next to it (or
    # vice versa) would just read as a rendering bug, not a feature.
    cell_marker_registry: list[tuple[str, float, str]] = []

    n_projected = 0
    for c in cells_to_project:
        # Fixed SCREEN-pixel radius, not a real-world (meter) one -- tried
        # meter-based (folium.Circle, zoom-scaling) in the test repo first
        # to stop a cell marker from visibly outgrowing its own shrinking
        # storm raster as you zoomed out, but that traded one problem for
        # another: a marker shrinking in lockstep with the raster invites
        # comparing the two at every zoom level, and a floored, approximate
        # equal-area circle never matches the raster's actual (often
        # irregular) footprint closely enough to survive that comparison.
        # Settled instead on treating it like a basemap PLACE LABEL -- the
        # text for a city name stays the same pixel size at every zoom
        # level (what changes with zoom is which labels are dense enough to
        # show at all, which build_cell_zoom_declutter_script below
        # handles), it never grows or shrinks to match the city's real
        # footprint. So CircleMarker (pixels, zoom-fixed), sized by
        # area_km2 only to give bigger/stronger cells a modestly bigger
        # fixed dot than small ones (the way a capital gets bigger label
        # text than a village) -- never by zoom level. Tested in the
        # 500km test repo through a real widespread-storm day before being
        # carried over here, per explicit instruction.
        km_per_px = PRODUCTS[c.product]["km_per_px"]
        area_km2 = c.pixel_count / (km_per_px ** 2)
        radius_px = min(6.0 + area_km2 ** 0.5 * 0.4, 14.0)
        fill_color = "#2ca02c" if do_fuse else PRODUCT_STYLE.get(c.product, {}).get("color", "#08306b")
        label_prefix = "Fused cell" if do_fuse else f"{c.product.upper()} cell"
        circle = folium.CircleMarker(
            location=c.centroid_latlon, radius=radius_px,
            # Semi-transparent on purpose -- these markers sit directly on
            # top of the reflectivity raster (the actual storm shape drawn
            # a few lines up via ImageOverlay), and at full opacity a cell
            # marker fully occults whatever real echo pattern is under it,
            # which is most of what a viewer actually wants to see. Also
            # tuned and confirmed in the test repo first (dropped from
            # opaque 1.0/1.0 to 0.85/0.55, then further to 0.6/0.3, which
            # is what's carried over here).
            color="white", weight=3, opacity=0.6,
            fill=True, fill_color=fill_color, fill_opacity=0.3,
            popup=f"{label_prefix} #{c.id} — {c.max_dbz:.0f} dBZ now, trend: {c.trend}",
        )
        circle.add_to(m)
        cell_marker_registry.append((circle.get_name(), area_km2, "circle"))

        if not c.velocity_kmh or c.velocity_kmh[0] < min_speed_kmh:
            continue  # no reliable motion yet, or effectively stationary
        speed_kmh, bearing_deg = c.velocity_kmh

        radar = PRODUCT_RADAR.get(c.product)
        max_range = ARROW_MAX_RANGE_KM.get(radar)
        if max_range is not None:
            site = RADAR_SITES[radar]
            dist_km = _haversine_km((site["site_lat"], site["site_lon"]), c.centroid_latlon)
            if dist_km > max_range:
                continue  # beyond the full-resolution disc -- marker only, no arrow

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
        arrow = folium.Marker(
            location=(arrow_lat, arrow_lon),
            icon=build_bearing_arrow_icon(bearing_deg, "#333333"),
            tooltip=f"{label_prefix} #{c.id} heading {bearing_deg:.0f}° "
                    f"at {speed_kmh:.0f} km/h",
        )
        arrow.add_to(m)
        cell_marker_registry.append((arrow.get_name(), area_km2, "marker"))
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
    m.get_root().script.add_child(folium.Element(
        build_cell_zoom_declutter_script(m.get_name(), cell_marker_registry)))

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
#
# koc_maxz (Kochi MAXZ) dropped per explicit instruction -- that radar is
# under maintenance, so polling it would just serve an increasingly stale
# (or outright broken) frame under a "live" banner. cni_maxz (Chennai DWR
# MAXZ) and kkl_ppz (Karaikal Extended -- the 250-500km ring kkl_maxz's own
# image can't reach, via its mask_within_km=250.0 field) promoted here from
# the 500km test repo per explicit instruction, after testing there; both
# were already fully defined in this file (PRODUCTS/PRODUCT_STYLE/
# PRODUCT_RADAR/ARROW_MAX_RANGE_KM all already had entries for them) so
# this line is the only change needed to turn them on for real here.
POLLED_PRODUCTS = ("maxz", "kkl_maxz", "kkl_ppz", "cni_maxz", "cni_ppz")

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


def _archive_frame_sort_seconds(path: Path) -> float:
    """Epoch seconds to order an archived frame by when its ORDER WASN'T
    RECORDED (see archived_frames below): the obs_time in a dated name, or
    the poll-receipt time in a number-only name. Only a fallback -- those
    two are different clocks (IMD publishes a frame roughly 20 minutes
    after its own obs_time), so they can't be trusted to interleave
    correctly; the recorded order in state is what's authoritative."""
    t = _obs_time_from_archive_filename(path)
    if t is not None:
        return t.timestamp()
    try:
        return float(int(path.stem.rsplit("_", 1)[-1]))
    except ValueError:
        return 0.0


def archived_frames(product: str, state: dict | None = None) -> list[Path]:
    """This product's archived frames, OLDEST FIRST, in the order they were
    actually archived -- the last one is the most recently saved frame.

    Replaces a plain `sorted(ARCHIVE_DIR.glob(...))` that used to be
    repeated at each call site. A frame is named after its obs_time when
    that could be read ("kkl_maxz_20261001T235222Z.gif") and after the
    poll time as a bare epoch number when it couldn't
    ("kkl_maxz_1790911304.gif"), and "1..." sorts BEFORE "2026..." as
    text. So whenever a radar's timestamp became unreadable, every new
    frame sorted to the FRONT of the list -- the "oldest" end:
      - prune_archive deleted each new frame the moment it was saved,
        keeping the last few dated frames forever instead;
      - _load_previous_dbz_for_flow and the fetch-failure fallback both
        took the last dated frame as "the previous frame", however many
        hours old it had become.
    Seen live on 2026-10-02: Karaikal MAXZ's time stopped being readable
    after 23:52Z, new frames kept arriving for 3+ hours, none survived in
    the archive, and motion was being measured against the 23:52Z frame
    with a dt of one polling cycle.

    Order comes from state[product]["archive_order"], which poll_and_decode
    appends to each time it saves a frame -- that's the real acquisition
    order, independent of how any frame happens to be named. Frames on
    disk that aren't in that list (everything archived before this
    existed, or when `state` isn't passed) go first, ordered by
    _archive_frame_sort_seconds."""
    on_disk = {f.name: f for f in ARCHIVE_DIR.glob(f"{product}_*.gif")}
    recorded = ((state or {}).get(product) or {}).get("archive_order") or []
    known = [on_disk[name] for name in recorded if name in on_disk]
    known_names = {f.name for f in known}
    unrecorded = sorted((f for name, f in on_disk.items() if name not in known_names),
                        key=_archive_frame_sort_seconds)
    return unrecorded + known


def prune_archive(product: str, state: dict | None = None) -> None:
    """Keep only the most recent KEEP_FRAMES_PER_PRODUCT archived frames for
    `product` -- this repo commits every new frame as a binary file, so an
    unbounded archive would make the repo grow forever for no benefit this
    pipeline currently uses (nothing here builds an animation from it yet).
    "Most recent" by archived_frames' order, not by file name."""
    frames = archived_frames(product, state)
    for old in frames[:-KEEP_FRAMES_PER_PRODUCT]:
        old.unlink()
    if state is not None and product in state:
        state[product]["archive_order"] = [f.name for f in frames[-KEEP_FRAMES_PER_PRODUCT:]]


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


def _load_previous_dbz_for_flow(product: str, state: dict | None = None) -> np.ndarray | None:
    """Load and decode the most recently archived frame for `product`
    (BEFORE this cycle's new frame gets saved into the same archive --
    caller must call this first) so compute_optical_flow has something to
    diff the new frame against. Returns None if there's no prior frame yet
    (first-ever run for this product) or if the archived file fails to
    open/decode for any reason -- optical flow is simply skipped for this
    cycle in that case, same as "no prior track" already does elsewhere."""
    existing = archived_frames(product, state)
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

    # Very fine-resolution products (NIOT MAXZ is ~10.6 px per km) move far
    # more PIXELS between two frames than Farneback's window can follow: a
    # 30 km/h storm shifts ~80 px in 15 min, and the window (25) is only
    # reliable to about 15-20 px, so measured speeds came out near zero and
    # directions were close to random. Measure on a copy shrunk to about
    # 1 px/km instead (a 30 km/h storm is then ~8 px) and scale the vectors
    # back to full-resolution pixels. Checked with known synthetic shifts:
    # 10-45 km/h come out within ~2% in the right direction (unchanged
    # products, at or below ~2 px/km, take the original path below).
    px_per_km = PRODUCTS[product]["km_per_px"]     # despite its name: pixels per km
    shrink = int(px_per_km // 1.0) if px_per_km >= 2.0 else 1
    if shrink >= 2:
        h, w = prev_gray.shape
        small = (max(w // shrink, 1), max(h // shrink, 1))
        flow_small = cv2.calcOpticalFlowFarneback(
            cv2.resize(prev_gray, small, interpolation=cv2.INTER_AREA),
            cv2.resize(curr_gray, small, interpolation=cv2.INTER_AREA), None,
            pyr_scale=0.5, levels=4, winsize=25, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0,
        )
        return cv2.resize(flow_small, (w, h), interpolation=cv2.INTER_LINEAR) * float(shrink)

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

    km_per_px = PRODUCTS[product]["km_per_px"]   # NB: really PIXELS per km (pixel_to_latlon divides by it too), so km = px / this
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
    prev_dbz_for_flow = _load_previous_dbz_for_flow(product, state)

    now = time.time()
    pstate["last_new_frame_ts"] = now
    pstate["last_obs_time"] = obs_time.isoformat() if obs_time else None
    pstate["last_frame_hash"] = frame_hash
    tag = obs_time.strftime("%Y%m%dT%H%M%SZ") if obs_time else int(now)
    fname = ARCHIVE_DIR / f"{product}_{tag}.gif"
    fname.write_bytes(raw)
    # Record the order frames were actually saved in -- see archived_frames
    # for why the file name alone can't be relied on for that.
    pstate["archive_order"] = [n for n in pstate.get("archive_order", []) if n != fname.name] + [fname.name]
    print(f"[poll] {product}: new frame saved: {fname} (obs_time={obs_time})")
    prune_archive(product, state)

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


# ---------------------------------------------------------------------------
# Past 3h/6h/12h radar loop (animated GIFs)
#
# Produces three looping GIFs -- the fused (NIOT+Karaikal+Kochi) reflectivity
# picture over the last 3, 6 and 12 hours -- on a fixed geographic extent
# chosen to keep all three radars' range circles in frame together (same
# extent as a screenshot of the live storm_forecast_map.html at its
# zoom_start=7 multi-radar view: Mangaluru/Shivamogga in the NW down to
# Kanyakumari/northern Sri Lanka in the south, out to Chennai/the Bay of
# Bengal in the NE). Lives in THIS (production) repo only, built from
# POLLED_PRODUCTS -- not wired into the 500km test repo.
#
# Design, and why it's shaped this way:
#
# - A frame is captured every ordinary 15-minute cycle (same cadence as
#   everything else here), reusing that cycle's ALREADY-fetched/decoded
#   _last_dbz -- no extra network calls, no re-decoding archived frames
#   later. This needs a full 12h+ of frame history to animate from, so
#   mosaic_frames/ is committed back to the repo between runs exactly like
#   state/ and archive/ already are (see nowcast.yml's git-auto-commit step)
#   -- without that, every run would again see "no previous frames", same
#   failure mode KEEP_FRAMES_PER_PRODUCT's docstring already called out for
#   a from-scratch animation feature.
#
# - The three GIFs are REBUILT every cycle, matching the ~10-minute cadence
#   the radar data itself updates on -- so the loop's last frame is never
#   more than one cycle stale. (This used to be gated to once an hour to
#   save CPU/FTP traffic, back when stale edge caching made more frequent
#   updates pointless anyway -- now that Cloudflare purging actually works,
#   there's no reason to hold the loop back from the same freshness as the
#   live map, and re-encoding a few dozen frames every ~10 minutes is cheap.)
#
# - The built GIFs + HTML page are written straight into the COMMITTED
#   radar_loop/ dir (survives between runs, like state/archive) and copied
#   into output/ every cycle for upload -- see publish_radar_loop()'s
#   docstring for why that copy-forward still matters even though rebuild
#   now happens every cycle too (first-ever run before radar_loop/ exists,
#   and any cycle where capture_mosaic_frame() came up short on frames).

MOSAIC_EDGE_MARGIN_KM = 50.0

def _compute_mosaic_bounds(margin_km: float = MOSAIC_EDGE_MARGIN_KM) -> tuple[float, float, float, float]:
    """(south, west, north, east) degrees -- each edge is 50km beyond ONE
    specific radar's own scan boundary in that direction, per explicit
    instruction, not just "whichever radar happens to reach furthest
    there" (NIOT's own east edge is actually very slightly further out
    than Karaikal's, for instance -- deliberately not used for the east
    edge anyway):
      - north: NIOT's own northern range-circle edge (it's the
        northernmost radar of the three)
      - west & south: Kochi's own western/southern range-circle edges
        (the westernmost/southernmost radar)
      - east: Karaikal's own eastern range-circle edge (not NIOT's,
        even though NIOT's is marginally further east -- Karaikal is
        what this crop is anchored to on that side)
    Longitude degrees-per-km depends on latitude (cos(lat)), so each
    edge uses ITS OWN radar's latitude for that conversion, not a single
    shared approximation across the whole map."""
    def _north_edge(radar, product):
        site = RADAR_SITES[radar]
        return site["site_lat"] + (PRODUCTS[product]["range_km"] + margin_km) / 111.32

    def _south_edge(radar, product):
        site = RADAR_SITES[radar]
        return site["site_lat"] - (PRODUCTS[product]["range_km"] + margin_km) / 111.32

    def _east_edge(radar, product):
        site = RADAR_SITES[radar]
        km_per_deg_lon = 111.32 * math.cos(math.radians(site["site_lat"]))
        return site["site_lon"] + (PRODUCTS[product]["range_km"] + margin_km) / km_per_deg_lon

    def _west_edge(radar, product):
        site = RADAR_SITES[radar]
        km_per_deg_lon = 111.32 * math.cos(math.radians(site["site_lat"]))
        return site["site_lon"] - (PRODUCTS[product]["range_km"] + margin_km) / km_per_deg_lon

    north = _north_edge("niot", "maxz")
    west = _west_edge("kochi", "koc_maxz")
    south = _south_edge("kochi", "koc_maxz")
    east = _east_edge("karaikal", "kkl_maxz")
    return (south, west, north, east)


MOSAIC_BOUNDS = _compute_mosaic_bounds()  # (south, west, north, east) degrees
MOSAIC_ZOOM = 7                           # matches build_forecast_map's zoom_start for 2+ radars

MOSAIC_TILE_URL = CARTO_VOYAGER_URL
MOSAIC_BASEMAP_PATH = Path("mosaic_basemap.png")
MOSAIC_FRAMES_DIR = Path("mosaic_frames")
RADAR_LOOP_DIR = Path("radar_loop")          # persisted (committed) -- see module docstring above
RADAR_LOOP_WINDOWS_HOURS = (3, 6, 12)
MOSAIC_RETAIN_HOURS = 13.0                   # 12h window + 1h slack for a late/skipped cycle
GIF_FRAME_DURATION_MS = 1000   # ~1 fps -- 300ms played too fast to read a storm's actual movement


def _mercator_px(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    """Standard slippy-map (Web Mercator) global pixel coordinates at a
    given zoom, tile size 256px -- the same projection CARTO/OSM tiles are
    served in, used here purely for placing things (overlays, markers,
    range circles) onto the stitched tile basemap in pixel space."""
    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n * 256.0
    lat_rad = math.radians(max(min(lat, 85.05), -85.05))  # mercator's own valid range
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n * 256.0
    return x, y


def _mosaic_tile_range() -> tuple[int, int, int, int]:
    """(x_min, y_min, x_max, y_max) tile indices at MOSAIC_ZOOM needed to
    COVER MOSAIC_BOUNDS -- just picks which tiles to fetch/stitch; the
    saved basemap is then cropped down to the exact bounds (see
    fetch_basemap_mosaic), not left at this tile grid's own coarser
    edges. At zoom 7 a tile is ~2.8 degrees (~300km) across, so without
    that crop the "50km past each radar's edge" margin could silently
    balloon to several hundred km on whichever side happens to fall
    closest to a tile boundary -- exactly what was asked NOT to happen."""
    south, west, north, east = MOSAIC_BOUNDS
    x0, y0 = _mercator_px(west, north, MOSAIC_ZOOM)
    x1, y1 = _mercator_px(east, south, MOSAIC_ZOOM)
    return (int(x0 // 256), int(y0 // 256), int(x1 // 256), int(y1 // 256))


def _mosaic_origin_px() -> tuple[float, float]:
    """Exact (not tile-floored) pixel coordinates of MOSAIC_BOUNDS' own
    (west, north) corner at MOSAIC_ZOOM -- the true origin of the saved,
    CROPPED basemap image. mosaic_lonlat_to_px measures every point
    against this, not the coarser tile grid's corner, so what's on disk
    really does start exactly at the requested bounds."""
    _, west, north, _ = MOSAIC_BOUNDS
    return _mercator_px(west, north, MOSAIC_ZOOM)


def mosaic_lonlat_to_px(lon: float, lat: float) -> tuple[float, float]:
    """Pixel coordinates of (lon, lat) within the saved, cropped
    MOSAIC_BASEMAP_PATH image -- every overlay/marker/circle drawn onto a
    mosaic frame goes through this one function, so they all agree with
    the basemap and with each other by construction."""
    ox, oy = _mosaic_origin_px()
    x, y = _mercator_px(lon, lat, MOSAIC_ZOOM)
    return x - ox, y - oy


def fetch_basemap_mosaic() -> Image.Image:
    """Stitches CARTO Voyager tiles covering MOSAIC_BOUNDS, then crops
    the result down to MOSAIC_BOUNDS' own exact pixel rectangle (tiles
    only decide what gets fetched -- at zoom 7 a tile is ~300km across,
    so leaving the image at the tile grid's own coarser edges instead of
    cropping it would silently turn "50km past each radar's edge" into
    however much slack that radar's edge happened to leave before the
    next tile boundary, which could be hundreds of km on some sides and
    far less on others). Cached to MOSAIC_BASEMAP_PATH, fetched ONCE ever
    (the basemap itself never changes) and committed to the repo like
    state/archive, so every subsequent run just loads it straight off
    disk with no network call at all. Only hits the tile server again if
    that cached file is ever missing (first run, bounds changed and the
    stale file was removed, or someone deletes it)."""
    if MOSAIC_BASEMAP_PATH.exists():
        return Image.open(MOSAIC_BASEMAP_PATH).convert("RGB")

    x_min, y_min, x_max, y_max = _mosaic_tile_range()
    n_x, n_y = x_max - x_min + 1, y_max - y_min + 1
    print(f"[mosaic] fetching {n_x}x{n_y} basemap tiles at zoom {MOSAIC_ZOOM} (one-time, then cached)")
    canvas = Image.new("RGB", (n_x * 256, n_y * 256), (220, 230, 235))
    subdomains = "abcd"
    for tx in range(x_min, x_max + 1):
        for ty in range(y_min, y_max + 1):
            url = MOSAIC_TILE_URL.format(s=subdomains[(tx + ty) % len(subdomains)],
                                          z=MOSAIC_ZOOM, x=tx, y=ty)
            try:
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                tile = Image.open(io.BytesIO(resp.content)).convert("RGB")
            except Exception as exc:
                print(f"[mosaic] tile ({tx},{ty}) fetch failed ({exc}) -- leaving blank")
                continue
            canvas.paste(tile, ((tx - x_min) * 256, (ty - y_min) * 256))

    # Crop the tile-aligned canvas down to MOSAIC_BOUNDS' own exact pixel
    # rectangle -- see this function's docstring for why that crop matters.
    ox, oy = _mosaic_origin_px()
    south, west, north, east = MOSAIC_BOUNDS
    ex, ey = _mercator_px(east, south, MOSAIC_ZOOM)
    left, top = int(round(ox - x_min * 256.0)), int(round(oy - y_min * 256.0))
    right, bottom = int(round(ex - x_min * 256.0)), int(round(ey - y_min * 256.0))
    cropped = canvas.crop((left, top, right, bottom))
    cropped.save(MOSAIC_BASEMAP_PATH)
    return cropped


def _mosaic_circle_radius_px(center_lat: float, range_km: float) -> float:
    """Pixel radius of a `range_km` geographic circle centered at
    center_lat, in mosaic pixel space. Mercator is locally conformal (it
    preserves shape/angles at any given point, just not area across the
    whole map), so a real-world circle really does map to a circle here --
    this just needs ONE correct radius, measured by projecting a point
    range_km due north of the center and taking the pixel distance."""
    cx, cy = mosaic_lonlat_to_px(0.0, center_lat)  # lon irrelevant, only need the y-scale at this latitude
    dlat = (range_km / 111.32)
    _, cy2 = mosaic_lonlat_to_px(0.0, center_lat + dlat)
    return abs(cy - cy2)


def render_mosaic_frame(products: tuple, capture_time: datetime) -> Image.Image:
    """One flat PNG frame for the radar loop: cached CARTO basemap, each
    fresh product's reflectivity draped on via the same corner-to-corner
    rectangle placement Leaflet's own ImageOverlay uses in the live map
    (stretch dbz_array_to_png's output between its plot_bounds_latlon()
    corners, remapped into mosaic pixel space) -- not a true per-pixel
    reprojection, but that's exactly what the live map itself already
    does, so this stays visually consistent with it rather than
    introducing a second, subtly-different rendering of the same data.
    Then the same dashed range rings + site markers as the live map, and a
    timestamp caption so a frame is still readable in isolation (e.g. the
    first frame of a loop, before playback)."""
    base = fetch_basemap_mosaic().copy().convert("RGBA")

    radars_shown = sorted({PRODUCT_RADAR[p] for p in products})
    now_utc = datetime.now(timezone.utc)
    fresh_products = {
        p for p in products
        if _last_obs_time_seen.get(p) is None
        or (now_utc - _last_obs_time_seen[p]).total_seconds() / 60.0 <= STALE_OBS_MINUTES
    }
    rasters = _mask_stale_overlap({p: a for p, a in _last_dbz.items() if p in fresh_products}, products)

    for product, dbz in rasters.items():
        overlay_path = f"_mosaic_overlay_{product}.png"
        dbz_array_to_png(dbz, product, overlay_path)
        overlay = Image.open(overlay_path)
        south, west, north, east = plot_bounds_latlon(product)
        x0, y0 = mosaic_lonlat_to_px(west, north)
        x1, y1 = mosaic_lonlat_to_px(east, south)
        w, h = max(int(round(x1 - x0)), 1), max(int(round(y1 - y0)), 1)
        overlay = overlay.resize((w, h), Image.BILINEAR)
        base.alpha_composite(overlay, dest=(int(round(x0)), int(round(y0))))
        try:
            Path(overlay_path).unlink()
        except OSError:
            pass

    draw = ImageDraw.Draw(base)
    drawn_ranges = set()
    for product in products:
        range_km = PRODUCTS[product]["range_km"]
        radar = PRODUCT_RADAR[product]
        key = (radar, range_km)
        if key in drawn_ranges:
            continue
        drawn_ranges.add(key)
        site = RADAR_SITES[radar]
        cx, cy = mosaic_lonlat_to_px(site["site_lon"], site["site_lat"])
        r = _mosaic_circle_radius_px(site["site_lat"], range_km)
        # Thin/dashed-looking (drawn as a dotted arc via short segments) --
        # same "reference context, not data" reasoning as the live map's
        # own faint range rings.
        n_dots = max(int(2 * math.pi * r / 10), 12)
        for i in range(n_dots):
            if i % 2:
                continue
            theta = 2 * math.pi * i / n_dots
            px, py = cx + r * math.cos(theta), cy + r * math.sin(theta)
            draw.ellipse([px - 1, py - 1, px + 1, py + 1], fill=(85, 85, 85, 160))

    for radar in radars_shown:
        site = RADAR_SITES[radar]
        cx, cy = mosaic_lonlat_to_px(site["site_lon"], site["site_lat"])
        draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(30, 60, 150, 255), outline=(255, 255, 255, 255), width=1)

    # Plain hyphen, not an em-dash -- the default bitmap font (and some
    # fallback fonts) has no glyph for it and silently renders mojibake
    # instead.
    caption = f"Chennai Rains radar - {capture_time.strftime('%Y-%m-%d %H:%M')} UTC"
    try:
        # size= (Pillow 9.2+) scales PIL's own bundled font -- no system
        # font file dependency, so this is identical on this dev sandbox
        # and the GitHub Actions runner. The original bare
        # load_default() is a fixed ~10px bitmap font: fine on a frame
        # viewed at its native 1024px, but unreadable once a GIF this
        # size gets shown at a smaller display width on the page --
        # exactly what was reported ("incorporate the timestamps" really
        # meant "make them legible"). 34px reads clearly even scaled down
        # for a phone-width page.
        font = ImageFont.load_default(size=34)
    except TypeError:
        # Pillow <9.2 -- no size= param at all. Fall back to the tiny
        # bitmap font rather than erroring the whole render.
        font = ImageFont.load_default()
    except Exception:
        font = None
    text_bbox = draw.textbbox((0, 0), caption, font=font) if font else (0, 0, len(caption) * 6, 11)
    pad = 10
    draw.rectangle([4, base.height - (text_bbox[3] - text_bbox[1]) - pad * 2 - 4,
                    4 + (text_bbox[2] - text_bbox[0]) + pad * 2, base.height - 4],
                   fill=(0, 0, 0, 170))
    draw.text((4 + pad, base.height - (text_bbox[3] - text_bbox[1]) - pad - 4), caption,
               fill=(255, 255, 255, 255), font=font)

    return base.convert("RGB")


def capture_mosaic_frame() -> None:
    """Called once per ordinary poll cycle (from run_pipeline) -- renders
    and archives this cycle's mosaic frame, then prunes anything older
    than MOSAIC_RETAIN_HOURS. Filenamed by wall-clock CAPTURE time (not
    any product's own obs_time, which can be None if a timestamp failed to
    parse this cycle) so frames always sort chronologically and the
    windowing logic below always has something to compare against."""
    MOSAIC_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    now_utc = datetime.now(timezone.utc)
    frame = render_mosaic_frame(POLLED_PRODUCTS, now_utc)
    path = MOSAIC_FRAMES_DIR / f"{now_utc.strftime('%Y%m%dT%H%M%SZ')}.png"
    frame.save(path)
    print(f"[mosaic] saved {path}")

    cutoff = now_utc - timedelta(hours=MOSAIC_RETAIN_HOURS)
    for existing in MOSAIC_FRAMES_DIR.glob("*.png"):
        try:
            ts = datetime.strptime(existing.stem, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if ts < cutoff:
            existing.unlink()


def _mosaic_frames_since(hours: float) -> list[Path]:
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(hours=hours)
    frames = []
    for p in sorted(MOSAIC_FRAMES_DIR.glob("*.png")):
        try:
            ts = datetime.strptime(p.stem, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if ts >= cutoff:
            frames.append(p)
    return frames


RADAR_LOOP_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chennai Rains — Radar Loop</title>
<style>
  body {{ margin: 0; padding: 24px 16px 48px; background: #0b1220; color: #e8edf4;
         font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }}
  h1 {{ text-align: center; font-size: 1.3em; margin: 0 0 4px; }}
  p.sub {{ text-align: center; color: #9fb0c3; margin: 0 0 24px; font-size: 0.9em; }}
  /* Stacked top-to-bottom, not side-by-side -- each loop gets the full
     page width instead of being squeezed to a third of it, so the
     reflectivity/range-circle detail (and the timestamp burned into
     each frame) actually reads at a glance instead of needing a
     click-to-zoom. */
  .loops {{ display: flex; flex-direction: column; align-items: center; gap: 32px; max-width: 1100px; margin: 0 auto; }}
  figure {{ margin: 0; background: #121b2e; border-radius: 10px; padding: 10px; width: 100%; }}
  figure img {{ width: 100%; height: auto; border-radius: 6px; display: block; }}
  figcaption {{ text-align: center; margin-top: 8px; font-weight: 600; color: #cdd9e8; }}
  a {{ color: #7fb2ff; }}
</style>
</head>
<body>
  <h1>Chennai Rains Radar Loop</h1>
  <p class="sub">Fused NIOT + Karaikal + Kochi reflectivity · updated every ~10 min · {generated}</p>
  <div class="loops">
    {figures}
  </div>
  <p class="sub" style="margin-top:28px;">
    <a href="storm_forecast_map.html">Live storm map →</a> ·
    <a href="https://chennairains.com">chennairains.com</a>
  </p>
</body>
</html>
"""


def publish_radar_loop(rebuild: bool) -> None:
    """Writes the three radar-loop GIFs + their HTML page into the
    PERSISTED radar_loop/ dir (committed back to the repo, like
    state/archive/mosaic_frames) and copies them into output/ for upload.

    run_pipeline() now passes rebuild=True every cycle, so the GIFs stay as
    fresh as the live map (matching the ~10-minute polling cadence). The
    rebuild=False path still exists for the rare cycle where
    capture_mosaic_frame() didn't produce a usable new frame (e.g. a fetch
    failure with no archived fallback) -- in that case this just re-copies
    whatever's already in radar_loop/ into output/ unchanged, rather than
    rebuilding GIFs with no new content to show. That copy has to run every
    cycle regardless of whether content changed: output/ starts empty on
    every fresh checkout, and FTP-Deploy-Action deletes any remote file its
    own state tracks that's missing from this run's local-dir -- skipping
    it would make the page/GIFs flicker in and out of existence. First-ever
    run (radar_loop/ doesn't exist yet) always rebuilds regardless of the
    gate, so the page exists from the start rather than waiting for one."""
    RADAR_LOOP_DIR.mkdir(parents=True, exist_ok=True)
    gif_paths = {h: RADAR_LOOP_DIR / f"radar_loop_{h}h.gif" for h in RADAR_LOOP_WINDOWS_HOURS}
    html_path = RADAR_LOOP_DIR / "radar_loop.html"

    if rebuild or not html_path.exists() or any(not p.exists() for p in gif_paths.values()):
        figures = []
        for hours in RADAR_LOOP_WINDOWS_HOURS:
            frame_paths = _mosaic_frames_since(hours)
            out_path = gif_paths[hours]
            if len(frame_paths) < 2:
                print(f"[radar-loop] only {len(frame_paths)} frame(s) within {hours}h yet -- "
                      f"skipping that GIF this run (self-heals once more history has accumulated)")
                continue
            frames = [Image.open(p).convert("RGB") for p in frame_paths]
            frames[0].save(out_path, save_all=True, append_images=frames[1:],
                            duration=GIF_FRAME_DURATION_MS, loop=0, optimize=True)
            figures.append(
                f'<figure><img src="{out_path.name}" alt="Past {hours}h radar loop">'
                f'<figcaption>Past {hours} Hours</figcaption></figure>'
            )
            print(f"[radar-loop] built {out_path} from {len(frames)} frames")
        if figures:
            html_path.write_text(RADAR_LOOP_HTML_TEMPLATE.format(
                generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                figures="\n    ".join(figures),
            ))

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    for src in [html_path, *gif_paths.values()]:
        if src.exists():
            (OUTPUT_HTML.parent / src.name).write_bytes(src.read_bytes())


# === BOT DATA EXPORT (not from the notebook) ===
#
# A machine-readable copy of what the map shows, for the query bot ("will
# it rain in Pallikaranai in the next hour?"). The map itself only carries
# this information as a picture; tracked cells (state/prev_cells.json) only
# exist for cores at/above cell_dbz_threshold (30-35 dBZ), so lighter rain
# over a place would otherwise be invisible to anything reading the data.
#
# Two files, both written into output/ so the existing FTP step uploads
# them alongside the map with no workflow change:
#
#   OUTPUT_BOT_JSON -- small index: which radars were used and how old
#       each one is, the grid layout, the fused strong cells (same ones the
#       map draws), and every connected rain area down to the weakest echo
#       IMD's own images carry (20 dBZ), each with its motion.
#   OUTPUT_BOT_GRID -- gzip of raw bytes: reflectivity on a regular lat/lon
#       grid, one layer as observed plus one per lead time in
#       BOT_LEAD_TIMES_MIN, moved along the measured motion.
#
# Purely additive: nothing here feeds back into detection, tracking, state
# or the map. run_pipeline() calls it inside a try/except so a problem
# here can never stop the map from being published.
import gzip

OUTPUT_BOT_JSON = Path("output/nowcast_bot.json")
OUTPUT_BOT_GRID = Path("output/nowcast_bot_grid.bin.gz")

BOT_GRID_STEP_DEG = 0.02            # ~2.2 km; fine enough for a locality, small enough to ship every cycle
# Grid layers every 10 minutes, not just at 30/60/90: a small cell moving at
# 30 km/h crosses a 5 km circle in about 20 minutes, so with only half-hour
# snapshots it could pass right over a place between two of them and never
# show up there (seen in testing: a 3 km cell due over Velachery 8 minutes
# out read as "dry" at 0, 30, 60 and 90). At 10-minute steps nothing moving
# under MAX_PLAUSIBLE_CELL_SPEED_KMH can cross that circle unseen. The
# 0-minute layer is the observed picture moved up to "now" (each radar's
# frame is 10-45 minutes old by the time this runs).
BOT_LEAD_TIMES_MIN = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90)
BOT_CELL_LEAD_TIMES_MIN = (30, 60, 90)   # projected positions listed per cell; same as build_forecast_map
# A grid node only counts as echo if at least this share of the radar pixels
# around it have echo. Taking the strongest pixel alone turned every stray
# pixel into a whole 5 sq km node: three specks about 1 km across came out
# as a 48 sq km "rain area", and NIOT's 95 m pixels made it worse.
BOT_MIN_ECHO_FRACTION = 0.25
# Optical flow between two frames is only trusted when they are a sensible
# distance apart in time: too close and a one-pixel wobble reads as a fast
# storm, too far and the pattern has changed too much to match.
BOT_MOTION_MIN_DT_MIN = 4.0
BOT_MOTION_MAX_DT_MIN = 45.0
# How far one storm's measured motion is spread to its surroundings. 40 km
# was too far: a stationary patch 53 km from a storm moving at 30 km/h was
# given 13 km/h of the storm's motion. At 20 km it measures 1 km/h.
BOT_MOTION_SMOOTH_KM = 20.0
BOT_MOTION_MIN_ECHO_PX = 25         # fewer echo pixels than this is not enough to measure motion from
BOT_RAIN_AREA_MIN_KM2 = 20.0        # smaller patches are left in the grid but not listed as an area
BOT_MAX_RAIN_AREAS = 100
BOT_NO_ECHO = 0
BOT_NO_COVERAGE = 255
# Products left out of the bot export altogether (they stay on the map).
# For a radar whose picture is not yet trusted enough to answer questions
# from: its echo, coverage, cells and motion are all kept out of the grid,
# and it is listed in "radars" with status "excluded".
BOT_EXCLUDED_PRODUCTS: tuple = ()
_IST = timezone(timedelta(hours=5, minutes=30))


def _bot_fresh_products(products: tuple, now_utc: datetime) -> list[str]:
    """Same rule build_forecast_map uses to decide what is drawn: a product
    with a known observation time older than STALE_OBS_MINUTES is dropped;
    one whose time could not be read is kept."""
    return [p for p in products
            if p in _last_dbz and (
                _last_obs_time_seen.get(p) is None
                or (now_utc - _last_obs_time_seen[p]).total_seconds() / 60.0 <= STALE_OBS_MINUTES)]


def _bot_grid_axes(products: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Grid node latitudes (north to south) and longitudes (west to east)
    covering every polled product's range circle. Built from ALL polled
    products, not just this cycle's fresh ones, so the layout stays the
    same from run to run."""
    south = west = np.inf
    north = east = -np.inf
    for p in products:
        site = site_for(p)
        r = PRODUCTS[p]["range_km"]
        dlat = r / 111.0
        dlon = r / (111.0 * np.cos(np.radians(site["site_lat"])))
        south, north = min(south, site["site_lat"] - dlat), max(north, site["site_lat"] + dlat)
        west, east = min(west, site["site_lon"] - dlon), max(east, site["site_lon"] + dlon)
    step = BOT_GRID_STEP_DEG
    north, south = np.ceil(north / step) * step, np.floor(south / step) * step
    west, east = np.floor(west / step) * step, np.ceil(east / step) * step
    nrows = int(round((north - south) / step)) + 1
    ncols = int(round((east - west) / step)) + 1
    return north - np.arange(nrows) * step, west + np.arange(ncols) * step


def _bot_pixel_index(product: str, lat2d: np.ndarray, lon2d: np.ndarray, shape: tuple):
    """For every grid node: which pixel of this product's decoded array it
    falls on, and whether the product genuinely covers that node (inside
    the image, inside range_km, and outside any inner disc the product
    cedes to a sharper one via mask_within_km)."""
    cfg = PRODUCTS[product]
    px, py = latlon_to_pixel(lat2d, lon2d, product)
    x = np.round(px - cfg["plot_bbox"][0]).astype(int)
    y = np.round(py - cfg["plot_bbox"][1]).astype(int)
    inside = (x >= 0) & (x < shape[1]) & (y >= 0) & (y < shape[0])
    site = site_for(product)
    dist = _haversine_km((site["site_lat"], site["site_lon"]), (lat2d, lon2d))
    cover = inside & (dist <= cfg["range_km"])
    if cfg.get("mask_within_km") is not None:
        cover &= dist >= cfg["mask_within_km"]
    return x, y, cover


def _bot_regrid_dbz(dbz: np.ndarray, product: str, lat2d: np.ndarray, lon2d: np.ndarray):
    """Product's reflectivity on the common grid: dBZ where there is echo,
    -1 where the radar looks and sees nothing, NaN where it does not look.
    Each node looks at the pixels within roughly one grid cell (not just
    the single nearest pixel), so a small echo on a fine-resolution product
    such as NIOT is not skipped over by the coarser grid: it takes the
    strongest of them, provided enough of them have echo at all
    (BOT_MIN_ECHO_FRACTION) -- otherwise a lone pixel would be blown up
    into a whole node."""
    # Window wide enough that every pixel belongs to the window of the node
    # nearest to it (half a grid cell in pixels, plus the half pixel lost to
    # rounding the node onto the pixel grid).
    half_cell_px = BOT_GRID_STEP_DEG * 111.0 * PRODUCTS[product]["km_per_px"] / 2.0
    k = 2 * int(np.floor(half_cell_px + 0.5)) + 1
    echo = ~np.isnan(dbz)
    src = np.where(echo, dbz, -1.0)
    if k > 1:
        share = ndimage.uniform_filter(echo.astype(float), size=k)
        src = np.where(share >= BOT_MIN_ECHO_FRACTION - 1e-9, ndimage.maximum_filter(src, size=k), -1.0)
    x, y, cover = _bot_pixel_index(product, lat2d, lon2d, dbz.shape)
    out = np.full(lat2d.shape, np.nan)
    out[cover] = src[y[cover], x[cover]]
    return out


def _bot_optical_flow(prev_dbz: np.ndarray, curr_dbz: np.ndarray, product: str) -> np.ndarray:
    """Farneback flow like compute_optical_flow, with two differences that
    matter for following WEAK rain and for frames that are far apart in
    time. Kept separate so the tracked cells' own velocities
    (compute_optical_flow) are not changed.

    Grayscale: there, no-echo and the weakest echo (20 dBZ, the bottom of
    IMD's scale) both map to black, so the outline of a weak rain area is
    invisible to the flow and its motion comes out too slow (a 27 dBZ patch
    moving at 30 km/h measured 24). Here no-echo is black and any echo
    starts well above it, so a weak patch has an edge to follow.

    Window: winsize=25 loses track once a storm has moved more than about
    15 px between frames -- with frames 30-40 min apart a 35-40 km/h storm
    measured 12-18% slow. A wider window (and more pyramid levels) measures
    the same cases to within 1 km/h."""
    vmin, vmax = product_value_range(product)
    span = (vmax - vmin) or 1.0

    def to_gray(dbz: np.ndarray) -> np.ndarray:
        level = 90.0 + 165.0 * (np.clip(np.nan_to_num(dbz, nan=vmin), vmin, vmax) - vmin) / span
        return np.where(np.isnan(dbz), 0.0, level).astype(np.uint8)

    return cv2.calcOpticalFlowFarneback(
        to_gray(prev_dbz), to_gray(curr_dbz), None,
        pyr_scale=0.5, levels=5, winsize=45, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0,
    )


def _bot_product_motion(product: str, curr_dbz: np.ndarray, lat2d: np.ndarray, lon2d: np.ndarray):
    """Motion of this product's echo (weak echo included) from optical flow
    between its current frame and the previous archived one. Returns
    (weight, u_kmh_east, v_kmh_north) on the common grid plus a short
    status string; weight is 0 where there is nothing to measure.

    Frames are chosen by the time in their file name, never by sort order,
    and only when the current frame's own time is known -- so a frame with
    an unreadable timestamp simply contributes no motion instead of a wrong
    one."""
    zeros = np.zeros(lat2d.shape)
    t_curr = _last_obs_time_seen.get(product)
    if t_curr is None:
        return zeros, zeros, zeros, "no motion: time of the latest frame could not be read"
    best = None
    for f in ARCHIVE_DIR.glob(f"{product}_*.gif"):
        t = _obs_time_from_archive_filename(f)
        if t is None or t >= t_curr:
            continue
        dt = (t_curr - t).total_seconds() / 60.0
        if BOT_MOTION_MIN_DT_MIN <= dt <= BOT_MOTION_MAX_DT_MIN and (best is None or t > best[0]):
            best = (t, f, dt)
    if best is None:
        return zeros, zeros, zeros, (f"no motion: no earlier frame {BOT_MOTION_MIN_DT_MIN:.0f}-"
                                     f"{BOT_MOTION_MAX_DT_MIN:.0f} min before the latest one")
    _, prev_file, dt = best
    try:
        img = Image.open(prev_file)
        img.load()
        prev_dbz = decode_reflectivity(img, product, build_lut_from_colorbar(img, product))
    except Exception as e:
        return zeros, zeros, zeros, f"no motion: earlier frame could not be decoded ({e})"
    if prev_dbz.shape != curr_dbz.shape:
        return zeros, zeros, zeros, "no motion: frame size changed"
    flow = _bot_optical_flow(prev_dbz, curr_dbz, product)

    km_per_px = PRODUCTS[product]["km_per_px"]
    # Same sign convention as sample_cell_velocity_from_flow: +x is east, +y is south.
    u = flow[..., 0] / km_per_px / (dt / 60.0)
    v = -flow[..., 1] / km_per_px / (dt / 60.0)
    valid = (~np.isnan(curr_dbz) | ~np.isnan(prev_dbz)) & (np.hypot(u, v) <= MAX_PLAUSIBLE_CELL_SPEED_KMH)
    if int(valid.sum()) < BOT_MOTION_MIN_ECHO_PX:
        return zeros, zeros, zeros, "no motion: too little echo to measure"

    # Average (not strongest) motion over roughly one grid cell, ignoring pixels with no echo.
    k = max(1, int(round(BOT_GRID_STEP_DEG * 111.0 * km_per_px)))
    w_px = ndimage.uniform_filter(valid.astype(float), size=k)
    with np.errstate(invalid="ignore", divide="ignore"):
        u_px = ndimage.uniform_filter(np.where(valid, u, 0.0), size=k) / w_px
        v_px = ndimage.uniform_filter(np.where(valid, v, 0.0), size=k) / w_px
    x, y, cover = _bot_pixel_index(product, lat2d, lon2d, curr_dbz.shape)
    w, ug, vg = zeros.copy(), zeros.copy(), zeros.copy()
    w[cover] = w_px[y[cover], x[cover]]
    ug[cover] = np.nan_to_num(u_px[y[cover], x[cover]])
    vg[cover] = np.nan_to_num(v_px[y[cover], x[cover]])
    return w, ug, vg, f"ok: {dt:.0f} min between frames"


def _bot_speed_bearing(u: float, v: float) -> tuple[float, float]:
    return round(float(np.hypot(u, v)), 1), round(float(np.degrees(np.arctan2(u, v)) % 360), 0)


def export_bot_data(products: tuple | None = None, now_utc: datetime | None = None) -> dict | None:
    """Write OUTPUT_BOT_JSON + OUTPUT_BOT_GRID for this cycle (see the block
    comment above). Returns the JSON content, or None if there was nothing
    to export."""
    products = products or POLLED_PRODUCTS
    now_utc = now_utc or datetime.now(timezone.utc)
    included = tuple(p for p in products if p not in BOT_EXCLUDED_PRODUCTS)
    fresh = _bot_fresh_products(included, now_utc)
    lats, lons = _bot_grid_axes(included or products)
    lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")
    step = BOT_GRID_STEP_DEG

    radars, regridded, age_h = [], {}, {}
    w_sum = np.zeros(lat2d.shape); u_sum = np.zeros(lat2d.shape); v_sum = np.zeros(lat2d.shape)
    for p in products:
        obs = _last_obs_time_seen.get(p)
        age = (now_utc - obs).total_seconds() / 60.0 if obs is not None else None
        entry = {
            "product": p, "radar": PRODUCT_RADAR[p],
            "label": PRODUCT_STYLE.get(p, {}).get("label", p.upper()),
            "range_km": PRODUCTS[p]["range_km"],
            "obs_time_utc": obs.isoformat(timespec="seconds") if obs is not None else None,
            "obs_time_ist": obs.astimezone(_IST).strftime("%d %b %H:%M IST") if obs is not None else None,
            "age_min": round(age, 0) if age is not None else None,
        }
        if p in BOT_EXCLUDED_PRODUCTS:
            entry.update(status="excluded", used=False, motion="no motion: left out of the bot export")
        elif p not in _last_dbz:
            entry.update(status="missing", used=False, motion="no motion: no frame this cycle")
        elif p not in fresh:
            entry.update(status="stale", used=False, motion="no motion: frame too old to use")
        else:
            entry.update(status="fresh" if obs is not None else "time_unknown", used=True)
            regridded[p] = _bot_regrid_dbz(_last_dbz[p], p, lat2d, lon2d)
            # A frame with no readable time is treated as taken just now (same as the map does).
            age_h[p] = max(age, 0.0) / 60.0 if age is not None else 0.0
            w, u, v, note = _bot_product_motion(p, _last_dbz[p], lat2d, lon2d)
            entry["motion"] = note
            w_sum += w; u_sum += w * u; v_sum += w * v
        radars.append(entry)

    if not regridded:
        print("[bot-export] no usable radar this cycle -- writing an index that says so, no grid")

    # ---- one smooth motion field for the whole grid ----
    # Measured only where there is echo; spread outward (normalized
    # convolution, so empty ground doesn't drag speeds toward zero) and,
    # beyond the reach of any measurement, filled with the overall average
    # -- a storm has to be able to arrive somewhere that is dry right now.
    motion_ok = bool(w_sum.sum() > 0)
    if motion_ok:
        sigma = BOT_MOTION_SMOOTH_KM / (step * 111.0)
        gw = ndimage.gaussian_filter(w_sum, sigma)
        mean_u, mean_v = float(u_sum.sum() / w_sum.sum()), float(v_sum.sum() / w_sum.sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            U = np.where(gw > 1e-6, ndimage.gaussian_filter(u_sum, sigma) / gw, mean_u)
            V = np.where(gw > 1e-6, ndimage.gaussian_filter(v_sum, sigma) / gw, mean_v)
    else:
        U = V = None
        mean_u = mean_v = 0.0

    # ---- layers ----
    def composite(lead_min):
        """Strongest value over all used radars at each node. lead_min None =
        exactly as observed. Otherwise each radar's picture is moved forward
        by (its own age + lead), so every layer refers to the same clock
        time even though the radars' frames were taken at different times.

        Echo is carried FORWARD: every echo node moves along the motion
        measured at that node. (The first version looked BACKWARD from each
        destination using the motion at the destination. Where the motion
        field changes over a short distance -- a stationary shower next to
        a moving storm -- ground between the two has an in-between motion,
        looks back, finds the stationary shower and copies it: a ghost
        echo drifting off a shower that isn't moving. In testing that put
        a stationary patch 28 km out of place at 90 minutes.)

        Coverage still looks backward: a node counts as "seen, dry" only if
        the place its weather is coming from was inside radar coverage;
        otherwise nothing is known about what will arrive there."""
        best = np.full(lat2d.shape, np.nan)
        for p, grid in regridded.items():
            if lead_min is None:
                moved = grid
            else:
                hours = age_h[p] + lead_min / 60.0
                src_lat = lat2d - V * hours / 111.0
                src_lon = lon2d - U * hours / (111.0 * np.cos(np.radians(lat2d)))
                r = np.round((lats[0] - src_lat) / step).astype(int)
                c = np.round((src_lon - lons[0]) / step).astype(int)
                ok = (r >= 0) & (r < grid.shape[0]) & (c >= 0) & (c < grid.shape[1])
                seen = np.zeros(grid.shape, dtype=bool)
                seen[ok] = ~np.isnan(grid[r[ok], c[ok]])
                moved = np.where(seen, -1.0, np.nan)

                er, ec = np.where(np.nan_to_num(grid, nan=-1.0) > 0)
                if er.size:
                    dr = np.round(er - V[er, ec] * hours / 111.0 / step).astype(int)   # north is toward row 0
                    dc = np.round(ec + U[er, ec] * hours / (111.0 * np.cos(np.radians(lat2d[er, ec]))) / step).astype(int)
                    inb = (dr >= 0) & (dr < grid.shape[0]) & (dc >= 0) & (dc < grid.shape[1])
                    echo = np.full(grid.shape, -1.0)
                    np.maximum.at(echo, (dr[inb], dc[inb]), grid[er[inb], ec[inb]])
                    # nodes that moved by slightly different amounts can leave one-node gaps inside a storm
                    echo = ndimage.grey_closing(echo, size=3)
                    arrived = echo > 0
                    moved[arrived] = echo[arrived]
            best = np.fmax(best, moved)
        return best

    def to_bytes(layer):
        out = np.full(layer.shape, BOT_NO_COVERAGE, dtype=np.uint8)
        seen = ~np.isnan(layer)
        out[seen] = BOT_NO_ECHO
        echo = seen & (layer > 0)
        out[echo] = np.clip(np.round(layer[echo]), 1, 254).astype(np.uint8)
        return out

    layers_meta, blobs = [], []
    observed = composite(None) if regridded else None
    if observed is not None:
        layers_meta.append({"name": "observed", "lead_min": 0,
                            "note": "latest frame from each radar, as drawn on the map"})
        blobs.append(to_bytes(observed))
        if motion_ok:
            for lead in BOT_LEAD_TIMES_MIN:
                valid = now_utc + timedelta(minutes=lead)
                layers_meta.append({"name": f"plus_{lead}", "lead_min": lead,
                                    "valid_utc": valid.isoformat(timespec="seconds"),
                                    "valid_ist": valid.astimezone(_IST).strftime("%d %b %H:%M IST")})
                blobs.append(to_bytes(composite(lead)))

    # ---- fused strong cells: the same ones the map draws ----
    cells_out = []
    fresh_cells = [c for p in fresh for c in _prev_cells.get(p, [])]
    multi = len({PRODUCT_RADAR[p] for p in included}) > 1
    for c in (cluster_cells(fresh_cells) if multi else fresh_cells):
        radar = PRODUCT_RADAR.get(c.product)
        site = RADAR_SITES[radar]
        dist = float(_haversine_km((site["site_lat"], site["site_lon"]), c.centroid_latlon))
        max_range = ARROW_MAX_RANGE_KM.get(radar)
        # Same test the map applies before drawing a direction arrow.
        reliable = bool(c.velocity_kmh and c.velocity_kmh[0] >= 5.0
                        and (max_range is None or dist <= max_range))
        item = {
            "id": c.id, "radar": radar, "product": c.product,
            "lat": round(float(c.centroid_latlon[0]), 4), "lon": round(float(c.centroid_latlon[1]), 4),
            "max_dbz": round(float(c.max_dbz), 1),
            "area_km2": round(c.pixel_count / (PRODUCTS[c.product]["km_per_px"] ** 2), 1),
            "trend": c.trend, "elevated": bool(c.elevated),
            "confirmed_at_surface": bool(c.confirmed_at_surface),
            "speed_kmh": c.velocity_kmh[0] if c.velocity_kmh else None,
            "bearing_deg": c.velocity_kmh[1] if c.velocity_kmh else None,
            "motion_reliable": reliable,
        }
        if reliable:
            item["projected"] = {
                str(lead): [round(float(x), 4) for x in
                            project_forward(*c.centroid_latlon, c.velocity_kmh[0], c.velocity_kmh[1], lead)]
                for lead in BOT_CELL_LEAD_TIMES_MIN}
        cells_out.append(item)

    # ---- every connected rain area, weak ones included ----
    areas_out = []
    if observed is not None:
        echo = np.nan_to_num(observed, nan=-1.0) > 0
        labeled, n = ndimage.label(echo, structure=np.ones((3, 3)))
        cell_km2 = (step * 111.0) ** 2 * np.cos(np.radians(lat2d))
        idx = np.arange(1, n + 1)
        if n:
            area = ndimage.sum(cell_km2, labeled, idx)
            keep = [i for i in np.argsort(-area) if area[i] >= BOT_RAIN_AREA_MIN_KM2][:BOT_MAX_RAIN_AREAS]
            slices = ndimage.find_objects(labeled)
            for i in keep:
                sl = slices[i]
                m = labeled[sl] == idx[i]
                vals = observed[sl][m]
                la, lo = lat2d[sl][m], lon2d[sl][m]
                item = {
                    "lat": round(float(np.average(la, weights=vals)), 3),
                    "lon": round(float(np.average(lo, weights=vals)), 3),
                    "area_km2": round(float(area[i]), 0),
                    "max_dbz": round(float(vals.max()), 0),
                    "mean_dbz": round(float(vals.mean()), 0),
                    "south": round(float(la.min()), 2), "north": round(float(la.max()), 2),
                    "west": round(float(lo.min()), 2), "east": round(float(lo.max()), 2),
                    "has_strong_core": bool(vals.max() >= 35),
                }
                if motion_ok:
                    au, av = float(U[sl][m].mean()), float(V[sl][m].mean())
                    item["speed_kmh"], item["bearing_deg"] = _bot_speed_bearing(au, av)
                areas_out.append(item)

    grid_meta = None
    if blobs:
        OUTPUT_BOT_GRID.parent.mkdir(parents=True, exist_ok=True)
        # mtime=0 keeps the file byte-identical when the data is, so the FTP step skips re-uploading it
        with open(OUTPUT_BOT_GRID, "wb") as fh:
            with gzip.GzipFile(fileobj=fh, mode="wb", mtime=0) as gz:
                gz.write(b"".join(b.tobytes() for b in blobs))
        grid_meta = {
            "file": OUTPUT_BOT_GRID.name, "compression": "gzip",
            "lat_north": round(float(lats[0]), 4), "lon_west": round(float(lons[0]), 4),
            "step_deg": step, "nrows": len(lats), "ncols": len(lons), "layers": layers_meta,
        }
    elif OUTPUT_BOT_GRID.exists():
        OUTPUT_BOT_GRID.unlink()

    mean_speed, mean_bearing = _bot_speed_bearing(mean_u, mean_v)
    doc = {
        "schema": 1,
        "version": now_utc.strftime("%Y%m%dT%H%M%SZ"),
        "generated_utc": now_utc.isoformat(timespec="seconds"),
        "generated_ist": now_utc.astimezone(_IST).strftime("%d %b %Y %H:%M IST"),
        "stale_after_min": STALE_OBS_MINUTES,
        "radars": radars,
        "grid": grid_meta,
        "motion": {
            "available": motion_ok,
            "overall_speed_kmh": mean_speed if motion_ok else None,
            "overall_bearing_deg": mean_bearing if motion_ok else None,
            "note": ("bearing is the direction storms are moving TOWARD, in degrees clockwise from north"
                     if motion_ok else
                     "no radar had two usable frames, so only the observed layer is written this cycle"),
        },
        "cells": cells_out,
        "rain_areas": areas_out,
        "how_to_read": (
            "Grid file: gunzip, then one unsigned byte per node, layers stacked in the order listed. "
            "byte position = layer_index * nrows * ncols + row * ncols + col, with "
            "row = round((lat_north - lat) / step_deg) and col = round((lon - lon_west) / step_deg). "
            "0 = radar coverage but no echo, 255 = no usable radar coverage (do not read as dry), "
            "1-254 = reflectivity in dBZ. IMD's images start at 20 dBZ, so rain lighter than that is not seen. "
            "Rough guide: 20-29 light, 30-39 moderate, 40-49 heavy, 50+ very heavy. "
            "plus_NN layers are the observed picture moved along the measured motion with no growth or decay, "
            "so check a neighbourhood that widens with lead time (about 5 km at 30 min, 10 km at 60, 15 km at 90). "
            "Do not answer from this file if generated_utc is more than 30 minutes old. "
            "Fetch the grid file with ?v=<version> to avoid a cached copy."),
    }
    OUTPUT_BOT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_BOT_JSON.write_text(json.dumps(doc, indent=1))
    print(f"[bot-export] wrote {OUTPUT_BOT_JSON} ({len(cells_out)} cells, {len(areas_out)} rain areas, "
          f"{len(layers_meta)} grid layers, motion {'ok' if motion_ok else 'unavailable'})")
    return doc


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
            existing = archived_frames(product, state)
            # A frame whose obs_time couldn't be read is treated as fresh
            # everywhere downstream (see build_forecast_map's
            # fresh_products) -- right for a frame fetched this cycle, but
            # the newest archived frame can now BE such a frame, and if
            # fetches keep failing it would stay on the map as "time
            # unavailable" indefinitely. Its file name still records when
            # it was polled, so refuse the fallback once that is older
            # than the same staleness limit everything else uses.
            if existing and _obs_time_from_archive_filename(existing[-1]) is None:
                polled_s = _archive_frame_sort_seconds(existing[-1])
                age_min = (time.time() - polled_s) / 60.0
                if age_min > STALE_OBS_MINUTES:
                    print(f"[poll] {product}: fetch failed, and the last archived frame {existing[-1]} "
                          f"has no readable time and was polled {age_min:.0f} min ago -- not using it")
                    existing = []
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

    # Past 3h/6h/12h radar loop -- frame captured every cycle (reusing the
    # _last_dbz this cycle already fetched/decoded above, no extra work),
    # and the GIFs are now REBUILT every cycle too, matching the ~10-minute
    # cadence the radar data itself updates on -- see this module's
    # docstring above capture_mosaic_frame() for why this used to be gated
    # to once an hour and why that's no longer necessary.
    # Data copy of this cycle for the query bot / social alerts -- see the BOT
    # DATA EXPORT block above. Written into output/ so the existing FTP step
    # uploads it. Never allowed to take the map down with it.
    try:
        export_bot_data(POLLED_PRODUCTS)
    except Exception as e:
        print(f"[bot-export] failed (the map above is unaffected): {e!r}")

    capture_mosaic_frame()
    publish_radar_loop(rebuild=True)
    print("Updated radar loop (3h/6h/12h)")

    # .htaccess (Cache-Control headers -- see that file's own comments)
    # needs to land in output/ on EVERY cycle too, for the same reason
    # radar_loop/'s files do: output/ is empty on every fresh checkout,
    # and FTP-Deploy-Action deletes any remote file missing from a run's
    # local-dir. It's static (committed, not generated), so this is a
    # plain copy, not a rebuild.
    htaccess_src = Path(".htaccess")
    if htaccess_src.exists():
        (OUTPUT_HTML.parent / ".htaccess").write_bytes(htaccess_src.read_bytes())


if __name__ == "__main__":
    run_pipeline()
