"""Months of alkenone production — ``checkSeasonality.m`` (PORTING.md CS-01..03)."""
from __future__ import annotations

import numpy as np

from .utils import as_column

#: Polygons and months, from checkSeasonality.m:23-35. (lon, lat) vertices.
MEDITERRANEAN = (np.array([-5.5, 3, 45, 45, -5.5]), np.array([36.25, 47.5, 47.5, 30, 30]))
NORTH_ATLANTIC = (np.array([-55, -50, 20, 10, -4.5, -4.5]),
                  np.array([48, 70, 70, 62.5, 58.2, 48]))
#: North Pacific longitudes are 0-360, so the polygon can cross the dateline.
NORTH_PACIFIC = (np.array([135, 135, 250, 232, 180]), np.array([45, 70, 70, 52.5, 45]))

ANNUAL = np.arange(1, 13)
MONTHS_MEDITERRANEAN = np.array([1, 2, 3, 4, 5, 11, 12])
MONTHS_NORTH_ATLANTIC = np.array([8, 9, 10])
MONTHS_NORTH_PACIFIC = np.array([6, 7, 8])


def inpolygon(x, y, xv, yv) -> np.ndarray:
    """MATLAB ``inpolygon``: inside **or on the edge** (CS-02).

    Ray casting for the interior, plus an explicit on-segment test, because
    MATLAB counts boundary points as in and a bare ray cast is arbitrary there.

    Args:
        x, y: Query points.
        xv, yv: Polygon vertices; closed or not.

    Returns:
        Boolean array, the shape of `x`.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    xv, yv = np.asarray(xv, dtype=float), np.asarray(yv, dtype=float)
    x1, y1 = xv, yv
    x2, y2 = np.roll(xv, -1), np.roll(yv, -1)
    px, py = x[..., None], y[..., None]
    crosses = (y1 > py) != (y2 > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        xcross = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
    inside = (crosses & (px < xcross)).sum(axis=-1) % 2 == 1
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    scale = np.maximum(np.hypot(x2 - x1, y2 - y1), 1.0)
    on_line = np.abs(cross) <= 1e-12 * scale * np.maximum(1.0, np.abs(px) + np.abs(py))
    within = ((px >= np.minimum(x1, x2)) & (px <= np.maximum(x1, x2))
              & (py >= np.minimum(y1, y2)) & (py <= np.maximum(y1, y2)))
    on_edge = (on_line & within).any(axis=-1)
    return inside | on_edge


def check_seasonality(lat, lon) -> list[np.ndarray]:
    """Months of production at each location (``checkSeasonality.m``).

    Annual (1-12) everywhere except the three regions where TT2018 found the
    calibration to be seasonal. Later regions override earlier ones, in
    MATLAB's order: Mediterranean, North Atlantic, North Pacific.

    Args:
        lat: Latitudes, decimal degrees.
        lon: Longitudes, decimal degrees, -180 to 180 or 0 to 360.

    Returns:
        One array of months per location, like MATLAB's cell array.
    """
    lat, lon = as_column(lat, "lat"), as_column(lon, "lon")
    if lat.shape != lon.shape:
        raise ValueError(f"lat and lon differ in length: {lat.size} vs {lon.size}")
    lon360 = np.where(lon < 0, lon + 360, lon)
    months = [ANNUAL.copy() for _ in range(lat.size)]
    for mask, m in ((inpolygon(lon, lat, *MEDITERRANEAN), MONTHS_MEDITERRANEAN),
                    (inpolygon(lon, lat, *NORTH_ATLANTIC), MONTHS_NORTH_ATLANTIC),
                    (inpolygon(lon360, lat, *NORTH_PACIFIC), MONTHS_NORTH_PACIFIC)):
        for i in np.flatnonzero(mask):
            months[i] = m.copy()
    return months
