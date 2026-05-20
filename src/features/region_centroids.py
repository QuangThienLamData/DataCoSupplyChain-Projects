"""Lat/lon centroids for the regions we need to score storm exposure on.

For **hurricane-belt states** (Gulf + Atlantic coast) we use the largest
coastal-metro centroid: that's where most population, ports, and economic
activity sit, and it's where landfalling storms actually disrupt
operations. A population-weighted state centroid would put Texas near
Waco — far enough inland that Harvey looks ~330 km away even though it
parked over Houston for a week.

For **non-coastal states** we use the geographic centroid; storms don't
realistically reach them, so the exact point barely matters.

Sources: Wikipedia metro coordinates (coastal); US geographic centroids
(interior).
"""
from __future__ import annotations

# (lat, lon). Hurricane-belt states use their primary coastal metro;
# interior states use the geographic centroid.
US_STATE_CENTROIDS: dict[str, tuple[float, float]] = {
    # --- Gulf Coast + Atlantic coast (coastal-metro proxy) ---
    "TX": (29.7604, -95.3698),    # Houston
    "LA": (29.9511, -90.0715),    # New Orleans
    "MS": (30.3960, -88.8853),    # Biloxi
    "AL": (30.6954, -88.0399),    # Mobile
    "FL": (27.9506, -82.4572),    # Tampa (central peninsula coast)
    "GA": (32.0809, -81.0912),    # Savannah
    "SC": (32.7765, -79.9311),    # Charleston
    "NC": (34.2257, -77.9447),    # Wilmington
    "VA": (36.8529, -75.9780),    # Virginia Beach
    "MD": (38.9784, -76.4922),    # Annapolis
    "DE": (38.7726, -75.1180),    # Lewes
    "NJ": (39.3643, -74.4229),    # Atlantic City
    "NY": (40.7128, -74.0060),    # New York City
    "CT": (41.3083, -72.9279),    # New Haven
    "RI": (41.4901, -71.3128),    # Newport
    "MA": (41.6688, -70.2962),    # Cape Cod
    # --- Caribbean / Pacific exposure ---
    "HI": (21.3099, -157.8581),   # Honolulu
    # --- Interior states (geographic centroid, no realistic exposure) ---
    "AR": (34.8938, -92.4426),
    "AZ": (33.7712, -111.3877),
    "CA": (36.116203, -119.681564),
    "CO": (39.0598, -105.3111),
    "DC": (38.8951, -77.0369),
    "IA": (42.0115, -93.2105),
    "ID": (44.2405, -114.4788),
    "IL": (40.3495, -88.9861),
    "IN": (39.8494, -86.2583),
    "KS": (38.5266, -96.7265),
    "KY": (37.6681, -84.6701),
    "MI": (43.3266, -84.5361),
    "MN": (45.6945, -93.9002),
    "MO": (38.4561, -92.2884),
    "MT": (46.9219, -110.4544),
    "ND": (47.5289, -99.7840),
    "NE": (41.1254, -98.2681),
    "NH": (43.4525, -71.5639),
    "NM": (34.8405, -106.2485),
    "NV": (38.3135, -117.0554),
    "OH": (40.3888, -82.7649),
    "OK": (35.5653, -96.9289),
    "OR": (44.5720, -122.0709),
    "PA": (40.5908, -77.2098),
    "SD": (44.2998, -99.4388),
    "TN": (35.7478, -86.6923),
    "UT": (40.1500, -111.8624),
    "VT": (44.0459, -72.7107),
    "WA": (47.4009, -121.4905),
    "WI": (44.2685, -89.6165),
    "WV": (38.4912, -80.9545),
    "WY": (42.7559, -107.3025),
}

# Country-level centroids (capital/main population mass). DataCo's
# customer base includes Puerto Rico explicitly; the rest cover the
# Atlantic basin's exposed coast in case we expand.
COUNTRY_CENTROIDS: dict[str, tuple[float, float]] = {
    # Spanish-labelled keys match DataCo's customer_country values
    "Puerto Rico": (18.220833, -66.590149),  # San Juan area
    "EE. UU.": (37.0902, -95.7129),          # geographic US centroid (fallback)
    "Mexico": (19.4326, -99.1332),
    "Cuba": (21.5218, -77.7812),
    "República Dominicana": (18.7357, -70.1627),
    "Haiti": (18.9712, -72.2852),
    "Jamaica": (18.1096, -77.2975),
    "Honduras": (15.2000, -86.2419),
    "Nicaragua": (12.8654, -85.2072),
    "Belize": (17.1899, -88.4976),
    "Bahamas": (25.0343, -77.3963),
}

# Subset of US states that actually appear in DataCo customer_state
# (from the SQL check). All standard 2-letter codes.
DATACO_US_STATES: list[str] = [
    "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA",
    "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD",
    "MI", "MN", "MO", "MT", "NC", "ND", "NJ", "NM", "NV", "NY",
    "OH", "OK", "OR", "PA", "RI", "SC", "TN", "TX", "UT", "VA",
    "WA", "WI", "WV",
]
