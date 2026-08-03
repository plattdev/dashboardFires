"""
Spanish Wildfire Tracker — NASA FIRMS
======================================================
Streamlit dashboard that visualises active fire detections across Spain
using NASA FIRMS satellite data (VIIRS) and overlays Spanish protected
natural areas (ENP) from MITECO GeoParquet files.  It also integrates a
100 m population raster from WorldPop to compute fire-exposure KPIs,
and EFFIS burnt-area polygons from the Copernicus Emergency Management
Service for cumulative 2026 seasonal data.

Data sources
------------
- NASA FIRMS NRT & SP VIIRS feeds (CSV via API)
- MITECO ENP boundaries (GeoParquet, EPSG:32628 / EPSG:32630)
- WorldPop Population Grid 100 m (GeoTIFF, EPSG:4326, CC BY 4.0 / ODbL)
- EFFIS / Copernicus EMS burnt areas (Shapefile via JRC REST API)
"""

# ── Standard-library imports ──────────────────────────────────────────────────
from streamlit.elements.lib import built_in_chart_utils
import datetime                        # date arithmetic for date picker & FIRMS query window
import hashlib                         # MD5 hash of fire data for caching KPI results
import io                              # in-memory bytes buffer for reading EFFIS zip
import json                            # convert GeoPandas GeoJSON strings into dicts for PyDeck
from concurrent.futures import ThreadPoolExecutor  # background prefetch of heavy datasets
from pathlib import Path               # cross-platform file-system path handling

# ── Third-party imports ───────────────────────────────────────────────────────
import geopandas as gpd                # spatial joins & CRS reprojections (ENP layer)
import numpy as np                     # vectorised array math (colors, population masks)
import pandas as pd                    # DataFrames — core data structure throughout the app
import pydeck as pdk                   # 3D WebGL map rendering inside Streamlit - the engine powering the entire interactive map- without it none of the ENP, burnt area or population map would render
import rasterio                        # read the WorldPop population GeoTIFF raster
import requests                        # HTTP client for downloading EFFIS burnt-area data
from rasterio.enums import Resampling  # resampling strategy when downscaling the raster
from scipy.spatial import cKDTree      # fast spatial proximity queries for population KPIs
import streamlit as st                 # the web-app framework that runs everything


# ── Page Configuration ────────────────────────────────────────────────────────
# Must be the very first Streamlit command in the script.
st.set_page_config(page_title="Spanish Wildfire Tracker", layout="wide")

# NB: If I want to create a separate CSS style this way
# def _inject_css(path: Path) -> None:
#     """Read a CSS file and inject it into the Streamlit page."""
#     st.markdown(f"<style>{path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
# _inject_css(Path("css/styles.css"))

# Inline CSS — hides the Deploy button & tightens padding.
st.markdown(
    """
    <style>
    /* Hide the Deploy button */
    [data-testid="stAppDeployButton"],
    .stAppDeployButton { display: none !important; }

    /* Reduce whitespace around the main content area */
    .main .block-container,
    [data-testid="stMainBlockContainer"],
    .stMainBlockContainer {
        padding-top: 1rem !important;
        padding-bottom: 0 !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }

    /* Tighten spacing between vertical blocks */
    div[data-testid="stVerticalBlock"] > div { gap: 0.1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
POPULATION_TIFF = DATA_DIR / "SpainPopulation2026_CN_100m_R2025A_v1.tif"

# Downsample the 100 m raster to ~1 km for the heatmap layer (10× = 1000 m).
_POP_DOWNSAMPLE_FACTOR = 10

# Approximate degree buffer for ~5 km at Spain's latitude (~40°N).
_DEG_BUFFER_5KM = 0.045




# ═════════════════════════════════════════════════════════════════════════════
#  CACHED DATA LOADERS
#  --------------------------------------------------------------------------
#  Streamlit re-runs the entire script on every user interaction.
#  @st.cache_resource keeps heavy objects (GeoDataFrames, file handles) in
#  memory across reruns.  @st.cache_data caches serialisable return values
#  (dicts, DataFrames) and is invalidated when inputs change.
# ═════════════════════════════════════════════════════════════════════════════


@st.cache_resource()
def load_enp_geodataframes() -> dict[str, gpd.GeoDataFrame]:
    """Load ENP (Espacios Naturales Protegidos) GeoParquet files from disk.

    Returns a dict keyed by region name ('canarias', 'peninsula') with each
    GeoDataFrame in its original projected CRS (UTM 28N / 30N).  These
    projected CRS are required later for accurate distance-based spatial joins.
    """
    files = {"canarias": "Enp2025_c.parquet", "peninsula": "Enp2025_p.parquet"}
    return {
        key: gpd.read_parquet(DATA_DIR / fname)
        for key, fname in files.items()
        if (DATA_DIR / fname).exists()
    }


@st.cache_data()
def build_enp_geojson() -> dict | None:
    """Build a GeoJSON dict of all ENP areas in WGS-84 for PyDeck rendering.

    Simplifies the geometries by 150 m in their native metric CRS (UTM) to
    reduce polygon vertex count, then reprojects to EPSG:4326 for the map.
    Returns None if no ENP data is available.
    """
    enp_gdfs = load_enp_geodataframes()
    if not enp_gdfs:
        return None

    parts: list[gpd.GeoDataFrame] = []
    for gdf in enp_gdfs.values():
        simplified = gdf.copy()
        simplified["geometry"] = simplified["geometry"].simplify(150)
        parts.append(simplified.to_crs(epsg=4326))

    combined = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True), geometry="geometry", crs="EPSG:4326"
    )

    # Plain-text tooltip for PyDeck hover
    if {"SITE_NAME", "ODESIGNATE"}.issubset(combined.columns):
        combined["tooltip_text"] = (
            combined["ODESIGNATE"].fillna("Espacio Protegido")
            + " — "
            + combined["SITE_NAME"].fillna("")
        )

    return json.loads(combined.to_json())


@st.cache_resource(show_spinner=False)
def load_population_heatmap_data() -> pd.DataFrame | None:
    """Load the WorldPop raster, downsample to ~1 km, return a DataFrame.

    Uses rasterio's decimated read so the full 93 MB raster is never held
    in memory at full resolution.  Returns columns: lat, lon, population.
    Returns None if the TIFF file is missing.
    """
    if not POPULATION_TIFF.exists():
        return None

    with rasterio.open(POPULATION_TIFF) as src:
        out_shape = (
            src.height // _POP_DOWNSAMPLE_FACTOR,
            src.width  // _POP_DOWNSAMPLE_FACTOR,
        )
        data = src.read(1, out_shape=out_shape, resampling=Resampling.average)

        # Scale average back to approximate sum for the aggregated cell
        data *= _POP_DOWNSAMPLE_FACTOR ** 2
        nodata = src.nodata
        transform = src.transform * src.transform.scale(
            src.width  / data.shape[1],
            src.height / data.shape[0],
        )

    # Keep only valid (non-nodata, > 0) cells
    mask = (data > 0) if nodata is None else ((data != nodata) & (data > 0))
    rows, cols = np.nonzero(mask)
    xs, ys = rasterio.transform.xy(transform, rows, cols)

    return pd.DataFrame({
        "lat": np.asarray(ys, dtype=np.float32),
        "lon": np.asarray(xs, dtype=np.float32),
        "population": data[rows, cols].astype(np.float32),
    })


@st.cache_resource(show_spinner=False)
def _open_population_raster() -> rasterio.DatasetReader | None:
    """Keep the population raster file handle open for point-sampling.

    Cached with @st.cache_resource so the handle is shared across reruns
    instead of opening/closing the 93 MB file every time.
    Returns None if the TIFF file is missing.
    """
    if not POPULATION_TIFF.exists():
        return None
    return rasterio.open(POPULATION_TIFF)


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_effis_burnt_areas() -> gpd.GeoDataFrame | None:
    """Download the EFFIS current-season burnt-area shapefile from JRC.

    Downloads the zip archive, reads the shapefile with GeoPandas,
    filters for Spain (COUNTRY_CODE == 'ES') in 2026, and reprojects to
    EPSG:4326 for map display. Caches to local parquet for fast subsequent loads.

    Returns
    -------
    gpd.GeoDataFrame in EPSG:4326 with burnt-area polygons, or None on error.
    """
    cache_path = DATA_DIR / "effis_burnt_areas_2026.parquet"
    if cache_path.exists():
        try:
            return gpd.read_parquet(cache_path)
        except Exception:
            pass

    url = (
        "https://maps.effis.emergency.copernicus.eu/effis"
        "?service=WFS&request=getfeature&typename=ms:modis.ba.poly"
        "&version=1.1.0&outputformat=SHAPEZIP"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=120)
        resp.raise_for_status()

        # Read directly from the zip archive in memory
        gdf = gpd.read_file(io.BytesIO(resp.content))
    except Exception as exc:
        print(f"Error fetching EFFIS data: {exc}")
        return None

    if gdf.empty:
        return None

    # Filter for Spain
    country_col = None
    for candidate in ["COUNTRY", "COUNTRY_CODE", "country"]:
        if candidate in gdf.columns:
            country_col = candidate
            break

    if country_col:
        gdf = gdf[gdf[country_col].astype(str).str.upper().isin(["ES", "SPAIN"])]

    # Filter by date — EFFIS uses INITIALDATE or FIREDATE
    date_col = None
    for candidate in ["INITIALDATE", "FIREDATE", "INIT_DATE"]:
        if candidate in gdf.columns:
            date_col = candidate
            break

    if date_col is not None:
        gdf[date_col] = pd.to_datetime(gdf[date_col], errors="coerce")
        gdf = gdf[gdf[date_col] >= "2026-01-01"]

    if gdf.empty:
        return None

    # Reproject to WGS-84 for PyDeck
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    # Simplify for rendering performance (100 m tolerance in degrees ≈ 0.001°)
    gdf["geometry"] = gdf["geometry"].simplify(0.001)

    # Build tooltip
    area_col = None
    for candidate in ["AREA_HA", "AREA", "area_ha"]:
        if candidate in gdf.columns:
            area_col = candidate
            break

    if area_col and date_col:
        gdf["tooltip_text"] = (
            "Burnt Area — "
            + gdf[date_col].dt.strftime("%Y-%m-%d").fillna("")
            + " — "
            + gdf[area_col].astype(str)
            + " ha"
        )
    else:
        gdf["tooltip_text"] = "2026 Burnt Area"

    # Cache to local parquet
    try:
        if date_col is not None:
            gdf[date_col] = gdf[date_col].astype(str)
        # Convert any other problematic columns to string if necessary
        for col in gdf.columns:
            if col not in ["geometry", "tooltip_text", area_col, country_col]:
                gdf[col] = gdf[col].astype(str)
        gdf.to_parquet(cache_path)
    except Exception as exc:
        print(f"Warning: could not cache EFFIS data to parquet: {exc}")

    return gdf


# ── Background Prefetch ───────────────────────────────────────────────────────
# Kick off data loading in background threads *now*, while Streamlit renders
# the title and controls.  By the time the map needs this data, the cached
# loaders will already have completed and the results sit in the cache.
_prefetch_pool = ThreadPoolExecutor(max_workers=4)
_prefetch_pool.submit(load_enp_geodataframes)
_prefetch_pool.submit(build_enp_geojson)
_prefetch_pool.submit(load_population_heatmap_data)
_prefetch_pool.submit(fetch_effis_burnt_areas)


# ── FIRMS API Loader ──────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_europe_fires(api_key: str, start_date: str, day_range: int, source: str) -> pd.DataFrame:
    """Fetch VIIRS fire detections for Europe from the NASA FIRMS Area API.

    Parameters
    ----------
    api_key    : NASA FIRMS MAP_KEY (from .streamlit/secrets.toml).
    start_date : ISO date string (YYYY-MM-DD) for the query start.
    day_range  : Number of days forward from start_date.
    source     : FIRMS dataset id ('VIIRS_SNPP_NRT' or 'VIIRS_SNPP_SP').

    Returns
    -------
    pd.DataFrame filtered to nominal/high confidence detections only.
    Empty DataFrame on error or no data.
    """
    bbox = "-20.0,25.0,35.0,65.0"  # Europe + Mediterranean + North Africa
    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{api_key}/{source}/{bbox}/{day_range}/{start_date}"
    )
    try:
        data = pd.read_csv(url)
    except Exception as exc:
        st.error(f"Error connecting to NASA FIRMS API: {exc}")
        return pd.DataFrame()

    if data.empty or "confidence" not in data.columns:
        return pd.DataFrame()

    # FIRMS uses 'n'/'h' for NRT feeds, 'nominal'/'high' for SP feeds
    return data[data["confidence"].isin(["nominal", "high", "n", "h"])]


# ═════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════


def _hash_fire_df(fires_df: pd.DataFrame) -> str:
    """Create a lightweight hash of fire data for use as a cache key.

    Hashes only the coordinate columns so the KPI cache is invalidated when
    fire locations change but not when unrelated columns differ.
    """
    raw = pd.util.hash_pandas_object(fires_df[["latitude", "longitude"]]).values.tobytes()
    return hashlib.md5(raw).hexdigest()


@st.cache_data(show_spinner=False)
def compute_eco_fires(fire_hash: str, _fires_df: pd.DataFrame) -> int:
    """Count fires falling inside a Spanish protected natural area (ENP).

    Creates fire point geometries with the vectorised gpd.points_from_xy()
    (C-backed, ~10× faster than a Python loop of shapely.Point calls), then
    performs a spatial join with the 'intersects' predicate — only fires whose
    location falls within (or touches the boundary of) an ENP polygon are counted.

    Parameters
    ----------
    fire_hash : MD5 hash of the fire coordinates (used as cache key).
    _fires_df  : Fire DataFrame with 'latitude' and 'longitude' columns.

    Returns
    -------
    int — number of unique fires inside any ENP boundary.
    """
    enp_gdfs = load_enp_geodataframes()
    if not enp_gdfs or _fires_df.empty:
        return 0

    # Vectorised geometry creation — no Python loop needed
    fires_gdf = gpd.GeoDataFrame(
        _fires_df,
        geometry=gpd.points_from_xy(_fires_df["longitude"], _fires_df["latitude"]),
        crs="EPSG:4326",
    )

    hit_indices: set[int] = set()
    for enp_gdf in enp_gdfs.values():
        fires_projected = fires_gdf.to_crs(enp_gdf.crs)
        joined = gpd.sjoin(fires_projected, enp_gdf, predicate="intersects")
        hit_indices.update(joined.index.tolist())

    return len(hit_indices)


@st.cache_data(show_spinner=False)
def compute_population_kpis(fire_hash: str, _fires_df: pd.DataFrame) -> dict[str, float | int]:
    """Compute population-exposure KPIs by sampling the WorldPop raster.

    Uses scipy's cKDTree for a single vectorised proximity query instead of
    iterating over each fire in a Python loop.  This is orders of magnitude
    faster when there are hundreds of fire detections.

    Parameters
    ----------
    fire_hash : MD5 hash of the fire coordinates (used as cache key).
    _fires_df  : Fire DataFrame with 'latitude' and 'longitude' columns.

    Returns
    -------
    dict with keys:
        exposed_pop  – estimated people within ~5 km of any active fire.
        mean_density – average population value at fire-pixel locations.
    """
    result: dict[str, float | int] = {"exposed_pop": 0, "mean_density": 0.0}

    # ── 1. Exposed population (vectorised with cKDTree) ───────────────────
    pop_df = load_population_heatmap_data()
    if pop_df is not None and not pop_df.empty and not _fires_df.empty:
        # Build a KD-tree of population cell coordinates
        pop_coords = np.column_stack([pop_df["lat"].values, pop_df["lon"].values])
        tree = cKDTree(pop_coords)

        # Query all population cells within the degree buffer of any fire
        fire_coords = np.column_stack([_fires_df["latitude"].values, _fires_df["longitude"].values])
        nearby_sets = tree.query_ball_point(fire_coords, r=_DEG_BUFFER_5KM)

        # Flatten to unique population-cell indices and sum their population
        nearby_indices = set()
        for idx_list in nearby_sets:
            nearby_indices.update(idx_list)

        if nearby_indices:
            result["exposed_pop"] = int(pop_df["population"].values[list(nearby_indices)].sum())

    # ── 2. Mean density at fire locations (point-samples the full raster) ─
    src = _open_population_raster()
    if src is not None and not _fires_df.empty:
        try:
            coords = list(zip(_fires_df["longitude"], _fires_df["latitude"]))
            sampled = np.array([val[0] for val in src.sample(coords)], dtype=np.float32)
            if src.nodata is not None:
                sampled[sampled == src.nodata] = 0.0
            sampled[sampled < 0] = 0.0
            result["mean_density"] = float(np.mean(sampled)) if len(sampled) else 0.0
        except Exception:
            pass  # leave mean_density at 0.0

    return result


# ── Color assignment (vectorised) ─────────────────────────────────────────────

_COLOR_BINS = [
    (pd.Timedelta(hours=24), [139, 0,   0, 200]),   # Dark Red  — ≤ 24 h
    (pd.Timedelta(days=2),   [255, 0,   0, 200]),    # Red       — 24–48 h
    (pd.Timedelta(days=4),   [255, 165, 0, 200]),    # Orange    — 2–4 days
]
_DEFAULT_COLOR = [255, 255, 0, 200]                    # Yellow    — > 4 days


def assign_colors(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'color_rgba' column based on how recent each fire detection is.

    Uses np.select for vectorised performance — no row-wise .apply() needed.
    Colors go from dark red (most recent) through red, orange, to yellow.
    """
    dt_max = df["acq_datetime"].max()
    diff = dt_max - df["acq_datetime"]

    conditions = [diff <= threshold for threshold, _ in _COLOR_BINS]
    choices    = [color for _, color in _COLOR_BINS]

    indices    = np.select(conditions, range(len(choices)), default=len(choices))
    all_colors = choices + [_DEFAULT_COLOR]
    df["color_rgba"] = [all_colors[int(i)] for i in indices]
    return df


# ── Spain bounding-box filter ────────────────────────────────────────────────

def filter_spain(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only fires inside Spain (mainland + Balearics + Canary Islands).

    Uses simple lat/lon bounding boxes — fast and sufficient for this use case.
    """
    mainland = (
        (df["latitude"] >= 36.0)  & (df["latitude"] <= 44.0)
        & (df["longitude"] >= -9.5) & (df["longitude"] <= 4.5)
    )
    canaries = (
        (df["latitude"] >= 27.0)  & (df["latitude"] <= 30.0)
        & (df["longitude"] >= -19.0) & (df["longitude"] <= -13.0)
    )
    return df[mainland | canaries]


# ═════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("""
    <style>
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 0rem !important;
    }
    footer {visibility: hidden;} /* Hide default Streamlit footer to prevent vertical scrolling */
    </style>
""", unsafe_allow_html=True)
st.markdown("#### Spanish Wildfire Tracker — NASA FIRMS")
st.markdown(
    "**Context:** This dashboard tracks active thermal anomalies and cumulative burned area in Spain (sourced from NASA FIRMS and Copernicus EFFIS)."
    " While human ignitions—both negligent and intentional—remain the primary trigger, climate-driven fuel dryness, coupled with rural land abandonment and dense biomass accumulation, enables localized ignitions to rapidly escalate into uncontrollable, high-intensity megafires.   "
)

# ── Controls + KPIs row (side by side) ────────────────────────────────────────
# Left column: date picker + slider.  Right columns: KPI metrics.
# Ratios [1, 0.67, 0.67, 0.67] keep ctrl_col at ~1/3 width (matching the
# histogram column below) so the KPIs start at the map's left edge.
ctrl_col, k1, k2, k3 = st.columns([1, 0.67, 0.67, 0.67], gap="large")

with ctrl_col:
    selected_date  = st.date_input("End date:", datetime.date.today())
    selected_range = st.slider("Days to look back:", min_value=1, max_value=3, value=3)

# Decide FIRMS dataset: SP for data older than 30 days, NRT otherwise
start_date      = selected_date - datetime.timedelta(days=selected_range - 1)
days_from_today = (datetime.date.today() - selected_date).days
source          = "VIIRS_SNPP_SP" if days_from_today > 30 else "VIIRS_SNPP_NRT"

# ── Fetch fire data ───────────────────────────────────────────────────────────
europe_df = fetch_europe_fires(
    st.secrets["FIRMS_API_KEY"],
    start_date.strftime("%Y-%m-%d"),
    selected_range,
    source,
)

if europe_df.empty:
    st.warning("No fire data available for the selected date range.")
    st.stop()

# ── Prepare fire DataFrame ────────────────────────────────────────────────────
europe_df["acq_datetime"] = pd.to_datetime(
    europe_df["acq_date"] + " " + europe_df["acq_time"].astype(str).str.zfill(4),
    format="%Y-%m-%d %H%M",
)

# Plain-text tooltip shown on hover over fire dots
europe_df["tooltip_text"] = (
    "Fire Detection — Date: "
    + europe_df["acq_datetime"].dt.strftime("%Y-%m-%d %H:%M")
)

europe_df = assign_colors(europe_df)

# ── Spain-only subset for KPIs ────────────────────────────────────────────────
spain_df    = filter_spain(europe_df)
total_fires = len(spain_df)
fire_hash   = _hash_fire_df(spain_df)

try:
    eco_fires = compute_eco_fires(fire_hash, spain_df)
except Exception as exc:
    st.error(f"Error processing ENP spatial layer: {exc}")
    eco_fires = 0

try:
    pop_kpis = compute_population_kpis(fire_hash, spain_df)
except Exception as exc:
    st.error(f"Error computing population KPIs: {exc}")
    pop_kpis = {"exposed_pop": 0, "mean_density": 0.0}

# ── EFFIS burnt-area KPI ──────────────────────────────────────────────────────
effis_gdf = fetch_effis_burnt_areas()
effis_total_ha = 0.0
effis_count = 0
if effis_gdf is not None and not effis_gdf.empty:
    effis_count = len(effis_gdf)
    for candidate in ["AREA_HA", "AREA", "area_ha"]:
        if candidate in effis_gdf.columns:
            numeric_area = pd.to_numeric(effis_gdf[candidate], errors="coerce").fillna(0.0)
            effis_total_ha = float(numeric_area.sum())
            break

# ── Render KPIs (in the columns defined above) ───────────────────────────────
with k1:
    st.metric("Total Active Fires (High Conf.)", total_fires)
    st.metric("Fires in Protected Areas", eco_fires)
with k2:
    st.metric("People within 5 km of Fires", f"{pop_kpis['exposed_pop']:,} people" if pop_kpis["exposed_pop"] else "—")
    st.metric("Avg. Population at Fire Sites", f"{pop_kpis['mean_density']:.1f} people / ha" if pop_kpis["mean_density"] else "—")
with k3:
    st.metric("Total Burnt Area 2026 (EFFIS)", f"{effis_total_ha:,.0f} ha" if effis_total_ha else "—")
    st.metric("Total EFFIS Fires (2026)", effis_count if effis_count else "—")

# Global KPI font size reduction and robust Toggle CSS
st.markdown("""
<style>
div[data-testid="stMetricValue"] {
    font-size: 1.2rem !important;
}
div[data-testid="stMetricValue"] > div {
    font-size: 1.2rem !important;
}
div[data-testid="stMetricLabel"] p {
    font-size: 0.8rem !important;
}

/* Protected Areas Toggle */
div[data-testid="stElementContainer"]:has(.enp-anchor) + div[data-testid="stElementContainer"] div[data-testid="stToggle"] input:checked + div {
    background-color: #6B8E23 !important;
}
div[data-testid="stElementContainer"]:has(.enp-anchor) + div[data-testid="stElementContainer"] div[data-testid="stWidgetLabel"] p {
    color: #6B8E23 !important;
    font-weight: 500;
}

/* Population Toggle */
div[data-testid="stElementContainer"]:has(.pop-anchor) + div[data-testid="stElementContainer"] div[data-testid="stToggle"] input:checked + div {
    background-color: #9E7C97 !important;
}
div[data-testid="stElementContainer"]:has(.pop-anchor) + div[data-testid="stElementContainer"] div[data-testid="stWidgetLabel"] p {
    color: #9E7C97 !important;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# EFFIS burnt areas are now permanent
show_effis = True

# ── Map & Histogram Layout ────────────────────────────────────────────────────
hist_col, map_col = st.columns([1, 1.4], gap="large")

with map_col:
    # ── Legend & Layer toggles ────────────────────────────────────────────────
    t_leg, t1, t2 = st.columns([2.8, 1, 1], vertical_alignment="center")
    with t_leg:
        st.markdown(
            "<div style='margin: 0px; font-size: 0.85em; display: flex; align-items: center; gap: 6px; flex-wrap: wrap;'>"
            "<b>Legend:</b> "
            '<span style="color:#8B0000; font-size: 1.1em; vertical-align: middle;">⬤</span> ≤ 24 h &nbsp;|&nbsp; '
            '<span style="color:#FF0000; font-size: 1.1em; vertical-align: middle;">⬤</span> 24–48 h &nbsp;|&nbsp; '
            '<span style="color:#FFA500; font-size: 1.1em; vertical-align: middle;">⬤</span> 2–4 days &nbsp;|&nbsp; '
            '<span style="color:#FFFF00; font-size: 1.1em; vertical-align: middle;">⬤</span> > 4 days &nbsp;|&nbsp; '
            '<span style="display: inline-block; width: 14px; height: 12px; background-color: #B22222; vertical-align: middle; margin-right: 3px; border-radius: 1px;"></span> Burnt Areas'
            "</div>",
            unsafe_allow_html=True,
        )
    with t1:
        st.markdown('<div class="enp-anchor"></div>', unsafe_allow_html=True)
        show_enp = st.toggle("Show Protected Areas", value=False)
    with t2:
        st.markdown('<div class="pop-anchor"></div>', unsafe_allow_html=True)
        show_pop = st.toggle("Show Population", value=False)

# ── Build map layers ──────────────────────────────────────────────────────────
layers: list[pdk.Layer] = []

# Fire scatter dots (always shown)
layers.append(
    pdk.Layer(
        "ScatterplotLayer",
        data=europe_df,
        get_position="[longitude, latitude]",
        get_color="color_rgba",
        get_radius=1000,
        radius_min_pixels=4,
        radius_max_pixels=8,
        pickable=True,
        opacity=0.8,
        filled=True,
    )
)

# Protected natural areas (green polygons)
if show_enp:
    enp_geojson = build_enp_geojson()
    if enp_geojson:
        layers.append(
            pdk.Layer(
                "GeoJsonLayer",
                data=enp_geojson,
                pickable=True,
                stroked=True,
                filled=True,
                get_fill_color="[34, 139, 34, 40]",
                get_line_color="[34, 139, 34, 255]",
                line_width_min_pixels=1,
            )
        )

# EFFIS burnt areas (dark red polygons)
if show_effis and effis_gdf is not None and not effis_gdf.empty:
    effis_geojson = json.loads(effis_gdf.to_json())
    layers.insert(
        0,
        pdk.Layer(
            "GeoJsonLayer",
            data=effis_geojson,
            pickable=True,
            stroked=True,
            filled=True,
            get_fill_color="[178, 34, 34, 80]",     # firebrick, semi-transparent
            get_line_color="[178, 34, 34, 200]",
            line_width_min_pixels=1,
        ),
    )

# Population heatmap (inserted at index 0 so it renders behind everything)
if show_pop:
    pop_df = load_population_heatmap_data()
    if pop_df is not None and not pop_df.empty:
        layers.insert(
            0,
            pdk.Layer(
                "HeatmapLayer",
                data=pop_df,
                get_position="[lon, lat]",
                get_weight="population",
                radiusPixels=35,
                intensity=1,
                threshold=0.05,
                opacity=0.6,
                color_range=[
                    [0, 0, 128], [0, 128, 255], [0, 255, 128],
                    [255, 255, 0], [255, 128, 0], [255, 0, 0],
                ],
            ),
        )
    else:
        st.warning("Population raster not found in data/ folder.")


with hist_col:
    if effis_gdf is not None and not effis_gdf.empty:
        st.markdown("#### Monthly Burnt Area (2026)")
        date_c = next((c for c in ["FIREDATE", "INITIALDATE", "INIT_DATE"] if c in effis_gdf.columns), None)
        area_c = next((c for c in ["AREA_HA", "AREA", "area_ha"] if c in effis_gdf.columns), None)

        if date_c and area_c:
            chart_df = effis_gdf.copy()
            chart_df["Month"] = pd.to_datetime(chart_df[date_c]).dt.month_name()
            chart_df["Month_Num"] = pd.to_datetime(chart_df[date_c]).dt.month
            
            # Ensure area is numeric before summing
            chart_df[area_c] = pd.to_numeric(chart_df[area_c], errors="coerce").fillna(0.0)
            
            # Sum up area_ha by month
            monthly = chart_df.groupby(["Month_Num", "Month"], as_index=False)[area_c].sum()
            monthly = monthly.sort_values("Month_Num")
            
            # Format X-axis and Y-axis for better readability
            monthly["Month"] = monthly["Month_Num"].astype(str).str.zfill(2) + " - " + monthly["Month"]
            monthly["Area (x1000 ha)"] = monthly[area_c] / 1000.0
            
            # Properly specify x and y, and limit height to avoid vertical scrolling
            st.bar_chart(monthly, x="Month", y="Area (x1000 ha)", color="#B22222", height=450)
        else:
            st.info("Monthly data not available.")
    else:
        st.info("No EFFIS data available.")

    # Data sources below the bar chart
    st.markdown(
        "<div style='margin-top: 15px; margin-bottom: 0px; font-size: 0.75em; color: gray; line-height: 1.4;'>"
        "<b>Data sources:</b> NASA FIRMS (VIIRS active fires filtered for nominal/high confidence) | MITECO (Protected Natural Areas) | "
        "<a href='https://www.worldpop.org/' style='color: gray; text-decoration: none;'>WorldPop</a> Population Data "
        "(<a href='https://creativecommons.org/licenses/by/4.0/' style='color: gray; text-decoration: none;'>CC BY 4.0</a> / "
        "<a href='https://opendatacommons.org/licenses/odbl/' style='color: gray; text-decoration: none;'>ODbL</a>) | "
        "<a href='https://effis.jrc.ec.europa.eu/' style='color: gray; text-decoration: none;'>EFFIS 2026 (Copernicus EMS)</a> Burnt Areas"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Map Rendering (Right Column) ──────────────────────────────────────────────
# Render the main WebGL interactive map using PyDeck inside the right layout column.
with map_col:
    st.pydeck_chart(
        pdk.Deck(
            # 'layers' contains all active map layers (Scatterplot fire dots, ENP polygons, EFFIS burnt areas, Heatmap)
            layers=layers,
            # Initial camera position centered over Spain (lat: 40.0, lon: -3.0) at zoom level 5
            # pitch=0 sets a flat top-down 2D perspective (increase pitch to e.g. 45 for 3D tilt)
            initial_view_state=pdk.ViewState(
                latitude=40.0, longitude=-3.0, zoom=5, pitch=0
            ),
            # Interactive hover tooltip styling — displays formatted text when hovering over data features
            tooltip={
                "html": "{tooltip_text}",
                "style": {
                    "backgroundColor": "darkgrey",
                    "color": "white",
                    "fontFamily": "'Arial', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
                    "fontSize": "14px",
                    "fontWeight": "500",
                    "borderRadius": "8px",
                    "padding": "8px 12px",
                    "boxShadow": "0 10px 15px -3px rgba(0, 0, 0, 0.3)",
                    "border": "1px solid rgba(255, 255, 255, 0.15)",
                },
            },
            height=680, # Larger square map height
        ),
        # Automatically expand map width to fill the column container
        use_container_width=True
    )
    

