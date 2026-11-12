"""
Spanish Wildfire Tracker
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
import datetime                        # date arithmetic for date picker & FIRMS query window
import hashlib                         # MD5 hash of fire data for caching KPI results
import io                              # in-memory bytes buffer for reading EFFIS zip
import json                            # convert GeoPandas GeoJSON strings into dicts for PyDeck
import logging                         # structured runtime diagnostics instead of bare print()
import time                            # check local EFFIS parquet cache age
from pathlib import Path               # cross-platform file-system path handling

# ── Third-party imports ───────────────────────────────────────────────────────
import altair as alt                   # customized chart axes and font sizes
import geopandas as gpd                # spatial joins & CRS reprojections (ENP layer)
import numpy as np                     # vectorised array math (colors, population masks)
import pandas as pd                    # DataFrames — core data structure throughout the app
import pydeck as pdk                   # 3D WebGL map rendering inside Streamlit
import rasterio                        # read the WorldPop population GeoTIFF raster
import requests                        # HTTP client for downloading EFFIS burnt-area data
from rasterio.enums import Resampling  # resampling strategy when downscaling the raster
from scipy.spatial import cKDTree      # fast spatial proximity queries for population KPIs
import streamlit as st                 # the web-app framework that runs everything

logger = logging.getLogger(__name__)


# ── Page Configuration ────────────────────────────────────────────────────────
# Must be the very first Streamlit command in the script.
st.set_page_config(page_title="Spanish Wildfire Tracker", layout="wide")

# ── Redirección automática desde Streamlit Cloud hacia Fly.io ─────────────────
# Si un reclutador accede desde el enlace antiguo de tu CV (dashboardfires.streamlit.app),
# se redirige inmediatamente a la versión rápida 24/7 de Fly.io.
st.markdown("""
<script>
    (function() {
        try {
            var host = window.top.location.hostname;
            if (host.includes("streamlit.app")) {
                window.top.location.replace("https://spanish-wildfire-tracker.fly.dev/");
            }
        } catch (e) {
            // Si el iframe bloquea el acceso a window.top, redirige el frame actual
            if (window.location.hostname.includes("streamlit.app")) {
                window.location.replace("https://spanish-wildfire-tracker.fly.dev/");
            }
        }
    })();
</script>
""", unsafe_allow_html=True)

# ── All CSS in a single block ─────────────────────────────────────────────────
# Consolidated from previously scattered st.markdown(<style>) calls to avoid
# specificity conflicts and make the stylesheet easy to maintain.
st.markdown("""
<style>
/* ── Global background ─────────────────────────────────────────────────────── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"] {
    background-color: #D5D8DC !important;
}

/* ── Hide deploy button & footer ───────────────────────────────────────────── */
[data-testid="stAppDeployButton"],
.stAppDeployButton { display: none !important; }
footer { visibility: hidden; }

/* ── Container padding ─────────────────────────────────────────────────────── */
[data-testid="stMainBlockContainer"] {
    padding: 2rem 2rem 0 2rem !important;
    max-width: 100% !important;
}

/* ── Tighten vertical block spacing ────────────────────────────────────────── */
div[data-testid="stVerticalBlock"] > div { gap: 0.1rem; }

/* ── KPI metric cards ──────────────────────────────────────────────────────── */
.kpi-card {
    background-color: #F0F2F6;
    border-radius: 8px;
    padding: 8px 14px;
    margin-bottom: 6px;
    border: 1px solid rgba(0, 0, 0, 0.05);
}
.kpi-card .kpi-label {
    font-size: 0.72rem;
    font-weight: 500;
    color: #6c757d;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin: 0;
    line-height: 1.3;
}
.kpi-card .kpi-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #666666;
    letter-spacing: -0.02em;
    margin: 2px 0 0 0;
    line-height: 1.2;
}

/* ── Multiselect tag colours ───────────────────────────────────────────────── */
span[data-baseweb="tag"]:has(span[title="Protected Areas"]) {
    background-color: #dcedc8 !important;
}
span[data-baseweb="tag"]:has(span[title="Population"]) {
    background-color: #d1c4e9 !important;
}
span[data-baseweb="tag"] span,
span[data-baseweb="tag"] svg {
    color: #000000 !important;
    fill: #000000 !important;
}

/* ── Compact multiselect widget ────────────────────────────────────────────── */
div[data-testid="stMultiSelect"] {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}
div[data-testid="stMultiSelect"] label {
    display: none !important;
}
div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
    font-size: 0.85rem !important;
    padding-top: 2px !important;
    padding-bottom: 2px !important;
}
span[data-baseweb="tag"] {
    height: 24px !important;
    font-size: 0.78rem !important;
    padding-left: 6px !important;
    padding-right: 6px !important;
}

/* ── Chart styling ─────────────────────────────────────────────────────────── */
[data-testid="stArrowVegaLiteChart"],
[data-testid="stVegaLiteChart"] {
    border-radius: 8px !important;
    overflow: hidden !important;
    border: 1px solid rgba(0, 0, 0, 0.05) !important;
}

/* ── Reusable utility classes ──────────────────────────────────────────────── */
.spacer-sm { margin-top: 5px; }
.spacer-md { margin-top: 15px; }

.status-note {
    background-color: #F0F2F6;
    border-left: 3px solid #6c757d;
    border-radius: 6px;
    padding: 6px 12px;
    margin-top: 6px;
    margin-bottom: 4px;
    font-size: 0.8rem;
    color: #495057;
    line-height: 1.4;
}

.legend-bar {
    margin: 0;
    font-size: 0.85em;
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
}
.legend-swatch {
    display: inline-block;
    width: 14px;
    height: 12px;
    background-color: #B22222;
    vertical-align: middle;
    margin-right: 3px;
    border-radius: 1px;
}

.data-sources {
    margin-top: 15px;
    margin-bottom: 0;
    font-size: 0.75em;
    color: gray;
    line-height: 1.4;
}
.data-sources a {
    color: gray;
    text-decoration: none;
}

/* ── PyDeck map iframe responsive height ───────────────────────────────────── */
[data-testid="stDeckGlJsonChart"] iframe {
    min-height: 480px;
}
@media (max-width: 768px) {
    [data-testid="stDeckGlJsonChart"] iframe {
        min-height: 75vh !important;
        height: 75vh !important;
    }
}
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
POPULATION_TIFF = DATA_DIR / "SpainPopulation2026_CN_100m_R2025A_v1.tif"
POPULATION_PARQUET = DATA_DIR / "spain_population_1km.parquet"
EFFIS_PARQUET = DATA_DIR / "effis_burnt_areas_2026.parquet"
_CURRENT_YEAR = datetime.date.today().year

# Downsample the 100 m raster to ~1 km for the heatmap layer (10× = 1000 m).
_POP_DOWNSAMPLE_FACTOR = 10

# Approximate degree buffer for ~5 km at Spain's latitude (~40°N).
_DEG_BUFFER_5KM = 0.045

# NASA FIRMS Area API limits and bounding box (Europe + Mediterranean + North Africa)
_FIRMS_MAX_DAYS = 5
_FIRMS_BBOX = "-20.0,25.0,35.0,65.0"
_VALID_CONFIDENCE = ("nominal", "high", "n", "h")

# Candidate column names in EFFIS shapefiles
_EFFIS_DATE_COLS = ("FIREDATE", "INITIALDATE", "INIT_DATE")
_EFFIS_AREA_COLS = ("AREA_HA", "AREA", "area_ha")
_EFFIS_COUNTRY_COLS = ("COUNTRY", "COUNTRY_CODE", "country")
_EFFIS_CACHE_TTL_SECONDS = 86_400  # Refresh local parquet every 24 hours


def _first_matching_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Return the first column name from candidates present in df, or None."""
    return next((col for col in candidates if col in df.columns), None)


# ═════════════════════════════════════════════════════════════════════════════
#  CACHED DATA LOADERS
#  --------------------------------------------------------------------------
#  Streamlit re-runs the entire script on every user interaction.
#  @st.cache_resource keeps heavy objects (GeoDataFrames, spatial trees) in
#  memory across reruns.  @st.cache_data caches serialisable return values
#  (dicts, DataFrames) and is invalidated when inputs change.
# ═════════════════════════════════════════════════════════════════════════════


@st.cache_resource(show_spinner=False)
def load_enp_geodataframes() -> dict[str, gpd.GeoDataFrame]:
    """Load ENP (Espacios Naturales Protegidos) GeoParquet files from disk.

    Returns a dict keyed by region name ('canarias', 'peninsula') with each
    GeoDataFrame in its original projected CRS (UTM 28N / 30N).  These
    projected CRS are required later for accurate distance-based spatial joins.
    """
    files = {"canarias": "Enp2025_c.parquet", "peninsula": "Enp2025_p.parquet"}
    cols = ["SITE_NAME", "ODESIGNATE", "geometry"]
    return {
        key: gpd.read_parquet(DATA_DIR / fname, columns=cols)
        for key, fname in files.items()
        if (DATA_DIR / fname).exists()
    }


@st.cache_data(show_spinner=False)
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
        combined = combined[["geometry", "tooltip_text"]]
    else:
        combined = combined[["geometry"]]

    return json.loads(combined.to_json())


@st.cache_data(show_spinner=False)
def load_population_heatmap_data() -> pd.DataFrame | None:
    """Load the ~1 km downsampled WorldPop grid (lat, lon, population).

    Reads from the pre-aggregated parquet file first (~2.7 MB, <20 MB RAM) to
    avoid decompressing the 523M-pixel GeoTIFF into memory on 512 MB VMs.
    Falls back to downsampling the GeoTIFF on the fly if the parquet is absent.
    """
    if POPULATION_PARQUET.exists():
        return pd.read_parquet(POPULATION_PARQUET)

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


@st.cache_data(show_spinner=False)
def build_population_heatmap_df() -> pd.DataFrame | None:
    """Build a compact DataFrame (lon, lat, population) for PyDeck HeatmapLayer.

    Aggregates the 485k-row 1 km grid onto a ~3 km (0.03°) visual grid so
    pdk.Deck.to_json() produces a ~6 MB WebSocket payload (~173 MB RAM)
    instead of a 63.5 MB payload (616 MB RAM that OOM-kills 512 MB VMs),
    while keeping 100% of the 1 km grid intact for KPI calculations.
    """
    pop_df = load_population_heatmap_data()
    if pop_df is None or pop_df.empty:
        return None

    hm = pop_df.copy()
    hm["lat"] = (hm["lat"] / 0.03).round() * 0.03
    hm["lon"] = (hm["lon"] / 0.03).round() * 0.03
    hm = hm.groupby(["lat", "lon"], as_index=False)["population"].sum()
    hm = hm[hm["population"] >= 5].copy()
    hm["lat"] = hm["lat"].astype(np.float32).round(3)
    hm["lon"] = hm["lon"].astype(np.float32).round(3)
    hm["population"] = hm["population"].astype(np.float32).round(0)
    return hm


def _touch_safely(path: Path) -> None:
    """Update file mtime if possible so failed refreshes don't retry on every rerun."""
    try:
        if path.exists():
            path.touch()
    except OSError:
        pass


def _refresh_effis_parquet(cache_path: Path) -> None:
    """Download the latest EFFIS current-season burnt-area shapefile and save to parquet."""
    url = (
        "https://maps.effis.emergency.copernicus.eu/effis"
        "?service=WFS&request=getfeature&typename=ms:modis.ba.poly"
        "&version=1.1.0&outputformat=SHAPEZIP"
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        )
    }

    try:
        try:
            resp = requests.get(url, headers=headers, timeout=25)
        except requests.exceptions.SSLError:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            resp = requests.get(url, headers=headers, timeout=25, verify=False)
        resp.raise_for_status()
        gdf = gpd.read_file(io.BytesIO(resp.content), bbox=(-19.0, 27.0, 4.5, 44.0))
    except Exception as exc:
        logger.warning("Could not refresh EFFIS data from WFS: %s", exc)
        _touch_safely(cache_path)
        return

    if gdf.empty:
        _touch_safely(cache_path)
        return

    # Filter for Spain
    country_col = _first_matching_col(gdf, _EFFIS_COUNTRY_COLS)
    if country_col:
        gdf = gdf[gdf[country_col].astype(str).str.upper().isin(["ES", "SPAIN"])]

    # Filter by date — EFFIS uses INITIALDATE or FIREDATE
    date_col = _first_matching_col(gdf, _EFFIS_DATE_COLS)
    if date_col is not None:
        gdf[date_col] = pd.to_datetime(gdf[date_col], errors="coerce")
        gdf = gdf[gdf[date_col] >= f"{_CURRENT_YEAR}-01-01"]

    if gdf.empty:
        _touch_safely(cache_path)
        return

    # Reproject to WGS-84 for PyDeck
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    # Simplify for rendering performance (100 m tolerance in degrees ≈ 0.001°)
    gdf["geometry"] = gdf["geometry"].simplify(0.001)

    # Build tooltip and normalize essential columns
    area_col = _first_matching_col(gdf, _EFFIS_AREA_COLS)
    if area_col and date_col:
        gdf["tooltip_text"] = (
            "Burnt Area — "
            + gdf[date_col].dt.strftime("%Y-%m-%d").fillna("")
            + " — "
            + gdf[area_col].astype(str)
            + " ha"
        )
    else:
        gdf["tooltip_text"] = f"{_CURRENT_YEAR} Burnt Area"

    try:
        if date_col is not None:
            gdf[date_col] = gdf[date_col].astype(str)
        keep_cols = [c for c in (date_col, area_col, "tooltip_text", "geometry") if c and c in gdf.columns]
        gdf[keep_cols].to_parquet(cache_path)
    except Exception as exc:
        logger.warning("Could not cache EFFIS data to parquet: %s", exc)


def _get_effis_cache_mtime() -> float:
    """Ensure the EFFIS parquet cache is fresh (< 24 h) and return its mtime.

    Passing this mtime into @st.cache_data functions guarantees that Streamlit
    invalidates its in-memory cache immediately whenever the parquet updates.
    """
    needs_refresh = (
        not EFFIS_PARQUET.exists()
        or (time.time() - EFFIS_PARQUET.stat().st_mtime) >= _EFFIS_CACHE_TTL_SECONDS
    )
    if needs_refresh:
        _refresh_effis_parquet(EFFIS_PARQUET)
    return round(EFFIS_PARQUET.stat().st_mtime, 2) if EFFIS_PARQUET.exists() else 0.0


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_effis_burnt_areas(cache_mtime: float = 0.0) -> gpd.GeoDataFrame | None:
    """Load the EFFIS current-season burnt-area GeoDataFrame for Spain (EPSG:4326).

    Keyed on cache_mtime so any update to the local parquet file immediately
    refreshes the histogram, KPIs, and map layers.
    """
    if not EFFIS_PARQUET.exists():
        _refresh_effis_parquet(EFFIS_PARQUET)
    if EFFIS_PARQUET.exists():
        try:
            return gpd.read_parquet(EFFIS_PARQUET)
        except Exception as exc:
            logger.warning("Error reading EFFIS parquet cache: %s", exc)
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def build_effis_geojson(cache_mtime: float = 0.0) -> dict | None:
    """Serialize the EFFIS burnt-area GeoDataFrame to a lean cached GeoJSON dict."""
    effis_gdf = fetch_effis_burnt_areas(cache_mtime)
    if effis_gdf is None or effis_gdf.empty:
        return None
    cols = [c for c in ("geometry", "tooltip_text") if c in effis_gdf.columns]
    return json.loads(effis_gdf[cols].to_json())


def get_effis_last_fire_date(effis_gdf: gpd.GeoDataFrame | None) -> datetime.date | None:
    """Return the latest wildfire date recorded in the EFFIS Spain dataset."""
    if effis_gdf is None or effis_gdf.empty:
        return None
    date_col = _first_matching_col(effis_gdf, _EFFIS_DATE_COLS)
    if not date_col:
        return None
    max_ts = pd.to_datetime(effis_gdf[date_col], errors="coerce").max()
    return None if pd.isna(max_ts) else max_ts.date()


def _extract_effis_fires_for_window(
    effis_gdf: gpd.GeoDataFrame | None,
    from_date: datetime.date,
    to_date: datetime.date,
) -> pd.DataFrame:
    """Extract EFFIS fire events falling within [from_date, to_date] as point detections.

    Used when NASA FIRMS NRT has no nominal/high VIIRS detections for a date window
    where EFFIS has confirmed wildfires in Spain (e.g. smaller autumn/winter fires),
    so active-fire KPIs and map points stay consistent with EFFIS records.
    """
    if effis_gdf is None or effis_gdf.empty:
        return pd.DataFrame()

    date_col = _first_matching_col(effis_gdf, _EFFIS_DATE_COLS)
    if not date_col:
        return pd.DataFrame()

    dt_series = pd.to_datetime(effis_gdf[date_col], errors="coerce")
    mask = (dt_series.dt.date >= from_date) & (dt_series.dt.date <= to_date)
    subset = effis_gdf.loc[mask]
    if subset.empty:
        return pd.DataFrame()

    rep_pts = subset.geometry.representative_point()
    acq_dt = dt_series.loc[mask]
    return pd.DataFrame({
        "latitude": rep_pts.y.to_numpy(dtype=np.float64),
        "longitude": rep_pts.x.to_numpy(dtype=np.float64),
        "acq_date": acq_dt.dt.strftime("%Y-%m-%d").to_numpy(),
        "acq_time": acq_dt.dt.strftime("%H%M").to_numpy(),
        "confidence": "high",
    })


# ── FIRMS API Loader ──────────────────────────────────────────────────────────

def _read_firms_csv(url: str) -> pd.DataFrame:
    """Download and filter a NASA FIRMS CSV feed to nominal/high confidence."""
    try:
        data = pd.read_csv(url)
    except Exception:
        return pd.DataFrame()

    if data.empty or "confidence" not in data.columns:
        return pd.DataFrame()

    # FIRMS uses 'n'/'h' for NRT feeds, 'nominal'/'high' for SP feeds
    return data[data["confidence"].isin(_VALID_CONFIDENCE)].copy()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_europe_fires(api_key: str, start_date: str, day_range: int, source: str) -> pd.DataFrame:
    """Fetch VIIRS fire detections for Europe from the NASA FIRMS Area API.

    Parameters
    ----------
    api_key    : NASA FIRMS MAP_KEY (from .streamlit/secrets.toml).
    start_date : ISO date string (YYYY-MM-DD) for the query start.
                 Always anchored on yesterday or earlier so NRT data exists.
    day_range  : Number of days forward from start_date.
    source     : FIRMS dataset id ('VIIRS_SNPP_NRT' or 'VIIRS_SNPP_SP').

    Returns
    -------
    pd.DataFrame filtered to nominal/high confidence detections only.
    Empty DataFrame on error or no data.
    """
    clamped_range = max(1, min(day_range, _FIRMS_MAX_DAYS))
    base_url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}"

    # Try primary source first; fall back to alternate feed if inside NRT/SP transition window
    fallback_source = "VIIRS_SNPP_NRT" if source == "VIIRS_SNPP_SP" else "VIIRS_SNPP_SP"
    for candidate_source in (source, fallback_source):
        url = f"{base_url}/{candidate_source}/{_FIRMS_BBOX}/{clamped_range}/{start_date}"
        df = _read_firms_csv(url)
        if not df.empty:
            return df

    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def find_last_fire_date(
    api_key: str,
    yesterday: datetime.date,
    effis_mtime: float = 0.0,
) -> datetime.date:
    """Determine the most recent date with recorded fire activity in Spain.

    Checks both NASA FIRMS recent detections and the EFFIS dataset,
    returning whichever has the most recent Spanish wildfire date (<= yesterday).
    """
    candidates: list[datetime.date] = []

    effis_last = get_effis_last_fire_date(fetch_effis_burnt_areas(effis_mtime))
    if effis_last is not None:
        candidates.append(min(effis_last, yesterday))

    recent_start = (yesterday - datetime.timedelta(days=_FIRMS_MAX_DAYS - 1)).strftime("%Y-%m-%d")
    spain_recent = filter_spain(
        fetch_europe_fires(api_key, recent_start, _FIRMS_MAX_DAYS, "VIIRS_SNPP_NRT")
    )
    if not spain_recent.empty:
        candidates.append(min(pd.to_datetime(spain_recent["acq_date"]).max().date(), yesterday))

    return max(candidates) if candidates else yesterday


# ═════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════


def _hash_fire_df(fires_df: pd.DataFrame) -> str:
    """Create a lightweight hash of fire data for use as a cache key.

    Hashes only the coordinate columns so the KPI cache is invalidated when
    fire locations change but not when unrelated columns differ.
    """
    raw = pd.util.hash_pandas_object(fires_df[["latitude", "longitude"]]).to_numpy().tobytes()
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

    fires_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(_fires_df["longitude"], _fires_df["latitude"]),
        crs="EPSG:4326",
    )

    hit_indices: set[int] = set()
    for enp_gdf in enp_gdfs.values():
        fires_projected = fires_gdf.to_crs(enp_gdf.crs)
        joined = gpd.sjoin(fires_projected, enp_gdf[["geometry"]], predicate="intersects")
        hit_indices.update(joined.index)

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
    if _fires_df.empty:
        return result

    # ── 1. Exposed population (vectorised with cKDTree) ───────────────────
    pop_df = load_population_heatmap_data()
    if pop_df is not None and not pop_df.empty:
        pop_coords = np.column_stack([pop_df["lat"].to_numpy(), pop_df["lon"].to_numpy()])
        tree = cKDTree(pop_coords)

        fire_coords = np.column_stack([
            _fires_df["latitude"].to_numpy(),
            _fires_df["longitude"].to_numpy(),
        ])
        nearby_sets = tree.query_ball_point(fire_coords, r=_DEG_BUFFER_5KM)
        nearby_indices = set().union(*nearby_sets)

        if nearby_indices:
            pop_vals = pop_df["population"].to_numpy()
            result["exposed_pop"] = int(pop_vals[list(nearby_indices)].sum())

    # ── 2. Mean density at fire locations (point-samples the full raster) ─
    if POPULATION_TIFF.exists():
        try:
            with rasterio.open(POPULATION_TIFF) as src:
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
    (pd.Timedelta(hours=24), [255, 75, 75, 255]),    # Streamlit Coral Red (Solid) — ≤ 24 h
    (pd.Timedelta(days=2),   [253, 127, 128, 255]),  # Coral Red lightened (~70%)  — 24–48 h
    (pd.Timedelta(days=4),   [251, 179, 180, 255]),  # Coral Red lightened (~40%)  — 2–4 days
]
_DEFAULT_COLOR = [249, 223, 224, 255]                # Coral Red lightened (~15%)  — > 4 days (softest)


def assign_colors(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'color_rgba' column based on how recent each fire detection is.

    Uses np.select for vectorised performance — no row-wise .apply() needed.
    Colors go from dark red (most recent) through red, orange, to yellow.
    """
    if df.empty:
        return df

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
    if df.empty:
        return df

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


def _kpi_card(label: str, value) -> None:
    """Render a single KPI as a clean HTML card — no st.metric hacks needed."""
    st.markdown(
        f"<div class='kpi-card'>"
        f"<p class='kpi-label'>{label}</p>"
        f"<p class='kpi-value'>{value}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )


st.markdown("#### Spanish Wildfire Tracker  ")

# ── Date range state & defaults ───────────────────────────────────────────────
# NASA FIRMS API allows a maximum window of 5 days per request.
# NRT data has ~1-day latency, so "yesterday" is the latest available date.
# When there are no active fires today (e.g. in winter), default the 'To' date
# automatically to the date of the last recorded fire.
_yesterday = datetime.date.today() - datetime.timedelta(days=1)
_api_key = st.secrets["FIRMS_API_KEY"]
_effis_mtime = _get_effis_cache_mtime()
effis_gdf = fetch_effis_burnt_areas(_effis_mtime)
_last_fire_date = find_last_fire_date(_api_key, _yesterday, _effis_mtime)
_default_to = _last_fire_date
_default_from = _default_to - datetime.timedelta(days=2)  # 3-day window

if (
    "from_date" not in st.session_state
    or "to_date" not in st.session_state
    or st.session_state.get("_last_fire_anchor") != _last_fire_date
):
    st.session_state["_last_fire_anchor"] = _last_fire_date
    st.session_state["from_date"] = _default_from
    st.session_state["to_date"] = _default_to


def _on_from_date_change() -> None:
    """Keep 'To' within the valid [from_date, from_date + 4 days] window."""
    f_date: datetime.date = st.session_state["from_date"]
    t_date: datetime.date = st.session_state["to_date"]
    max_to = min(f_date + datetime.timedelta(days=_FIRMS_MAX_DAYS - 1), _yesterday)
    if t_date < f_date or t_date > max_to:
        st.session_state["to_date"] = min(f_date + datetime.timedelta(days=2), max_to)


def _on_to_date_change() -> None:
    """Keep 'From' within the valid [to_date - 4 days, to_date] window."""
    f_date: datetime.date = st.session_state["from_date"]
    t_date: datetime.date = st.session_state["to_date"]
    min_from = t_date - datetime.timedelta(days=_FIRMS_MAX_DAYS - 1)
    if f_date > t_date or f_date < min_from:
        st.session_state["from_date"] = t_date - datetime.timedelta(days=2)


# ── Layout Definition ─────────────────────────────────────────────────────────
hist_col, map_col = st.columns([1, 1.4], gap="large")

with hist_col:
    # Context section in two columns
    ctx_col1, ctx_col2 = st.columns(2, gap="medium")
    with ctx_col1:
        st.markdown("**Context:** This dashboard tracks active thermal anomalies and cumulative burned area in Spain (sourced from NASA FIRMS and Copernicus EFFIS). While human ignitions—both negligent and intentional—remain the primary trigger,")
    with ctx_col2:
        st.markdown(" climate-driven fuel dryness, coupled with rural land abandonment and dense biomass accumulation, enables localized ignitions to rapidly escalate into uncontrollable, high-intensity megafires.")

    st.markdown("<div class='spacer-sm'></div>", unsafe_allow_html=True)

    # ── Date range picker ────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        from_date = st.date_input(
            "From:",
            key="from_date",
            max_value=_yesterday,
            on_change=_on_from_date_change,
            help="Earliest detection date to show.",
        )
    with c2:
        to_date = st.date_input(
            "To:",
            key="to_date",
            max_value=_yesterday,
            on_change=_on_to_date_change,
            help=f"Latest detection date (max {_FIRMS_MAX_DAYS}-day window; defaults to last recorded fire).",
        )

    # Ensure range stays clamped within [1, _FIRMS_MAX_DAYS]
    if to_date < from_date:
        to_date = from_date
    selected_range = min((to_date - from_date).days + 1, _FIRMS_MAX_DAYS)
    days_from_today = (datetime.date.today() - from_date).days
    source = "VIIRS_SNPP_SP" if days_from_today > 30 else "VIIRS_SNPP_NRT"
    start_date_str = from_date.strftime("%Y-%m-%d")

# ── Fetch fire data (with EFFIS fallback for confirmed Spanish fires) ────────
europe_df = fetch_europe_fires(
    _api_key,
    start_date_str,
    selected_range,
    source,
)

# If FIRMS has no Spanish detections in the window but EFFIS recorded fires on
# those dates, supplement with the EFFIS fire centroids so KPIs and map update.
if filter_spain(europe_df).empty:
    effis_window_df = _extract_effis_fires_for_window(effis_gdf, from_date, to_date)
    if not effis_window_df.empty:
        europe_df = (
            pd.concat([europe_df, effis_window_df], ignore_index=True)
            if not europe_df.empty
            else effis_window_df
        )

# ── Prepare fire DataFrame & Spain-only subset for KPIs ──────────────────────
if not europe_df.empty:
    europe_df = europe_df.copy()
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
    spain_df = filter_spain(europe_df)
else:
    spain_df = pd.DataFrame(columns=["latitude", "longitude"])

total_fires = len(spain_df)

if not spain_df.empty:
    fire_hash = _hash_fire_df(spain_df)
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
else:
    eco_fires = 0
    pop_kpis = {"exposed_pop": 0, "mean_density": 0.0}

# ── UX-friendly status note when in off-season / no active fires ─────────────
with hist_col:
    if spain_df.empty:
        st.markdown(
            f"<div class='status-note'>"
            f"No active fires detected in Spain for the selected dates — "
            f"displaying 2026 cumulative burnt areas on the map "
            f"(last recorded fire: <b>{_last_fire_date:%Y-%m-%d}</b>)."
            f"</div>",
            unsafe_allow_html=True,
        )
    elif _last_fire_date < _yesterday and to_date == _last_fire_date:
        st.markdown(
            f"<div class='status-note'>"
            f"No active fires today — calendar set to the last recorded fire "
            f"(<b>{_last_fire_date:%Y-%m-%d}</b>)."
            f"</div>",
            unsafe_allow_html=True,
        )

# ── EFFIS burnt-area KPI ──────────────────────────────────────────────────────
effis_total_ha = 0.0
effis_count = 0
if effis_gdf is not None and not effis_gdf.empty:
    effis_count = len(effis_gdf)
    area_col = _first_matching_col(effis_gdf, _EFFIS_AREA_COLS)
    if area_col:
        numeric_area = pd.to_numeric(effis_gdf[area_col], errors="coerce").fillna(0.0)
        effis_total_ha = float(numeric_area.sum())

# ── KPIs, Legend & Layers (right column) ──────────────────────────────────────
with map_col:
    k1, k2, k3 = st.columns([1, 1, 1], gap="medium")
    with k1:
        _kpi_card("Total Active Fires (High Confidence)", total_fires)
        _kpi_card("Fires in Protected Areas", eco_fires)
    with k2:
        _kpi_card(
            "People within 5 km of Fires",
            f"{pop_kpis['exposed_pop']:,}" if total_fires > 0 else "—",
        )
        _kpi_card(
            "Avg. Population at Fire Sites",
            f"{pop_kpis['mean_density']:.1f} people / ha" if total_fires > 0 else "—",
        )
    with k3:
        _kpi_card(
            f"Total Burnt Area {_CURRENT_YEAR} (EFFIS)",
            f"{effis_total_ha:,.0f} ha" if effis_total_ha else "—",
        )
        _kpi_card(
            f"Total EFFIS Fires ({_CURRENT_YEAR})",
            effis_count if effis_count else "—",
        )

    st.markdown("<div class='spacer-md'></div>", unsafe_allow_html=True)

    # ── Legend & Layer toggles ────────────────────────────────────────────────
    t_leg, t_layers = st.columns([2.1, 1.0], vertical_alignment="center")
    with t_leg:
        st.markdown(
            "<div class='legend-bar'>"
            "<b>Legend:</b> "
            '<span style="color:#FF4B4B; font-size: 1.1em; vertical-align: middle;">⬤</span> ≤ 24 h &nbsp;|&nbsp; '
            '<span style="color:#FD7F80; font-size: 1.1em; vertical-align: middle;">⬤</span> 24–48 h &nbsp;|&nbsp; '
            '<span style="color:#FBB3B4; font-size: 1.1em; vertical-align: middle;">⬤</span> 2–4 days &nbsp;|&nbsp; '
            '<span style="color:#F9DFE0; font-size: 1.1em; vertical-align: middle;">⬤</span> > 4 days &nbsp;|&nbsp; '
            f'<span class="legend-swatch"></span> Burnt Area {_CURRENT_YEAR}'
            "</div>",
            unsafe_allow_html=True,
        )
    with t_layers:
        selected_layers = st.multiselect(
            "Extra Layers",
            options=["Protected Areas", "Population"],
            default=[],
            label_visibility="collapsed",
            placeholder="Add layers...",
        )
        show_enp = "Protected Areas" in selected_layers
        show_pop = "Population" in selected_layers

# ── Build map layers ──────────────────────────────────────────────────────────
layers: list[pdk.Layer] = []

# Fire scatter dots (shown whenever active detections exist in the window)
if not europe_df.empty:
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=europe_df,
            get_position="[longitude, latitude]",
            get_fill_color="color_rgba",
            get_radius=1000,
            radius_min_pixels=4,
            radius_max_pixels=8,
            pickable=True,
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

# EFFIS burnt areas (dark red polygons — always shown)
effis_geojson = build_effis_geojson(_effis_mtime)
if effis_geojson is not None:
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
    pop_hm_df = build_population_heatmap_df()
    if pop_hm_df is not None and not pop_hm_df.empty:
        layers.insert(
            0,
            pdk.Layer(
                "HeatmapLayer",
                data=pop_hm_df,
                get_position="[lon, lat]",
                get_weight="population",
                radiusPixels=35,
                intensity=1,
                threshold=0.05,
                opacity=0.6,
                color_range=[
                    [240, 248, 255],  # White / Ice Blue (lowest density)
                    [125, 211, 252],  # Light Blue
                    [14, 165, 233],   # Electric Blue
                    [124, 58, 237],   # Electric Purple
                    [91, 33, 182],    # Deep Purple (highest density)
                ],
            ),
        )
    else:
        st.warning("Population raster not found in data/ folder.")

# ── Bar chart (left column) ──────────────────────────────────────────────────
with hist_col:
    if effis_gdf is not None and not effis_gdf.empty:
        st.markdown(f"#### Monthly Burnt Area ({_CURRENT_YEAR})")
        date_c = _first_matching_col(effis_gdf, _EFFIS_DATE_COLS)
        area_c = _first_matching_col(effis_gdf, _EFFIS_AREA_COLS)

        if date_c and area_c:
            chart_df = effis_gdf[[date_c, area_c]].copy()
            parsed_dates = pd.to_datetime(chart_df[date_c], errors="coerce")
            chart_df["Month_Num"] = parsed_dates.dt.month

            # Ensure area is numeric before summing
            chart_df[area_c] = pd.to_numeric(chart_df[area_c], errors="coerce").fillna(0.0)

            # Build full sequence of months from January up to the current month of the year
            today = datetime.date.today()
            max_data_month = (
                int(chart_df["Month_Num"].max())
                if not chart_df["Month_Num"].isna().all()
                else 1
            )
            last_month = max(today.month, max_data_month) if today.year == _CURRENT_YEAR else 12
            all_months = pd.DataFrame({
                "Month_Num": list(range(1, last_month + 1)),
                "Month": [
                    datetime.date(_CURRENT_YEAR, m, 1).strftime("%B")
                    for m in range(1, last_month + 1)
                ],
            })

            # Sum up area_ha and fire count by month and merge onto the full month sequence
            monthly_agg = (
                chart_df.groupby("Month_Num", as_index=False)
                .agg(
                    Burnt_Ha=(area_c, "sum"),
                    Fires=(area_c, "count"),
                )
            )
            monthly = all_months.merge(monthly_agg, on="Month_Num", how="left")
            monthly["Burnt_Ha"] = monthly["Burnt_Ha"].fillna(0.0)
            monthly["Fires"] = monthly["Fires"].fillna(0).astype(int)
            monthly = monthly.sort_values("Month_Num")

            # Format Y-axis for better readability (in thousands of ha), with a minimum
            # visible bar height (~2% of peak) whenever Burnt_Ha > 0 so months with smaller
            # autumn/winter fires (e.g. Jan, Oct, Nov) remain clearly visible alongside July.
            monthly["Area (x1000 ha)"] = monthly["Burnt_Ha"] / 1000.0
            max_area_k = float(monthly["Area (x1000 ha)"].max()) or 1.0
            min_visible_k = max_area_k * 0.02
            monthly["Display_Area"] = np.where(
                monthly["Burnt_Ha"] > 0,
                np.maximum(monthly["Area (x1000 ha)"], min_visible_k),
                0.0,
            )
            monthly["Label"] = [
                f"{ha / 1000:.1f}k" if ha >= 1000 else (f"{ha:,.0f} ha" if ha > 0 else "")
                for ha in monthly["Burnt_Ha"]
            ]

            base_chart = alt.Chart(monthly).encode(
                x=alt.X(
                    "Month:N",
                    title=None,
                    sort=monthly["Month"].tolist(),
                    axis=alt.Axis(
                        labelFontSize=12,
                        titleFontSize=13,
                        labelAngle=-45,
                        labelLimit=80,
                    ),
                ),
                tooltip=[
                    alt.Tooltip("Month:N", title="Month"),
                    alt.Tooltip("Burnt_Ha:Q", title="Burnt Area (ha)", format=",.0f"),
                    alt.Tooltip("Fires:Q", title="EFFIS Fires", format=",d"),
                ],
            )

            bars = base_chart.mark_bar(color="#B22222").encode(
                y=alt.Y(
                    "Display_Area:Q",
                    title="Area (x1000 ha)",
                    scale=alt.Scale(domain=[0, max_area_k * 1.12]),
                    axis=alt.Axis(labelFontSize=12, titleFontSize=13),
                ),
            )

            labels = base_chart.mark_text(
                align="center",
                baseline="bottom",
                dy=-4,
                fontSize=10,
                color="#495057",
            ).encode(
                y=alt.Y("Display_Area:Q"),
                text=alt.Text("Label:N"),
            )

            bar_chart_obj = (
                (bars + labels)
                .properties(
                    height=400,
                    padding={"top": 20, "left": 15, "right": 15, "bottom": 30},
                )
                .configure_view(stroke=None, fill="#F0F2F6")
                .configure(background="#F0F2F6")
            )
            st.altair_chart(bar_chart_obj, width="stretch", theme=None)
        else:
            st.info("Monthly data not available.")
    else:
        st.info("No EFFIS data available.")

    # Data sources below the bar chart
    st.markdown(
        "<div class='data-sources'>"
        "<b>Data sources:</b> NASA FIRMS (VIIRS active fires filtered for nominal/high confidence) | "
        "MITECO (Protected Natural Areas) | "
        "<a href='https://www.worldpop.org/'>WorldPop</a> Population Data "
        "(<a href='https://creativecommons.org/licenses/by/4.0/'>CC BY 4.0</a> / "
        "<a href='https://opendatacommons.org/licenses/odbl/'>ODbL</a>) | "
        f"<a href='https://effis.jrc.ec.europa.eu/'>EFFIS {_CURRENT_YEAR} (Copernicus EMS)</a> Burnt Areas."
        "</div>",
        unsafe_allow_html=True,
    )

# ── Map Rendering (Right Column) ──────────────────────────────────────────────
with map_col:
    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(
                latitude=40.0, longitude=-3.0, zoom=5, pitch=0
            ),
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
        ),
        width="stretch",
        height=600,
    )
