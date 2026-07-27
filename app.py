"""
European Wildfire Tracker — NASA FIRMS
======================================
Streamlit dashboard that visualises active fire detections across Spain
using NASA FIRMS satellite data (VIIRS) and overlays Spanish protected
natural areas (ENP) from MITECO GeoParquet files. It also integrates a
100 m population raster from WorldPop to compute fire-exposure KPIs.

Data sources:
- NASA FIRMS NRT & SP VIIRS feeds (CSV via API)
- MITECO ENP boundaries (GeoParquet, EPSG:32628 / EPSG:32630)
- WorldPop Population Grid 100 m (GeoTIFF, EPSG:4326, CC BY 4.0 / ODbL)
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk
import rasterio
from rasterio.enums import Resampling
import streamlit as st
from shapely.geometry import Point

# ---------------------------------------------------------------------------
# 1. PAGE CONFIG & STYLING
# ---------------------------------------------------------------------------
st.set_page_config(page_title="European Wildfire Tracker", layout="wide")


def _inject_css(path: Path) -> None:
    """Read a CSS file and inject it into the Streamlit page."""
    st.markdown(f"<style>{path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


_inject_css(Path("css/styles.css"))

# ---------------------------------------------------------------------------
# 2. CACHED DATA LOADERS (top-level, as per Streamlit best practices)
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
POPULATION_TIFF = DATA_DIR / "SpainPopulation2026_CN_100m_R2025A_v1.tif"
_POP_DOWNSAMPLE_FACTOR = 10  # 100 m → ~1 km
_POP_DENSITY_THRESHOLD = 10  # hab/pixel threshold for "populated area" KPI


@st.cache_resource(show_spinner=False)
def load_enp_geodataframes() -> dict[str, gpd.GeoDataFrame]:
    """Load ENP (Espacios Naturales Protegidos) GeoParquet files.

    Returns a dict keyed by region name ('canarias', 'peninsula') with the
    GeoDataFrames in their original projected CRS (UTM 28N / 30N).
    """
    result: dict[str, gpd.GeoDataFrame] = {}
    for key, filename in [("canarias", "Enp2025_c.parquet"), ("peninsula", "Enp2025_p.parquet")]:
        path = DATA_DIR / filename
        if path.exists():
            result[key] = gpd.read_parquet(path)
    return result


@st.cache_data(show_spinner=False)
def build_enp_geojson() -> dict | None:
    """Simplify ENP geometries and return a GeoJSON dict in WGS-84 for PyDeck.

    Simplification uses a 150 m tolerance in the native metric projection
    before reprojecting to EPSG:4326.
    """
    enp_gdfs = load_enp_geodataframes()
    if not enp_gdfs:
        return None

    parts: list[gpd.GeoDataFrame] = []
    for gdf in enp_gdfs.values():
        simplified = gdf.copy()
        simplified["geometry"] = simplified["geometry"].simplify(150)
        parts.append(simplified.to_crs(epsg=4326))

    if not parts:
        return None

    combined = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs="EPSG:4326")

    # Tooltip column for PyDeck (plain text, no HTML)
    if {"SITE_NAME", "ODESIGNATE"}.issubset(combined.columns):
        combined["tooltip_text"] = (
            combined["ODESIGNATE"].fillna("Espacio Protegido") + " — " + combined["SITE_NAME"].fillna("")
        )
    return json.loads(combined.to_json())


@st.cache_resource(show_spinner=False)
def load_population_heatmap_data() -> pd.DataFrame | None:
    """Load the WorldPop population raster, downsample to ~1 km, and return a
    DataFrame of ``(lat, lon, population)`` for PyDeck HeatmapLayer.

    Uses ``rasterio`` decimated reads so the full 93 MB raster is never held
    in memory at full resolution.
    """
    if not POPULATION_TIFF.exists():
        return None

    with rasterio.open(POPULATION_TIFF) as src:
        # Decimated read — resolution goes from 100 m to ~1 km
        out_shape = (
            1,
            src.height // _POP_DOWNSAMPLE_FACTOR,
            src.width // _POP_DOWNSAMPLE_FACTOR,
        )
        data = src.read(
            1,
            out_shape=(out_shape[1], out_shape[2]),
            resampling=Resampling.average,
        )
        # Scale average back to sum for the aggregated cell
        data = data * (_POP_DOWNSAMPLE_FACTOR ** 2)
        nodata = src.nodata
        transform = src.transform * src.transform.scale(
            src.width / data.shape[1],
            src.height / data.shape[0],
        )

    # Build coordinate arrays for valid (non-nodata, > 0) cells
    if nodata is not None:
        mask = (data != nodata) & (data > 0)
    else:
        mask = data > 0

    rows, cols = np.nonzero(mask)
    xs, ys = rasterio.transform.xy(transform, rows, cols)

    return pd.DataFrame({
        "lat": np.asarray(ys, dtype=np.float32),
        "lon": np.asarray(xs, dtype=np.float32),
        "population": data[rows, cols].astype(np.float32),
    })


@st.cache_resource(show_spinner=False)
def _open_population_raster() -> rasterio.DatasetReader | None:
    """Keep the population raster open for point-sampling (shared across reruns)."""
    if not POPULATION_TIFF.exists():
        return None
    return rasterio.open(POPULATION_TIFF)


def compute_population_kpis(
    fires_df: pd.DataFrame,
) -> dict[str, float | int]:
    """Compute population-exposure KPIs by sampling the WorldPop raster.

    Returns a dict with keys:
    - ``exposed_pop``   – estimated people within 5 km of an active fire
    - ``mean_density``  – average population value at fire-pixel locations
    - ``fires_in_pop``  – count of fires where pixel population > threshold
    """
    result: dict[str, float | int] = {
        "exposed_pop": 0,
        "mean_density": 0.0,
        "fires_in_pop": 0,
    }
    src = _open_population_raster()
    if src is None or fires_df.empty:
        return result

    coords = list(zip(fires_df["longitude"], fires_df["latitude"]))
    sampled = np.array(
        [val[0] for val in src.sample(coords)], dtype=np.float32
    )
    # Replace nodata with 0
    if src.nodata is not None:
        sampled[sampled == src.nodata] = 0.0
    sampled[sampled < 0] = 0.0

    result["mean_density"] = float(np.mean(sampled)) if len(sampled) else 0.0
    result["fires_in_pop"] = int(np.sum(sampled > _POP_DENSITY_THRESHOLD))

    # Exposed population: use the downsampled heatmap data with a
    # simple distance filter (~5 km ≈ 0.045° at Spain's latitude)
    pop_df = load_population_heatmap_data()
    if pop_df is not None and not pop_df.empty:
        deg_buf = 0.045  # ~5 km
        total_exposed = 0.0
        # Vectorised approach: for each fire, find pop cells within buffer
        fire_lats = fires_df["latitude"].values
        fire_lons = fires_df["longitude"].values
        pop_lats = pop_df["lat"].values
        pop_lons = pop_df["lon"].values
        pop_vals = pop_df["population"].values

        # Mark population cells that are near ANY fire
        near_any = np.zeros(len(pop_df), dtype=bool)
        for i in range(len(fire_lats)):
            dist_mask = (
                (np.abs(pop_lats - fire_lats[i]) < deg_buf)
                & (np.abs(pop_lons - fire_lons[i]) < deg_buf)
            )
            near_any |= dist_mask
        total_exposed = float(pop_vals[near_any].sum())
        result["exposed_pop"] = int(total_exposed)

    return result


@st.cache_data(ttl=3600)
def fetch_europe_fires(api_key: str, start_date: str, day_range: int, source: str) -> pd.DataFrame:
    """Fetch VIIRS fire detections for Europe from the NASA FIRMS Area API.

    Parameters
    ----------
    api_key : str
        NASA FIRMS MAP_KEY.
    start_date : str
        ISO date string (YYYY-MM-DD) for the start of the query window.
    day_range : int
        Number of days forward from *start_date*.
    source : str
        FIRMS dataset identifier (e.g. ``VIIRS_SNPP_NRT`` or ``VIIRS_SNPP_SP``).

    Returns
    -------
    pd.DataFrame
        Filtered to nominal/high confidence detections. Empty DataFrame on error.
    """
    # Bounding box for Europe, Mediterranean, and North Africa (lon_min, lat_min, lon_max, lat_max)
    bbox = "-20.0,25.0,35.0,65.0"
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/{source}/{bbox}/{day_range}/{start_date}"
    try:
        data = pd.read_csv(url)
    except Exception as exc:
        st.error(f"Error connecting to NASA FIRMS API: {exc}")
        return pd.DataFrame()

    if data.empty or "confidence" not in data.columns:
        return pd.DataFrame()

    # FIRMS API returns 'n'/'h' for NRT and 'nominal'/'high' for SP
    return data[data["confidence"].isin(["nominal", "high", "n", "h"])]


def compute_eco_fires(fires_df: pd.DataFrame) -> int:
    """Count fires within 5 km of a Spanish protected natural area (ENP).

    Uses projected CRS spatial joins for accuracy (UTM 28N for Canarias,
    UTM 30N for the peninsula).
    """
    enp_gdfs = load_enp_geodataframes()
    if not enp_gdfs or fires_df.empty:
        return 0

    geometry = [Point(xy) for xy in zip(fires_df["longitude"], fires_df["latitude"])]
    fires_gdf = gpd.GeoDataFrame(fires_df, geometry=geometry, crs="EPSG:4326")

    near_indices: set[int] = set()
    for enp_gdf in enp_gdfs.values():
        fires_projected = fires_gdf.to_crs(enp_gdf.crs)
        joined = gpd.sjoin_nearest(fires_projected, enp_gdf, max_distance=5000)
        near_indices.update(joined.index.tolist())

    return len(near_indices)


# ---------------------------------------------------------------------------
# Vectorised color assignment helpers
# ---------------------------------------------------------------------------

# Time-based color thresholds (most recent → oldest)
_COLOR_BINS = [
    (pd.Timedelta(hours=24), [139, 0, 0, 200]),   # Dark Red  — ≤ 24 h
    (pd.Timedelta(days=2),   [255, 0, 0, 200]),    # Red       — 24–48 h
    (pd.Timedelta(days=4),   [255, 165, 0, 200]),  # Orange    — 2–4 days
]
_DEFAULT_COLOR = [255, 255, 0, 200]                 # Yellow    — > 4 days


def assign_colors(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``color_rgba`` column based on time since the most recent detection.

    Uses ``np.select`` for vectorised performance instead of row-wise ``apply``.
    """
    dt_max = df["acq_datetime"].max()
    diff = dt_max - df["acq_datetime"]

    conditions = [diff <= threshold for threshold, _ in _COLOR_BINS]
    choices = [color for _, color in _COLOR_BINS]

    indices = np.select(conditions, range(len(choices)), default=len(choices))
    all_colors = choices + [_DEFAULT_COLOR]
    df["color_rgba"] = [all_colors[int(i)] for i in indices]
    return df


# ---------------------------------------------------------------------------
# 3. DASHBOARD LAYOUT
# ---------------------------------------------------------------------------

st.title("European Wildfire Tracker — NASA FIRMS")
st.subheader("Spain — Active Fire Detections")

# --- Top Row: Controls & KPIs ----------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    selected_date = st.date_input("End date:", datetime.date.today())
    selected_range = st.slider("Days to look back:", min_value=1, max_value=5, value=5)

# Determine FIRMS dataset source: SP for data older than 30 days, NRT otherwise
start_date = selected_date - datetime.timedelta(days=selected_range - 1)
days_from_today = (datetime.date.today() - selected_date).days
source = "VIIRS_SNPP_SP" if days_from_today > 30 else "VIIRS_SNPP_NRT"

europe_df = fetch_europe_fires(
    st.secrets["FIRMS_API_KEY"],
    start_date.strftime("%Y-%m-%d"),
    selected_range,
    source,
)

# --- KPI row ---------------------------------------------------------------

if not europe_df.empty:
    # Parse acquisition datetime
    europe_df["acq_datetime"] = pd.to_datetime(
        europe_df["acq_date"] + " " + europe_df["acq_time"].astype(str).str.zfill(4),
        format="%Y-%m-%d %H%M",
    )

    # Plain-text tooltip for PyDeck
    europe_df["tooltip_text"] = (
        "Fire Detection — Date: "
        + europe_df["acq_datetime"].dt.strftime("%Y-%m-%d %H:%M")
        + " — Confidence: "
        + europe_df["confidence"].astype(str)
    )

    europe_df = assign_colors(europe_df)

    # Rough bounding box to isolate Spain (excluding North Africa, capturing Canaries and Balearics)
    # Mainland/Balearics: lat 36.0 to 44.0, lon -9.5 to 4.5
    # Canary Islands: lat 27.0 to 30.0, lon -19.0 to -13.0
    spain_mask = (
        ((europe_df['latitude'] >= 36.0) & (europe_df['latitude'] <= 44.0) & (europe_df['longitude'] >= -9.5) & (europe_df['longitude'] <= 4.5)) |
        ((europe_df['latitude'] >= 27.0) & (europe_df['latitude'] <= 30.0) & (europe_df['longitude'] >= -19.0) & (europe_df['longitude'] <= -13.0))
    )
    spain_only_df = europe_df[spain_mask]

    # Compute KPIs (Only for Spain)
    total_fires = len(spain_only_df)
    try:
        # compute_eco_fires already filters by distance to Spanish ENP polygons
        eco_fires = compute_eco_fires(spain_only_df)
    except Exception as exc:
        st.error(f"Error processing ENP spatial layer: {exc}")
        eco_fires = 0

    # Compute population KPIs
    try:
        pop_kpis = compute_population_kpis(spain_only_df)
    except Exception as exc:
        st.error(f"Error computing population KPIs: {exc}")
        pop_kpis = {"exposed_pop": 0, "mean_density": 0.0, "fires_in_pop": 0}

    with col2:
        st.markdown(
            f"""
            <div style="background-color: rgba(255, 165, 0, 0.2); padding: 10px; border-radius: 8px; height: 100%; border: 1px solid rgba(255, 165, 0, 0.5);">
                <p style="margin-top: 0; margin-bottom: 5px; color: #333; font-size: 0.9em; font-weight: 600;">Total High-Confidence Fires</p>
                <p style="font-size: 1.8em; font-weight: bold; margin-bottom: 0; color: #ff8c00; line-height: 1;">{total_fires}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div style="background-color: rgba(34, 139, 34, 0.2); padding: 10px; border-radius: 8px; height: 100%; border: 1px solid rgba(34, 139, 34, 0.5);">
                <p style="margin-top: 0; margin-bottom: 5px; color: #333; font-size: 0.9em; font-weight: 600;">Fires < 5 km from Protected Areas</p>
                <p style="font-size: 1.8em; font-weight: bold; margin-bottom: 0; color: #228b22; line-height: 1;">{eco_fires}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --- Population KPI row ------------------------------------------------
    pop_col1, pop_col2, pop_col3 = st.columns(3)

    with pop_col1:
        exposed = pop_kpis["exposed_pop"]
        exposed_fmt = f"{exposed:,.0f}" if exposed else "—"
        st.markdown(
            f"""
            <div style="background-color: rgba(100, 100, 255, 0.15); padding: 10px; border-radius: 8px; height: 100%; border: 1px solid rgba(100, 100, 255, 0.5);">
                <p style="margin-top: 0; margin-bottom: 5px; color: #333; font-size: 0.9em; font-weight: 600;">👥 Population Exposed (5 km)</p>
                <p style="font-size: 1.8em; font-weight: bold; margin-bottom: 0; color: #4a4aff; line-height: 1;">{exposed_fmt}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with pop_col2:
        mean_d = pop_kpis["mean_density"]
        mean_fmt = f"{mean_d:.1f}" if mean_d else "—"
        st.markdown(
            f"""
            <div style="background-color: rgba(150, 80, 200, 0.15); padding: 10px; border-radius: 8px; height: 100%; border: 1px solid rgba(150, 80, 200, 0.5);">
                <p style="margin-top: 0; margin-bottom: 5px; color: #333; font-size: 0.9em; font-weight: 600;">📊 Mean Pop. Density at Fires</p>
                <p style="font-size: 1.8em; font-weight: bold; margin-bottom: 0; color: #9650c8; line-height: 1;">{mean_fmt} <span style="font-size: 0.5em;">hab/px</span></p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with pop_col3:
        fires_pop = pop_kpis["fires_in_pop"]
        st.markdown(
            f"""
            <div style="background-color: rgba(220, 50, 50, 0.15); padding: 10px; border-radius: 8px; height: 100%; border: 1px solid rgba(220, 50, 50, 0.5);">
                <p style="margin-top: 0; margin-bottom: 5px; color: #333; font-size: 0.9em; font-weight: 600;">🏘️ Fires in Populated Areas (&gt;{_POP_DENSITY_THRESHOLD} hab/px)</p>
                <p style="font-size: 1.8em; font-weight: bold; margin-bottom: 0; color: #dc3232; line-height: 1;">{fires_pop}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
else:
    st.warning("No fire data available for the selected date range.")

# --- Legend & Toggles (same row) -------------------------------------------
leg_col, tog_col1, tog_col2 = st.columns([2, 1, 1])

with leg_col:
    st.markdown(
        "**Legend:** "
        '<span style="color:#8B0000">⬤</span> ≤ 24 h &nbsp;|&nbsp; '
        '<span style="color:#FF0000">⬤</span> 24–48 h &nbsp;|&nbsp; '
        '<span style="color:#FFA500">⬤</span> 2–4 days &nbsp;|&nbsp; '
        '<span style="color:#FFFF00">⬤</span> > 4 days',
        unsafe_allow_html=True,
    )

with tog_col1:
    show_enp = st.toggle("Show Protected Areas (MITECO)", value=False)

with tog_col2:
    show_pop = st.toggle("Show Population Density (WorldPop)", value=False)

# --- Map -------------------------------------------------------------------
layers: list[pdk.Layer] = []

if not europe_df.empty:
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
    else:
        st.warning("ENP data not found in the data/ folder.")

if show_pop:
    pop_df = load_population_heatmap_data()
    if pop_df is not None and not pop_df.empty:
        layers.insert(
            0,  # render below fire points
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
                    [0, 0, 128],       # dark blue  – low
                    [0, 128, 255],     # sky blue
                    [0, 255, 128],     # cyan-green
                    [255, 255, 0],     # yellow
                    [255, 128, 0],     # orange
                    [255, 0, 0],       # red        – high
                ],
            ),
        )
    else:
        st.warning("Population raster not found in data/ folder.")

st.pydeck_chart(
    pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=40.0, longitude=-3.0, zoom=5, pitch=0),
        tooltip={"html": "{tooltip_text}", "style": {"backgroundColor": "steelblue", "color": "white"}},
    )
)

st.caption(
    "Data sources: NASA FIRMS (VIIRS) | MITECO (Protected Natural Areas) | "
    "[WorldPop](https://www.worldpop.org/) Population Data ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) / [ODbL](https://opendatacommons.org/licenses/odbl/))"
)
