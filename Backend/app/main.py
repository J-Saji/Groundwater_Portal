from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from typing import Optional
from pydantic import BaseModel
import json
import traceback
from datetime import datetime, timedelta
import calendar
from shapely.geometry import Point, shape
from shapely.prepared import prep
import math
import pandas as pd
import numpy as np
from scipy.stats import linregress
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy.interpolate import griddata
from scipy.stats import linregress
import statsmodels.api as sm
import time 
import geopandas as gpd
from shapely.geometry import Point, shape
from shapely.prepared import prep
from shapely.ops import unary_union
from shapely.validation import make_valid
from shapely.geometry import box
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict
import pickle
import base64
from dotenv import load_dotenv
from typing import Optional, List, Any # Updated imports
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import httpx
from testingragsql import query_agent
load_dotenv()
DATABASE_URL = os.environ.get("DB_URI")
if not DATABASE_URL:
    print("ERROR: DB_URI not set in .env file")
    exit(1)
# Optional dependencies
try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    print("WARNING: ruptures not available - changepoint detection disabled")

try:
    import pymannkendall as mk
    MK_AVAILABLE = True
except ImportError:
    MK_AVAILABLE = False
    print("WARNING: pymannkendall not available - Mann-Kzendall test disabled")

# ============= NEW: SIMPLE CACHE CLASS =============
class SimpleCache:
    def __init__(self, ttl_seconds=3600):
        self.cache = {}
        self.ttl = ttl_seconds
    
    def get(self, key):
        if key in self.cache:
            value, timestamp = self.cache[key]
            if (datetime.now() - timestamp).seconds < self.ttl:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key, value):
        self.cache[key] = (value, datetime.now())
    
    def clear_old(self):
        """Clean expired entries"""
        now = datetime.now()
        expired = [k for k, (v, ts) in self.cache.items() 
                   if (now - ts).seconds >= self.ttl]
        for k in expired:
            del self.cache[k]

# Global cache instances
TIMESERIES_CACHE = SimpleCache(ttl_seconds=3600)  # 1 hour
WELLS_CACHE = SimpleCache(ttl_seconds=1800)       # 30 minutes
GRACE_CACHE = SimpleCache(ttl_seconds=7200)       # 2 hours
RAINFALL_CACHE = SimpleCache(ttl_seconds=7200)    # 2 hours

# Thread pool for parallel queries
EXECUTOR = ThreadPoolExecutor(max_workers=4)

# Now your existing app = FastAPI(...) continues

app = FastAPI(
    title="GeoHydro API - Wells, GRACE, Rainfall & AI Chatbot",
    version="6.1.0",
    description="Groundwater monitoring API with integrated AI chatbot and unified timeseries (Wells + GRACE + Rainfall) - FIXED NaN Issues"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables from Backend/.env
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)

print("✅ GeoHydro API Started!")

@app.on_event("startup")
async def start_cache_cleanup():
    """Clean expired cache entries every 10 minutes"""
    async def cleanup_task():
        while True:
            await asyncio.sleep(600)  # 10 minutes
            TIMESERIES_CACHE.clear_old()
            WELLS_CACHE.clear_old()
            GRACE_CACHE.clear_old()
            RAINFALL_CACHE.clear_old()
            print("🧹 Cache cleanup complete")
    
    asyncio.create_task(cleanup_task())


# =============================================================================
# GRACE BAND MAPPING
# =============================================================================

GRACE_BAND_MAPPING = {}

def auto_detect_grace_bands():
    global GRACE_BAND_MAPPING
    print("🔍 Auto-detecting GRACE band mappings...")
    try:
        with engine.connect() as conn:
            total_bands_query = text("SELECT ST_NumBands(rast) FROM public.grace_lwe WHERE rid = 1;")
            total_bands = conn.execute(total_bands_query).fetchone()[0]
            base_date = datetime(2002, 1, 1)
            for band in range(1, total_bands + 1):
                check_query = text(f"""
                    SELECT COUNT(*) FROM (
                        SELECT (ST_PixelAsCentroids(rast, {band})).val as val
                        FROM public.grace_lwe WHERE rid = 1 LIMIT 10
                    ) sub WHERE val IS NOT NULL;
                """)
                has_data = conn.execute(check_query).fetchone()[0] > 0
                if has_data:
                    estimated_days = 106.5 + (band - 1) * ((8597.5 - 106.5) / (total_bands - 1))
                    estimated_date = base_date + timedelta(days=estimated_days)
                    GRACE_BAND_MAPPING[(estimated_date.year, estimated_date.month)] = band
                    if band % 20 == 0:
                        print(f"  Checked {band}/{total_bands} bands...")
            print(f"✅ Detected {len(GRACE_BAND_MAPPING)} available GRACE months")
    except Exception as e:
        print(f"⚠️  Error auto-detecting GRACE bands: {e}")

def grace_date_to_band(year: int, month: int) -> int:
    if (year, month) not in GRACE_BAND_MAPPING:
        raise ValueError(f"No GRACE data for {year}-{month:02d}")
    return GRACE_BAND_MAPPING[(year, month)]

auto_detect_grace_bands()

# =============================================================================
# RAINFALL TABLE DETECTION
# =============================================================================

RAINFALL_TABLES = {}

def detect_rainfall_tables():
    global RAINFALL_TABLES
    print("🔍 Detecting rainfall tables...")
    try:
        with engine.connect() as conn:
            query = text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name LIKE 'rainfall_%'
                ORDER BY table_name;
            """)
            result = conn.execute(query)
            for row in result:
                table_name = row[0]
                import re
                match = re.search(r'rainfall_(\d{4})', table_name)
                if match:
                    year = int(match.group(1))
                    RAINFALL_TABLES[year] = table_name
            print(f"✅ Detected {len(RAINFALL_TABLES)} rainfall years: {sorted(RAINFALL_TABLES.keys())}")
    except Exception as e:
        print(f"⚠️  Error detecting rainfall tables: {e}")

detect_rainfall_tables()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
RESOURCE_TO_MCM_FACTOR = 1.0

def get_gwr_for_year(
    year: int,
    boundary_geojson: str = None
) -> pd.DataFrame:
    """
    Query GWR data for a specific year from database
    Returns DataFrame with district, state, annual_resource, geometry
    
    Currently supports: 2017 only (expandable)
    """
    
    # ✅ Validate year
    if year not in AVAILABLE_GWR_YEARS:
        print(f"WARNING: No GWR data for year {year}. Available: {AVAILABLE_GWR_YEARS}")
        return pd.DataFrame()
    
    # ✅ Get correct table name
    table_name = get_gwr_table_for_year(year)
    
    where_clause = ""
    params = {}
    
    if boundary_geojson:
        where_clause = """
            AND ST_Intersects(
                ST_MakeValid(geometry),
                ST_GeomFromGeoJSON(:boundary_geojson)
            )
        """
        params["boundary_geojson"] = boundary_geojson
    
    # ✅ Query with dynamic table name
    query = text(f"""
        SELECT 
            "District" as district,
            "state" as state,
            "Annual_Replenishable_Groundwater_Resource" as annual_resource,
            ST_AsGeoJSON(ST_MakeValid(geometry)) as geometry,
            ST_Area(ST_Transform(ST_MakeValid(geometry), 32643)) as area_m2
        FROM {table_name}
        WHERE "Annual_Replenishable_Groundwater_Resource" IS NOT NULL
        {where_clause}
    """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(query, params)
            
            records = []
            for row in result:
                try:
                    geom_json = json.loads(row[3])
                    
                    # Parse annual resource value
                    annual_resource = float(row[2]) if row[2] is not None else 0.0
                    
                    records.append({
                        "district": row[0],
                        "state": row[1],
                        "annual_resource_mcm": annual_resource * RESOURCE_TO_MCM_FACTOR,  # ✅ NEW
                        "area_m2": float(row[4]) if row[4] else 0.0,
                        "geometry": geom_json
                    })
                except Exception as e:
                    print(f"Error parsing GWR row: {e}")
                    continue
            
            return pd.DataFrame(records)
    
    except Exception as e:
        print(f"Error querying GWR for year {year}: {e}")
        return pd.DataFrame()


def clip_gwr_to_boundary(
    gwr_df: pd.DataFrame,
    boundary_geojson: str
) -> pd.DataFrame:
    """
    Clip GWR polygons to boundary (matching Dash's robust_clip_to_boundary logic)
    """
    if gwr_df.empty or not boundary_geojson:
        return gwr_df
    
    try:
        boundary_geom = shape(json.loads(boundary_geojson))
        prepared_boundary = prep(boundary_geom)
        
        clipped_records = []
        
        for idx, row in gwr_df.iterrows():
            try:
                # Get original geometry
                gwr_geom = shape(row['geometry'])
                
                # Check if it intersects
                if prepared_boundary.intersects(gwr_geom):
                    # Clip to boundary
                    clipped_geom = gwr_geom.intersection(boundary_geom)
                    
                    # Skip if empty after clipping
                    if clipped_geom.is_empty:
                        continue
                    
                    # Update geometry
                    clipped_row = row.copy()
                    clipped_row['geometry'] = clipped_geom.__geo_interface__
                    clipped_records.append(clipped_row)
            
            except Exception as e:
                print(f"Error clipping GWR polygon: {e}")
                continue
        
        return pd.DataFrame(clipped_records)
    
    except Exception as e:
        print(f"Error in clip_gwr_to_boundary: {e}")
        return gwr_df
    
def day_of_year(year: int, month: int, day: int) -> int:
    return datetime(year, month, day).timetuple().tm_yday

def get_month_day_range(year: int, month: int):
    first_day = day_of_year(year, month, 1)
    last_day_of_month = calendar.monthrange(year, month)[1]
    last_day = day_of_year(year, month, last_day_of_month)
    return first_day, last_day

def get_boundary_geometry(state: str = None, district: str = None):
    if district and state:
        boundary_query = text("""
            SELECT ST_AsGeoJSON(geometry) as geojson
            FROM district_state
            WHERE UPPER("District") = UPPER(:district)
            AND UPPER("State") = UPPER(:state)
            LIMIT 1;
        """)
        params = {"district": district, "state": state}
    elif state:
        boundary_query = text("""
            SELECT ST_AsGeoJSON(ST_Union(geometry)) as geojson
            FROM district_state
            WHERE UPPER("State") = UPPER(:state);
        """)
        params = {"state": state}
    else:
        return None
    try:
        with engine.connect() as conn:
            boundary_row = conn.execute(boundary_query, params).fetchone()
        if not boundary_row:
            return None
        geojson = json.loads(boundary_row[0])
        geometry = shape(geojson)
        return prep(geometry)
    except Exception as e:
        print(f"Error getting boundary: {e}")
        return None

def get_boundary_geojson(state: str = None, district: str = None):
    """Get boundary GeoJSON with robust error handling"""
    if district and state:
        boundary_query = text("""
            SELECT ST_AsGeoJSON(ST_MakeValid(geometry)) as geojson
            FROM district_state
            WHERE UPPER("District") = UPPER(:district)
            AND UPPER("State") = UPPER(:state)
            LIMIT 1;
        """)
        params = {"district": district, "state": state}
    elif state:
        # ✅ FIX: Multiple approaches for state-level union
        boundary_query = text("""
            WITH valid_geoms AS (
                SELECT ST_MakeValid(geometry) as geom
                FROM district_state
                WHERE UPPER("State") = UPPER(:state)
                AND geometry IS NOT NULL
            ),
            buffered AS (
                -- Buffer by 0 often fixes topology issues
                SELECT ST_Buffer(geom, 0) as geom
                FROM valid_geoms
            )
            SELECT ST_AsGeoJSON(
                CASE 
                    -- Try union first
                    WHEN ST_IsValid(ST_Union(geom)) THEN ST_Union(geom)
                    -- If that fails, try collecting
                    WHEN ST_IsValid(ST_Collect(geom)) THEN ST_Collect(geom)
                    -- Last resort: just take first geometry
                    ELSE (SELECT geom FROM buffered LIMIT 1)
                END
            ) as geojson
            FROM buffered;
        """)
        params = {"state": state}
    else:
        return None
    
    try:
        with engine.connect() as conn:
            boundary_row = conn.execute(boundary_query, params).fetchone()
        
        if not boundary_row or not boundary_row[0]:
            print(f"⚠️  No boundary found for state={state}, district={district}")
            return None
        
        # Validate the returned GeoJSON
        try:
            geojson_test = json.loads(boundary_row[0])
            if not geojson_test.get('coordinates'):
                print(f"⚠️  Empty coordinates in boundary")
                return None
        except json.JSONDecodeError:
            print(f"⚠️  Invalid GeoJSON returned")
            return None
        
        return boundary_row[0]
    
    except Exception as e:
        print(f"Error getting boundary: {e}")
        return None

def filter_points_by_boundary(points: list, prepared_geom) -> list:
    if prepared_geom is None:
        return points
    filtered = []
    for point in points:
        pt = Point(point["longitude"], point["latitude"])
        if prepared_geom.contains(pt) or prepared_geom.touches(pt):
            filtered.append(point)
    return filtered

def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    from math import radians, sin, cos, sqrt, atan2
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def get_boundary_centroid(state: str = None, district: str = None):
    if district and state:
        query = text("""
            SELECT ST_Y(ST_Centroid(geometry)) as lat, ST_X(ST_Centroid(geometry)) as lon
            FROM district_state
            WHERE UPPER("District") = UPPER(:district)
            AND UPPER("State") = UPPER(:state)
            LIMIT 1;
        """)
        params = {"district": district, "state": state}
    elif state:
        query = text("""
            SELECT ST_Y(ST_Centroid(ST_Union(geometry))) as lat, 
                   ST_X(ST_Centroid(ST_Union(geometry))) as lon
            FROM district_state
            WHERE UPPER("State") = UPPER(:state);
        """)
        params = {"state": state}
    else:
        return None
    try:
        with engine.connect() as conn:
            result = conn.execute(query, params).fetchone()
            if result:
                return {"lat": float(result[0]), "lon": float(result[1])}
    except Exception as e:
        print(f"Error getting centroid: {e}")
    return None

def apply_fallback_nearest_points(all_points: list, centroid: dict, max_distance_km: int = 50, max_points: int = 100) -> list:
    if not centroid:
        return []
    points_with_distance = []
    for point in all_points:
        distance = haversine_distance(
            centroid["lon"], centroid["lat"],
            point["longitude"], point["latitude"]
        )
        if distance <= max_distance_km:
            point["distance_km"] = round(distance, 2)
            points_with_distance.append(point)
    points_with_distance.sort(key=lambda p: p["distance_km"])
    return points_with_distance[:max_points]

def categorize_water_level(depth):
    if depth is None or depth < 0:
        return 'Recharge'
    elif depth < 30:
        return 'Shallow (0-30m)'
    elif depth < 60:
        return 'Moderate (30-60m)'
    elif depth < 100:
        return 'Deep (60-100m)'
    else:
        return 'Very Deep (>100m)'
    
def compute_gridded_density(wells_gdf, selection_boundary, bounds, radius_km=20.0, max_cells=1600):
    """
    Compute absolute density grid (sites per 1000 km²) clipped to AOI
    
    Args:
        wells_gdf: GeoDataFrame with well data
        selection_boundary: GeoDataFrame with boundary geometry (or None)
        bounds: [minx, miny, maxx, maxy]
        radius_km: Search radius in kilometers
        max_cells: Maximum grid resolution
    
    Returns: DataFrame with columns [x, y, density_per_1000km2]
    """
    if wells_gdf is None or wells_gdf.empty:
        return pd.DataFrame(columns=['x', 'y', 'density_per_1000km2'])
    
    # Get unique site locations
    site_locs = wells_gdf.groupby('site_id')[['latitude', 'longitude']].first()
    if site_locs.empty:
        return pd.DataFrame(columns=['x', 'y', 'density_per_1000km2'])
    
    coords_deg = site_locs[['latitude', 'longitude']].values
    
    # Create grid
    x_min, y_min, x_max, y_max = bounds
    width = max(x_max - x_min, 1e-6)
    height = max(y_max - y_min, 1e-6)
    area_deg2 = width * height
    
    # Adaptive resolution
    if area_deg2 <= 20.0:
        max_cells = 2500
    else:
        max_cells = 1600
    
    aspect = width / (height + 1e-9)
    nx = max(int(np.sqrt(max_cells * max(aspect, 1e-3))), 18)
    ny = max(int(max_cells / max(nx, 1)), 18)
    
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    GX, GY = np.meshgrid(xs, ys)
    
    grid_lat = GY.flatten()
    grid_lon = GX.flatten()
    
    # Clip grid to AOI boundary
    if selection_boundary is not None and not selection_boundary.empty:
        try:
            union_geom =unary_union(selection_boundary.geometry)
            P = prep(union_geom)
            keep = np.fromiter(
                (P.contains(Point(lo, la)) or P.touches(Point(lo, la)) 
                 for lo, la in zip(grid_lon, grid_lat)),
                dtype=bool,
                count=grid_lon.size
            )
            grid_lat = grid_lat[keep]
            grid_lon = grid_lon[keep]
        except Exception as e:
            print(f"Warning: Grid clipping failed: {e}")
    
    if grid_lon.size == 0:
        return pd.DataFrame(columns=['x', 'y', 'density_per_1000km2'])
    
    # Compute density at each grid point using haversine
    area_km2 = np.pi * (radius_km ** 2)
    
    # Convert to radians for haversine
    coords_rad = np.deg2rad(coords_deg)
    grid_coords_rad = np.deg2rad(np.column_stack([grid_lat, grid_lon]))
    
    # Use NearestNeighbors with radius
    nbrs = NearestNeighbors(
        radius=(radius_km / 6371.0),
        metric='haversine',
        algorithm='ball_tree'
    ).fit(coords_rad)
    
    neighbors = nbrs.radius_neighbors(grid_coords_rad, return_distance=False)
    
    # Count neighbors at each grid point
    counts = np.array([len(idx) for idx in neighbors], dtype=float)
    
    # Convert to density per 1000 km²
    density_per_1000km2 = (counts / area_km2) * 1000.0
    
    # Build result dataframe
    result = pd.DataFrame({
        'x': grid_lon,
        'y': grid_lat,
        'density_per_1000km2': density_per_1000km2
    })
    
    return result

def interpolate_grid_points(points, value_key="lwe_cm"):
    """
    Increase point density through spatial interpolation
    - For 1-2 points: triple the count (1→3, 2→6)
    - For 3+ points: ~2x density with interpolation
    """
    if not points:
        return points
    
    # Special handling for 1-2 points: just triple them with small offsets
    if len(points) <= 2:
        print(f"Small region interpolation: {len(points)} points → tripling to {len(points) * 3}")
        expanded_points = []
        
        # Use smaller offset and spread in multiple directions to stay within boundary
        offset = 0.03  # Reduced from 0.08 to 0.03 degrees (~3km)
        
        for original_point in points:
            # Original point
            expanded_points.append(original_point.copy())
            
            # Add 2 nearby points in different directions (NE and SW pattern)
            # This creates a small cluster around the original point
            for dx, dy in [(offset, offset), (-offset, -offset)]:
                new_point = original_point.copy()
                new_point['longitude'] = original_point['longitude'] + dx
                new_point['latitude'] = original_point['latitude'] + dy
                expanded_points.append(new_point)
        
        print(f"Small region SUCCESS: {len(points)} → {len(expanded_points)} points")
        return expanded_points
    
    # Regular interpolation for 3+ points
    coords = np.array([[p['longitude'], p['latitude']] for p in points])
    values = np.array([p[value_key] for p in points])
    
    # Calculate grid spacing
    unique_lons = np.unique(coords[:, 0])
    unique_lats = np.unique(coords[:, 1])
    
    print(f"Interpolation input: {len(points)} points, {len(unique_lons)} unique lons, {len(unique_lats)} unique lats")
    
    if len(unique_lons) < 2 or len(unique_lats) < 2:
        print(f"Interpolation skipped: points don't form a 2D grid (lons={len(unique_lons)}, lats={len(unique_lats)})")
        return points
    
    lon_spacing = np.median(np.diff(np.sort(unique_lons)))
    lat_spacing = np.median(np.diff(np.sort(unique_lats)))
    
    # Create grid with 1.5x density (gives ~2.25x total points)
    density_factor = 1.5
    new_lon_spacing = lon_spacing / density_factor
    new_lat_spacing = lat_spacing / density_factor
    
    lon_min, lon_max = coords[:, 0].min(), coords[:, 0].max()
    lat_min, lat_max = coords[:, 1].min(), coords[:, 1].max()
    
    new_lons = np.arange(lon_min, lon_max + new_lon_spacing, new_lon_spacing)
    new_lats = np.arange(lat_min, lat_max + new_lat_spacing, new_lat_spacing)
    
    print(f"Creating interpolation grid: {len(new_lons)}x{len(new_lats)} = {len(new_lons) * len(new_lats)} potential points")
    
    grid_lon, grid_lat = np.meshgrid(new_lons, new_lats)
    
    # Interpolate values
    try:
        grid_values = griddata(coords, values, (grid_lon, grid_lat), method='linear', fill_value=np.nan)
    except Exception as e:
        print(f"Interpolation griddata failed: {e}")
        return points
    
    # Convert to point list
    interpolated_points = []
    for i in range(len(new_lats)):
        for j in range(len(new_lons)):
            if not np.isnan(grid_values[i, j]):
                point = {
                    'longitude': float(grid_lon[i, j]),
                    'latitude': float(grid_lat[i, j]),
                    value_key: round(float(grid_values[i, j]), 3)
                }
                # Copy other fields from first point
                for key in points[0].keys():
                    if key not in ['longitude', 'latitude', value_key]:
                        point[key] = points[0][key]
                interpolated_points.append(point)
    
    print(f"Interpolation SUCCESS: {len(points)} → {len(interpolated_points)} points (density x{density_factor})")
    return interpolated_points if len(interpolated_points) > 0 else points


def build_monthly_site_matrix(wells_df):
    """
    Build monthly site matrix (date × site_id) for composite GWL series
    """
    if wells_df is None or wells_df.empty:
        return pd.DataFrame()
    
    if 'date' not in wells_df.columns:
        return pd.DataFrame()
    
    df = wells_df[['site_id', 'date', 'gwl']].copy()
    df['date'] = pd.to_datetime(df['date'])
    
    df = df.set_index('date').groupby('site_id').resample('MS').mean().reset_index()
    mat = df.pivot(index='date', columns='site_id', values='gwl').sort_index()
    
    return mat


def haversine_distance_vectorized(lat1, lon1, lat2, lon2):
    """Calculate haversine distance in km (vectorized)"""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c


def idw_weights_from_distances(distances, power=2, eps=1e-12):
    """Compute IDW weights from distances"""
    w = 1.0 / np.power(distances + eps, power)
    w[np.isinf(w)] = 1 / eps
    wsum = np.sum(w, axis=1, keepdims=True)
    wsum[wsum == 0] = eps
    return w / wsum

# =============================================================================
# UNIFIED TIMESERIES HELPERS (FIXED)
# =============================================================================

def query_wells_monthly(boundary_geojson: str = None):
    """OPTIMIZED: Added caching"""
    # Check cache first
    cache_key = f"wells_monthly_{hash(boundary_geojson) if boundary_geojson else 'all'}"
    cached = WELLS_CACHE.get(cache_key)
    if cached is not None:
        print(f"  ✓ Wells (cached): {len(cached)} months")
        return cached
    
    # Original query logic
    params = {}
    where_clauses = []

    if boundary_geojson:
        where_clauses.append("""
            ST_Intersects(
                ST_GeomFromGeoJSON(:boundary_geojson),
                ST_SetSRID(ST_MakePoint(
                    COALESCE(NULLIF("LON", 0), "LONGITUD_1"),
                    COALESCE(NULLIF("LAT", 0), "LATITUDE_1")
                ), 4326)
            )
        """)
        params["boundary_geojson"] = boundary_geojson

    where_clauses.append('"GWL" IS NOT NULL')
    where_clause = ("WHERE " + "\n AND ".join(where_clauses)) if where_clauses else ""

    query = text(f"""
        SELECT 
            DATE_TRUNC('month', "Date") as period,
            AVG("GWL") as avg_gwl,
            COUNT(*) as count
        FROM groundwater_level
        {where_clause}
        GROUP BY DATE_TRUNC('month', "Date")
        ORDER BY period;
    """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(query, params)
            df = pd.DataFrame([
                {"period": row[0], "avg_gwl": float(row[1]), "count": int(row[2])}
                for row in result
            ])
            
            # Store in cache
            WELLS_CACHE.set(cache_key, df)
            
            if not df.empty:
                print(f"  ✓ Wells (fresh): {len(df)} months retrieved")
            return df
    except Exception as e:
        print(f"  ❌ Wells query error: {e}")
        return pd.DataFrame(columns=["period", "avg_gwl", "count"])


def query_grace_monthly(boundary_geojson: str = None):
    """
    OPTIMIZED: Query all GRACE years at once, return monthly results
    Caches entire dataset per boundary
    """
    # Check cache first
    cache_key = f"grace_monthly_{hash(boundary_geojson) if boundary_geojson else 'all'}"
    cached = GRACE_CACHE.get(cache_key)
    if cached is not None:
        print(f"  ✓ GRACE (cached): {len(cached)} months")
        return cached
    
    if not GRACE_BAND_MAPPING:
        print("  ⚠️  GRACE band mapping is empty!")
        return pd.DataFrame(columns=['period', 'avg_tws'])
    
    print(f"  🛰️  Querying GRACE: {len(GRACE_BAND_MAPPING)} months (year-based batching)...")
    
    # Group bands by year for batch processing
    years_dict = {}
    for (year, month), band in GRACE_BAND_MAPPING.items():
        if year not in years_dict:
            years_dict[year] = []
        years_dict[year].append((month, band))
    
    grace_data = []
    
    # Query each year's data in one go
    for year, month_band_list in sorted(years_dict.items()):
        try:
            # Build band list for this year
            bands_str = ', '.join([str(band) for _, band in month_band_list])
            
            # Single query for entire year
            if boundary_geojson:
                query = text(f"""
                    WITH monthly_data AS (
                        SELECT 
                            band_num,
                            SUM(lwe_cm * COS(RADIANS(latitude))) / NULLIF(SUM(COS(RADIANS(latitude))), 0) as weighted_avg
                        FROM (
                            SELECT 
                                unnest(ARRAY[{bands_str}]) as band_num,
                                ST_Y(ST_SetSRID((ST_PixelAsCentroids(rast, unnest(ARRAY[{bands_str}]))).geom, 4326)) as latitude,
                                (ST_PixelAsCentroids(rast, unnest(ARRAY[{bands_str}]))).val as lwe_cm,
                                ST_SetSRID((ST_PixelAsCentroids(rast, unnest(ARRAY[{bands_str}]))).geom, 4326) as geom
                            FROM public.grace_lwe
                            WHERE rid = 1
                        ) pixels
                        WHERE lwe_cm IS NOT NULL
                        AND ST_Intersects(geom, ST_GeomFromGeoJSON(:boundary))
                        GROUP BY band_num
                    )
                    SELECT band_num, weighted_avg FROM monthly_data
                """)
                params = {"boundary": boundary_geojson}
            else:
                query = text(f"""
                    WITH monthly_data AS (
                        SELECT 
                            band_num,
                            SUM(lwe_cm * COS(RADIANS(latitude))) / NULLIF(SUM(COS(RADIANS(latitude))), 0) as weighted_avg
                        FROM (
                            SELECT 
                                unnest(ARRAY[{bands_str}]) as band_num,
                                ST_Y(ST_SetSRID((ST_PixelAsCentroids(rast, unnest(ARRAY[{bands_str}]))).geom, 4326)) as latitude,
                                (ST_PixelAsCentroids(rast, unnest(ARRAY[{bands_str}]))).val as lwe_cm
                            FROM public.grace_lwe
                            WHERE rid = 1
                        ) pixels
                        WHERE lwe_cm IS NOT NULL
                        GROUP BY band_num
                    )
                    SELECT band_num, weighted_avg FROM monthly_data
                """)
                params = {}
            
            with engine.connect() as conn:
                result = conn.execute(query, params)
                
                # Map band numbers back to months
                band_to_month = {band: month for month, band in month_band_list}
                
                for row in result:
                    band_num = row[0]
                    avg_tws = row[1]
                    
                    if avg_tws is not None and band_num in band_to_month:
                        month = band_to_month[band_num]
                        grace_data.append({
                            "period": datetime(year, month, 1),
                            "avg_tws": float(avg_tws)
                        })
            
            print(f"    ✓ Year {year}: {len([m for m, _ in month_band_list])} months")
            
        except Exception as e:
            print(f"    ⚠️  Year {year} error: {str(e)[:150]}")
            continue
    
    df = pd.DataFrame(grace_data) if grace_data else pd.DataFrame(columns=['period', 'avg_tws'])
    
    # Store in cache
    GRACE_CACHE.set(cache_key, df)
    
    print(f"  ✓ GRACE (fresh): {len(df)}/{len(GRACE_BAND_MAPPING)} months retrieved")
    return df

def query_rainfall_monthly(boundary_geojson: str = None):
    """
    OPTIMIZED: Parallel year processing for 5-6x speedup
    Queries multiple years simultaneously with ThreadPoolExecutor
    """
    # Check cache first
    cache_key = f"rainfall_monthly_{hash(boundary_geojson) if boundary_geojson else 'all'}"
    cached = RAINFALL_CACHE.get(cache_key)
    if cached is not None:
        print(f"  ✓ Rainfall (cached): {len(cached)} months")
        return cached
    
    if not RAINFALL_TABLES:
        print("  ⚠️  No rainfall tables found!")
        return pd.DataFrame(columns=['period', 'avg_rainfall'])
    
    print(f"  🌧️  Querying Rainfall: {len(RAINFALL_TABLES)} years (parallel processing)...")
    
    def process_year(year_table_tuple):
        """Process a single year's rainfall data"""
        year, table_name = year_table_tuple
        year_data = []
        
        try:
            for month in range(1, 13):
                first_day, last_day = get_month_day_range(year, month)
                bands = list(range(first_day, last_day + 1))
                bands_str = ','.join(map(str, bands))
                
                # Single query for entire month (all days)
                if boundary_geojson:
                    query = text(f"""
                        WITH daily_pixels AS (
                            SELECT 
                                ST_X(ST_SetSRID((ST_PixelAsCentroids(rast, unnest(ARRAY[{bands_str}]))).geom, 4326)) as lon,
                                ST_Y(ST_SetSRID((ST_PixelAsCentroids(rast, unnest(ARRAY[{bands_str}]))).geom, 4326)) as lat,
                                (ST_PixelAsCentroids(rast, unnest(ARRAY[{bands_str}]))).val as val
                            FROM {table_name}
                            WHERE rid = 1
                        )
                        SELECT 
                            SUM(val * COS(RADIANS(lat))) / NULLIF(SUM(COS(RADIANS(lat))), 0) as weighted_avg
                        FROM daily_pixels
                        WHERE val IS NOT NULL 
                        AND val >= 0 
                        AND val <= 500
                        AND ST_Intersects(
                            ST_SetSRID(ST_MakePoint(lon, lat), 4326),
                            ST_GeomFromGeoJSON(:boundary)
                        )
                    """)
                    params = {"boundary": boundary_geojson}
                else:
                    query = text(f"""
                        WITH daily_pixels AS (
                            SELECT 
                                ST_Y(ST_SetSRID((ST_PixelAsCentroids(rast, unnest(ARRAY[{bands_str}]))).geom, 4326)) as lat,
                                (ST_PixelAsCentroids(rast, unnest(ARRAY[{bands_str}]))).val as val
                            FROM {table_name}
                            WHERE rid = 1
                        )
                        SELECT 
                            SUM(val * COS(RADIANS(lat))) / NULLIF(SUM(COS(RADIANS(lat))), 0) as weighted_avg
                        FROM daily_pixels
                        WHERE val IS NOT NULL 
                        AND val >= 0 
                        AND val <= 500
                    """)
                    params = {}
                
                with engine.connect() as conn:
                    result = conn.execute(query, params).fetchone()
                    
                    if result and result[0] is not None:
                        year_data.append({
                            "period": datetime(year, month, 1),
                            "avg_rainfall": round(float(result[0]), 2)
                        })
            
            print(f"    ✓ Year {year}: {len(year_data)} months")
            return year_data
            
        except Exception as e:
            print(f"    ⚠️  Year {year} error: {str(e)[:150]}")
            return []
    
    # Process years in parallel (max 6 workers to avoid overwhelming DB)
    all_year_items = list(sorted(RAINFALL_TABLES.items()))
    rainfall_data = []
    
    with ThreadPoolExecutor(max_workers=6) as executor:
        # Submit all year processing tasks
        futures = [executor.submit(process_year, year_item) for year_item in all_year_items]
        
        # Collect results as they complete
        for future in futures:
            year_result = future.result()
            rainfall_data.extend(year_result)
    
    df = pd.DataFrame(rainfall_data) if rainfall_data else pd.DataFrame(columns=['period', 'avg_rainfall'])
    
    # Sort by period (since parallel processing may complete out of order)
    if not df.empty:
        df = df.sort_values('period').reset_index(drop=True)
    
    # Store in cache
    RAINFALL_CACHE.set(cache_key, df)
    
    print(f"  ✓ Rainfall (fresh): {len(df)} months retrieved")
    return df


AVAILABLE_GWR_YEARS = [2017]  # ← Add new years here when you get them
def get_available_gwr_years():
    """Returns list of years with GWR data available"""
    return AVAILABLE_GWR_YEARS

# Year-to-table mapping (for future multi-year support)
GWR_TABLE_MAP = {
    2017: "ground_water_resource_sod_dt_2017"
    # Future: 2020: "ground_water_resource_sod_dt_2020"
}

def get_gwr_table_for_year(year: int) -> str:
    """Get table name for a specific year"""
    if year not in GWR_TABLE_MAP:
        raise ValueError(f"No GWR data for year {year}")
    return GWR_TABLE_MAP[year]

def get_gwr_table_for_year(year: int) -> str:
    """Get table name for a specific year"""
    if year not in GWR_TABLE_MAP:
        raise ValueError(f"No GWR data for year {year}. Available: {AVAILABLE_GWR_YEARS}")
    return GWR_TABLE_MAP[year]
# =============================================================================
# ROOT ENDPOINT
# =============================================================================

@app.get("/")
def root():
    return {
        "title": "GeoHydro API - Groundwater Monitoring System with AI Chatbot",
        "version": "6.1.0 (FIXED - GRACE & Rainfall NaN Issues Resolved)",
        "description": "API for groundwater data including wells, GRACE, rainfall, and AI chatbot with unified timeseries",
        "chatbot_enabled": CHATBOT_ENABLED,
        "endpoints": {
            "diagnostics": [
                "GET /api/debug/raster-info - Diagnostic endpoint for troubleshooting"
            ],
            "geography": [
                "GET /api/states - List all states",
                "GET /api/districts/{state} - Get districts by state",
                "GET /api/district/{state}/{district} - Get specific district"
            ],
            "aquifers": [
                "GET /api/aquifers - All aquifers",
                "GET /api/aquifers/state/{state} - Aquifers by state",
                "GET /api/aquifers/district/{state}/{district} - Aquifers by district"
            ],
            "wells": [
                "GET /api/wells - Get well measurements (spatial filtering)",
                "GET /api/wells/timeseries - UNIFIED monthly timeseries (Wells + GRACE + Rainfall)",
                "GET /api/wells/summary - Regional statistics",
                "GET /api/wells/storage - Pre/post-monsoon storage",
                "GET /api/wells/years - Available year range (dynamic)"
            ],
            "gwr": [
                "GET /api/gwr/available-years - Get years with GWR data",  # ← ADD THIS
                "GET /api/gwr - Get GWR data for specific year",
                "GET /api/gwr/timeseries - GWR time series",
                "GET /api/gwr/summary - Comprehensive GWR summary",
                "GET /api/wells/storage-vs-gwr - Storage vs GWR comparison"
            ],
            "grace": [
                "GET /api/grace/available - Available GRACE months",
                "GET /api/grace - GRACE TWS data"
            ],
            "rainfall": [
                "GET /api/rainfall - Rainfall data"
            ]
        },
        "docs": "/docs",
        "database_info": {
            "grace_months_available": len(GRACE_BAND_MAPPING),
            "rainfall_years_available": len(RAINFALL_TABLES)
        }
    }

# =============================================================================
# DIAGNOSTIC ENDPOINT
# =============================================================================

@app.get("/api/debug/raster-info")
async def debug_raster_info(
    state: Optional[str] = None,
    district: Optional[str] = None
):
    try:
        boundary_geojson = get_boundary_geojson(state, district)
        
        if not boundary_geojson:
            return {"error": "Could not get boundary for specified state/district"}
        
        grace_check = None
        grace_intersects = None
        grace_test = None
        grace_sample = []
        
        try:
            with engine.connect() as conn:
                grace_check = conn.execute(text("""
                    SELECT 
                        rid,
                        ST_NumBands(rast) as num_bands,
                        ST_AsText(ST_Envelope(rast::geometry)) as bbox,
                        ST_SRID(rast::geometry) as srid,
                        ST_Width(rast) as width,
                        ST_Height(rast) as height
                    FROM public.grace_lwe
                    WHERE rid = 1
                """)).fetchone()
                
                grace_intersects = conn.execute(text("""
                    SELECT ST_Intersects(
                        rast::geometry,
                        ST_GeomFromGeoJSON(:boundary)
                    ) as intersects
                    FROM public.grace_lwe
                    WHERE rid = 1
                """), {"boundary": boundary_geojson}).fetchone()
                
                grace_sample = conn.execute(text("""
                    WITH pixels AS (
                        SELECT 
                            (ST_PixelAsCentroids(rast, 1)).*
                        FROM public.grace_lwe
                        WHERE rid = 1
                        LIMIT 10
                    )
                    SELECT 
                        ST_X(geom) as lon,
                        ST_Y(geom) as lat,
                        val
                    FROM pixels
                    WHERE val IS NOT NULL
                """)).fetchall()
                
                grace_test = conn.execute(text("""
                    WITH pixels AS (
                        SELECT 
                            (ST_PixelAsCentroids(rast, 1)).*
                        FROM public.grace_lwe
                        WHERE rid = 1
                    )
                    SELECT 
                        COUNT(*) as total_pixels,
                        COUNT(*) FILTER (WHERE val IS NOT NULL) as non_null_pixels,
                        COUNT(*) FILTER (
                            WHERE val IS NOT NULL 
                            AND ST_Intersects(geom, ST_GeomFromGeoJSON(:boundary))
                        ) as intersecting_pixels,
                        AVG(val) FILTER (
                            WHERE val IS NOT NULL 
                            AND ST_Intersects(geom, ST_GeomFromGeoJSON(:boundary))
                        ) as avg_value
                    FROM pixels
                """), {"boundary": boundary_geojson}).fetchone()
        except Exception as e:
            print(f"GRACE check error: {e}")
        
        rainfall_info = []
        for year, table_name in list(RAINFALL_TABLES.items())[:3]:
            try:
                with engine.connect() as conn:
                    rf_check = conn.execute(text(f"""
                        SELECT 
                            ST_NumBands(rast) as num_bands,
                            ST_AsText(ST_Envelope(rast::geometry)) as bbox
                        FROM {table_name}
                        WHERE rid = 1
                        LIMIT 1
                    """)).fetchone()
                    
                    rf_test = conn.execute(text(f"""
                        WITH pixels AS (
                            SELECT 
                                (ST_PixelAsCentroids(rast, 1)).*
                            FROM {table_name}
                            WHERE rid = 1
                        )
                        SELECT 
                            COUNT(*) FILTER (
                                WHERE val IS NOT NULL 
                                AND ST_Intersects(geom, ST_GeomFromGeoJSON(:boundary))
                            ) as intersecting_pixels
                        FROM pixels
                    """), {"boundary": boundary_geojson}).fetchone()
                    
                    rainfall_info.append({
                        "year": year,
                        "table": table_name,
                        "bands": rf_check[0] if rf_check else None,
                        "bbox": rf_check[1] if rf_check else None,
                        "intersecting_pixels": rf_test[0] if rf_test else 0
                    })
            except Exception as e:
                rainfall_info.append({
                    "year": year,
                    "table": table_name,
                    "error": str(e)[:200]
                })
        
        return {
            "status": "success",
            "boundary": {
                "state": state,
                "district": district,
                "has_geojson": boundary_geojson is not None
            },
            "grace": {
                "rid": grace_check[0] if grace_check else None,
                "num_bands": grace_check[1] if grace_check else None,
                "bbox": grace_check[2] if grace_check else None,
                "srid": grace_check[3] if grace_check else None,
                "dimensions": f"{grace_check[4]}x{grace_check[5]}" if grace_check else None,
                "boundary_intersects": grace_intersects[0] if grace_intersects else None,
                "test_query": {
                    "total_pixels": grace_test[0] if grace_test else 0,
                    "non_null_pixels": grace_test[1] if grace_test else 0,
                    "intersecting_pixels": grace_test[2] if grace_test else 0,
                    "avg_value": float(grace_test[3]) if grace_test and grace_test[3] else None
                },
                "sample_pixels": [
                    {"lon": p[0], "lat": p[1], "val": p[2]}
                    for p in grace_sample
                ] if grace_sample else []
            },
            "rainfall": {
                "tables_found": len(RAINFALL_TABLES),
                "years_range": f"{min(RAINFALL_TABLES.keys())}-{max(RAINFALL_TABLES.keys())}" if RAINFALL_TABLES else None,
                "sample_tables": rainfall_info
            },
            "grace_band_mapping": {
                "total_bands": len(GRACE_BAND_MAPPING),
                "sample": dict(list(GRACE_BAND_MAPPING.items())[:5]),
                "latest": dict(list(GRACE_BAND_MAPPING.items())[-5:])
            }
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# =============================================================================
# GEOGRAPHY ENDPOINTS
# =============================================================================

@app.get("/api/states")
def get_all_states():
    query = text("""
        SELECT DISTINCT "State"
        FROM public.district_state
        WHERE "State" IS NOT NULL AND "State" <> ''
        ORDER BY "State";
    """)
    try:
        with engine.connect() as conn:
            result = conn.execute(query)
            states = [{"State": row[0]} for row in result]
            return states
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/districts/{state_name}")
def get_districts(state_name: str):
    query = text("""
        SELECT 
            "District",
            "State",
            ST_AsGeoJSON(geometry) AS geojson,
            ST_Y(ST_Centroid(geometry)) AS center_lat,
            ST_X(ST_Centroid(geometry)) AS center_lng
        FROM public.district_state
        WHERE UPPER("State") = UPPER(:state_name)
        ORDER BY "District";
    """)
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"state_name": state_name})
            districts = [
                {
                    "district_name": row[0],
                    "State": row[1],
                    "geometry": json.loads(row[2]),
                    "center": [row[3], row[4]] if row[3] and row[4] else None
                }
                for row in result
            ]
            if not districts:
                raise HTTPException(status_code=404, detail=f"No districts found")
            return districts
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/gwr/available-years")
def get_gwr_available_years():
    """Get all years with GWR data available"""
    try:
        years = get_available_gwr_years()
        
        if not years:
            return {
                "status": "error",
                "message": "No GWR data found in database",
                "years": []
            }
        
        return {
            "status": "success",
            "min_year": min(years),
            "max_year": max(years),
            "total_years": len(years),
            "years": years
        }
    
    except Exception as e:
        print(f"GWR years error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gwr")
def get_gwr_data(
    year: int = Query(2017, ge=2017, le=2030, description="Year for GWR data"),
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    clip_to_boundary: bool = Query(True, description="Clip polygons to exact boundary")
):
    """
    Get Ground Water Resource (GWR) data for a specific year
    
    Currently available: 2017 only
    Returns GeoJSON with annual replenishable groundwater resource (MCM) per polygon
    """
    
    if year not in AVAILABLE_GWR_YEARS:
        raise HTTPException(
            status_code=404,
            detail=f"GWR data not available for {year}. Available years: {AVAILABLE_GWR_YEARS}"
        )
    
    try:
        # Get boundary
        boundary_geojson = None
        if state or district:
            boundary_geojson = get_boundary_geojson(state, district)
            if not boundary_geojson:
                raise HTTPException(
                    status_code=404,
                    detail=f"Boundary not found for state={state}, district={district}"
                )
        
        # Query GWR data
        gwr_df = get_gwr_for_year(year, boundary_geojson)
        
        # ✅ CHECK IF EMPTY FIRST (before calculating zmin/zmax)
        if gwr_df.empty:
            return {
                "data_type": "gwr",
                "year": year,
                "filters": {"state": state, "district": district},
                "status": "no_data",
                "message": f"No GWR data found for year {year}",
                "count": 0,
                "total_resource_mcm": 0,
                "geojson": {
                    "type": "FeatureCollection",
                    "features": []
                }
            }
        
        # ✅ NOW SAFE TO CALCULATE (gwr_df is not empty)
        zmin = float(gwr_df["annual_resource_mcm"].min())
        zmax = float(gwr_df["annual_resource_mcm"].max())
        
        # Handle edge case where all values are the same
        if zmax - zmin < 0.001:
            zmin = zmin - 0.1 * abs(zmin) if zmin != 0 else 0
            zmax = zmax + 0.1 * abs(zmax) if zmax != 0 else 1
        
        # Optionally clip to exact boundary (like Dash does)
        if clip_to_boundary and boundary_geojson:
            gwr_df = clip_gwr_to_boundary(gwr_df, boundary_geojson)
        
        # Build GeoJSON
        features = []
        for idx, row in gwr_df.iterrows():
            features.append({
                "type": "Feature",
                "id": str(idx),
                "properties": {
                    "district": row["district"],
                    "state": row["state"],
                    "annual_resource_mcm": round(float(row["annual_resource_mcm"]), 2),
                    "area_m2": float(row["area_m2"])
                },
                "geometry": row["geometry"]
            })
        
        geojson_data = {
            "type": "FeatureCollection",
            "features": features
        }
        
        # Calculate statistics
        total_resource = gwr_df["annual_resource_mcm"].sum()
        
        # ✅ NOW ALL VARIABLES EXIST - RETURN RESPONSE
        return {
            "data_type": "gwr",
            "year": year,
            "filters": {
                "state": state,
                "district": district,
                "clip_to_boundary": clip_to_boundary
            },
            "status": "success",
            "statistics": {
                "total_resource_mcm": round(float(total_resource), 2),
                "mean_resource_mcm": round(float(gwr_df["annual_resource_mcm"].mean()), 2),
                "min_resource_mcm": round(zmin, 2),
                "max_resource_mcm": round(zmax, 2),
                "num_districts": int(gwr_df["district"].nunique()),
                "total_area_km2": round(float(gwr_df["area_m2"].sum()) / 1_000_000, 2),
                # ✅ ADD COLOR RANGE FOR FRONTEND NORMALIZATION
                "color_range": {
                    "zmin": round(zmin, 2),
                    "zmax": round(zmax, 2)
                }
            },
            "count": len(features),
            "geojson": geojson_data
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"GWR Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gwr/timeseries")
def get_gwr_timeseries(
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None)
):
    """
    Get GWR time series for all available years
    """
    try:
        # Get boundary
        boundary_geojson = None
        if state or district:
            boundary_geojson = get_boundary_geojson(state, district)
            if not boundary_geojson:
                raise HTTPException(
                    status_code=404,
                    detail=f"Boundary not found for state={state}, district={district}"
                )
        
        # ✅ Get available years dynamically
        available_years = get_available_gwr_years()
        
        if not available_years:
            raise HTTPException(status_code=404, detail="No GWR data available")
        
        # Query each year
        timeseries_data = []
        
        for year in available_years:
            gwr_df = get_gwr_for_year(year, boundary_geojson)
            
            if not gwr_df.empty:
                # Sum total resource for the year
                total_resource = gwr_df["annual_resource_mcm"].sum()
                
                if pd.notna(total_resource) and total_resource > 0:
                    timeseries_data.append({
                        "year": int(year),
                        "annual_resource_mcm": round(float(total_resource), 2),
                        "num_districts": int(gwr_df["district"].nunique())
                    })
        
        if not timeseries_data:
            return {
                "data_type": "gwr_timeseries",
                "filters": {"state": state, "district": district},
                "count": 0,
                "data": [],
                "message": "No GWR data in selected region"
            }
        
        # Sort by year
        timeseries_data.sort(key=lambda x: x["year"])
        
        # Calculate statistics
        resources = [d["annual_resource_mcm"] for d in timeseries_data]
        
        return {
            "data_type": "gwr_timeseries",
            "filters": {"state": state, "district": district},
            "statistics": {
                "mean_annual_resource_mcm": round(float(np.mean(resources)), 2),
                "total_years": len(timeseries_data),
                "year_range": {
                    "min": min(d["year"] for d in timeseries_data),
                    "max": max(d["year"] for d in timeseries_data)
                }
            },
            "count": len(timeseries_data),
            "data": timeseries_data
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"GWR Timeseries Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


def get_gwr_table_for_year(year: int) -> str:
    """Get table name for a specific year"""
    if year not in GWR_TABLE_MAP:
        raise ValueError(f"No GWR data for year {year}. Available: {AVAILABLE_GWR_YEARS}")
    return GWR_TABLE_MAP[year]


@app.get("/api/district/{state_name}/{district_name}")
def get_district(state_name: str, district_name: str):
    query = text("""
        SELECT 
            "District",
            "State",
            ST_AsGeoJSON(geometry) AS geojson,
            ST_Y(ST_Centroid(geometry)) AS center_lat,
            ST_X(ST_Centroid(geometry)) AS center_lng,
            ST_XMin(geometry), ST_YMin(geometry), 
            ST_XMax(geometry), ST_YMax(geometry)
        FROM public.district_state
        WHERE UPPER("District") = UPPER(:district_name)
        AND UPPER("State") = UPPER(:state_name)
        LIMIT 1;
    """)
    try:
        with engine.connect() as conn:
            row = conn.execute(query, {
                "district_name": district_name,
                "state_name": state_name
            }).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="District not found")
            return {
                "district_name": row[0],
                "State": row[1],
                "geometry": json.loads(row[2]),
                "center": [row[3], row[4]],
                "bounds": {
                    "min": [row[6], row[5]],
                    "max": [row[8], row[7]]
                }
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# AQUIFERS ENDPOINTS
# =============================================================================

@app.get("/api/aquifers")
def get_all_aquifers():
    query = text("""
        SELECT 
            aquifer, aquifers, zone_m, mbgl, avg_mbgl,
            m2_perday, m3_per_day, yeild__, per_cm, state,
            ST_AsGeoJSON(geometry) AS geojson,
            ST_Y(ST_Centroid(geometry)) AS center_lat,
            ST_X(ST_Centroid(geometry)) AS center_lng
        FROM public.aquifers
        ORDER BY aquifer;
    """)
    try:
        with engine.connect() as conn:
            result = conn.execute(query)
            aquifers = [
                {
                    "aquifer": row[0], "aquifers": row[1], "zone_m": row[2],
                    "mbgl": row[3], "avg_mbgl": row[4], "m2_perday": row[5],
                    "m3_per_day": row[6], "yeild": row[7], "per_cm": row[8],
                    "state": row[9], "geometry": json.loads(row[10]),
                    "center": [row[11], row[12]] if row[11] and row[12] else None
                }
                for row in result
            ]
            return aquifers
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/aquifers/state/{state_name}")
def get_aquifers_by_state(state_name: str):
    state_boundary_query = text("""
        SELECT ST_AsGeoJSON(ST_Union(ST_MakeValid(geometry))) as geojson
        FROM public.district_state
        WHERE UPPER("State") = UPPER(:state_name);
    """)
    
    try:
        with engine.connect() as conn:
            state_row = conn.execute(state_boundary_query, {"state_name": state_name}).fetchone()
            
            if not state_row or not state_row[0]:
                raise HTTPException(status_code=404, detail=f"State not found: {state_name}")
            
            aquifer_query = text("""
                SELECT 
                    a.aquifer, a.aquifers, a.zone_m, a.mbgl, a.avg_mbgl,
                    a.m2_perday, a.m3_per_day, a.yeild__, a.per_cm, a.state,
                    ST_AsGeoJSON(ST_MakeValid(a.geometry)) AS geojson,
                    ST_Y(ST_Centroid(a.geometry)) AS center_lat,
                    ST_X(ST_Centroid(a.geometry)) AS center_lng,
                    ST_Area(ST_Transform(ST_MakeValid(a.geometry), 3857)) AS area_sqm
                FROM public.aquifers a
                WHERE ST_Intersects(
                    ST_MakeValid(a.geometry),
                    ST_GeomFromGeoJSON(:state_geojson)
                )
                AND LOWER(a.state) LIKE LOWER(:state_pattern)
                ORDER BY a.aquifer;
            """)
            
            # Create very flexible pattern for state name matching
            # Handles: "Dadra, Nagar Haveli, Daman & Diu" vs "DADRA,NAGAR HAVELI,dAMAN & DIU"
            # Replace commas with space (not empty) to preserve word boundaries
            normalized = state_name.replace(',', ' ').replace(' and ', ' ').replace(' & ', ' ').replace('  ', ' ').strip()
            # Split into words and join with wildcards
            words = [w.strip() for w in normalized.split() if w.strip()]
            state_pattern = '%' + '%'.join(words) + '%'
            print(f"  🔍 Aquifer filter: state_name='{state_name}' → pattern='{state_pattern}'")
            
            result = conn.execute(aquifer_query, {
                "state_geojson": state_row[0], 
                "state_name": state_name,
                "state_pattern": state_pattern
            })
            
            aquifers = []
            for row in result:
                try:
                    geom_json = json.loads(row[10])
                    aquifers.append({
                        "aquifer": row[0], "aquifers": row[1], "zone_m": row[2],
                        "mbgl": row[3], "avg_mbgl": row[4], "m2_perday": row[5],
                        "m3_per_day": row[6], "yeild": row[7], "per_cm": row[8],
                        "state": row[9], "geometry": geom_json,
                        "center": [row[11], row[12]] if row[11] and row[12] else None,
                        "area_sqm": float(row[13]) if row[13] else 0
                    })
                except Exception as e:
                    print(f"Error parsing aquifer geometry: {e}")
                    continue
            
            # Fallback: If no aquifers found with state name filter, try spatial-only
            if not aquifers:
                print(f"No aquifers found for {state_name} with name filter, trying spatial-only...")
                fallback_query = text("""
                    SELECT 
                        a.aquifer, a.aquifers, a.zone_m, a.mbgl, a.avg_mbgl,
                        a.m2_perday, a.m3_per_day, a.yeild__, a.per_cm, a.state,
                        ST_AsGeoJSON(ST_MakeValid(a.geometry)) AS geojson,
                        ST_Y(ST_Centroid(a.geometry)) AS center_lat,
                        ST_X(ST_Centroid(a.geometry)) AS center_lng,
                        ST_Area(ST_Transform(ST_MakeValid(a.geometry), 3857)) AS area_sqm
                    FROM public.aquifers a
                    WHERE ST_Intersects(
                        ST_MakeValid(a.geometry),
                        ST_GeomFromGeoJSON(:state_geojson)
                    )
                    ORDER BY a.aquifer
                    LIMIT 50;
                """)
                fallback_result = conn.execute(fallback_query, {"state_geojson": state_row[0]})
                for row in fallback_result:
                    try:
                        geom_json = json.loads(row[10])
                        aquifers.append({
                            "aquifer": row[0], "aquifers": row[1], "zone_m": row[2],
                            "mbgl": row[3], "avg_mbgl": row[4], "m2_perday": row[5],
                            "m3_per_day": row[6], "yeild": row[7], "per_cm": row[8],
                            "state": row[9], "geometry": geom_json,
                            "center": [row[11], row[12]] if row[11] and row[12] else None,
                            "area_sqm": float(row[13]) if row[13] else 0
                        })
                    except Exception as e:
                        print(f"Error parsing fallback aquifer geometry: {e}")
                        continue
            
            if not aquifers:
                raise HTTPException(status_code=404, detail=f"No aquifers found for state: {state_name}")
            
            return aquifers
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting state aquifers: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/aquifers/district/{state_name}/{district_name}")
def get_aquifers_by_district(state_name: str, district_name: str):
    district_query = text("""
        SELECT ST_AsGeoJSON(ST_MakeValid(geometry)) as geojson
        FROM public.district_state
        WHERE UPPER("District") = UPPER(:district_name)
        AND UPPER("State") = UPPER(:state_name)
        LIMIT 1;
    """)
    
    try:
        with engine.connect() as conn:
            district_row = conn.execute(district_query, {
                "district_name": district_name,
                "state_name": state_name
            }).fetchone()
            
            if not district_row:
                raise HTTPException(status_code=404, detail=f"District not found: {district_name}, {state_name}")
            
            aquifer_query = text("""
                SELECT 
                    a.aquifer, a.aquifers, a.zone_m, a.mbgl, a.avg_mbgl,
                    a.m2_perday, a.m3_per_day, a.yeild__, a.per_cm, a.state,
                    ST_AsGeoJSON(
                        ST_MakeValid(
                            ST_Intersection(
                                ST_MakeValid(a.geometry),
                                ST_GeomFromGeoJSON(:district_geojson)
                            )
                        )
                    ) AS geojson,
                    ST_Y(ST_Centroid(
                        ST_Intersection(
                            ST_MakeValid(a.geometry),
                            ST_GeomFromGeoJSON(:district_geojson)
                        )
                    )) AS center_lat,
                    ST_X(ST_Centroid(
                        ST_Intersection(
                            ST_MakeValid(a.geometry),
                            ST_GeomFromGeoJSON(:district_geojson)
                        )
                    )) AS center_lng,
                    ST_Area(
                        ST_Transform(
                            ST_Intersection(
                                ST_MakeValid(a.geometry),
                                ST_GeomFromGeoJSON(:district_geojson)
                            ),
                            3857
                        )
                    ) AS area_sqm
                FROM public.aquifers a
                WHERE ST_Intersects(
                    ST_MakeValid(a.geometry),
                    ST_GeomFromGeoJSON(:district_geojson)
                )
                AND LOWER(a.state) LIKE LOWER(:state_pattern)
                ORDER BY a.aquifer;
            """)
            
            # Create very flexible pattern for state name matching
            normalized = state_name.replace(',', ' ').replace(' and ', ' ').replace(' & ', ' ').replace('  ', ' ').strip()
            words = [w.strip() for w in normalized.split() if w.strip()]
            state_pattern = '%' + '%'.join(words) + '%'
            
            result = conn.execute(aquifer_query, {
                "district_geojson": district_row[0], 
                "state_name": state_name,
                "state_pattern": state_pattern
            })
            
            aquifers = []
            for row in result:
                try:
                    geom_json = json.loads(row[10])
                    aquifers.append({
                        "aquifer": row[0], "aquifers": row[1], "zone_m": row[2],
                        "mbgl": row[3], "avg_mbgl": row[4], "m2_perday": row[5],
                        "m3_per_day": row[6], "yeild": row[7], "per_cm": row[8],
                        "state": row[9], "geometry": geom_json,
                        "center": [row[11], row[12]] if row[11] and row[12] else None,
                        "area_sqm": float(row[13]) if row[13] else 0
                    })
                except Exception as e:
                    print(f"Error parsing aquifer geometry: {e}")
                    continue
            
            # Fallback: If no aquifers found with state name filter, try spatial-only
            if not aquifers:
                print(f"No aquifers found for {district_name} with name filter, trying spatial-only...")
                fallback_query = text("""
                    SELECT 
                        a.aquifer, a.aquifers, a.zone_m, a.mbgl, a.avg_mbgl,
                        a.m2_perday, a.m3_per_day, a.yeild__, a.per_cm, a.state,
                        ST_AsGeoJSON(ST_MakeValid(a.geometry)) AS geojson,
                        ST_Y(ST_Centroid(a.geometry)) AS center_lat,
                        ST_X(ST_Centroid(a.geometry)) AS center_lng,
                        ST_Area(ST_Transform(ST_MakeValid(a.geometry), 3857)) AS area_sqm
                    FROM public.aquifers a
                    WHERE ST_Intersects(
                        ST_MakeValid(a.geometry),
                        ST_GeomFromGeoJSON(:district_geojson)
                    )
                    ORDER BY a.aquifer
                    LIMIT 50;
                """)
                fallback_result = conn.execute(fallback_query, {"district_geojson": district_row[0]})
                for row in fallback_result:
                    try:
                        geom_json = json.loads(row[10])
                        aquifers.append({
                            "aquifer": row[0], "aquifers": row[1], "zone_m": row[2],
                            "mbgl": row[3], "avg_mbgl": row[4], "m2_perday": row[5],
                            "m3_per_day": row[6], "yeild": row[7], "per_cm": row[8],
                            "state": row[9], "geometry": geom_json,
                            "center": [row[11], row[12]] if row[11] and row[12] else None,
                            "area_sqm": float(row[13]) if row[13] else 0
                        })
                    except Exception as e:
                        print(f"Error parsing fallback aquifer geometry: {e}")
                        continue
            
            if not aquifers:
                raise HTTPException(status_code=404, detail=f"No aquifers found for district: {district_name}")
            
            return aquifers
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting district aquifers: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# GROUNDWATER WELLS ENDPOINTS
# =============================================================================

@app.get("/api/wells")
def get_groundwater_wells(
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    year: Optional[int] = Query(None, ge=1990, le=2025),
    month: Optional[int] = Query(None, ge=1, le=12),
    season: Optional[str] = Query(None, regex="^(PREMONSOON|MONSOON|POSTMONS_1)$"),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    max_points: int = Query(5000, ge=100, le=50000)
):
    where_conditions = []
    params = {}
    
    if year:
        with engine.connect() as conn:
            year_check = text("""
                SELECT EXISTS(
                    SELECT 1 FROM groundwater_level 
                    WHERE EXTRACT(YEAR FROM "Date") = :year 
                    LIMIT 1
                )
            """)
            has_data = conn.execute(year_check, {"year": year}).fetchone()[0]
            
            if not has_data:
                return {
                    "data_type": "groundwater_wells",
                    "filters": {
                        "state": state,
                        "district": district,
                        "year": year,
                        "month": month,
                        "season": season,
                        "start_date": start_date,
                        "end_date": end_date
                    },
                    "spatial_filter": "disabled",
                    "count": 0,
                    "wells": [],
                    "message": f"No well data available for year {year}"
                }
    
    boundary_geojson = None
    if district and state:
        boundary_geojson = get_boundary_geojson(state, district)
        if not boundary_geojson:
            raise HTTPException(status_code=404, detail=f"District not found: {district}, {state}")
    elif state:
        boundary_geojson = get_boundary_geojson(state, None)
        if not boundary_geojson:
            raise HTTPException(status_code=404, detail=f"State not found: {state}")
    
    if boundary_geojson:
        where_conditions.append("""
            ST_Intersects(
                ST_GeomFromGeoJSON(:boundary_geojson),
                ST_SetSRID(ST_MakePoint(
                    CASE 
                        WHEN "LON" IS NOT NULL AND "LON" != 0 THEN "LON"
                        ELSE "LONGITUD_1"
                    END,
                    CASE 
                        WHEN "LAT" IS NOT NULL AND "LAT" != 0 THEN "LAT"
                        ELSE "LATITUDE_1"
                    END
                ), 4326)
            )
        """)
        params["boundary_geojson"] = boundary_geojson
    
    if year and month:
        where_conditions.append('EXTRACT(YEAR FROM "Date") = :year')
        where_conditions.append('EXTRACT(MONTH FROM "Date") = :month')
        params["year"] = year
        params["month"] = month
    elif year:
        where_conditions.append('EXTRACT(YEAR FROM "Date") = :year')
        params["year"] = year
    
    if season:
        where_conditions.append('UPPER("Season") = UPPER(:season)')
        params["season"] = season
    
    if start_date:
        where_conditions.append('"Date" >= :start_date')
        params["start_date"] = start_date
    
    if end_date:
        where_conditions.append('"Date" <= :end_date')
        params["end_date"] = end_date
    
    where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
    
    query = text(f"""
        SELECT 
            "Date",
            "GWL",
            CASE 
                WHEN "LAT" IS NOT NULL AND "LAT" != 0 THEN "LAT"
                ELSE "LATITUDE_1"
            END as latitude,
            CASE 
                WHEN "LON" IS NOT NULL AND "LON" != 0 THEN "LON"
                ELSE "LONGITUD_1"
            END as longitude,
            "STATE" as state,
            "DISTRICT" as district,
            "Season",
            "SITE_NAME",
            "SITE_TYPE",
            "AQUIFER"
        FROM public.groundwater_level
        WHERE {where_clause}
        AND "GWL" IS NOT NULL
        AND "Date" IS NOT NULL
        AND (
            ("LAT" IS NOT NULL AND "LAT" != 0 AND "LON" IS NOT NULL AND "LON" != 0)
            OR
            ("LATITUDE_1" IS NOT NULL AND "LATITUDE_1" != 0 AND "LONGITUD_1" IS NOT NULL AND "LONGITUD_1" != 0)
        )
        ORDER BY "Date" DESC
        LIMIT :max_points;
    """)
    
    params["max_points"] = max_points
    
    try:
        with engine.connect() as conn:
            result = conn.execute(query, params)
            
            wells = []
            for row in result:
                date_val = row[0]
                gwl_val = float(row[1]) if row[1] is not None else None
                lat = float(row[2]) if row[2] is not None else None
                lon = float(row[3]) if row[3] is not None else None
                
                if gwl_val is None or lat is None or lon is None:
                    continue
                
                site_id = f"{round(lat, 5)}_{round(lon, 5)}"
                category = categorize_water_level(gwl_val)
                
                wells.append({
                    "site_id": site_id,
                    "date": date_val.isoformat() if hasattr(date_val, 'isoformat') else str(date_val),
                    "year": date_val.year if hasattr(date_val, 'year') else None,
                    "month": date_val.month if hasattr(date_val, 'month') else None,
                    "gwl": round(gwl_val, 2),
                    "gwl_category": category,
                    "latitude": round(lat, 6),
                    "longitude": round(lon, 6),
                    "state": row[4],
                    "district": row[5],
                    "season": row[6],
                    "site_name": row[7],
                    "site_type": row[8],
                    "aquifer": row[9],
                    "hovertext": f"Date: {date_val.strftime('%Y-%m-%d') if hasattr(date_val, 'strftime') else date_val}<br>GWL: {gwl_val:.2f} m<br>Category: {category}"
                })
            
            return {
                "data_type": "groundwater_wells",
                "filters": {
                    "state": state,
                    "district": district,
                    "year": year,
                    "month": month,
                    "season": season,
                    "start_date": start_date,
                    "end_date": end_date
                },
                "spatial_filter": "enabled" if boundary_geojson else "disabled",
                "count": len(wells),
                "wells": wells
            }
    
    except Exception as e:
        print(f"Error getting wells: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/wells/timeseries")
def get_wells_timeseries(
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    aggregation: str = Query("monthly", regex="^(monthly|yearly)$"),
    view: str = Query("raw", regex="^(raw|seasonal|deseasonalized)$")
):
    print(f"\n🔄 Timeseries query: state={state}, district={district}, view={view}")
    start_time = datetime.now()
    
    # ===== OPTIMIZATION: Check if we can compute from cached raw data =====
    raw_cache_key = f"timeseries_raw_data_{state}_{district}"
    view_cache_key = f"timeseries_{state}_{district}_{view}"
    
    # Check if final view is cached
    cached_view_result = TIMESERIES_CACHE.get(view_cache_key)
    if cached_view_result is not None:
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"  ✅ Returned {view} view from cache in {elapsed:.2f}s")
        cached_view_result["cache_hit"] = True
        cached_view_result["response_time_seconds"] = round(elapsed, 2)
        return cached_view_result
    
    # Check if raw merged data is cached (for computing transformations)
    cached_raw_data = TIMESERIES_CACHE.get(raw_cache_key)
    
    if cached_raw_data is not None:
        print(f"  ✓ Using cached raw data to compute {view} view")
        
        # ✅ FIX: Deserialize the pickled DataFrame
        try:
            combined = pickle.loads(base64.b64decode(cached_raw_data.encode('utf-8')))
            print(f"  ✓ Deserialized cached DataFrame: {len(combined)} rows")
        except Exception as e:
            print(f"  ⚠️ Failed to deserialize cache: {e}")
            # Fall back to fresh fetch
            fetch_needed = True
        else:
            # Successfully loaded from cache
            fetch_needed = False
    else:
        # Need to fetch fresh data
        print(f"  📊 Fetching fresh data...")
        fetch_needed = True
    
    # ===== FETCH DATA (only if not cached) =====
    if fetch_needed:
        boundary_geojson = get_boundary_geojson(state, district) if (state or district) else None
        
        if view in ["seasonal", "deseasonalized"]:
            aggregation = "monthly"
        
        # Parallel queries
        print("  📊 Fetching data (parallel)...")
        wells_future = EXECUTOR.submit(query_wells_monthly, boundary_geojson)
        grace_future = EXECUTOR.submit(query_grace_monthly, boundary_geojson)
        rainfall_future = EXECUTOR.submit(query_rainfall_monthly, boundary_geojson)
        
        wells_df = wells_future.result()
        grace_df = grace_future.result()
        rainfall_df = rainfall_future.result()
        
        # Merge
        print("  🔗 Merging timeseries...")
        if not wells_df.empty:
            combined = wells_df.copy()
        else:
            combined = pd.DataFrame(columns=["period"])
        
        if not grace_df.empty:
            combined = combined.merge(grace_df, on="period", how="outer")
        else:
            combined["avg_tws"] = np.nan
        
        if not rainfall_df.empty:
            combined = combined.merge(rainfall_df, on="period", how="outer")
        else:
            combined["avg_rainfall"] = np.nan
        
        combined = combined.sort_values("period").reset_index(drop=True)
        
        if combined.empty:
            return {
                "view": view,
                "aggregation": aggregation,
                "filters": {"state": state, "district": district},
                "count": 0,
                "timeseries": [],
                "statistics": None,
                "error": "No data available for this region"
            }
        
        # Interpolate
        print("  🔧 Interpolating missing values...")
        combined = combined.set_index("period")
        combined = combined.interpolate(method="time", limit_direction="both")
        combined = combined.reset_index()
        
        print(f"  ✓ Merged timeseries: {len(combined)} months")

# Serialize DataFrame to preserve dtypes
        serialized_df = base64.b64encode(pickle.dumps(combined)).decode('utf-8')
        TIMESERIES_CACHE.set(raw_cache_key, serialized_df)
        print(f"  💾 Cached raw data for future {view} computations")
    
    # ===== NOW PROCESS THE VIEW (using either cached or fresh data) =====
    
    if view in ["seasonal", "deseasonalized"]:
            # Only proceed if we have enough data
            if len(combined) < 24:
                return {
                    "view": view,
                    "aggregation": aggregation,
                    "filters": {"state": state, "district": district},
                    "count": 0,
                    "timeseries": [],
                    "statistics": None,
                    "error": f"Need at least 24 months of data for {view} analysis (found {len(combined)})"
                }

            print(f"  📈 Performing {view} decomposition (Vectorized)...")
            df = combined.copy()
            
            # Ensure period is datetime index for decomposition
            df = df.set_index("period").sort_index()
            
            # Initialize statistics dict
            statistics = {}

            # 1. Process GWL
            if "avg_gwl" in df.columns and not df["avg_gwl"].isna().all():
                try:
                    # Interpolate just for decomposition stability
                    series = df["avg_gwl"].interpolate(limit_direction='both')
                    decomp = seasonal_decompose(series, model='additive', period=12)
                    
                    if view == "seasonal":
                        df["gwl_seasonal"] = decomp.seasonal
                    else: # deseasonalized
                        # Trend + Residual = Deseasonalized (Original - Seasonal)
                        df["gwl_deseasonalized"] = series - decomp.seasonal
                        
                        # Calculate Trend Statistics (on valid deseasonalized data)
                        valid_y = df["gwl_deseasonalized"].dropna()
                        if len(valid_y) > 1:
                            x = np.arange(len(valid_y))
                            slope, _, r_value, _, _ = linregress(x, valid_y.values)
                            statistics["gwl_trend"] = {
                                "slope_per_year": round(float(slope * 12), 4),
                                "r_squared": round(float(r_value ** 2), 3),
                                "direction": "declining" if slope > 0 else "recovering"
                            }
                except Exception as e:
                    print(f"GWL decomp error: {e}")

            # 2. Process GRACE
            if "avg_tws" in df.columns and not df["avg_tws"].isna().all():
                try:
                    series = df["avg_tws"].interpolate(limit_direction='both')
                    decomp = seasonal_decompose(series, model='additive', period=12)
                    
                    if view == "seasonal":
                        df["grace_seasonal"] = decomp.seasonal
                    else:
                        df["grace_deseasonalized"] = series - decomp.seasonal
                        
                        # Statistics
                        valid_y = df["grace_deseasonalized"].dropna()
                        if len(valid_y) > 1:
                            x = np.arange(len(valid_y))
                            slope, _, r_value, _, _ = linregress(x, valid_y.values)
                            statistics["grace_trend"] = {
                                "slope_per_year": round(float(slope * 12), 4),
                                "r_squared": round(float(r_value**2), 3)
                            }
                except Exception as e:
                    print(f"GRACE decomp error: {e}")

            # 3. Process Rainfall
            if "avg_rainfall" in df.columns and not df["avg_rainfall"].isna().all():
                try:
                    series = df["avg_rainfall"].fillna(0) # Rainfall usually 0 if nan
                    decomp = seasonal_decompose(series, model='additive', period=12)
                    
                    if view == "seasonal":
                        df["rainfall_seasonal"] = decomp.seasonal
                    else:
                        df["rainfall_deseasonalized"] = series - decomp.seasonal
                except Exception as e:
                    print(f"Rainfall decomp error: {e}")

            # Prepare final output
            df = df.reset_index()
            # Convert date to string for JSON serialization
            df["date"] = df["period"].apply(lambda x: x.isoformat())
            
            # Replace NaN with None for JSON compliance
            df = df.replace({np.nan: None})
            
            # Convert to dictionary records
            output_timeseries = df.to_dict("records")
            
            # Add monthly rainfall totals for Raw view (matches Dash reference)
            if view == "raw":
                for record in output_timeseries:
                    if record.get("avg_rainfall") is not None:
                        period = pd.to_datetime(record["date"])
                        days_in_month = period.days_in_month
                        record["monthly_rainfall_total_mm"] = round(
                            record["avg_rainfall"] * days_in_month, 2
                        )

            result = {
                "view": view,
                "aggregation": "monthly",
                "filters": {"state": state, "district": district},
                "count": len(output_timeseries),
                "chart_config": {
                    "gwl_chart_type": "line",
                    "grace_chart_type": "line",
                    "rainfall_chart_type": "line",
                    "gwl_y_axis_reversed": True if view != "seasonal" else False # Don't reverse seasonal component usually
                },
                "timeseries": output_timeseries,
                "statistics": statistics,
                "interpretations": {
                    "gwl_trend": {
                        "what_is_slope": "Rate of groundwater level change over time",
                        "slope_meaning": {
                            "positive_slope": "Declining groundwater (water table getting DEEPER - concerning)",
                            "negative_slope": "Recovering groundwater (water table getting SHALLOWER - good sign)"
                        },
                        "r_squared_meaning": "How well the trendline fits the data (0-1 scale, higher = better fit)",
                        "r_squared_ranges": {
                            "0.0_to_0.3": "Weak trend - high variability, trend not reliable",
                            "0.3_to_0.7": "Moderate trend - data follows trend with some variation",
                            "0.7_to_1.0": "Strong trend - data closely follows the trend"
                        },
                        "current_values": statistics.get("gwl_trend", {})
                    } if "gwl_trend" in statistics else None,
                    "grace_trend": {
                        "what_is_slope": "Rate of total water storage change (groundwater + soil moisture + surface water)",
                        "slope_meaning": {
                            "positive_slope": "Water storage increasing (more water in the ground)",
                            "negative_slope": "Water storage decreasing (water being depleted)"
                        },
                        "r_squared_meaning": "How well the trendline fits the data",
                        "current_values": statistics.get("grace_trend", {})
                    } if "grace_trend" in statistics else None,
                    "actionable_insights": []
                },
                "cache_hit": False
            }
            
            # Add actionable insights based on trends
            insights = []
            if "gwl_trend" in statistics:
                gwl = statistics["gwl_trend"]
                slope = gwl["slope_per_year"]
                r2 = gwl.get("r_squared", 0)
                
                # Use R² threshold instead of p-value (R² > 0.3 indicates meaningful trend)
                if r2 > 0.3:  # Meaningful trend
                    if slope > 0:  # Declining (bad)
                        severity = "CRITICAL" if slope > 2 else ("HIGH" if slope > 1 else "MODERATE")
                        insights.append({
                            "severity": severity,
                            "metric": "Groundwater Level",
                            "finding": f"Water table declining at {abs(slope):.2f}m per year",
                            "meaning": "Groundwater is being depleted faster than it's being recharged",
                            "recommendation": "Consider: Rainwater harvesting, reduced pumping, managed aquifer recharge",
                            "confidence": "High" if r2 > 0.7 else "Moderate"
                        })
                    else:  # Recovering (good)
                        insights.append({
                            "severity": "POSITIVE",
                            "metric": "Groundwater Level",
                            "finding": f"Water table recovering at {abs(slope):.2f}m per year",
                            "meaning": "Groundwater recharge exceeds extraction",
                            "recommendation": "Maintain current water management practices",
                            "confidence": "High" if r2 > 0.7 else "Moderate"
                        })
                else:
                    insights.append({
                        "severity": "INFO",
                        "metric": "Groundwater Level",
                        "finding": "No clear long-term trend detected",
                        "meaning": "Water levels relatively stable or highly variable (Low R² = weak trend fit)"
                    })
            
            if "grace_trend" in statistics:
                grace = statistics["grace_trend"]
                slope = grace["slope_per_year"]
                r2 = grace.get("r_squared", 0)
                
                if abs(slope) > 0.5 and r2 > 0.5:  # Meaningful trend
                    if slope < 0:  # Decreasing storage (bad)
                        insights.append({
                            "severity": "WARNING",
                            "metric": "Total Water Storage (GRACE)",
                            "finding": f"Water storage declining at {abs(slope):.2f} cm per year",
                            "meaning": "Combined groundwater, soil moisture, and surface water decreasing",
                            "recommendation": "Comprehensive water conservation needed across all sources"
                        })
                    else:  # Increasing storage (good)
                        insights.append({
                            "severity": "POSITIVE",
                            "metric": "Total Water Storage (GRACE)",
                            "finding": f"Water storage increasing at {slope:.2f} cm per year",
                            "meaning": "Overall water availability improving"
                        })
            
            result["interpretations"]["actionable_insights"] = insights
            
            # Cache this view
            TIMESERIES_CACHE.set(view_cache_key, result)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            result["response_time_seconds"] = round(elapsed, 2)
            print(f"  ✅ Complete in {elapsed:.2f}s")
            
            return result
    
    else:  # view == "raw"
        print(f"  📊 Returning raw data (no decomposition needed)...")
        df = combined.copy()
        
        # Prepare final output
        df["date"] = df["period"].apply(lambda x: x.isoformat())
        
        # Replace NaN with None for JSON compliance
        df = df.replace({np.nan: None})
        
        # Convert to dictionary records
        output_timeseries = df.to_dict("records")
        
        # Add monthly rainfall totals for Raw view (matches Dash reference)
        for record in output_timeseries:
            if record.get("avg_rainfall") is not None:
                period = pd.to_datetime(record["date"])
                days_in_month = period.days_in_month
                record["monthly_rainfall_total_mm"] = round(
                    record["avg_rainfall"] * days_in_month, 2
                )
        
        result = {
            "view": view,
            "aggregation": "monthly",
            "filters": {"state": state, "district": district},
            "count": len(output_timeseries),
            "chart_config": {
                "gwl_chart_type": "line",
                "grace_chart_type": "line",
                "rainfall_chart_type": "bar",  # Bar for raw view
                "gwl_y_axis_reversed": True
            },
            "timeseries": output_timeseries,
            "statistics": None,  # No statistics for raw view
            "interpretations": None,  # No interpretations for raw view
            "cache_hit": False
        }
        
        # Cache this view
        TIMESERIES_CACHE.set(view_cache_key, result)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        result["response_time_seconds"] = round(elapsed, 2)
        print(f"  ✅ Raw view complete in {elapsed:.2f}s")
        
        return result
        
@app.get("/api/wells/summary")
def get_wells_summary(
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None)
):
    
    boundary_geojson = get_boundary_geojson(state, district) if (state or district) else None
    
    where_clause = ""
    params = {}
    
    if boundary_geojson:
        where_clause = """
            WHERE ST_Intersects(
                ST_GeomFromGeoJSON(:boundary_geojson),
                ST_SetSRID(ST_MakePoint(
                    COALESCE(NULLIF("LON", 0), "LONGITUD_1"),
                    COALESCE(NULLIF("LAT", 0), "LATITUDE_1")
                ), 4326)
            )
        """
        params["boundary_geojson"] = boundary_geojson
    
    query = text(f"""
        SELECT 
            DATE_TRUNC('month', "Date") as month,
            AVG("GWL") as avg_gwl
        FROM groundwater_level
        {where_clause}
        AND "GWL" IS NOT NULL
        GROUP BY DATE_TRUNC('month', "Date")
        ORDER BY month;
    """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(query, params)
            
            data = [(row[0], float(row[1])) for row in result]
            
            if len(data) < 2:
                return {
                    "filters": {"state": state, "district": district},
                    "error": "Insufficient data for analysis"
                }
            
            dates = [d[0] for d in data]
            gwl_values = [d[1] for d in data]
            
            x = np.array([(d - dates[0]).days for d in dates])
            y = np.array(gwl_values)
            
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            
            slope_per_year = slope * 365.25
            r_squared = r_value ** 2
            
            return {
                "filters": {"state": state, "district": district},
                "statistics": {
                    "mean_gwl": round(float(np.mean(y)), 2),
                    "min_gwl": round(float(np.min(y)), 2),
                    "max_gwl": round(float(np.max(y)), 2),
                    "std_gwl": round(float(np.std(y)), 2)
                },
                "trend": {
                    "slope_m_per_year": round(slope_per_year, 4),
                    "r_squared": round(r_squared, 3),
                    "p_value": round(p_value, 4),
                    "trend_direction": "declining" if slope_per_year > 0 else "recovering",
                    "significance": "significant" if p_value < 0.05 else "not_significant"
                },
                "temporal_coverage": {
                    "start_date": dates[0].isoformat(),
                    "end_date": dates[-1].isoformat(),
                    "months_of_data": len(data),
                    "span_years": round((dates[-1] - dates[0]).days / 365.25, 1)
                }
            }
    
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/wells/storage-vs-gwr")
def get_storage_vs_gwr(
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    start_year: Optional[int] = Query(None),
    end_year: Optional[int] = Query(None)
):
    """
    Enhanced storage analysis that includes GWR comparison
    Matches the Dash code's storage plot with GWR overlay
    """
    try:
        boundary_geojson = get_boundary_geojson(state, district) if (state or district) else None
        
        # === 1. GET AQUIFER PROPERTIES FOR SPECIFIC YIELD ===
        where_aquifer = ""
        params = {}
        
        if boundary_geojson:
            where_aquifer = "WHERE ST_Intersects(ST_MakeValid(geometry), ST_GeomFromGeoJSON(:boundary_geojson))"
            params["boundary_geojson"] = boundary_geojson
        
        aquifer_query = text(f"""
            SELECT 
                ST_Area(ST_Transform(ST_MakeValid(geometry), 32643)) as area_m2,
                yeild__
            FROM aquifers
            {where_aquifer}
        """)
        
        total_area_m2 = 0.0
        weighted_sy_sum = 0.0
        
        with engine.connect() as conn:
            result = conn.execute(aquifer_query, params if boundary_geojson else {})
            
            for row in result:
                area = float(row[0]) if row[0] else 0
                yield_str = str(row[1]).lower() if row[1] else ""
                
                # Parse specific yield from yield string
                sy = 0.05  # Default
                try:
                    import re
                    numbers = re.findall(r'\d+\.?\d*', yield_str)
                    if numbers:
                        vals = [float(n) for n in numbers]
                        sy = sum(vals) / len(vals) / 100.0  # Convert percentage to decimal
                except:
                    sy = 0.05
                
                total_area_m2 += area
                weighted_sy_sum += (area * sy)
        
        if total_area_m2 == 0:
            return {
                "data_type": "storage_vs_gwr",
                "filters": {"state": state, "district": district},
                "status": "error",
                "message": "No aquifer data available for this region",
                "statistics": {},
                "count": 0,
                "data": []
            }
        
        specific_yield = weighted_sy_sum / total_area_m2
        
        # === 2. CALCULATE STORAGE FROM WELLS ===
        where_wells = ""
        if boundary_geojson:
            where_wells = """
                WHERE ST_Intersects(
                    ST_GeomFromGeoJSON(:boundary_geojson),
                    ST_SetSRID(ST_MakePoint(
                        COALESCE(NULLIF("LON", 0), "LONGITUD_1"),
                        COALESCE(NULLIF("LAT", 0), "LATITUDE_1")
                    ), 4326)
                )
            """
        
        year_filter_start = 'AND EXTRACT(YEAR FROM "Date") >= :start_year' if start_year else ""
        year_filter_end = 'AND EXTRACT(YEAR FROM "Date") <= :end_year' if end_year else ""
        
        storage_query = text(f"""
            SELECT 
                EXTRACT(YEAR FROM "Date") as year,
                AVG(CASE WHEN EXTRACT(MONTH FROM "Date") IN (4,5,6) THEN "GWL" END) as pre_gwl,
                AVG(CASE WHEN EXTRACT(MONTH FROM "Date") IN (10,11,12) THEN "GWL" END) as post_gwl
            FROM groundwater_level
            {where_wells}
            AND "GWL" IS NOT NULL
            {year_filter_start}
            {year_filter_end}
            GROUP BY EXTRACT(YEAR FROM "Date")
            HAVING AVG(CASE WHEN EXTRACT(MONTH FROM "Date") IN (4,5,6) THEN "GWL" END) IS NOT NULL
            AND AVG(CASE WHEN EXTRACT(MONTH FROM "Date") IN (10,11,12) THEN "GWL" END) IS NOT NULL
            ORDER BY year;
        """)
        
        if start_year:
            params["start_year"] = start_year
        if end_year:
            params["end_year"] = end_year
        
        storage_years = []
        
        with engine.connect() as conn:
            result = conn.execute(storage_query, params)
            
            for row in result:
                year = int(row[0])
                pre_gwl = float(row[1])
                post_gwl = float(row[2])
                
                # Storage calculation (positive = depletion, negative = recharge)
                fluctuation_m = pre_gwl - post_gwl
                storage_change_mcm = (fluctuation_m * total_area_m2 * specific_yield) / 1_000_000
                
                storage_years.append({
                    "year": year,
                    "pre_monsoon_gwl": round(pre_gwl, 2),
                    "post_monsoon_gwl": round(post_gwl, 2),
                    "fluctuation_m": round(fluctuation_m, 2),
                    "storage_change_mcm": round(storage_change_mcm, 2)
                })
        
        # === 3. GET GWR DATA ===
        available_gwr_years = get_available_gwr_years()
        gwr_years = []
        
        for year in available_gwr_years:
            # Apply year filters if provided
            if start_year and year < start_year:
                continue
            if end_year and year > end_year:
                continue
            
            gwr_df = get_gwr_for_year(year, boundary_geojson)
            
            if not gwr_df.empty:
                # ✅ Apply conversion factor here
                total_resource = gwr_df["annual_resource_mcm"].sum() * RESOURCE_TO_MCM_FACTOR
                
                if pd.notna(total_resource) and total_resource > 0:
                    gwr_years.append({
                        "year": int(year),
                        "annual_resource_mcm": round(float(total_resource), 2)
                    })
        
        # === 4. MERGE STORAGE AND GWR DATA ===
        all_years = sorted(set(
            [s["year"] for s in storage_years] + 
            [g["year"] for g in gwr_years]
        ))
        
        merged_data = []
        
        for year in all_years:
            year_data = {"year": year}
            
            # Add storage data if available
            storage_match = next((s for s in storage_years if s["year"] == year), None)
            if storage_match:
                year_data["storage_change_mcm"] = storage_match["storage_change_mcm"]
                year_data["pre_monsoon_gwl"] = storage_match["pre_monsoon_gwl"]
                year_data["post_monsoon_gwl"] = storage_match["post_monsoon_gwl"]
            else:
                year_data["storage_change_mcm"] = None
                year_data["pre_monsoon_gwl"] = None
                year_data["post_monsoon_gwl"] = None
            
            # Add GWR data if available
            gwr_match = next((g for g in gwr_years if g["year"] == year), None)
            if gwr_match:
                year_data["gwr_resource_mcm"] = gwr_match["annual_resource_mcm"]
            else:
                year_data["gwr_resource_mcm"] = None
            
            merged_data.append(year_data)
        
        # === 5. CALCULATE STATISTICS ===
        storage_values = [d["storage_change_mcm"] for d in merged_data if d["storage_change_mcm"] is not None]
        gwr_values = [d["gwr_resource_mcm"] for d in merged_data if d["gwr_resource_mcm"] is not None]
        
        statistics = {
            "aquifer_properties": {
                "total_area_km2": round(total_area_m2 / 1_000_000, 2),
                "area_weighted_specific_yield": round(specific_yield, 4)
            }
        }
        
        if storage_values:
            statistics["storage"] = {
                "years_with_data": len(storage_values),
                "avg_annual_storage_change_mcm": round(np.mean(storage_values), 2),
                "total_fluctuation_mcm": round(np.sum(storage_values), 2)
            }
        
        if gwr_values:
            statistics["gwr"] = {
                "years_with_data": len(gwr_values),
                "avg_annual_resource_mcm": round(np.mean(gwr_values), 2),
                "total_resource_mcm": round(np.sum(gwr_values), 2)
            }
        
        return {
            "data_type": "storage_vs_gwr",
            "filters": {
                "state": state,
                "district": district,
                "start_year": start_year,
                "end_year": end_year
            },
            "statistics": statistics,
            "count": len(merged_data),
            "data": merged_data,
            "note": "Storage represents seasonal fluctuation (MCM). GWR represents annual replenishable resource (MCM)."
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Storage vs GWR Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/wells/years")
def get_wells_years():
    query = text("""
        SELECT 
            MIN(EXTRACT(YEAR FROM "Date")) as min_year,
            MAX(EXTRACT(YEAR FROM "Date")) as max_year
        FROM groundwater_level
        WHERE "Date" IS NOT NULL;
    """)
    try:
        with engine.connect() as conn:
            result = conn.execute(query).fetchone()
            if result and result[0] and result[1]:
                min_year = int(result[0])
                max_year = int(result[1])
                return {
                    "min_year": min_year,
                    "max_year": max_year,
                    "span": max_year - min_year + 1,
                    "source": "database",
                    "description": f"Wells data available from {min_year} to {max_year}"
                }
            else:
                return {
                    "min_year": 1994,
                    "max_year": 2024,
                    "span": 31,
                    "source": "fallback",
                    "description": "No data found, using default range"
                }
    except Exception as e:
        print(f"Error getting year range: {str(e)}")
        return {
            "min_year": 1994,
            "max_year": 2024,
            "span": 31,
            "source": "fallback_error",
            "error": str(e)
        }

# =============================================================================
# GRACE ENDPOINTS
# =============================================================================

@app.get("/api/grace/available")
def get_grace_available():
    if not GRACE_BAND_MAPPING:
        return {"error": "GRACE mappings not loaded yet"}
    by_year = {}
    for (year, month), band in sorted(GRACE_BAND_MAPPING.items()):
        if year not in by_year:
            by_year[year] = []
        by_year[year].append(month)
    return {
        "total_months": len(GRACE_BAND_MAPPING),
        "years": sorted(by_year.keys()),
        "by_year": by_year,
        "date_range": {
            "start": f"{min(GRACE_BAND_MAPPING.keys())[0]}-{min(GRACE_BAND_MAPPING.keys())[1]:02d}",
            "end": f"{max(GRACE_BAND_MAPPING.keys())[0]}-{max(GRACE_BAND_MAPPING.keys())[1]:02d}"
        }
    }

@app.get("/api/grace")
def get_grace(
    year: int = Query(..., ge=2002, le=2025),
    month: Optional[int] = Query(None, ge=1, le=12),
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    sample_rate: int = Query(1, ge=1, le=20),
    max_points: int = Query(10000, ge=100, le=50000)
):
    
    if month is not None:
        try:
            band = grace_date_to_band(year, month)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        bands = [band]
        available_months = 1
        available_month_list = [month]
        description = f"{year}-{month:02d}"
    else:
        bands = []
        available_month_list = []
        for m in range(1, 13):
            try:
                band = grace_date_to_band(year, m)
                with engine.connect() as conn:
                    check_query = text(f"""
                        SELECT COUNT(*) FROM (
                            SELECT (ST_PixelAsCentroids(rast, {band})).val as val
                            FROM public.grace_lwe WHERE rid = 1 LIMIT 10
                        ) sub WHERE val IS NOT NULL;
                    """)
                    has_data = conn.execute(check_query).fetchone()[0] > 0
                    if has_data:
                        bands.append(band)
                        available_month_list.append(m)
            except:
                continue
        if not bands:
            return {
                "data_type": "grace_tws",
                "year": year,
                "month": None,
                "status": "no_data",
                "count": 0,
                "points": []
            }
        available_months = len(bands)
        description = f"{year} annual ({available_months}/12 months)"
    
    prepared_boundary = get_boundary_geometry(state, district)
    all_points = []
    
    try:
        with engine.connect() as conn:
            for band_num in bands:
                query = text(f"""
                    SELECT 
                        ST_X(ST_SetSRID((ST_PixelAsCentroids(rast, {band_num})).geom, 4326)) as longitude,
                        ST_Y(ST_SetSRID((ST_PixelAsCentroids(rast, {band_num})).geom, 4326)) as latitude,
                        (ST_PixelAsCentroids(rast, {band_num})).val as lwe_cm
                    FROM public.grace_lwe
                    WHERE rid = 1
                """)
                result = conn.execute(query)
                band_points = [
                    {"longitude": float(row[0]), "latitude": float(row[1]), "lwe_cm": float(row[2])}
                    for row in result if row[2] is not None
                ]
                all_points.extend(band_points)
        
        if len(bands) > 1:
            from collections import defaultdict
            location_values = defaultdict(list)
            for point in all_points:
                key = (point["longitude"], point["latitude"])
                location_values[key].append(point["lwe_cm"])
            all_points = []
            for (lon, lat), values in location_values.items():
                cell_area_km2 = 625 * math.cos(math.radians(lat))
                avg_lwe = sum(values) / len(values)
                all_points.append({
                    "longitude": lon,
                    "latitude": lat,
                    "lwe_cm": round(avg_lwe, 3),
                    "cell_area_km2": round(cell_area_km2, 2)
                })
        else:
            for point in all_points:
                lat = point["latitude"]
                cell_area_km2 = 625 * math.cos(math.radians(lat))
                point["cell_area_km2"] = round(cell_area_km2, 2)
                point["lwe_cm"] = round(point["lwe_cm"], 3)
        
        fallback_used = False
        fallback_message = None
        
        if prepared_boundary:
            filtered_points = filter_points_by_boundary(all_points, prepared_boundary)
            if len(filtered_points) == 0:
                centroid = get_boundary_centroid(state, district)
                if centroid:
                    filtered_points = apply_fallback_nearest_points(all_points, centroid, 50, 100)
                    fallback_used = True
                    fallback_message = f"Region too small. Showing {len(filtered_points)} nearest points within 50km."
        else:
            filtered_points = all_points
        
        regional_avg = None
        total_area = None
        if len(filtered_points) > 0:
            total_weighted_lwe = sum(p["lwe_cm"] * p["cell_area_km2"] for p in filtered_points)
            total_area = sum(p["cell_area_km2"] for p in filtered_points)
            regional_avg = round(total_weighted_lwe / total_area, 3) if total_area > 0 else None
        
        
        # Apply interpolation to increase point density (handles all cases: 1→3, 2→6, 3+→~2x)
        try:
            filtered_points = interpolate_grid_points(filtered_points, value_key="lwe_cm")
            
            # Re-filter to remove any interpolated points that went outside boundary
            if prepared_boundary and len(filtered_points) > 0:
                points_before_refilter = len(filtered_points)
                filtered_points = filter_points_by_boundary(filtered_points, prepared_boundary)
                points_removed = points_before_refilter - len(filtered_points)
                if points_removed > 0:
                    print(f"Boundary re-filter: Removed {points_removed} interpolated points outside boundary")
        except Exception as e:
            print(f"Interpolation skipped: {e}")
        
        
        if sample_rate > 1 and len(filtered_points) > 0:
            filtered_points = filtered_points[::sample_rate]
        
        if len(filtered_points) > max_points:
            filtered_points = filtered_points[:max_points]
        
        return {
            "data_type": "grace_tws",
            "year": year,
            "month": month,
            "description": description,
            "status": "success" if not fallback_used else "partial_coverage",
            "fallback_used": fallback_used,
            "fallback_message": fallback_message,
            "months_available": available_months,
            "available_month_list": available_month_list,
            "regional_average_cm": regional_avg,
            "total_area_km2": round(total_area, 2) if total_area else None,
            "count": len(filtered_points),
            "points": filtered_points
        }
    
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# RAINFALL ENDPOINTS
# =============================================================================

@app.get("/api/rainfall")
def get_rainfall(
    year: int = Query(..., ge=1994, le=2024),
    month: Optional[int] = Query(None, ge=1, le=12),
    day: Optional[int] = Query(None, ge=1, le=31),
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    sample_rate: int = Query(1, ge=1, le=50),
    max_points: int = Query(10000, ge=100, le=50000)
):
    
    table_name = f"rainfall_{year}"
    check_query = text("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = :table_name
        );
    """)
    
    try:
        with engine.connect() as conn:
            exists = conn.execute(check_query, {"table_name": table_name}).fetchone()[0]
        if not exists:
            raise HTTPException(status_code=404, detail=f"No rainfall data for year {year}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    if day is not None and month is not None:
        band = day_of_year(year, month, day)
        bands = [band]
        num_days = 1
        description = f"{year}-{month:02d}-{day:02d} (single day)"
        calc_mode = "single_day"
    elif month is not None:
        first_day, last_day = get_month_day_range(year, month)
        bands = list(range(first_day, last_day + 1))
        num_days = len(bands)
        description = f"{year}-{month:02d} (avg daily over {num_days} days)"
        calc_mode = "average_monthly"
    else:
        num_days = 366 if calendar.isleap(year) else 365
        bands = list(range(1, num_days + 1))
        description = f"{year} (avg daily over {num_days} days)"
        calc_mode = "average_annual"
    
    prepared_boundary = get_boundary_geometry(state, district)
    all_points = []
    
    try:
        with engine.connect() as conn:
            for i, band_num in enumerate(bands):
                query = text(f"""
                    SELECT 
                        ST_X(ST_SetSRID((ST_PixelAsCentroids(rast, {band_num})).geom, 4326)) as longitude,
                        ST_Y(ST_SetSRID((ST_PixelAsCentroids(rast, {band_num})).geom, 4326)) as latitude,
                        (ST_PixelAsCentroids(rast, {band_num})).val as rainfall_mm
                    FROM {table_name}
                    WHERE rid = 1
                """)
                result = conn.execute(query)
                band_points = [
                    {"longitude": float(row[0]), "latitude": float(row[1]), "rainfall_mm": float(row[2])}
                    for row in result if row[2] is not None
                ]
                all_points.extend(band_points)
        
        if len(bands) > 1:
            from collections import defaultdict
            location_values = defaultdict(list)
            for point in all_points:
                key = (point["longitude"], point["latitude"])
                location_values[key].append(point["rainfall_mm"])
            all_points = [
                {"longitude": lon, "latitude": lat, "rainfall_mm": round(sum(values) / len(values), 2), "days_averaged": len(values)}
                for (lon, lat), values in location_values.items()
            ]
        else:
            for point in all_points:
                point["rainfall_mm"] = round(point["rainfall_mm"], 2)
                point["days_averaged"] = 1
        
        fallback_used = False
        fallback_message = None
        
        if prepared_boundary:
            filtered_points = filter_points_by_boundary(all_points, prepared_boundary)
            if len(filtered_points) == 0:
                centroid = get_boundary_centroid(state, district)
                if centroid:
                    filtered_points = apply_fallback_nearest_points(all_points, centroid, 50, 100)
                    fallback_used = True
                    fallback_message = f"Region too small. Showing {len(filtered_points)} nearest points within 50km."
        else:
            filtered_points = all_points
        
        regional_avg = None
        if len(filtered_points) > 0:
            regional_avg = round(sum(p["rainfall_mm"] for p in filtered_points) / len(filtered_points), 2)
        
        
        # Apply interpolation to increase point density (handles all cases: 1→3, 2→6, 3+→~2x)
        try:
            filtered_points = interpolate_grid_points(filtered_points, value_key="rainfall_mm")
            
            # Re-filter to remove any interpolated points that went outside boundary
            if prepared_boundary and len(filtered_points) > 0:
                points_before_refilter = len(filtered_points)
                filtered_points = filter_points_by_boundary(filtered_points, prepared_boundary)
                points_removed = points_before_refilter - len(filtered_points)
                if points_removed > 0:
                    print(f"    → Boundary re-filter: Removed {points_removed} points outside boundary (kept {len(filtered_points)})")
        except Exception as e:
            print(f"Interpolation skipped: {e}")
        
        
        if sample_rate > 1 and len(filtered_points) > 0:
            filtered_points = filtered_points[::sample_rate]
        
        if len(filtered_points) > max_points:
            filtered_points = filtered_points[:max_points]
        
        return {
            "data_type": "rainfall",
            "year": year,
            "month": month,
            "day": day,
            "description": description,
            "status": "success" if not fallback_used else "partial_coverage",
            "fallback_used": fallback_used,
            "fallback_message": fallback_message,
            "calculation_method": calc_mode,
            "unit": "mm/day",
            "days_included": num_days,
            "regional_average_mm_per_day": regional_avg,
            "count": len(filtered_points),
            "points": filtered_points
        }
    
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# HEALTH CHECK
# =============================================================================
@app.post("/api/admin/clear-cache")
def clear_all_caches():
    """Clear all cached data (use after database updates)"""
    TIMESERIES_CACHE.cache.clear()
    WELLS_CACHE.cache.clear()
    GRACE_CACHE.cache.clear()
    RAINFALL_CACHE.cache.clear()
    return {
        "status": "success", 
        "message": "All caches cleared",
        "timestamp": datetime.now().isoformat()
    }
    
@app.get("/health")
def health_check():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        chatbot_status = "enabled" if CHATBOT_ENABLED else "disabled"
        
        return {
            "status": "healthy",
            "database": "connected",
            "grace_bands": len(GRACE_BAND_MAPPING),
            "rainfall_years": len(RAINFALL_TABLES),
            "chatbot": chatbot_status,
            "version": "6.2.0 - OPTIMIZED (Caching + Parallel)",
            "cache_stats": {
                "timeseries_entries": len(TIMESERIES_CACHE.cache),
                "wells_entries": len(WELLS_CACHE.cache),
                "grace_entries": len(GRACE_CACHE.cache),
                "rainfall_entries": len(RAINFALL_CACHE.cache)
            },
            "optimizations": [
                "In-memory caching (3600s TTL)",
                "Parallel GRACE/Rainfall queries (4 workers)",
                "SQL-based aggregations",
                "Auto cache cleanup (10min)"
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

def haversine_distance_vectorized(lat1, lon1, lat2, lon2):
    """Calculate haversine distance in km"""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def standardize_series(series):
    """Standardize a series to z-scores"""
    series = pd.Series(series).astype(float)
    if series.dropna().nunique() < 2:
        return pd.Series(np.zeros(len(series)))
    scaler = StandardScaler()
    return pd.Series(scaler.fit_transform(series.values.reshape(-1, 1)).flatten())

def get_wells_for_analysis(boundary_geojson: str = None):
    """Get all wells data for analysis"""
    where_clause = ""
    params = {}
    
    if boundary_geojson:
        where_clause = """
            WHERE ST_Intersects(
                ST_GeomFromGeoJSON(:boundary_geojson),
                ST_SetSRID(ST_MakePoint(
                    COALESCE(NULLIF("LON", 0), "LONGITUD_1"),
                    COALESCE(NULLIF("LAT", 0), "LATITUDE_1")
                ), 4326)
            )
        """
        params["boundary_geojson"] = boundary_geojson
    
    query = text(f"""
        SELECT 
            "Date",
            "GWL",
            COALESCE(NULLIF("LAT", 0), "LATITUDE_1") as latitude,
            COALESCE(NULLIF("LON", 0), "LONGITUD_1") as longitude,
            "STATE" as state,
            "DISTRICT" as district
        FROM groundwater_level
        {where_clause}
        AND "GWL" IS NOT NULL
        AND "Date" IS NOT NULL
        ORDER BY "Date"
    """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(query, params)
            df = pd.DataFrame([
                {
                    "date": row[0],
                    "gwl": float(row[1]),
                    "latitude": float(row[2]),
                    "longitude": float(row[3]),
                    "state": row[4],
                    "district": row[5],
                    "site_id": f"{round(float(row[2]), 5)}_{round(float(row[3]), 5)}"
                }
                for row in result
            ])
            return df
    except Exception as e:
        print(f"Error getting wells: {e}")
        return pd.DataFrame()

def get_aquifer_properties(boundary_geojson: str = None):
    """Get aquifer properties for the region"""
    where_clause = ""
    params = {}
    
    if boundary_geojson:
        where_clause = "WHERE ST_Intersects(ST_MakeValid(geometry), ST_GeomFromGeoJSON(:boundary_geojson))"
        params["boundary_geojson"] = boundary_geojson
    
    query = text(f"""
        SELECT 
            aquifer,
            aquifers as majoraquif,
            yeild__,
            ST_Area(ST_Transform(ST_MakeValid(geometry), 32643)) as area_m2,
            ST_AsGeoJSON(ST_Centroid(geometry)) as centroid
        FROM aquifers
        {where_clause}
    """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(query, params)
            aquifers = []
            for row in result:
                centroid = json.loads(row[4])
                aquifers.append({
                    "aquifer": row[0],
                    "majoraquif": row[1],
                    "yield": row[2],
                    "area_m2": float(row[3]) if row[3] else 0,
                    "latitude": centroid["coordinates"][1],
                    "longitude": centroid["coordinates"][0]
                })
            return pd.DataFrame(aquifers)
    except Exception as e:
        print(f"Error getting aquifers: {e}")
        return pd.DataFrame()

def get_aquifer_polygons(boundary_geojson: str = None):
    """Get aquifer polygons, optionally clipped to boundary"""
    where_clause = ""
    geometry_select = "ST_MakeValid(geometry)"
    params = {}
    
    if boundary_geojson:
        # Clip polygons to boundary AND filter to overlapping ones
        where_clause = """
            WHERE ST_Intersects(
                ST_MakeValid(geometry), 
                ST_GeomFromGeoJSON(:boundary_geojson)
            )
        """
        # Clip geometry to boundary (like robust_clip_to_boundary in Dash)
        geometry_select = """
            ST_Intersection(
                ST_MakeValid(geometry),
                ST_GeomFromGeoJSON(:boundary_geojson)
            )
        """
        params["boundary_geojson"] = boundary_geojson
    
    query = text(f"""
        SELECT 
            aquifer,
            aquifers as majoraquif,
            yeild__,
            ST_Area(ST_Transform({geometry_select}, 32643)) as area_m2,
            ST_AsGeoJSON({geometry_select}) as geometry
        FROM aquifers
        {where_clause}
    """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(query, params)
            aquifers = []
            
            for row in result:
                try:
                    geom_json = json.loads(row[4])
                    
                    # Skip empty geometries after clipping
                    if not geom_json.get('coordinates'):
                        continue
                    
                    aquifers.append({
                        "aquifer": row[0],
                        "majoraquif": row[1],
                        "yield": row[2],
                        "area_m2": float(row[3]) if row[3] else 0,
                        "geometry": geom_json
                    })
                except Exception as e:
                    print(f"Error parsing aquifer geometry: {e}")
                    continue
            
            return aquifers
            
    except Exception as e:
        print(f"Error getting aquifer polygons: {e}")
        print(traceback.format_exc())
        return []
# =============================================================================
# ADVANCED MODULE 1: AQUIFER SUITABILITY INDEX (ASI)
# =============================================================================

@app.get("/api/advanced/asi")
def aquifer_suitability_index(
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None)
):
    """Calculate Aquifer Suitability Index (ASI)"""
    try:
        boundary_geojson = get_boundary_geojson(state, district)
        aquifers_list = get_aquifer_polygons(boundary_geojson)
        
        if not aquifers_list:
            raise HTTPException(status_code=404, detail="No aquifer data for this region")
        
        aquifers_df = pd.DataFrame(aquifers_list)
        
        # Lithology-based specific yield mapping
        sy_map = {
            'alluvium': 0.10,
            'sandstone': 0.06,
            'limestone': 0.05,
            'basalt': 0.03,
            'granite': 0.02
        }
        
        def get_sy(row):
            yield_str = str(row.get('yield', '')).lower()
            try:
                import re
                numbers = re.findall(r'\d+\.?\d*', yield_str)
                if numbers:
                    vals = [float(n) for n in numbers]
                    return sum(vals) / len(vals) / 100.0
            except:
                pass
            
            majoraquif = str(row.get('majoraquif', '')).lower()
            for key, val in sy_map.items():
                if key in majoraquif:
                    return val
            return 0.04
        
        aquifers_df['specific_yield'] = aquifers_df.apply(get_sy, axis=1)
        
        # ✅ FIX 1: Handle edge case where all specific yields are the same
        sy = aquifers_df['specific_yield'].values
        
        # Check if there's any variation
        if len(np.unique(sy)) == 1:
            # All same value - assign middle score
            aquifers_df['asi_score'] = 2.5
            q_low, q_high = float(sy[0]), float(sy[0])
        else:
            # Normal quantile stretching
            q_low, q_high = np.quantile(sy, 0.05), np.quantile(sy, 0.95)
            
            # ✅ FIX 2: Ensure q_high > q_low
            if q_high - q_low < 1e-6:
                q_low, q_high = 0.01, 0.15
            
            aquifers_df['asi_score'] = (
                (sy.clip(q_low, q_high) - q_low) / (q_high - q_low) * 5.0
            ).clip(0, 5)
        
        # ✅ FIX 3: Replace any NaN/Inf values before JSON serialization
        aquifers_df['asi_score'] = aquifers_df['asi_score'].replace([np.inf, -np.inf], np.nan).fillna(2.5)
        aquifers_df['specific_yield'] = aquifers_df['specific_yield'].replace([np.inf, -np.inf], np.nan).fillna(0.04)
        
        # Build GeoJSON
        features = []
        for idx, row in aquifers_df.iterrows():
            features.append({
                "type": "Feature",
                "id": str(idx),
                "properties": {
                    "aquifer": row['aquifer'],
                    "majoraquif": row['majoraquif'],
                    "asi_score": float(row['asi_score']),  # Ensure it's a valid float
                    "specific_yield": float(row['specific_yield']),
                    "area_m2": float(row['area_m2'])
                },
                "geometry": row['geometry']
            })
        
        geojson_data = {
            "type": "FeatureCollection",
            "features": features
        }
        
        # ✅ FIX 4: Safe statistics calculation
        def safe_stat(series, func, default=0.0):
            """Calculate statistic safely, handling NaN/Inf"""
            try:
                val = func(series)
                if np.isnan(val) or np.isinf(val):
                    return default
                return round(float(val), 4)
            except:
                return default
        
        statistics = {
            "mean_asi": safe_stat(aquifers_df['asi_score'], np.mean, 2.5),
            "median_asi": safe_stat(aquifers_df['asi_score'], np.median, 2.5),
            "std_asi": safe_stat(aquifers_df['asi_score'], np.std, 0.0),
            "min_asi": safe_stat(aquifers_df['asi_score'], np.min, 0.0),
            "max_asi": safe_stat(aquifers_df['asi_score'], np.max, 5.0),
            "dominant_aquifer": str(aquifers_df['majoraquif'].mode().iloc[0]) if len(aquifers_df) > 0 else "Unknown",
            "avg_specific_yield": safe_stat(aquifers_df['specific_yield'], np.mean, 0.04),
            "total_area_km2": round(float(aquifers_df['area_m2'].sum()) / 1_000_000, 2)
        }
        
        # Calculate additional context metrics
        high_suit_pct = round((aquifers_df['asi_score'] >= 3.5).sum() / len(aquifers_df) * 100, 1)
        low_suit_pct = round((aquifers_df['asi_score'] < 2.5).sum() / len(aquifers_df) * 100, 1)
        
        # Determine regional assessment
        mean_asi = statistics["mean_asi"]
        if mean_asi > 3.5:
            regional_rating = "Excellent"
            rating_context = "exceptionally favorable geological conditions with high groundwater storage capacity"
        elif mean_asi > 2.5:
            regional_rating = "Good"
            rating_context = "favorable conditions with moderate to good storage potential in most areas"
        elif mean_asi > 1.5:
            regional_rating = "Moderate"
            rating_context = "mixed geological conditions requiring targeted groundwater development strategies"
        else:
            regional_rating = "Poor"
            rating_context = "challenging hard-rock terrain with limited natural storage capacity"
        
        return {
            "module": "ASI",
            "description": "Aquifer Suitability Index (0-5 scale)",
            "filters": {"state": state, "district": district},
            "statistics": statistics,
            "count": len(aquifers_df),
            "geojson": geojson_data,
            
            # ✅ ENHANCED METHODOLOGY
            "methodology": {
                "approach": "Lithology-based specific yield normalization",
                "quantile_stretch": {
                    "low": round(float(q_low), 5),
                    "high": round(float(q_high), 5)
                },
                "steps": [
                    f"1. Extracted {len(aquifers_df)} aquifer polygons from database",
                    f"2. Identified dominant lithology: {statistics.get('dominant_aquifer', 'N/A')}",
                    "3. Mapped lithology to specific yield (Alluvium=0.10, Sandstone=0.06, etc.)",
                    f"4. Normalized to 0-5 scale using quantile stretching (range: {round(float(q_low), 3)}-{round(float(q_high), 3)})"
                ],
                "formula": "ASI = ((Sy - q_low) / (q_high - q_low)) × 5",
                "data_source": "aquifers table",
                "scientific_basis": "Higher specific yield indicates better porosity and storage capacity, making aquifers more suitable for sustainable groundwater extraction."
            },
            
            # ✅ ENHANCED INTERPRETATION WITH NARRATIVES
            "interpretation": {
                "score_meaning": {
                    "0-1.5": "Poor suitability (hard rock, low storage)",
                    "1.5-2.5": "Low suitability (weathered rock)",
                    "2.5-3.5": "Moderate suitability (mixed aquifers)",
                    "3.5-5.0": "High suitability (alluvium, unconsolidated sediments)"
                },
                "regional_rating": regional_rating,
                "regional_narrative": f"This region exhibits {regional_rating.lower()} aquifer suitability with {rating_context}.",
                "high_suitability_percentage": high_suit_pct,
                "low_suitability_percentage": low_suit_pct,
                
                # Spatial distribution context
                "spatial_distribution": {
                    "high_quality_areas": f"{high_suit_pct}% of the region",
                    "challenging_areas": f"{low_suit_pct}% of the region",
                    "assessment": "highly heterogeneous" if abs(high_suit_pct - low_suit_pct) < 20 else ("predominantly favorable" if high_suit_pct > 50 else "predominantly challenging")
                },
                
                # Comparative context
                "comparative_context": {
                    "mean_vs_median": "balanced distribution" if abs(statistics["mean_asi"] - statistics["median_asi"]) < 0.3 else "skewed distribution",
                    "variability": "high variability" if statistics["std_asi"] > 1.0 else "moderate variability" if statistics["std_asi"] > 0.5 else "low variability",
                    "range_span": f"Score range spans {round(statistics['max_asi'] - statistics['min_asi'], 2)} points"
                }
            },
            
            # ✅ ENHANCED KEY INSIGHTS WITH ACTIONABLE CONTEXT
            "key_insights": [
                # Primary assessment
                f"🎯 Regional Suitability: {regional_rating} (Mean ASI: {statistics['mean_asi']}/5.0) - {rating_context}",
                
                 # Spatial breakdown
                f"📊 Spatial Analysis: {high_suit_pct}% high-suitability areas, {low_suit_pct}% challenging terrain",
                
                # Dominant geology
                f"🪨 Geology: Dominated by {statistics.get('dominant_aquifer', 'N/A')} with average specific yield of {round(statistics['avg_specific_yield'] * 100, 1)}%",
                
                # Scale context
                f"📏 Coverage: {statistics['total_area_km2']} km² analyzed across {len(aquifers_df)} aquifer units"
            ]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"ASI Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# ADVANCED MODULE 2: WELL NETWORK DENSITY ANALYSIS (POINT PROCESS)
# =============================================================================

@app.get("/api/advanced/network-density")
def well_network_density(
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    radius_km: float = Query(20.0, ge=5.0, le=100.0)
):
    """
    Analyze well network density and signal strength
    Returns TWO maps:
    1. Site-level: strength (|slope|/σ) with symbol size as local density
    2. Gridded: absolute density per 1000 km² clipped to AOI
    """
    try:
        boundary_geojson = get_boundary_geojson(state, district)
        wells_df = get_wells_for_analysis(boundary_geojson)
        
        if wells_df.empty:
            raise HTTPException(status_code=404, detail="No well data for this region")
        
        # Calculate slope and std for each site
        site_stats = []
        for site_id, group in wells_df.groupby('site_id'):
            if len(group) < 6:
                continue
            
            group = group.sort_values('date')
            days = (group['date'] - group['date'].min()).dt.days.values
            gwl = group['gwl'].values
            
            if len(np.unique(days)) < 2:
                continue
            
            slope, _, _, _, _ = linregress(days, gwl)
            std = np.std(gwl)
            
            site_stats.append({
                'site_id': site_id,
                'latitude': group['latitude'].iloc[0],
                'longitude': group['longitude'].iloc[0],
                'slope_m_per_year': slope * 365.25,
                'gwl_std': std,
                'strength': abs(slope * 365.25) / (std if std > 0 else 1e-6),
                'n_observations': len(group)
            })
        
        if not site_stats:
            raise HTTPException(status_code=404, detail="Insufficient data for analysis")
        
        sites_df = pd.DataFrame(site_stats)
        
        # Calculate local density using haversine
        coords = sites_df[['latitude', 'longitude']].values
        coords_rad = np.radians(coords)
        
        R_earth = 6371.0
        nbrs = NearestNeighbors(radius=radius_km/R_earth, metric='haversine').fit(coords_rad)
        neighbors = nbrs.radius_neighbors(coords_rad, return_distance=False)
        
        # Count neighbors (excluding self)
        density_counts = np.array([len(n) - 1 for n in neighbors])
        area_km2 = np.pi * (radius_km ** 2)
        sites_df['local_density_per_km2'] = density_counts / area_km2
        sites_df['neighbors_within_radius'] = density_counts
        
        # MAP 1: Site-level results
        map1_results = sites_df.to_dict('records')
        
        # MAP 2: Gridded density
        map2_results = []
        try:
            # Fetch boundary geometry using SQL
            selection_boundary = None
            
            if district and state:
                query = text("""
                    SELECT ST_AsGeoJSON(geometry) as geojson
                    FROM district_state
                    WHERE UPPER("District") = UPPER(:district)
                    AND UPPER("State") = UPPER(:state)
                    LIMIT 1;
                """)
                with engine.connect() as conn:
                    result = conn.execute(query, {"district": district, "state": state}).fetchone()
                    if result:
                        geom = shape(json.loads(result[0]))
                        selection_boundary = gpd.GeoDataFrame([{'geometry': geom}], crs="EPSG:4326")
            
            elif state:
                query = text("""
                    SELECT ST_AsGeoJSON(ST_Union(geometry)) as geojson
                    FROM district_state
                    WHERE UPPER("State") = UPPER(:state);
                """)
                with engine.connect() as conn:
                    result = conn.execute(query, {"state": state}).fetchone()
                    if result:
                        geom = shape(json.loads(result[0]))
                        selection_boundary = gpd.GeoDataFrame([{'geometry': geom}], crs="EPSG:4326")
            
            # Get bounds
            if selection_boundary is not None and not selection_boundary.empty:
                bounds = selection_boundary.total_bounds
            else:
                bounds = np.array([
                    wells_df['longitude'].min(),
                    wells_df['latitude'].min(),
                    wells_df['longitude'].max(),
                    wells_df['latitude'].max()
                ])
            
            # Compute gridded density
            grid_df = compute_gridded_density(
                wells_df, 
                selection_boundary, 
                bounds, 
                radius_km=radius_km
            )
            
            map2_results = grid_df.to_dict('records') if not grid_df.empty else []
            
        except Exception as e:
            print(f"Gridded density computation failed: {e}")
            print(traceback.format_exc())
            map2_results = []
        
        # Calculate enhanced metrics
        avg_strength = round(float(sites_df['strength'].mean()), 3)
        avg_density = round(float(sites_df['local_density_per_km2'].mean()), 4)
        strong_signal_count = int((sites_df['strength'] > 0.7).sum())
        strong_signal_pct = round((strong_signal_count / len(sites_df)) * 100, 1)
        
        # Determine quality assessment
        if avg_strength > 1.5:
            signal_quality = "Excellent"
            quality_context = "very high confidence in trend detection with minimal noise"
        elif avg_strength > 0.7:
            signal_quality = "Good"
            quality_context = "strong signal quality enabling reliable trend analysis"
        elif avg_strength > 0.3:
            signal_quality = "Moderate"
            quality_context = "detectable trends but with some noise requiring careful interpretation"
        else:
            signal_quality = "Poor"
            quality_context = "weak signals dominated by noise, making trend detection challenging"
        
        # Density assessment
        if avg_density > 0.05:
            coverage_rating = "Excellent"
            coverage_context = "dense monitoring network providing comprehensive spatial coverage"
        elif avg_density > 0.03:
            coverage_rating = "Good"
            coverage_context = "adequate monitoring coverage for regional analysis"
        elif avg_density > 0.01:
            coverage_rating = "Moderate"
            coverage_context = "sparse but functional monitoring network with coverage gaps"
        else:
            coverage_rating = "Poor"
            coverage_context = "very sparse monitoring requiring network expansion"
        
        return {
            "module": "NETWORK_DENSITY",
            "description": f"Well network analysis with {radius_km}km radius",
            "filters": {"state": state, "district": district},
            "parameters": {"radius_km": radius_km},
            "statistics": {
                "total_sites": len(map1_results),
                "avg_strength": avg_strength,
                "avg_local_density": avg_density,
                "median_observations": int(sites_df['n_observations'].median()),
                "grid_cells": len(map2_results)
            },
            
            # ✅ ENHANCED METHODOLOGY
            "methodology": {
                "approach": "Haversine-based spatial signal strength + local density",
                "steps": [
                    f"1. Analyzed {len(map1_results)} well sites with ≥6 monthly observations",
                    "2. Calculated trend slope (m/year) using linear regression",
                    "3. Computed signal strength = |slope| / standard_deviation",
                    f"4. Counted neighbors within {radius_km}km radius using haversine distance",
                    f"5. Generated gridded density map ({len(map2_results)} cells) clipped to region boundary"
                ],
                "signal_strength_formula": "strength = |trend_slope| / σ(GWL)",
                "density_formula": f"density = neighbors / (π × {radius_km}²) × 1000 km²",
                "scientific_basis": "Higher signal strength indicates clearer trends with less noise, enabling more confident groundwater trend analysis and forecasting."
            },
            
            # ✅ ENHANCED INTERPRETATION
            "interpretation": {
                "strength_levels": {
                    "0-0.3": "Weak signal (noisy data)",
                    "0.3-0.7": "Moderate signal (detectable trend)",
                    "0.7-1.5": "Strong signal (reliable trend)",
                    ">1.5": "Very strong signal (high confidence)"
                },
                "signal_quality_rating": signal_quality,
                "coverage_quality_rating": coverage_rating,
                "regional_narrative": f"This monitoring network exhibits {signal_quality.lower()} signal quality with {quality_context} and {coverage_rating.lower()} spatial coverage with {coverage_context}.",
                "strong_signal_percentage": strong_signal_pct,
                
                "spatial_distribution": {
                    "high_quality_sites": f"{strong_signal_pct}% have strong signals (>0.7)",
                    "network_density": f"{avg_density} sites/km²",
                    "assessment": "well-distributed" if avg_density > 0.03 else "clustered/sparse"
                },
                
                "comparative_context": {
                    "data_richness": f"Median {int(sites_df['n_observations'].median())} observations per site",
                    "signal_variation": "high variability" if sites_df['strength'].std() > 0.5 else "consistent signals",
                    "network_maturity": "mature" if int(sites_df['n_observations'].median()) > 50 else "developing"
                }
            },
            
            # ✅ ENHANCED KEY INSIGHTS
            "key_insights": [
                f"🎯 Network Quality: {signal_quality} signal quality ({avg_strength} avg strength) - {quality_context}",
                f"📊 Coverage Assessment: {coverage_rating} ({avg_density} sites/km²) - {coverage_context}",
                f"📈 High-Confidence Sites: {strong_signal_count} sites ({strong_signal_pct}%) have strong signals (>0.7)",
                f"💡 Data Maturity: Median {int(sites_df['n_observations'].median())} observations per site - {'excellent temporal coverage' if int(sites_df['n_observations'].median()) > 50 else 'growing dataset'}",
                f"📏 Analysis Scale: {len(map1_results)} monitoring sites across {len(map2_results)} grid cells"
            ],
            
            "map1_site_level": {
                "count": len(map1_results),
                "data": map1_results,
                "description": "Site-level strength with local density"
            },
            "map2_gridded": {
                "count": len(map2_results),
                "data": map2_results,
                "description": "Absolute density grid (sites per 1000 km²)"
            }
        }
   # ✅ ADD THESE LINES (they were missing!)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Network Density Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
# =============================================================================
# ADVANCED MODULE 3: SPATIO-TEMPORAL AQUIFER STRESS SCORE (SASS)
# =============================================================================

@app.get("/api/advanced/sass")
def aquifer_stress_score(
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    year: int = Query(..., ge=2002, le=2024),
    month: int = Query(..., ge=1, le=12)
):
    """
    Calculate composite stress score: SASS = 0.5*(-z_GWL) + 0.3*z_GRACE + 0.2*z_Rain
    """
    try:
        boundary_geojson = get_boundary_geojson(state, district)
        target_date = datetime(year, month, 15)
        tolerance = timedelta(days=60)
        
        # Get wells data
        wells_df = get_wells_for_analysis(boundary_geojson)
        if wells_df.empty:
            raise HTTPException(status_code=404, detail="No well data")
        
        wells_df['date_diff'] = abs(wells_df['date'] - target_date)
        near_wells = wells_df[wells_df['date_diff'] <= tolerance].sort_values('date_diff').groupby('site_id').first().reset_index()
        
        if near_wells.empty:
            raise HTTPException(status_code=404, detail="No wells near target date")
        
        # Standardize GWL
        gwl_z = standardize_series(near_wells['gwl'].values)
        
        # Get GRACE data
        try:
            band = grace_date_to_band(year, month)
            query = text(f"""
                SELECT 
                    ST_X(ST_SetSRID((ST_PixelAsCentroids(rast, {band})).geom, 4326)) as longitude,
                    ST_Y(ST_SetSRID((ST_PixelAsCentroids(rast, {band})).geom, 4326)) as latitude,
                    (ST_PixelAsCentroids(rast, {band})).val as tws
                FROM public.grace_lwe WHERE rid = 1
            """)
            
            with engine.connect() as conn:
                result = conn.execute(query)
                grace_points = pd.DataFrame([
                    {"longitude": r[0], "latitude": r[1], "tws": r[2]}
                    for r in result if r[2] is not None
                ])
            
            # Interpolate GRACE to well locations
            if not grace_points.empty:
                grace_interp = griddata(
                    grace_points[['longitude', 'latitude']].values,
                    grace_points['tws'].values,
                    near_wells[['longitude', 'latitude']].values,
                    method='linear'
                )
                grace_z = standardize_series(grace_interp)
            else:
                grace_z = pd.Series(np.zeros(len(near_wells)))
        except Exception as e:
            print(f"GRACE error in SASS: {e}")
            grace_z = pd.Series(np.zeros(len(near_wells)))
        
        # Get rainfall data
        try:
            table_name = f"rainfall_{year}"
            first_day, last_day = get_month_day_range(year, month)
            
            rainfall_values = []
            with engine.connect() as conn:
                for doy in range(first_day, last_day + 1):
                    query = text(f"""
                        SELECT 
                            ST_X(ST_SetSRID((ST_PixelAsCentroids(rast, {doy})).geom, 4326)) as lon,
                            ST_Y(ST_SetSRID((ST_PixelAsCentroids(rast, {doy})).geom, 4326)) as lat,
                            (ST_PixelAsCentroids(rast, {doy})).val as val
                        FROM {table_name} WHERE rid = 1
                    """)
                    result = conn.execute(query)
                    rainfall_values.extend([
                        {"longitude": r[0], "latitude": r[1], "val": r[2]}
                        for r in result if r[2] is not None and 0 <= r[2] <= 500
                    ])
            
            if rainfall_values:
                rain_df = pd.DataFrame(rainfall_values)
                rain_avg = rain_df.groupby(['longitude', 'latitude'])['val'].mean().reset_index()
                
                rain_interp = griddata(
                    rain_avg[['longitude', 'latitude']].values,
                    rain_avg['val'].values,
                    near_wells[['longitude', 'latitude']].values,
                    method='linear'
                )
                rain_z = standardize_series(rain_interp)
            else:
                rain_z = pd.Series(np.zeros(len(near_wells)))
        except Exception as e:
            print(f"Rainfall error in SASS: {e}")
            rain_z = pd.Series(np.zeros(len(near_wells)))
        
        # Calculate SASS
        sass = 0.5 * (-gwl_z.values) + 0.3 * grace_z.values + 0.2 * rain_z.values
        
        # Helper function to safely convert to float
        def safe_float(value, default=0.0):
            try:
                f = float(value)
                return default if (np.isnan(f) or np.isinf(f)) else f
            except (ValueError, TypeError):
                return default
        
        # Build results
        results = []
        for idx, row in near_wells.iterrows():
            results.append({
                "site_id": row['site_id'],
                "latitude": row['latitude'],
                "longitude": row['longitude'],
                "sass_score": round(safe_float(sass[idx]), 3),
                "gwl_stress": round(safe_float(-gwl_z.iloc[idx]), 3),
                "grace_z": round(safe_float(grace_z.iloc[idx]), 3),
                "rain_z": round(safe_float(rain_z.iloc[idx]), 3),
                "gwl": round(safe_float(row['gwl']), 2)
            })
        
        # Calculate statistics
        sass_values = [r['sass_score'] for r in results]
        
        # ✅ CREATE statistics VARIABLE BEFORE using it
        statistics = {
            "mean_sass": round(float(np.mean(sass_values)), 3) if sass_values else 0.0,
            "max_sass": round(float(np.max(sass_values)), 3) if sass_values else 0.0,
            "min_sass": round(float(np.min(sass_values)), 3) if sass_values else 0.0,
            "stressed_sites": int(sum(1 for s in sass_values if s > 1.0)),
            "critical_sites": int(sum(1 for s in sass_values if s > 2.0))
        }
        
        # Calculate enhanced metrics
        mean_sass = statistics["mean_sass"]
        stressed_pct = round((statistics["stressed_sites"] / len(results)) * 100, 1) if len(results) > 0 else 0
        critical_pct = round((statistics["critical_sites"] / len(results)) * 100, 1) if len(results) > 0 else 0
        
        # Determine stress assessment
        if mean_sass > 2.0:
            stress_rating = "Critical"
            stress_context = "severe aquifer stress requiring immediate emergency intervention"
            urgency = "URGENT"
        elif mean_sass > 1.0:
            stress_rating = "Moderate"
            stress_context = "moderate aquifer stress requiring proactive management"
            urgency = "HIGH"
        elif mean_sass > 0:
            stress_rating = "Low"
            stress_context = "low aquifer stress with adequate water availability"
            urgency = "MODERATE"
        else:
            stress_rating = "Favorable"
            stress_context = "better than average groundwater conditions"
            urgency = "LOW"
        
        # ✅ NOW RETURN THE RESPONSE
        return {
            "module": "SASS",
            "description": "Spatio-Temporal Aquifer Stress Score",
            "filters": {"state": state, "district": district, "year": year, "month": month},
            "formula": "SASS = 0.5×(−z_GWL) + 0.3×z_GRACE + 0.2×z_Rain",
            "statistics": statistics,
            "count": len(results),
            "data": results,
            
            "methodology": {
                "approach": "Multi-source composite stress index",
                "components": [
                    {
                        "name": "GWL Component",
                        "weight": 0.5,
                        "formula": "−z_GWL = −(GWL − mean) / std",
                        "meaning": "Negative z-score means deeper than normal (stressed)",
                        "data_source": f"Wells within 60 days of {year}-{month:02d}"
                    },
                    {
                        "name": "GRACE Component",
                        "weight": 0.3,
                        "formula": "z_GRACE = (TWS − mean) / std",
                        "meaning": "Positive = more water storage than average",
                        "data_source": f"GRACE satellite band for {year}-{month:02d}"
                    },
                    {
                        "name": "Rainfall Component",
                        "weight": 0.2,
                        "formula": "z_Rain = (Rain − mean) / std",
                        "meaning": "Positive = more rainfall than average",
                        "data_source": f"Monthly rainfall for {year}-{month:02d}"
                    }
                ],
                
                "calculation": "Each component standardized to z-scores, then weighted sum",
                "scientific_basis":"SASS integrates ground observations (GWL), satellite data (GRACE), and climate inputs (rainfall) to provide comprehensive multi-source stress assessment."

            },

            
            # ✅ ENHANCED INTERPRETATION
            "interpretation": {
                "stress_levels": {
                    "<0": "No stress (better than average)",
                    "0-1": "Low stress (monitor)",
                    "1-2": "Moderate stress (intervention recommended)",
                    ">2": "Critical stress (urgent action)"
                },
                "overall_status": stress_rating,
                "regional_narrative": f"This region exhibits {stress_rating.lower()} aquifer stress (mean SASS: {mean_sass}) indicating {stress_context} as of {year}-{month:02d}.",
                "stress_distribution": {
                    "critical_zones": f"{critical_pct}% ({statistics['critical_sites']} sites) with SASS > 2",
                    "moderate_stress": f"{stressed_pct - critical_pct}% with SASS 1-2",
                    "normal_conditions": f"{round(100 - stressed_pct, 1)}% with SASS < 1"
                },
                "spatial_assessment": "concentrated stress zones" if critical_pct > 20 else ("distributed stress" if stressed_pct > 30 else "stable"),
                "comparative_context": {
                    "severity_range": f"Max: {statistics['max_sass']}, Min: {statistics['min_sass']}",
                    "dominant_driver": "GWL depletion (50% weight)"
                }
            },
            
            # ✅ ENHANCED KEY INSIGHTS
            "key_insights": [
                f"🎯 Stress Assessment: {stress_rating} (Mean SASS: {mean_sass}) - {stress_context}",
                f"⚠️ Critical Zones: {statistics['critical_sites']} sites ({critical_pct}%) in critical stress (SASS > 2)",
                f"📊 Distribution: {stressed_pct}% stressed ({statistics['stressed_sites']} sites), {round(100 - stressed_pct, 1)}% normal",
                f"💧 Primary Driver: GWL depletion (50% weight) dominates stress signal",
                f"📅 Temporal Snapshot: {year}-{month:02d} across {len(results)} monitoring sites"
            ]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"SASS Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"SASS calculation failed: {str(e)}")

# =============================================================================
# ADVANCED MODULE 4: GRACE vs GROUND DIVERGENCE
# =============================================================================

@app.get("/api/advanced/grace-divergence")
def grace_divergence_analysis(
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    year: int = Query(..., ge=2002, le=2024),
    month: int = Query(..., ge=1, le=12)
):
    """
    Calculate divergence between GRACE satellite observations and ground well measurements
    Returns: z_GRACE - z_GWL on a pixel grid
    """
    try:
        # Helper function to safely convert to JSON-compliant float
        def safe_float(value, default=0.0):
            try:
                f = float(value)
                return default if (np.isnan(f) or np.isinf(f)) else f
            except (ValueError, TypeError):
                return default
        
        boundary_geojson = get_boundary_geojson(state, district)
        target_date = datetime(year, month, 15)
        
        # Get GRACE pixels
        band = grace_date_to_band(year, month)
        query = text(f"""
            SELECT 
                ST_X(ST_SetSRID((ST_PixelAsCentroids(rast, {band})).geom, 4326)) as longitude,
                ST_Y(ST_SetSRID((ST_PixelAsCentroids(rast, {band})).geom, 4326)) as latitude,
                (ST_PixelAsCentroids(rast, {band})).val as tws
            FROM public.grace_lwe WHERE rid = 1
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query)
            grace_df = pd.DataFrame([
                {"longitude": r[0], "latitude": r[1], "tws": r[2]}
                for r in result if r[2] is not None
            ])
        
        if grace_df.empty:
            raise HTTPException(status_code=404, detail="No GRACE data for this period")
        
        # Standardize GRACE
        grace_z = standardize_series(grace_df['tws'].values)
        grace_df['grace_z'] = grace_z
        
        # Get wells near target date
        wells_df = get_wells_for_analysis(boundary_geojson)
        tolerance = timedelta(days=60)
        wells_df['date_diff'] = abs(wells_df['date'] - target_date)
        near_wells = wells_df[wells_df['date_diff'] <= tolerance].sort_values('date_diff').groupby('site_id').first().reset_index()
        
        if near_wells.empty:
            raise HTTPException(status_code=404, detail="No wells near target date")
        
        # Standardize wells
        well_z = standardize_series(near_wells['gwl'].values)
        
        # Interpolate well z-scores to GRACE pixel locations
        well_z_grid = griddata(
            near_wells[['longitude', 'latitude']].values,
            well_z.values,
            grace_df[['longitude', 'latitude']].values,
            method='linear'
        )
        
        # Replace NaN from interpolation with 0
        well_z_grid = np.nan_to_num(well_z_grid, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate divergence
        grace_df['well_z_interpolated'] = well_z_grid
        grace_df['divergence'] = grace_df['grace_z'] - grace_df['well_z_interpolated']
        
        # Replace any NaN/Inf values in the DataFrame
        grace_df = grace_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Filter to boundary
        if boundary_geojson:
            prepared_boundary = get_boundary_geometry(state, district)
            if prepared_boundary:
                mask = grace_df.apply(
                    lambda row: prepared_boundary.contains(Point(row['longitude'], row['latitude'])),
                    axis=1
                )
                grace_df = grace_df[mask]
        
        # Ensure all values are JSON-compliant before creating results
        results = []
        for _, row in grace_df.iterrows():
            results.append({
                "latitude": safe_float(row['latitude']),
                "longitude": safe_float(row['longitude']),
                "divergence": round(safe_float(row['divergence']), 3),
                "grace_z": round(safe_float(row['grace_z']), 3),
                "well_z_interpolated": round(safe_float(row['well_z_interpolated']), 3),
                "tws": round(safe_float(row['tws']), 3)
            })
        statistics = {
            "mean_divergence": round(safe_float(grace_df['divergence'].mean()), 3),
            "positive_divergence_pixels": int((grace_df['divergence'] > 0).sum()),
            "negative_divergence_pixels": int((grace_df['divergence'] < 0).sum()),
            "max_divergence": round(safe_float(grace_df['divergence'].max()), 3),
            "min_divergence": round(safe_float(grace_df['divergence'].min()), 3),
            "total_pixels": len(grace_df),
            "mean_abs_divergence": round(safe_float(grace_df['divergence'].abs().mean()), 3)
        }
        
        # Calculate enhanced metrics
        total_pixels = len(grace_df)
        positive_div_pct = round((statistics['positive_divergence_pixels'] / total_pixels) * 100, 1) if total_pixels > 0 else 0
        negative_div_pct = round((statistics['negative_divergence_pixels'] / total_pixels) * 100, 1) if total_pixels > 0 else 0
        mean_abs_div = statistics['mean_abs_divergence']
        
        # Determine divergence quality
        if mean_abs_div > 1.0:
            divergence_rating = "High"
            context = "significant mismatch between satellite and ground observations requiring investigation"
            data_quality = "POOR"
        elif mean_abs_div > 0.5:
            divergence_rating = "Moderate"
            context = "some divergence present, indicating localized discrepancies"
            data_quality = "MODERATE"
        else:
            divergence_rating = "Low"
            context = "good agreement between GRACE satellite and well measurements"
            data_quality = "GOOD"
        return {
            "module": "GRACE_DIVERGENCE",
            "description": "Divergence between GRACE satellite and ground measurements",
            "filters": {"state": state, "district": district, "year": year, "month": month},
            "statistics": {
                "mean_divergence": round(safe_float(grace_df['divergence'].mean()), 3),
                "positive_divergence_pixels": int((grace_df['divergence'] > 0).sum()),
                "negative_divergence_pixels": int((grace_df['divergence'] < 0).sum()),
                "max_divergence": round(safe_float(grace_df['divergence'].max()), 3),
                "min_divergence": round(safe_float(grace_df['divergence'].min()), 3)
            },
            "count": len(results),
            "data": results,
            
            # ✅ ADDED
            "methodology": {
                "approach": "Pixel-level difference between satellite and ground observations",
                "steps": [
                    f"1. Retrieved GRACE TWS data for {year}-{month:02d} ({len(grace_df)} pixels)",
                    f"2. Retrieved well GWL data within 60 days of target date ({len(near_wells)} sites)",
                    "3. Standardized both datasets to z-scores (mean=0, std=1)",
                    "4. Interpolated well z-scores to GRACE pixel locations",
                    "5. Calculated divergence = z_GRACE − z_GWL for each pixel"
                ],
                "formula": "divergence = z_GRACE − z_well_interpolated",
                "scientific_basis": "Divergence analysis identifies spatial mismatches between satellite and ground observations, revealing data quality issues or hydrogeological anomalies."
            },
            
            # ✅ ENHANCED INTERPRETATION
            "interpretation": {
                "divergence_meaning": {
                    "positive": "GRACE shows more water than wells indicate (possible measurement mismatch)",
                    "negative": "Wells show more water than GRACE indicates (local vs regional difference)",
                    "near_zero": "Good agreement between satellite and ground"
                },
                "overall_agreement": data_quality,
                "regional_narrative": f"This region exhibits {divergence_rating.lower()} divergence ({mean_abs_div} mean absolute) with {context}.",
                "spatial_distribution": {
                    "positive_divergence_areas": f"{positive_div_pct}% of region (GRACE > GWL)",
                    "negative_divergence_areas": f"{negative_div_pct}% of region (GRACE < GWL)",
                    "assessment": "localized anomalies" if positive_div_pct < 30 else "widespread divergence"
                },
                "comparative_context": {
                    "data_agreement": "strong" if mean_abs_div < 0.5 else ("moderate" if mean_abs_div < 1.0 else "weak"),
                    "anomaly_distribution": f"{positive_div_pct}% positive vs {negative_div_pct}% negative"
                }
            },
            
            # ✅ ENHANCED KEY INSIGHTS
            "key_insights": [
                f"🎯 Divergence Assessment: {divergence_rating} ({mean_abs_div} mean absolute) - {context}",
                f"📊 Spatial Pattern: {positive_div_pct}% positive divergence, {negative_div_pct}% negative divergence",
                f"🛰️ Data Quality: {data_quality} agreement between GRACE and ground wells",
                f"📏 Analysis Coverage: {total_pixels} pixels analyzed across region"
            ]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Divergence Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
# =============================================================================
# ADVANCED MODULE 5: GWL FORECASTING
# =============================================================================
@app.get("/api/advanced/forecast")
def gwl_forecast_with_grace(
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    forecast_months: int = Query(12, ge=1, le=24),
    k_neighbors: int = Query(8, ge=3, le=20),
    grid_resolution: int = Query(50, ge=20, le=100)
):
    """
    GWL Forecasting using Wells + GRACE
    
    Method:
    1. Build monthly site matrix from wells
    2. For each grid cell, compute neighbor-weighted composite GWL series
    3. Deseasonalize GWL (extract trend+residual)
    4. Extract colocated GRACE series and deseasonalize
    5. OLS regression: y = a + b*t + c*gz (trend + GRACE anomaly)
    6. Forecast future trend + add back GWL seasonality
    7. Return Δ12m (change from last observed to forecast_months ahead)
    """
    try:
        print(f"\n🔮 Starting GRACE-integrated forecast: state={state}, district={district}")
        
        # Get boundary
        boundary_geojson = get_boundary_geojson(state, district)
        if not boundary_geojson:
            raise HTTPException(status_code=404, detail="Boundary not found")
        
        # Fetch boundary geometry
        selection_boundary = None
        if district and state:
            query = text("""
                SELECT ST_AsGeoJSON(geometry) as geojson
                FROM district_state
                WHERE UPPER("District") = UPPER(:district)
                AND UPPER("State") = UPPER(:state)
                LIMIT 1;
            """)
            with engine.connect() as conn:
                result = conn.execute(query, {"district": district, "state": state}).fetchone()
                if result:
                    geom = shape(json.loads(result[0]))
                    selection_boundary = gpd.GeoDataFrame([{'geometry': geom}], crs="EPSG:4326")
        
        elif state:
            query = text("""
                SELECT ST_AsGeoJSON(ST_Union(geometry)) as geojson
                FROM district_state
                WHERE UPPER("State") = UPPER(:state);
            """)
            with engine.connect() as conn:
                result = conn.execute(query, {"state": state}).fetchone()
                if result:
                    geom = shape(json.loads(result[0]))
                    selection_boundary = gpd.GeoDataFrame([{'geometry': geom}], crs="EPSG:4326")
        
        # Parse bounds
        if selection_boundary is not None and not selection_boundary.empty:
            bounds = selection_boundary.total_bounds
        else:
            bounds = np.array([68, 6, 97, 36])
        
        minx, miny, maxx, maxy = bounds
        print(f"  📍 Bounds: [{minx:.3f}, {miny:.3f}, {maxx:.3f}, {maxy:.3f}]")
        
        # Prepare boundary for filtering
        if selection_boundary is not None and not selection_boundary.empty:
            union_geom = unary_union(selection_boundary.geometry)
            prepared_boundary = prep(union_geom)
        else:
            prepared_boundary = None
        
        # Load wells
        print("  📊 Loading wells data...")
        wells_df = get_wells_for_analysis(boundary_geojson)
        if wells_df.empty:
            raise HTTPException(status_code=404, detail="No well data")
        
        print(f"  ✓ Found {len(wells_df)} well observations")
        
        # Build monthly matrix
        mat = build_monthly_site_matrix(wells_df)
        if mat.shape[0] < 24:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least 24 months of data, found {mat.shape[0]}"
            )
        
        print(f"  ✓ Built matrix: {len(mat)} months × {len(mat.columns)} sites")
        
        # Create grid
        print(f"  🗺️  Creating {grid_resolution}×{grid_resolution} grid...")
        width = max(maxx - minx, 1e-6)
        height = max(maxy - miny, 1e-6)
        
        xs = np.linspace(minx, maxx, grid_resolution)
        ys = np.linspace(miny, maxy, grid_resolution)
        GX, GY = np.meshgrid(xs, ys)
        
        grid_lons = GX.ravel()
        grid_lats = GY.ravel()
        
        # Filter grid to AOI
        if prepared_boundary is not None:
            inside_mask = np.array([
                prepared_boundary.contains(Point(lon, lat)) or 
                prepared_boundary.touches(Point(lon, lat))
                for lon, lat in zip(grid_lons, grid_lats)
            ])
            grid_lons = grid_lons[inside_mask]
            grid_lats = grid_lats[inside_mask]
        
        print(f"  ✓ Grid points inside AOI: {len(grid_lons)}")
        
        if len(grid_lons) == 0:
            raise HTTPException(status_code=404, detail="No grid points inside AOI")
        
        # Get site locations
        site_locs = wells_df.groupby('site_id')[['latitude', 'longitude']].first()
        site_ids = site_locs.index.values
        site_coords = site_locs[['latitude', 'longitude']].values
        
        mat = mat.reindex(columns=site_ids)
        
        # Build KNN
        print(f"  🎯 Building KNN model (k={k_neighbors})...")
        k_actual = min(k_neighbors, len(site_ids))
        nbrs = NearestNeighbors(n_neighbors=k_actual, metric='haversine', algorithm='ball_tree')
        nbrs.fit(np.radians(site_coords))
        
        dists_rad, idx = nbrs.kneighbors(np.radians(np.column_stack([grid_lats, grid_lons])))
        dists_km = dists_rad * 6371.0
        
        # Pre-load GRACE
        print(f"  🛰️  Loading GRACE data...")
        grace_times = []
        grace_data_list = []
        
        for (year, month), band in sorted(GRACE_BAND_MAPPING.items()):
            try:
                query = text(f"""
                    SELECT
                        ST_X(ST_SetSRID((ST_PixelAsCentroids(rast, {band})).geom, 4326)) as lon,
                        ST_Y(ST_SetSRID((ST_PixelAsCentroids(rast, {band})).geom, 4326)) as lat,
                        (ST_PixelAsCentroids(rast, {band})).val as tws
                    FROM public.grace_lwe
                    WHERE rid = 1
                """)
                
                with engine.connect() as conn:
                    result = conn.execute(query)
                    grace_pixels = pd.DataFrame([
                        {"lon": r[0], "lat": r[1], "tws": r[2]}
                        for r in result if r[2] is not None
                    ])
                
                if not grace_pixels.empty:
                    tws_interp = griddata(
                        grace_pixels[['lon', 'lat']].values,
                        grace_pixels['tws'].values,
                        np.column_stack([grid_lons, grid_lats]),
                        method='nearest'
                    )
                    
                    grace_times.append(datetime(year, month, 1))
                    grace_data_list.append(tws_interp)
            
            except Exception:
                continue
        
        if not grace_data_list:
            print("  ⚠️  No GRACE data available")
            grace_available = False
        else:
            grace_df = pd.DataFrame(
                np.column_stack(grace_data_list).T,
                index=pd.DatetimeIndex(grace_times),
                columns=range(len(grid_lons))
            )
            grace_df = grace_df.resample('MS').mean()
            grace_available = True
            print(f"  ✓ GRACE loaded: {len(grace_times)} months")
        
        # Forecast
        print(f"  🔮 Forecasting for {len(grid_lons)} grid points...")
        
        results = []
        successful = 0
        
        mat_values = mat.values
        mat_times = mat.index
        
        for q in range(len(grid_lons)):
            try:
                neighbor_idx = idx[q]
                neighbor_dist_km = dists_km[q]
                
                weights = idw_weights_from_distances(neighbor_dist_km.reshape(1, -1), power=2)[0]
                
                neighbor_series = mat_values[:, neighbor_idx]
                gwl_series = np.nansum(neighbor_series * weights, axis=1)
                
                s = pd.Series(gwl_series, index=mat_times)
                s = s.dropna()
                
                if len(s) < 24:
                    continue
                
                # Deseasonalize GWL
                s_clim = s.groupby(s.index.month).mean().reindex(range(1, 13)).interpolate().bfill().ffill()
                s_season = s.index.month.map(s_clim.to_dict())
                s_ds = (s - s_season).dropna()
                
                if len(s_ds) < 18:
                    continue
                
                # Get GRACE
                if grace_available and q in grace_df.columns:
                    gz = grace_df[q].dropna()
                    
                    if len(gz) >= 12:
                        gz.index = gz.index.to_period('M').to_timestamp()
                        gz = gz.groupby(gz.index).mean()
                        
                        gz_clim = gz.groupby(gz.index.month).mean().reindex(range(1, 13)).interpolate().bfill().ffill()
                        gz_season = gz.index.month.map(gz_clim.to_dict())
                        gz_ds = (gz - gz_season).dropna()
                    else:
                        gz_ds = pd.Series(index=s_ds.index, data=0.0)
                        gz_clim = pd.Series({m: 0.0 for m in range(1, 13)})
                else:
                    gz_ds = pd.Series(index=s_ds.index, data=0.0)
                    gz_clim = pd.Series({m: 0.0 for m in range(1, 13)})
                
                # OLS
                joined = pd.concat([
                    s_ds.rename('y'),
                    gz_ds.rename('gz')
                ], axis=1).dropna()
                
                if len(joined) < 18:
                    continue
                
                t0 = joined.index.min()
                joined['t'] = (joined.index - t0).days.values
                
                X = np.column_stack([
                    np.ones(len(joined)),
                    joined['t'].values,
                    joined['gz'].values
                ])
                y = joined['y'].values
                
                try:
                    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                except Exception:
                    continue
                
                a, b, c = beta
                
                # Forecast
                last_time = s.index.max()
                future_index = pd.date_range(
                    last_time + pd.offsets.MonthBegin(1),
                    periods=forecast_months,
                    freq='MS'
                )
                
                t_future = (future_index - t0).days.values
                gz_future = np.array([gz_clim.get(m, 0.0) for m in future_index.month])
                
                y_future_ds = a + b * t_future + c * gz_future
                s_season_future = np.array([s_clim.get(m, 0.0) for m in future_index.month])
                s_future = y_future_ds + s_season_future
                
                last_gwl = s.loc[last_time]
                final_gwl = s_future[-1]
                delta_forecast = final_gwl - last_gwl
                
                # R²
                y_pred = X @ beta
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                results.append({
                    "longitude": float(grid_lons[q]),
                    "latitude": float(grid_lats[q]),
                    "pred_delta_m": round(float(delta_forecast), 3),
                    "current_gwl": round(float(last_gwl), 2),
                    "forecast_gwl": round(float(final_gwl), 2),
                    "r_squared": round(float(r_squared), 3),
                    "trend_component": round(float(b * t_future[-1]), 3),
                    "grace_component": round(float(c * gz_future[-1]), 3) if grace_available else 0.0,
                    "n_months_training": len(joined)
                })
                
                successful += 1
                
                if successful % 100 == 0:
                    print(f"    ✓ Processed {successful}/{len(grid_lons)} points")
            
            except Exception as e:
                if successful < 3:
                    print(f"    ⚠️  Grid point {q} error: {str(e)[:100]}")
                continue
        
        print(f"  ✅ Forecast complete: {successful}/{len(grid_lons)} cells successful")
        
        if not results:
            raise HTTPException(
                status_code=500,
                detail="Forecast failed for all grid points"
            )
        
        results_df = pd.DataFrame(results)
        statistics = {
            "mean_change_m": round(float(results_df['pred_delta_m'].mean()), 3),
            "median_change_m": round(float(results_df['pred_delta_m'].median()), 3),
            "declining_cells": int((results_df['pred_delta_m'] > 0).sum()),
            "recovering_cells": int((results_df['pred_delta_m'] < 0).sum()),
            "mean_r_squared": round(float(results_df['r_squared'].mean()), 3),
            "mean_grace_contribution": round(float(results_df['grace_component'].mean()), 3) if grace_available else 0.0,
            "success_rate": round(successful / len(grid_lons) * 100, 1),
            "total_cells": len(results)
        }
        
        # Calculate enhanced metrics
        total_cells = len(results)
        declining_cells = statistics['declining_cells']
        recovering_cells = statistics['recovering_cells']
        mean_r2 = statistics['mean_r_squared']
        mean_grace_contrib = statistics['mean_grace_contribution']
        
        declining_pct = round((declining_cells / total_cells) * 100, 1) if total_cells > 0 else 0
        recovering_pct = round((recovering_cells / total_cells) * 100, 1) if total_cells > 0 else 0
        high_confidence = mean_r2 > 0.7
        
        # Determine forecast quality
        if mean_r2 > 0.7:
            forecast_rating = "Excellent"
            confidence_context = "high confidence predictions suitable for long-term planning"
        elif mean_r2 > 0.5:
            forecast_rating = "Good"
            confidence_context = "reliable predictions with moderate confidence"
        elif mean_r2 > 0.3:
            forecast_rating = "Moderate"
            confidence_context = "fair predictions requiring validation"
        else:
            forecast_rating = "Poor"
            confidence_context = "low confidence predictions, use with caution"        
        return {
            "module": "FORECAST_GWL_GRACE",
            "description": f"GWL forecast ({forecast_months}-month) using trend + GRACE",
            "method": "OLS with deseasonalized trend + GRACE anomaly",
            "filters": {"state": state, "district": district},
            "parameters": {
                "forecast_months": forecast_months,
                "k_neighbors": k_neighbors,
                "grid_resolution": grid_resolution,
                "grace_used": grace_available
            },
            "statistics": {
                "mean_change_m": round(float(results_df['pred_delta_m'].mean()), 3),
                "median_change_m": round(float(results_df['pred_delta_m'].median()), 3),
                "declining_cells": int((results_df['pred_delta_m'] > 0).sum()),
                "recovering_cells": int((results_df['pred_delta_m'] < 0).sum()),
                "mean_r_squared": round(float(results_df['r_squared'].mean()), 3),
                "mean_grace_contribution": round(float(results_df['grace_component'].mean()), 3) if grace_available else 0.0,
                "success_rate": round(successful / len(grid_lons) * 100, 1)
            },
            "count": len(results),
            "data": results_df.to_dict("records"),
            
            # ✅ ADDED
            "methodology": {
                "approach": "OLS regression on deseasonalized GWL with GRACE anomaly",
                "model": "y_future = a + b×time + c×GRACE_anomaly",
                "steps": [
                    f"1. Built monthly GWL matrix from {len(mat)} months of well data",
                    f"2. Created {grid_resolution}×{grid_resolution} grid ({len(grid_lons)} cells in AOI)",
                    f"3. For each cell, computed neighbor-weighted GWL using {k_neighbors} nearest wells",
                    "4. Deseasonalized GWL by removing monthly climatology (extract trend+residual)",
                    f"5. {'Interpolated GRACE TWS and deseasonalized' if grace_available else 'No GRACE data available'}",
                    f"6. Fit OLS: y = a + b×t + c×GRACE, forecast {forecast_months} months ahead",
                    "7. Added back GWL seasonality to forecast"
                ],
                "seasonality_handling": "Monthly climatology removed before trend fitting, added back for forecast",
                "scientific_basis": "OLS regression captures linear trend and climate signal (GRACE) to project future groundwater levels, accounting for seasonal variations."
            },
            
            # ✅ ENHANCED INTERPRETATION
            "interpretation": {
                "change_meaning": {
                    "positive": "Declining (water levels getting deeper)",
                    "negative": "Recovering (water levels getting shallower)"
                },
                "confidence": forecast_rating,
                "regional_narrative": f"Forecast model exhibits {forecast_rating.lower()} performance (R²: {mean_r2}) with {confidence_context}.",
                "trend_distribution": {
                    "declining_areas": f"{declining_pct}% ({declining_cells} cells) showing negative trends",
                    "recovering_areas": f"{recovering_pct}% ({recovering_cells} cells) showing recovery",
                    "stable_areas": f"{round(100 - declining_pct - recovering_pct, 1)}% stable"
                },
                "model_quality": {
                    "prediction_confidence": forecast_rating,
                    "r_squared": mean_r2,
                    "dominant_driver": "climate (GRACE)" if abs(mean_grace_contrib) > 0.5 else "local factors (trend)"
                },
                "grace_impact": f"GRACE contributes avg {mean_grace_contrib}m ({round(abs(mean_grace_contrib)/abs(statistics['mean_change_m'])*100 if statistics['mean_change_m'] != 0 else 0, 1)}% of total change)" if grace_available else "No GRACE data used",
                "comparative_context": {
                    "grace_contribution": f"{round(abs(mean_grace_contrib) * 100 / (abs(statistics['mean_change_m']) if statistics['mean_change_m'] != 0 else 1), 1)}% of variance explained by GRACE" if grace_available else "0%",
                    "forecast_reliability": "suitable for planning" if high_confidence else "requires validation"
                }
            },
            
            # ✅ ENHANCED KEY INSIGHTS
            "key_insights": [
                f"🎯 Forecast Quality: {forecast_rating} (R²: {mean_r2}) - {confidence_context}",
                f"📉 Declining Trend: {declining_cells} cells ({declining_pct}%) showing negative GWL trends over {forecast_months} months",
                f"📈 Recovery: {recovering_cells} cells ({recovering_pct}%) showing positive trends",
                f"🛰️ GRACE Contribution: {round(abs(mean_grace_contrib), 2)}m - {'climate-driven' if abs(mean_grace_contrib) > 0.5 else 'locally-driven'}" if grace_available else "🛰️ No GRACE data - trend-only forecast",
                f"📏 Model Coverage: {total_cells} grid cells with forecasts ({statistics['success_rate']}% success rate)"
            ]
        }
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Forecast error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
# =============================================================================
# ADVANCED MODULE 6: RECHARGE STRUCTURE RECOMMENDATION
# =============================================================================
@app.get("/api/advanced/recharge-planning")
def recharge_planning(
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    year: Optional[int] = Query(None, ge=1994, le=2024),
    month: Optional[int] = Query(None, ge=1, le=12)
):
    """
    Calculate MAR (Managed Aquifer Recharge) potential and recommend structure types
    """
    try:
        boundary_geojson = get_boundary_geojson(state, district)
        
        # Get aquifer properties
        aquifers_df = get_aquifer_properties(boundary_geojson)
        if aquifers_df.empty:
            raise HTTPException(status_code=404, detail="No aquifer data")
        
        total_area_m2 = aquifers_df['area_m2'].sum()
        total_area_km2 = total_area_m2 / 1_000_000
        
        # Determine dominant lithology using mode (most common value)
        dominant_lithology = 'alluvium'
        if not aquifers_df.empty and 'majoraquif' in aquifers_df.columns:
            lithology_series = aquifers_df['majoraquif'].astype(str).str.lower()
            mode_values = lithology_series.mode()
            if len(mode_values) > 0:
                dominant_lithology = mode_values.iloc[0]
                # Extract key lithology type
                if 'alluvium' in dominant_lithology:
                    dominant_lithology = 'alluvium'
                elif 'sandstone' in dominant_lithology:
                    dominant_lithology = 'sandstone'
                elif 'limestone' in dominant_lithology:
                    dominant_lithology = 'limestone'
                elif 'basalt' in dominant_lithology:
                    dominant_lithology = 'basalt'
                elif 'granite' in dominant_lithology:
                    dominant_lithology = 'granite'
                else:
                    dominant_lithology = 'alluvium'
        
        # Runoff coefficients by lithology (matching Dash: C_map)
        runoff_coeff_map = {
            'alluvium': 0.35,
            'sandstone': 0.30,
            'limestone': 0.28,
            'basalt': 0.25,
            'granite': 0.20
        }
        runoff_coeff = runoff_coeff_map.get(dominant_lithology, 0.25)
        
        # Get monsoon rainfall (matching Dash implementation)
        rainfall_m = 0.3  # Default 300mm monsoon
        if year and year in RAINFALL_TABLES:
            use_year = year
        else:
            use_year = max(RAINFALL_TABLES.keys()) if RAINFALL_TABLES else 2023
        
        try:
            table_name = f"rainfall_{use_year}"
            monsoon_months = [6, 7, 8, 9]  # June-September
            
            total_rainfall_mm = 0
            days_count = 0
            
            with engine.connect() as conn:
                for month_num in monsoon_months:
                    first_day, last_day = get_month_day_range(use_year, month_num)
                    
                    for doy in range(first_day, last_day + 1):
                        query = text(f"""
                            SELECT AVG((ST_PixelAsCentroids(rast, {doy})).val) as avg_val
                            FROM {table_name}
                            WHERE rid = 1
                        """)
                        result = conn.execute(query).fetchone()
                        if result[0]:
                            total_rainfall_mm += float(result[0])
                            days_count += 1
            
            # Calculate average and convert to meters
            if days_count > 0:
                rainfall_m = (total_rainfall_mm / days_count) / 1000.0
        except Exception:
            rainfall_m = 0.3  # Default 300mm monsoon
        
        # Calculate potential recharge (matching Dash formula exactly)
        capture_fraction = 0.15  # 15% capture efficiency (f_capture)
        potential_recharge_mcm = (total_area_m2 / 1_000_000) * rainfall_m * runoff_coeff * capture_fraction
        
        # Structure recommendations (matching Dash: struct_cap_m3 and allocation)
        struct_cap_m3 = {
            'Percolation tank': 50000,
            'Check dam': 30000,
            'Recharge shaft': 500,
            'Farm pond': 1000,
            'Gabion': 2500
        }
        
        # Structure mix and allocation fractions (matching Dash exactly)
        mix = ['Percolation tank', 'Check dam', 'Recharge shaft', 'Farm pond', 'Gabion']
        alloc_frac = [0.3, 0.3, 0.2, 0.15, 0.05]
        
        structure_plan = []
        for sname, frac in zip(mix, alloc_frac):
            vol_req_m3 = potential_recharge_mcm * 1_000_000 * frac
            n_units = max(1, int(vol_req_m3 / struct_cap_m3[sname])) if struct_cap_m3[sname] > 0 else 0
            
            structure_plan.append({
                "structure_type": sname,
                "recommended_units": n_units,
                "total_capacity_mcm": round(n_units * struct_cap_m3[sname] / 1_000_000, 3),
                "allocation_fraction": frac
            })
        
        # Site-specific recommendations based on stress (matching Dash logic)
        wells_df = get_wells_for_analysis(boundary_geojson)
        site_recommendations = []
        
        if not wells_df.empty and year and month:
            # Match Dash: target_date and tolerance
            target_date = pd.to_datetime(f"{year}-{month:02d}-15")
            tolerance = pd.Timedelta(days=60)
            
            wells_df = wells_df.copy()
            wells_df['dt_diff'] = (wells_df['date'] - target_date).abs()
            
            # Filter wells within tolerance and get closest reading per site
            near_wells = wells_df[wells_df['dt_diff'] <= tolerance].sort_values('dt_diff').groupby('site_id').head(1)
            
            if not near_wells.empty:
                # Calculate stress using negative z-score (matching Dash: -zscore_series)
                gwl_values = near_wells['gwl'].values
                gwl_mean = gwl_values.mean()
                gwl_std = gwl_values.std()
                
                if gwl_std > 0:
                    stress = -(gwl_values - gwl_mean) / gwl_std
                else:
                    stress = np.zeros_like(gwl_values)
                
                # Categorize stress (matching Dash bins exactly)
                stress_categories = pd.cut(
                    stress,
                    bins=[-np.inf, -0.5, 0, 0.5, np.inf],
                    labels=['Critical', 'Stressed', 'Moderate', 'Healthy']
                )
                
                # Recommendation map (matching Dash: rec_map)
                rec_map = {
                    'Critical': 'Recharge shaft',
                    'Stressed': 'Check dam',
                    'Moderate': 'Farm pond',
                    'Healthy': 'Percolation tank'
                }
                
                for idx, (_, row) in enumerate(near_wells.iterrows()):
                    category = str(stress_categories[idx])
                    site_recommendations.append({
                        "site_id": row['site_id'],
                        "latitude": row['latitude'],
                        "longitude": row['longitude'],
                        "stress_category": category,
                        "recommended_structure": rec_map.get(category, 'Percolation tank'),
                        "current_gwl": round(row['gwl'], 2)
                    })
        
        return {
            "module": "RECHARGE_PLANNING",
            "description": "MAR potential and structure recommendations",
            "filters": {"state": state, "district": district, "year": year, "month": month},
            "analysis_parameters": {
                "area_km2": round(total_area_km2, 2),
                "dominant_lithology": dominant_lithology,
                "runoff_coefficient": runoff_coeff,
                "monsoon_rainfall_m": round(rainfall_m, 3),
                "capture_fraction": capture_fraction,
                "year_analyzed": use_year
            },
            "potential": {
                "total_recharge_potential_mcm": round(potential_recharge_mcm, 2),
                "per_km2_mcm": round(potential_recharge_mcm / total_area_km2, 4) if total_area_km2 > 0 else 0
            },
            "structure_plan": structure_plan,
            "site_recommendations": site_recommendations[:100],
            "count": len(site_recommendations),
            
            # Calculate enhanced metrics for interpretation
            "_enhanced_metrics": {
                "total_structures": sum(s['recommended_units'] for s in structure_plan),
                "total_capacity_mcm": sum(s['total_capacity_mcm'] for s in structure_plan),
                "per_km2_potential": round(potential_recharge_mcm / total_area_km2, 4) if total_area_km2 > 0 else 0,
                "critical_sites": sum(1 for s in site_recommendations if s['stress_category'] == 'Critical'),
                "stressed_sites": sum(1 for s in site_recommendations if s['stress_category'] == 'Stressed')
            },
            
            # ✅ ADDED
            "methodology": {
                "approach": "Runoff-based MAR potential with structure allocation",
                "calculation_steps": [
                    f"1. Region area: {round(total_area_km2, 2)} km²",
                    f"2. Identified dominant lithology: {dominant_lithology} (from aquifer database)",
                    f"3. Applied runoff coefficient: {runoff_coeff} (lithology-based)",
                    f"4. Estimated monsoon rainfall: {round(rainfall_m, 3)}m (from {use_year} data)",
                    f"5. Assumed capture efficiency: {capture_fraction*100}% (typical for MAR structures)"
                ],
                "formula": "Potential (MCM) = Area × Rainfall × Runoff_Coeff × Capture_Fraction",
                "full_calculation": f"{round(total_area_km2, 2)} km² × {round(rainfall_m, 3)}m × {runoff_coeff} × {capture_fraction} = {round(potential_recharge_mcm, 2)} MCM",
                "scientific_basis": "Runoff-based MAR potential estimates capture efficiency of artificial structures in augmenting groundwater recharge during monsoon."
            },
            
            # ✅ ENHANCED INTERPRETATION
            "interpretation": {
                "structure_types": {
                    "Percolation tank": "Large capacity (50,000 m³), suitable for high-ASI alluvial zones",
                    "Check dam": "Medium capacity (30,000 m³), good for seasonal streams",
                    "Recharge shaft": "Small capacity (500 m³), for stressed urban sites",
                    "Farm pond": "Small capacity (1,000 m³), distributed rural recharge",
                    "Gabion": "Very small (2,500 m³), erosion control + recharge"
                },
                "allocation_strategy": "30% percolation tanks, 30% check dams, 20% shafts, 15% ponds, 5% gabions",
                "regional_narrative": f"Region shows {round(potential_recharge_mcm, 2)} MCM/year recharge potential across {round(total_area_km2, 2)} km² with {dominant_lithology} geology (runoff coeff: {runoff_coeff}).",
                "potential_distribution": {
                    "total_potential_mcm": round(potential_recharge_mcm, 2),
                    "per_km2_mcm": round(potential_recharge_mcm / total_area_km2, 4) if total_area_km2 > 0 else 0,
                    "implementation_scale": "large-scale" if potential_recharge_mcm > 100 else ("medium-scale" if potential_recharge_mcm > 20 else "small-scale"),
                    "monsoon_dependency": f"{round(rainfall_m*1000, 0)}mm monsoon rainfall"
                },
                "implementation_context": {
                    "structure_count": sum(s['recommended_units'] for s in structure_plan),
                    "dominant_type": max(structure_plan, key=lambda x: x['allocation_fraction'])['structure_type'] if structure_plan else "N/A",
                    "site_specific_count": len(site_recommendations),
                    "critical_priority_sites": sum(1 for s in site_recommendations if s['stress_category'] == 'Critical')
                },
                "comparative_context": {
                    "intensity": f"{round(potential_recharge_mcm / total_area_km2, 4)} MCM/km²",
                    "lithology_suitability": "excellent" if runoff_coeff >= 0.3 else ("good" if runoff_coeff >= 0.25 else "moderate"),
                    "rainfall_adequacy": "high" if rainfall_m > 0.4 else ("moderate" if rainfall_m > 0.25 else "limited")
                }
            },
            
            # ✅ ENHANCED KEY INSIGHTS
            "key_insights": [
                f"💧 Recharge Potential: {round(potential_recharge_mcm, 2)} MCM/year across {round(total_area_km2, 2)} km² ({round(potential_recharge_mcm / total_area_km2, 4) if total_area_km2 > 0 else 0} MCM/km²)",
                f"🏗️ Structure Plan: {sum(s['recommended_units'] for s in structure_plan)} total units - {max(structure_plan, key=lambda x: x['allocation_fraction'])['structure_type'] if structure_plan else 'N/A'} dominant (30%)",
                f"🌍 Geology: {dominant_lithology.capitalize()} with {runoff_coeff} runoff coefficient - {'excellent' if runoff_coeff >= 0.3 else 'good'} MAR suitability",
                f"📍 Site Priorities: {sum(1 for s in site_recommendations if s['stress_category'] == 'Critical')} critical sites need immediate recharge shafts" if site_recommendations else "📍 Run SASS module for stress-based site prioritization",
                f"🌧️ Monsoon Dependency: {round(rainfall_m*1000, 0)}mm rainfall - {'high' if rainfall_m > 0.4 else 'moderate'} recharge adequacy"
            ]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Recharge Planning Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
# =============================================================================
# ADVANCED MODULE 7: SIGNIFICANT TRENDS
# =============================================================================

@app.get("/api/advanced/significant-trends")
def significant_trends(
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    p_threshold: float = Query(0.1, ge=0.01, le=0.2)
):
    """
    Identify sites with statistically significant trends (Mann-Kendall or OLS)
    """
    try:
        boundary_geojson = get_boundary_geojson(state, district)
        wells_df = get_wells_for_analysis(boundary_geojson)
        
        if wells_df.empty:
            raise HTTPException(status_code=404, detail="No well data")
        
        significant_sites = []
        
        for site_id, group in wells_df.groupby('site_id'):
            if len(group) < 12:
                continue
            
            group = group.sort_values('date')
            series = group.set_index('date')['gwl'].resample('MS').mean().dropna()
            
            if len(series) < 12:
                continue
            
            # Try Mann-Kendall if available
            if MK_AVAILABLE:
                try:
                    result = mk.original_test(series.values)
                    slope = result.slope
                    p_value = result.p
                except:
                    # Fallback to OLS
                    x = np.arange(len(series))
                    slope, _, _, p_value, _ = linregress(x, series.values)
                    slope = slope * 12  # Convert to annual
            else:
                # OLS only
                x = np.arange(len(series))
                slope, _, _, p_value, _ = linregress(x, series.values)
                slope = slope * 12  # Convert to annual
            
            if p_value < p_threshold:
                significant_sites.append({
                    "site_id": site_id,
                    "latitude": group['latitude'].iloc[0],
                    "longitude": group['longitude'].iloc[0],
                    "slope_m_per_year": round(float(slope), 4),
                    "p_value": round(float(p_value), 4),
                    "trend_direction": "declining" if slope > 0 else "recovering",
                    "significance_level": "high" if p_value < 0.01 else ("medium" if p_value < 0.05 else "low"),
                    "n_months": len(series),
                    "date_range": f"{series.index.min().date()} to {series.index.max().date()}"
                })
        
        if not significant_sites:
            return {
                "module": "SIGNIFICANT_TRENDS",
                "description": f"No sites with significant trends (p < {p_threshold})",
                "filters": {"state": state, "district": district},
                "parameters": {"p_threshold": p_threshold},
                "count": 0,
                "data": []
            }
        
        sites_df = pd.DataFrame(significant_sites)
        
        return {
            "module": "SIGNIFICANT_TRENDS",
            "description": f"Sites with statistically significant trends (p < {p_threshold})",
            "filters": {"state": state, "district": district},
            "parameters": {
                "p_threshold": p_threshold,
                "method": "Mann-Kendall" if MK_AVAILABLE else "OLS",
                "min_months_required": 12
            },
            "statistics": {
                "total_significant": len(significant_sites),
                "declining": int((sites_df['slope_m_per_year'] > 0).sum()),
                "recovering": int((sites_df['slope_m_per_year'] < 0).sum()),
                "mean_slope": round(float(sites_df['slope_m_per_year'].mean()), 4),
                "high_significance": int((sites_df['p_value'] < 0.01).sum())
            },
            "count": len(significant_sites),
            "data": significant_sites,
            
            # Calculate enhanced metrics
            "_enhanced_metrics": {
                "total_sites": len(significant_sites),
                "declining_pct": round((sites_df['slope_m_per_year'] > 0).sum() / len(sites_df) * 100, 1) if len(sites_df) > 0 else 0,
                "recovering_pct": round((sites_df['slope_m_per_year'] < 0).sum() / len(sites_df) * 100, 1) if len(sites_df) > 0 else 0,
                "severe_decline": int((sites_df['slope_m_per_year'] > 0.5).sum()),
                "high_conf_sites": int((sites_df['p_value'] < 0.01).sum()),
                "mean_abs_slope": round(float(sites_df['slope_m_per_year'].abs().mean()), 4)
            },
            
            # ✅ ADDED
            "methodology": {
                "approach": f"{'Mann-Kendall trend test' if MK_AVAILABLE else 'OLS linear regression'} with significance testing",
                "steps": [
                    "1. Filtered wells to sites with ≥12 monthly observations",
                    "2. Resampled each site to monthly mean GWL",
                    f"3. Applied {'Mann-Kendall test' if MK_AVAILABLE else 'linear regression'} to detect monotonic trends",
                    f"4. Retained only sites with p-value < {p_threshold} (significant trends)",
                    "5. Classified as declining (slope > 0) or recovering (slope < 0)"
                ],
                "significance_levels": {
                    "p < 0.01": "High significance (99% confidence)",
                    "p < 0.05": "Medium significance (95% confidence)",
                    f"p < {p_threshold}": "Low significance (threshold for inclusion)"
                },
                "scientific_basis": f"{'Mann-Kendall non-parametric test detects monotonic trends without assuming data normality' if MK_AVAILABLE else 'Linear regression identifies long-term directional changes in groundwater levels'}"
            },
            
            # ✅ ENHANCED INTERPRETATION
            "interpretation": {
                "slope_meaning": {
                    "positive": "Declining (GWL getting deeper over time)",
                    "negative": "Recovering (GWL getting shallower over time)"
                },
                "trend_strength": f"Mean slope: {round(float(sites_df['slope_m_per_year'].mean()), 3)}m/year",
                "regional_narrative": f"Region exhibits {'CRITICAL' if (sites_df['slope_m_per_year'] > 0).sum() / len(sites_df) > 0.7 else ('CONCERNING' if (sites_df['slope_m_per_year'] > 0).sum() / len(sites_df) > 0.5 else 'STABLE')} trends with {round((sites_df['slope_m_per_year'] > 0).sum() / len(sites_df) * 100, 1)}% of significant sites showing decline.",
                "trend_distribution": {
                    "declining_sites": f"{int((sites_df['slope_m_per_year'] > 0).sum())} ({round((sites_df['slope_m_per_year'] > 0).sum() / len(sites_df) * 100, 1)}%)",
                    "recovering_sites": f"{int((sites_df['slope_m_per_year'] < 0).sum())} ({round((sites_df['slope_m_per_year'] < 0).sum() / len(sites_df) * 100, 1)}%)",
                    "severe_decline": f"{int((sites_df['slope_m_per_year'] > 0.5).sum())} sites > 0.5m/year",
                    "assessment": "system-wide decline" if (sites_df['slope_m_per_year'] > 0).sum() / len(sites_df) > 0.7 else "localized variability"
                },
                "confidence_analysis": {
                    "high_confidence": f"{int((sites_df['p_value'] < 0.01).sum())} sites (p<0.01)",
                    "medium_confidence": f"{int((sites_df['p_value'] < 0.05).sum() - (sites_df['p_value'] < 0.01).sum())} sites (0.01<p<0.05)",
                    "method_reliability": "excellent" if MK_AVAILABLE else "good"
                },
                "comparative_context": {
                    "mean_decline_rate": f"{round(float(sites_df[sites_df['slope_m_per_year'] > 0]['slope_m_per_year'].mean()), 3) if (sites_df['slope_m_per_year'] > 0).any() else 0}m/year",
                    "trend_severity": "SEVERE" if abs(sites_df['slope_m_per_year'].mean()) > 0.3 else ("MODERATE" if abs(sites_df['slope_m_per_year'].mean()) > 0.1 else "MILD"),
                    "sustainability_outlook": "unsustainable" if (sites_df['slope_m_per_year'] > 0).sum() / len(sites_df) > 0.7 else "manageable"
                }
            },
            
            # ✅ ENHANCED KEY INSIGHTS
            "key_insights": [
                f"📊 Trend Analysis: {len(significant_sites)} sites with statistically significant trends (p<{p_threshold}) - {round((sites_df['slope_m_per_year'] > 0).sum() / len(sites_df) * 100, 1)}% declining",
                f"📉 Declining Trend: {int((sites_df['slope_m_per_year'] > 0).sum())} sites showing long-term water level decline - {int((sites_df['slope_m_per_year'] > 0.5).sum())} SEVERE (>0.5m/year)",
                f"📈 Recovery: {int((sites_df['slope_m_per_year'] < 0).sum())} sites showing recovery trends - positive signs in {round((sites_df['slope_m_per_year'] < 0).sum() / len(sites_df) * 100, 1)}% of region",
                f"🎯 Confidence: {int((sites_df['p_value'] < 0.01).sum())} high-confidence sites (99%) using {'Mann-Kendall' if MK_AVAILABLE else 'OLS'} method",
                f"⚖️ System Health: Mean trend {round(float(sites_df['slope_m_per_year'].mean()), 3)}m/year - {'UNSUSTAINABLE' if (sites_df['slope_m_per_year'] > 0).sum() / len(sites_df) > 0.7 else 'CONCERNING' if (sites_df['slope_m_per_year'] > 0).sum() / len(sites_df) > 0.5 else 'STABLE'} outlook"
            ]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Significant Trends Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# ADVANCED MODULE 8: CHANGEPOINT DETECTION
# =============================================================================

@app.get("/api/advanced/changepoints")
def changepoint_detection(
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    penalty: float = Query(5.0, ge=1.0, le=20.0)
):
    """
    Detect structural breaks in GWL time series using PELT algorithm
    NOW RETURNS TWO MAPS: Changepoints + Coverage Diagnostics
    """
    if not RUPTURES_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Changepoint detection requires 'ruptures' package"
        )
    
    try:
        boundary_geojson = get_boundary_geojson(state, district)
        wells_df = get_wells_for_analysis(boundary_geojson)
        
        if wells_df.empty:
            raise HTTPException(status_code=404, detail="No well data")
        
        changepoint_sites = []
        coverage_sites = []
        analyzed_sites = []
        
        for site_id, group in wells_df.groupby('site_id'):
            group = group.sort_values('date')
            series = group.set_index('date')['gwl'].resample('MS').mean().interpolate().dropna()
            
            # ✅ COVERAGE INFO FOR ALL SITES (regardless of length)
            if len(series) > 0:
                coverage_sites.append({
                    "site_id": site_id,
                    "latitude": float(group['latitude'].iloc[0]),
                    "longitude": float(group['longitude'].iloc[0]),
                    "n_months": int(len(series)),
                    "date_start": series.index.min().date().isoformat(),
                    "date_end": series.index.max().date().isoformat(),
                    "span_years": round((series.index.max() - series.index.min()).days / 365.25, 1),
                    "analyzed": len(series) >= 24  # Flag if meets threshold
                })
            
            # ✅ CHANGEPOINT DETECTION (only for sites with ≥24 months)
            if len(series) < 24:
                continue
            
            analyzed_sites.append(site_id)
            
            try:
                algo = rpt.Pelt(model="l2").fit(series.values)
                breakpoints = algo.predict(pen=penalty)
                
                if len(breakpoints) > 1:
                    first_bp_idx = breakpoints[0]
                    if first_bp_idx < len(series):
                        first_bp_date = series.index[min(first_bp_idx, len(series) - 1)]
                        
                        changepoint_sites.append({
                            "site_id": site_id,
                            "latitude": float(group['latitude'].iloc[0]),
                            "longitude": float(group['longitude'].iloc[0]),
                            "changepoint_date": first_bp_date.date().isoformat(),
                            "changepoint_year": int(first_bp_date.year),
                            "changepoint_month": int(first_bp_date.month),
                            "n_breakpoints": len(breakpoints) - 1,
                            "all_breakpoints": [
                                series.index[min(bp, len(series)-1)].date().isoformat() 
                                for bp in breakpoints[:-1]
                            ],
                            "series_length": int(len(series))
                        })
            except Exception as e:
                print(f"Changepoint detection failed for {site_id}: {e}")
                continue
        
        # ✅ CALCULATE STATISTICS
        num_analyzed = len(analyzed_sites)
        detection_rate = round(len(changepoint_sites) / num_analyzed * 100, 1) if num_analyzed > 0 else 0
        
        # ✅ BUILD RESPONSE WITH BOTH MAPS
        return {
            "module": "CHANGEPOINTS",
            "description": "Structural break detection using PELT algorithm + Coverage Diagnostics",
            "filters": {"state": state, "district": district},
            "parameters": {
                "penalty": penalty,
                "algorithm": "PELT",
                "model": "l2 (mean shift detection)",
                "min_months_required": 24
            },
            
            # ✅ OVERALL STATISTICS
            "statistics": {
                "total_sites": len(coverage_sites),
                "sites_analyzed": num_analyzed,
                "sites_with_changepoints": len(changepoint_sites),
                "detection_rate": detection_rate,
                "avg_series_length": round(np.mean([s['n_months'] for s in coverage_sites]), 1) if coverage_sites else 0,
                "avg_span_years": round(np.mean([s['span_years'] for s in coverage_sites]), 1) if coverage_sites else 0,
                "sites_insufficient_data": len(coverage_sites) - num_analyzed
            },
            
            # ✅ MAP 1: CHANGEPOINTS (year of first break)
            "changepoints": {
                "count": len(changepoint_sites),
                "data": changepoint_sites,
                "description": "Sites with detected structural breaks (first breakpoint shown)"
            },
            
            # ✅ MAP 2: COVERAGE (months/span per site)
            "coverage": {
                "count": len(coverage_sites),
                "data": coverage_sites,  # Limit to 200 for performance
                "description": "Data coverage per site (shows power to detect changepoints)"
            },
            
            # ✅ METHODOLOGY
            "methodology": {
                "approach": "PELT (Pruned Exact Linear Time) algorithm for changepoint detection",
                "steps": [
                    f"1. Analyzed {len(coverage_sites)} wells in region",
                    f"2. Filtered to {num_analyzed} sites with ≥24 months of data",
                    "3. Resampled each site to monthly mean GWL and interpolated gaps",
                    f"4. Applied PELT with penalty={penalty} to detect mean shifts",
                    "5. Identified dates where GWL behavior changed abruptly",
                    "6. Created coverage map to show detection power per site"
                ],
                "penalty_meaning": f"Penalty={penalty}: {'Higher penalty = fewer breakpoints' if penalty > 10 else 'Lower penalty = more sensitive detection'}",
                "model_type": "L2 norm detects changes in mean level (not variance or slope)"
            },
            
            # ✅ INTERPRETATION
            "interpretation": {
                "changepoint_meaning": "Date when GWL trend changed direction or magnitude",
                "coverage_importance": "Sites with longer series (high months/span) have higher detection power",
                "causes": [
                    "Policy changes (pumping regulations)",
                    "Infrastructure (new wells, canals)",
                    "Climate shifts (drought, wet period)",
                    "Land use changes (urbanization, irrigation)"
                ]
            },
            
            # ✅ KEY INSIGHTS
            "key_insights": [
                f"{len(changepoint_sites)} sites ({detection_rate}% of analyzed) have structural breaks",
                f"Average series length: {round(np.mean([s['n_months'] for s in coverage_sites]), 1)} months",
                f"{len(coverage_sites) - num_analyzed} sites skipped (insufficient data < 24 months)",
                "Coverage map shows which sites have sufficient data for robust detection",
                "Changepoints indicate regime shifts in groundwater behavior"
            ]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Changepoint Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# ADVANCED MODULE 9: LAG CORRELATION (Rainfall -> GWL)
# =============================================================================

@app.get("/api/advanced/lag-correlation")
def rainfall_gwl_lag_correlation(
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    max_lag_months: int = Query(12, ge=1, le=24)
):
    """
    Find optimal lag between rainfall and GWL response for each site
    
    OPTIMIZATIONS APPLIED:
    1. Query rainfall ONCE (not per site)
    2. Pre-filter wells by minimum observations
    3. Vectorized date operations
    4. Early termination for invalid data
    """
    try:
        start_time = time.time()
        
        # === CONFIGURATION ===
        # Lowered threshold to include more wells (especially seasonal ones)
        # 10 months roughly equals 2.5 years of quarterly data
        MIN_OBSERVATIONS = 10  
        
        # Get boundary
        boundary_geojson = get_boundary_geojson(state, district)
        
        # ✅ OPTIMIZATION 1: Query rainfall ONCE for all sites
        rainfall_monthly = query_rainfall_monthly(boundary_geojson)
        if rainfall_monthly.empty:
            raise HTTPException(status_code=404, detail="No rainfall data")
        
        rainfall_series = rainfall_monthly.set_index('period')['avg_rainfall'].sort_index()
        print(f"Rainfall loaded: {time.time() - start_time:.2f}s")
        
        # Get wells data with filtering in SQL
        wells_df = get_wells_for_analysis(boundary_geojson)
        if wells_df.empty:
            raise HTTPException(status_code=404, detail="No well data")
        
        print(f"Wells loaded: {time.time() - start_time:.2f}s, count: {len(wells_df)}")
        
        # ✅ OPTIMIZATION 2: Pre-filter by site observation count
        site_counts = wells_df.groupby('site_id').size()
        
        # Use the relaxed threshold here
        valid_sites = site_counts[site_counts >= MIN_OBSERVATIONS].index
        wells_df = wells_df[wells_df['site_id'].isin(valid_sites)]
        
        if wells_df.empty:
            raise HTTPException(status_code=404, detail=f"No sites with sufficient data (min {MIN_OBSERVATIONS} observations)")
        
        print(f"After filtering: {len(valid_sites)} sites with ≥{MIN_OBSERVATIONS} observations")
        
        # ✅ OPTIMIZATION 3: Vectorized date operations
        wells_df['date'] = pd.to_datetime(wells_df['date'])
        wells_df = wells_df.sort_values(['site_id', 'date'])
        
        lag_results = []
        sites_processed = 0
        sites_skipped = 0
        
        for site_id, group in wells_df.groupby('site_id'):
            # Resample to monthly means to align with rainfall
            gwl_series = group.set_index('date')['gwl'].resample('MS').mean().dropna()
            
            # Check length against the relaxed threshold
            if len(gwl_series) < MIN_OBSERVATIONS:
                sites_skipped += 1
                continue
            
            # Find overlap with rainfall
            combined = pd.concat([
                gwl_series.rename('gwl'), 
                rainfall_series.rename('rain')
            ], axis=1).dropna()
            
            # Check minimum overlap against the relaxed threshold
            if len(combined) < MIN_OBSERVATIONS:
                sites_skipped += 1
                continue
            
            # Test different lags
            best_lag = 0
            best_corr = 0.0
            
            # Calculate correlation for each lag 
            for lag in range(0, max_lag_months + 1):
                try:
                    # Shift rainfall forward by lag months (Rain falls now -> impacts GWL later)
                    corr = combined['gwl'].corr(combined['rain'].shift(lag))
                    
                    # Check for valid correlation and if it's stronger (absolute value)
                    if pd.notna(corr) and abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_lag = lag
                except Exception:
                    continue
            
            # Only include sites with non-zero correlation
            if abs(best_corr) > 0.0:
                lag_results.append({
                    "site_id": site_id,
                    "latitude": float(group['latitude'].iloc[0]),
                    "longitude": float(group['longitude'].iloc[0]),
                    "best_lag_months": int(best_lag),
                    "correlation": round(float(best_corr), 3),
                    "abs_correlation": round(abs(float(best_corr)), 3),
                    "relationship": "positive" if best_corr > 0 else "negative",
                    "n_months_analyzed": int(len(combined))
                })
                sites_processed += 1
        
        print(f"Processing complete: {sites_processed} sites analyzed, {sites_skipped} skipped due to overlap/data gaps")
        print(f"Total time: {time.time() - start_time:.2f}s")
        
        if not lag_results:
            # More descriptive error message
            raise HTTPException(
                status_code=404, 
                detail=f"Insufficient overlap. Processed {len(valid_sites)} sites, but none had {MIN_OBSERVATIONS}+ months overlapping with rainfall."
            )
        
        # Create DataFrame for statistics
        results_df = pd.DataFrame(lag_results)
        
        # Calculate statistics
        lag_distribution = results_df['best_lag_months'].value_counts().sort_index().to_dict()
        
        return {
            "module": "LAG_CORRELATION",
            "description": f"Rainfall to GWL response lag analysis (0-{max_lag_months} months)",
            "filters": {"state": state, "district": district},
            "parameters": {
                "max_lag_months": max_lag_months,
                "min_months_required": MIN_OBSERVATIONS
            },
            "statistics": {
                "total_sites": len(lag_results),
                "sites_processed": sites_processed,
                "sites_skipped": sites_skipped,
                "mean_lag": round(float(results_df['best_lag_months'].mean()), 1),
                "median_lag": int(results_df['best_lag_months'].median()),
                "mean_abs_correlation": round(float(results_df['abs_correlation'].mean()), 3),
                "max_correlation": round(float(results_df['abs_correlation'].max()), 3),
                "min_correlation": round(float(results_df['abs_correlation'].min()), 3),
                "lag_distribution": lag_distribution
            },
            "count": len(lag_results),
            "data": lag_results[:500],
            
            # ✅ ADDED
            "methodology": {
                "approach": "Cross-correlation between rainfall and GWL at varying time lags",
                "steps": [
                    f"1. Retrieved monthly rainfall data for region",
                    f"2. Filtered wells to {len(valid_sites)} sites with ≥{MIN_OBSERVATIONS} observations",
                    "3. Resampled both datasets to monthly resolution",
                    "4. For each lag (0 to {max_lag_months} months):",
                    "   - Shifted rainfall forward by lag months",
                    "   - Calculated correlation with GWL",
                    "5. Selected lag with maximum absolute correlation"
                ],
                "lag_interpretation": "Lag = time between rainfall event and GWL response",
                "correlation_formula": "Pearson correlation coefficient between Rain(t-lag) and GWL(t)"
            },
            
            "interpretation": {
                "lag_meaning": {
                    "0_months": "Immediate response (shallow aquifer, direct recharge)",
                    "1-3_months": "Quick response (permeable formations)",
                    "4-6_months": "Moderate response (typical for most aquifers)",
                    "7-12_months": "Slow response (deep aquifers, low permeability)"
                },
                "correlation_sign": {
                    "positive": "More rain → deeper GWL (unusual, may indicate pumping increase)",
                    "negative": "More rain → shallower GWL (normal recharge response)"
                }
            },
            
            "key_insights": [
                f"Mean lag: {round(float(results_df['best_lag_months'].mean()), 1)} months (median: {int(results_df['best_lag_months'].median())})",
                f"Average correlation: {round(float(results_df['abs_correlation'].mean()), 2)}",
                f"Most common lag: {max(lag_distribution, key=lag_distribution.get)} months ({lag_distribution[max(lag_distribution, key=lag_distribution.get)]} sites)",
                f"Processed {sites_processed} sites, skipped {sites_skipped} (insufficient overlap)"
            ]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Lag Correlation Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
# =============================================================================
# ADVANCED MODULE 10: HOTSPOT CLUSTERING
# =============================================================================

@app.get("/api/advanced/hotspots")
def decline_hotspot_clustering(
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    debug: bool = Query(False, description="Enable debug output")
):
    """
    Identify spatially clustered declining well sites using DBSCAN
    Matches Dash implementation exactly
    """
    try:
        boundary_geojson = get_boundary_geojson(state, district)
        wells_df = get_wells_for_analysis(boundary_geojson)
        
        if wells_df.empty:
            raise HTTPException(status_code=404, detail="No well data")
        
        # Calculate slopes for each site
        site_slopes = []
        for site_id, group in wells_df.groupby('site_id'):
            if len(group) < 6:
                continue
            
            group = group.sort_values('date')
            days = (group['date'] - group['date'].min()).dt.days.values
            gwl = group['gwl'].values
            
            if len(np.unique(days)) < 2:
                continue
            
            slope, _, _, _, _ = linregress(days, gwl)
            
            site_slopes.append({
                'site_id': site_id,
                'latitude': group['latitude'].iloc[0],
                'longitude': group['longitude'].iloc[0],
                'slope_m_per_year': slope * 365.25
            })
        
        if not site_slopes:
            raise HTTPException(status_code=404, detail="Insufficient data for clustering")
        
        slopes_df = pd.DataFrame(site_slopes)
        
        # ✅ Filter to declining sites only (MATCHES DASH)
        declining = slopes_df[slopes_df['slope_m_per_year'] > 0].copy()
        
        # 🔍 DEBUG: Print total sites
        if debug:
            print(f"\n{'='*60}")
            print(f"HOTSPOT DEBUG: {state or 'All'} - {district or 'All'}")
            print(f"{'='*60}")
            print(f"Total sites with slopes: {len(slopes_df)}")
            print(f"Declining sites (slope > 0): {len(declining)}")
        
        # ✅ Get bounds from boundary (MATCHES DASH)
        if boundary_geojson:
            try:
                boundary_gdf = gpd.GeoDataFrame.from_features(
                    json.loads(boundary_geojson)['features'], 
                    crs="EPSG:4326"
                )
                bounds = boundary_gdf.total_bounds  # [x0, y0, x1, y1]
                if debug:
                    print(f"Bounds from boundary GeoJSON: {bounds}")
            except Exception as e:
                if debug:
                    print(f"Boundary parse error: {e}, using point cloud")
                coords = declining[['latitude', 'longitude']].values
                bounds = np.array([
                    coords[:, 1].min(),  # x0 (lon)
                    coords[:, 0].min(),  # y0 (lat)
                    coords[:, 1].max(),  # x1 (lon)
                    coords[:, 0].max()   # y1 (lat)
                ])
        else:
            coords = declining[['latitude', 'longitude']].values
            bounds = np.array([
                coords[:, 1].min(),
                coords[:, 0].min(),
                coords[:, 1].max(),
                coords[:, 0].max()
            ])
            if debug:
                print(f"Bounds from point cloud: {bounds}")
        
        # ✅ Calculate eps from diagonal in DEGREES (MATCHES DASH)
        x0, y0, x1, y1 = bounds
        diag = np.hypot(x1 - x0, y1 - y0)  # Diagonal in degrees
        eps_deg = np.clip(0.05 * diag, 0.02, 0.3)  # 5% of diagonal
        eps_rad = np.deg2rad(eps_deg)
        eps_km_approx = eps_deg * 111  # Approximate conversion at equator
        
        # 🔍 DEBUG: Print epsilon calculation
        if debug:
            print(f"\nEpsilon Calculation:")
            print(f"  Diagonal (degrees): {diag:.6f}")
            print(f"  5% of diagonal: {0.05 * diag:.6f}")
            print(f"  eps_deg (clipped 0.02-0.3): {eps_deg:.6f}")
            print(f"  eps_rad: {eps_rad:.6f}")
            print(f"  eps_km (approx): {eps_km_approx:.2f} km")
        
        # ✅ Adaptive min_samples (MATCHES DASH)
        min_samples = max(5, int(len(declining) * 0.04))  # 4% of declining sites
        
        # 🔍 DEBUG: Print min_samples
        if debug:
            print(f"\nMin Samples Calculation:")
            print(f"  4% of {len(declining)} declining sites = {len(declining) * 0.04:.1f}")
            print(f"  min_samples = max(5, {int(len(declining) * 0.04)}) = {min_samples}")
        
        if len(declining) < min_samples:
            raise HTTPException(
                status_code=404,
                detail=f"Only {len(declining)} declining sites found, need at least {min_samples}"
            )
        
        # ✅ DBSCAN clustering (MATCHES DASH)
        coords_rad = np.radians(declining[['latitude', 'longitude']].values)
        
        # 🔍 DEBUG: Print coords info
        if debug:
            print(f"\nCoordinates:")
            print(f"  Shape: {coords_rad.shape}")
            print(f"  Lat range: {declining['latitude'].min():.4f} to {declining['latitude'].max():.4f}")
            print(f"  Lon range: {declining['longitude'].min():.4f} to {declining['longitude'].max():.4f}")
        
        clustering = DBSCAN(eps=eps_rad, min_samples=min_samples, metric='haversine').fit(coords_rad)
        declining['cluster'] = clustering.labels_
        
        # 🔍 DEBUG: Print clustering results
        if debug:
            unique_clusters = sorted(declining['cluster'].unique())
            print(f"\nClustering Results:")
            print(f"  Unique cluster labels: {unique_clusters}")
            print(f"  Number of clusters (excluding -1): {len([c for c in unique_clusters if c != -1])}")
            print(f"  Noise points (cluster -1): {(declining['cluster'] == -1).sum()}")
            print(f"  Clustered points: {(declining['cluster'] != -1).sum()}")
        
        # Statistics per cluster
        cluster_stats = []
        for cluster_id in sorted(declining['cluster'].unique()):
            if cluster_id == -1:  # Noise
                continue
            
            cluster_sites = declining[declining['cluster'] == cluster_id]
            
            cluster_stats.append({
                "cluster_id": int(cluster_id),
                "n_sites": len(cluster_sites),
                "mean_slope": round(float(cluster_sites['slope_m_per_year'].mean()), 4),
                "max_slope": round(float(cluster_sites['slope_m_per_year'].max()), 4),
                "centroid_lat": round(float(cluster_sites['latitude'].mean()), 6),
                "centroid_lon": round(float(cluster_sites['longitude'].mean()), 6)
            })
            
            # 🔍 DEBUG: Print each cluster
            if debug:
                print(f"\n  Cluster {cluster_id}:")
                print(f"    Sites: {len(cluster_sites)}")
                print(f"    Mean slope: {cluster_sites['slope_m_per_year'].mean():.4f} m/yr")
                print(f"    Centroid: ({cluster_sites['latitude'].mean():.4f}, {cluster_sites['longitude'].mean():.4f})")
        
        if debug:
            print(f"{'='*60}\n")
        
        # Convert results
        results = declining.to_dict('records')
        
        response = {
            "module": "HOTSPOTS",
            "description": "Spatial clustering of declining well sites (Dash-aligned)",
            "filters": {"state": state, "district": district},
            "parameters": {
                "eps_deg": round(float(eps_deg), 6),
                "eps_km_approx": round(float(eps_km_approx), 2),
                "min_samples": int(min_samples),
                "algorithm": "DBSCAN",
                "metric": "haversine",
                "diagonal_deg": round(float(diag), 6)
            },
            "statistics": {
                "total_declining_sites": len(declining),
                "n_clusters": len(cluster_stats),
                "noise_points": int((declining['cluster'] == -1).sum()),
                "clustered_points": int((declining['cluster'] != -1).sum()),
                "clustering_rate": round(float((declining['cluster'] != -1).sum()) / float(len(declining)) * 100.0, 1)
            },
            "clusters": cluster_stats,
            "count": len(results),
            "data": results,
            
            "methodology": {
                "approach": "DBSCAN (Density-Based Spatial Clustering) - Dash-aligned",
                "steps": [
                    f"1. Calculated trend slopes for all sites (need ≥6 observations)",
                    f"2. Filtered to {len(declining)} declining sites (positive slope = deeper GWL)",
                    f"3. Calculated eps from AOI diagonal: {eps_deg:.4f}° ≈ {eps_km_approx:.1f}km",
                    f"4. Adaptive min_samples: {min_samples} (4% of declining sites, min 5)",
                    f"5. Applied DBSCAN with haversine distance metric",
                    f"6. Identified {len(cluster_stats)} clusters"
                ],
                "parameters_meaning": {
                    "eps": f"{eps_deg:.4f}° (≈{eps_km_approx:.1f}km) - Max distance between sites",
                    "min_samples": f"{min_samples} - Min sites per cluster (adaptive: 4% of dataset)"
                }
            },
            
            "key_insights": [
                f"{len(cluster_stats)} decline hotspots identified",
                f"{int((declining['cluster'] != -1).sum())} sites clustered ({round(int((declining['cluster'] != -1).sum()) / len(declining) * 100, 1)}% of declining sites)",
                f"{int((declining['cluster'] == -1).sum())} isolated decline sites (noise)",
                f"Largest cluster: {max(cluster_stats, key=lambda x: x['n_sites'])['n_sites'] if cluster_stats else 0} sites",
                f"Epsilon auto-scaled to {eps_deg:.4f}° based on AOI size"
            ]
        }
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Hotspot Clustering Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ADVANCED MODULES INDEX ENDPOINT
# =============================================================================

@app.get("/api/advanced")
def advanced_modules_index():
    """List all available advanced analysis modules"""
    return {
        "title": "Advanced Hydrogeological Analysis Modules",
        "description": "10 specialized modules for groundwater analysis",
        "modules": {
            "1_asi": {
                "name": "Aquifer Suitability Index",
                "endpoint": "/api/advanced/asi",
                "description": "0-5 score for storage/transmission potential",
                "requires": ["state OR district"],
                "dependencies": "none"
            },
            "2_network_density": {
                "name": "Well Network Density Analysis",
                "endpoint": "/api/advanced/network-density",
                "description": "Signal strength and local density mapping",
                "requires": ["state OR district"],
                "dependencies": "none"
            },
            "3_sass": {
                "name": "Spatio-Temporal Aquifer Stress Score",
                "endpoint": "/api/advanced/sass",
                "description": "Composite stress index using wells + GRACE + rainfall",
                "requires": ["state OR district", "year", "month"],
                "dependencies": "GRACE, rainfall"
            },
            "4_grace_divergence": {
                "name": "GRACE vs Ground Divergence",
                "endpoint": "/api/advanced/grace-divergence",
                "description": "Pixel-level difference between satellite and wells",
                "requires": ["year", "month"],
                "dependencies": "GRACE"
            },
            "5_forecast": {
                "name": "GWL Forecasting",
                "endpoint": "/api/advanced/forecast",
                "description": "12-month forward prediction using trend + GRACE",
                "requires": ["state OR district"],
                "dependencies": "GRACE (recommended)"
            },
            "6_recharge_planning": {
                "name": "Recharge Structure Recommendation",
                "endpoint": "/api/advanced/recharge-planning",
                "description": "MAR potential and structure mix",
                "requires": ["state OR district"],
                "dependencies": "rainfall (recommended)"
            },
            "7_significant_trends": {
                "name": "Significance-Aware Trend Mapping",
                "endpoint": "/api/advanced/significant-trends",
                "description": "Sites with statistically robust trends (Mann-Kendall/OLS)",
                "requires": ["state OR district"],
                "dependencies": "pymannkendall (optional)"
            },
            "8_changepoints": {
                "name": "Change-Point Detection",
                "endpoint": "/api/advanced/changepoints",
                "description": "Structural breaks using PELT algorithm",
                "requires": ["state OR district"],
                "dependencies": "ruptures (required)"
            },
            "9_lag_correlation": {
                "name": "Rainfall→GWL Lag Correlation",
                "endpoint": "/api/advanced/lag-correlation",
                "description": "Optimal response lag for each site",
                "requires": ["state OR district"],
                "dependencies": "rainfall"
            },
            "10_hotspots": {
                "name": "Decline Hotspot Clustering",
                "endpoint": "/api/advanced/hotspots",
                "description": "Spatial clustering of declining sites (DBSCAN)",
                "requires": ["state OR district"],
                "dependencies": "none"
            }
        },
        "optional_dependencies": {
            "ruptures": {
                "status": "available" if RUPTURES_AVAILABLE else "missing",
                "install": "pip install ruptures",
                "required_for": ["changepoints"]
            },
            "pymannkendall": {
                "status": "available" if MK_AVAILABLE else "missing",
                "install": "pip install pymannkendall",
                "required_for": ["significant_trends (enhanced)"]
            }
        },
        "usage_example": {
            "asi": "GET /api/advanced/asi?state=Chhattisgarh",
            "sass": "GET /api/advanced/sass?state=Chhattisgarh&year=2023&month=6",
            "forecast": "GET /api/advanced/forecast?district=Raipur&state=Chhattisgarh&forecast_months=12"
        }
    }

# =============================================================================
# RAG AGENT ENDPOINT
# =============================================================================

class AgentRequest(BaseModel):
    question: str
    map_context: Optional[dict] = None

class AgentResponse(BaseModel):
    answer: str
    success: bool
    error: Optional[str] = None

@app.post("/api/agent")
async def query_rag_agent(request: AgentRequest):
    """
    🤖 RAG Agent - Natural Language Groundwater Query System
    
    Query the intelligent RAG (Retrieval-Augmented Generation) agent that has access to:
    - 📚 Knowledge Base: Definitions, formulas, analysis methodologies
    - 🗄️ Database Tools: Groundwater wells, GRACE satellite, rainfall, aquifers
    - 📊 Timeseries Analysis: Trend detection, correlation analysis
    
    **Example Questions:**
    - "What is specific yield?"
    - "Show me groundwater levels in Kerala"
    - "Explain how ASI is calculated"
    - "What are the aquifers in Maharashtra?"
    - "Show me the Mann-Kendall trend for Punjab"
    
    **Parameters:**
    - question: Natural language question
    - map_context: (Optional) Current map view context from frontend
    
    **Returns:**
    - answer: The agent's response
    - success: Whether the query was successful
    - error: Error message if any
    """
    try:
        # Import the query_agent function from testingragsql
        
        
        print(f"\n🤖 Agent Query: {request.question[:100]}...")
        
        # Query the agent with optional map context
        result = query_agent(
            question=request.question,
            map_context=request.map_context,
            verbose=False
        )
        
        return {
            "answer": result.get("output", "No response generated"),
            "success": result.get("success", False),
            "error": result.get("error")
        }
    
    except ImportError as e:
        error_msg = f"RAG agent not available: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "answer": "The RAG agent is not available. Please ensure testingragsql.py is properly configured with Ollama running.",
            "success": False,
            "error": error_msg
        }
    
    except Exception as e:
        error_msg = f"Agent error: {str(e)}"
        print(f"❌ {error_msg}")
        print(traceback.format_exc())
        return {
            "answer": "An error occurred while processing your question. Please try rephrasing or check the server logs.",
            "success": False,
            "error": error_msg
        }

# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*80)
    print("🚀 GeoHydro API v6.1.0 - FIXED (GRACE & Rainfall NaN Issues Resolved)")
    print("="*80)
    print(f"📊 GRACE Months Available: {len(GRACE_BAND_MAPPING)}")
    print(f"🌧️  Rainfall Years Available: {len(RAINFALL_TABLES)}")
    print(f"🤖 RAG Agent: /api/agent endpoint available")
    print(f"🔗 Unified Timeseries: Wells + GRACE + Rainfall")
    print(f"✅ FIXED: GRACE/Rainfall queries now match working map overlay patterns")
    print(f"🔍 NEW: Diagnostic endpoint at /api/debug/raster-info")
    print("="*80)
    print("\n🌐 API Documentation: http://localhost:8000/docs")
    print("🏥 Health Check: http://localhost:8000/health")
    print("📊 Unified Timeseries: GET http://localhost:8000/api/wells/timeseries")
    print("🔍 Diagnostic: GET http://localhost:8000/api/debug/raster-info")
    print("\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)