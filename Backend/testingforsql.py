from sqlalchemy import create_engine, text
from langchain_ollama import ChatOllama
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Optional, List, Dict
import os
from dotenv import load_dotenv
import numpy as np
import json
import re
import pandas as pd
from datetime import datetime, timedelta
import calendar
from scipy.stats import linregress
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import chromadb

load_dotenv()

# =====================================================================
# GLOBAL THREAD POOL
# =====================================================================
EXECUTOR = ThreadPoolExecutor(max_workers=6)

# =====================================================================
# DATABASE CONNECTION
# =====================================================================

DATABASE_URL = os.environ.get("DB_URI")
if not DATABASE_URL:
    print("❌ Error: DB_URI not set in .env file")
    exit(1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

try:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("✅ Database connected\n")
except Exception as e:
    print(f"❌ Database connection failed: {e}")
    exit(1)


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def parse_numeric(value):
    """Safely parses messy database strings into a single float."""
    if not value:
        return 0.0
    
    s = str(value).strip().lower()
    
    if s in ['-', '', ' ', 'na', 'n/a', 'nil']:
        return 0.0
        
    try:
        if '-' in s:
            parts = [float(p.strip()) for p in s.split('-') if p.strip().replace('.','',1).isdigit()]
            if len(parts) >= 2:
                return sum(parts) / len(parts)
        
        if ' to ' in s:
            parts = [float(p.strip()) for p in s.split(' to ') if p.strip().replace('.','',1).isdigit()]
            if len(parts) >= 2:
                return sum(parts) / len(parts)

        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", s)
        if numbers:
            return float(numbers[0])
            
        return 0.0
        
    except Exception:
        return 0.0

def get_boundary_geojson(conn, state: str, district: str = None):
    """Fetch the boundary GeoJSON for a state or district."""
    if district and state:
        query = text("""
            SELECT ST_AsGeoJSON(ST_MakeValid(geometry)) as geojson
            FROM public.district_state
            WHERE UPPER("District") = UPPER(:district)
            AND UPPER("State") = UPPER(:state)
            LIMIT 1;
        """)
        params = {"district": district, "state": state}
    elif state:
        query = text("""
            SELECT ST_AsGeoJSON(ST_Union(ST_MakeValid(geometry))) as geojson
            FROM public.district_state
            WHERE UPPER("State") = UPPER(:state);
        """)
        params = {"state": state}
    else:
        return None
    
    result = conn.execute(query, params).fetchone()
    return result[0] if result else None

def get_month_day_range(year: int, month: int):
    """Get day-of-year range for a given month"""
    first_day = datetime(year, month, 1).timetuple().tm_yday
    last_day_of_month = calendar.monthrange(year, month)[1]
    last_day = datetime(year, month, last_day_of_month).timetuple().tm_yday
    return first_day, last_day

# =====================================================================
# GRACE BAND MAPPING
# =====================================================================

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

auto_detect_grace_bands()

# =====================================================================
# RAINFALL TABLE DETECTION
# =====================================================================

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
                match = re.search(r'rainfall_(\d{4})', table_name)
                if match:
                    year = int(match.group(1))
                    RAINFALL_TABLES[year] = table_name
            print(f"✅ Detected {len(RAINFALL_TABLES)} rainfall years: {sorted(RAINFALL_TABLES.keys())}")
    except Exception as e:
        print(f"⚠️  Error detecting rainfall tables: {e}")

detect_rainfall_tables()

def parse_agent_input(input_str):
    """
    Parse agent input that might be in format: state="Maharashtra", year=2015
    Returns a dictionary with extracted parameters.
    """
    if not input_str or not isinstance(input_str, str):
        return {}
    
    result = {}
    
    # Pattern to match: key="value" or key=value
    pattern = r'(\w+)\s*=\s*["\']?([^,"\'\n]+)["\']?'
    matches = re.findall(pattern, input_str)
    
    for key, value in matches:
        key = key.strip().lower()
        value = value.strip().strip('"').strip("'")
        
        # Try to convert to int
        if value.isdigit():
            result[key] = int(value)
        else:
            result[key] = value
    
    return result
# =====================================================================
# RAG SYSTEM INITIALIZATION
# =====================================================================

class RAGSystem:
    """Handles definitions and analysis knowledge base"""
    
    def __init__(self, definitions_file: str, analyses_file: str):
        print("\n📚 Initializing RAG Knowledge Base...")
        
        # Load data
        self.definitions = self._load_jsonl(definitions_file)
        self.analyses = self._load_jsonl(analyses_file)
        
        print(f"   ✓ Loaded {len(self.definitions)} definitions")
        print(f"   ✓ Loaded {len(self.analyses)} analyses")
        
        # Setup embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create vector stores
        client = chromadb.PersistentClient(path="./chroma_groundwater")
        
        # Definitions store
        def_texts = []
        for d in self.definitions:
            if 'term' in d and 'definition' in d:
                text = f"{d['term']}: {d['definition']}"
                if d.get('formula'):
                    text += f"\nFormula: {d['formula']}"
                if d.get('units'):
                    text += f"\nUnits: {d['units']}"
                def_texts.append(text)
        
        self.def_store = Chroma.from_texts(
            texts=def_texts,
            embedding=embeddings,
            client=client,
            collection_name="definitions"
        )
        
        # Analyses store
        analysis_texts = []
        for a in self.analyses:
            if 'analysis_name' in a and 'description' in a:
                text = f"{a['analysis_name']}: {a['description']}"
                if a.get('formula'):
                    text += f"\nFormula: {json.dumps(a['formula']) if isinstance(a['formula'], dict) else a['formula']}"
                if a.get('steps'):
                    text += f"\nSteps: {json.dumps(a['steps'])}"
                analysis_texts.append(text)
        
        self.analysis_store = Chroma.from_texts(
            texts=analysis_texts,
            embedding=embeddings,
            client=client,
            collection_name="analyses"
        )
        
        print("   ✓ RAG system ready\n")
    
    def _load_jsonl(self, filepath: str) -> List[Dict]:
        """Load JSONL file"""
        data = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except:
                            pass
        except FileNotFoundError:
            print(f"⚠️  File not found: {filepath}")
        return data
    
    def search_definition(self, term: str) -> str:
        """Search for term definition"""
        results = self.def_store.similarity_search(term, k=2)
        if not results:
            return f"No definition found for '{term}'"
        
        output = ["📚 DEFINITION:"]
        for doc in results:
            output.append(doc.page_content)
        return "\n".join(output)
    
    def search_analysis(self, name: str) -> str:
        """Search for analysis methodology"""
        # Try exact match first
        for analysis in self.analyses:
            if 'analysis_name' in analysis and name.lower() in analysis['analysis_name'].lower():
                output = [
                    f"🔬 {analysis['analysis_name']}",
                    f"Description: {analysis['description']}",
                ]
                
                if analysis.get('formula'):
                    formula = analysis['formula']
                    output.append(f"Formula: {json.dumps(formula, indent=2) if isinstance(formula, dict) else formula}")
                
                if analysis.get('steps'):
                    output.append("Steps:")
                    for i, step in enumerate(analysis['steps'], 1):
                        output.append(f"  {i}. {step}")
                
                return "\n".join(output)
        
        # Fallback to vector search
        results = self.analysis_store.similarity_search(name, k=1)
        if results:
            return f"🔬 ANALYSIS:\n{results[0].page_content}"
        return f"No analysis found for '{name}'"


# Initialize RAG (adjust paths as needed)
try:
    RAG = RAGSystem(
        definitions_file="D:/Repos/Work repo/NRSC Internship/dashboard/Backend/app/data/definitions.jsonl",
        analyses_file="D:/Repos/Work repo/NRSC Internship/dashboard/Backend/app/data/analysis_definition.jsonl"
    )
except Exception as e:
    print(f"⚠️  RAG system not available: {e}")
    RAG = None


# =====================================================================
# RAG TOOLS
# =====================================================================

@tool
def search_definition(term: str) -> str:
    """
    Search for groundwater term definitions.
    Use when user asks 'what is X?' or 'define X' or 'explain X'.
    
    Args:
        term: The groundwater term to define (e.g., "specific yield", "GWL", "aquifer")
    """
    if RAG is None:
        return "❌ Knowledge base not available"
    return RAG.search_definition(term)

@tool
def search_analysis(name: str) -> str:
    """
    Search for how analyses and calculations work.
    Use when user asks 'how is X calculated?' or 'explain X analysis' or 'what is X formula?'.
    
    Args:
        name: The analysis name (e.g., "ASI", "SASS", "Mann-Kendall", "water balance")
    """
    if RAG is None:
        return "❌ Knowledge base not available"
    return RAG.search_analysis(name)


# =====================================================================
# DATABASE TOOLS
# =====================================================================

@tool
def get_aquifer_properties(state: str, district: str = None) -> str:
    """
    Query aquifer lithology and properties for an Indian State or District.
    
    Args:
        state: State name (required, e.g., "Kerala", "Maharashtra")
        district: District name (optional)
    """
    # ✅ Handle dictionary input from LangChain
    if isinstance(state, str) and '=' in state:
        parsed = parse_agent_input(state)
        if parsed:
            state = parsed.get('state', '')
            district = parsed.get('district', district)
    
    # ✅ Handle dictionary input
    elif isinstance(state, dict):
        district = state.get('district', district)
        state = state.get('state', '')
    
    # Rest of the function stays the same...
    if state:
        state = str(state).strip().strip('"').strip("'").strip()
    if district:
        district = str(district).strip().strip('"').strip("'").strip() if district else None
    
    if not state or state.lower() in ['none', 'null', '']:
        return "❌ Error: State name is required"
    
    print(f"\n🔍 Querying aquifers: {state}, {district or 'all districts'}")
    
    try:
        with engine.connect() as conn:
            state_geojson = get_boundary_geojson(conn, state, district)
            
            if not state_geojson:
                return f"❌ Location not found: {state}"

            normalized = state.replace(',', ' ').replace('&', ' ').strip()
            state_pattern = '%' + '%'.join(normalized.split()) + '%'

            aquifer_query = text("""
                SELECT 
                    a.aquifers,
                    a.aquifer,
                    a.zone_m,
                    a.mbgl,
                    a.avg_mbgl,
                    a.yeild__,
                    ST_Area(ST_Transform(ST_MakeValid(a.geometry), 32643)) / 1000000.0 as area_km2
                FROM public.aquifers a
                WHERE ST_Intersects(
                    ST_MakeValid(a.geometry),
                    ST_GeomFromGeoJSON(:state_geojson)
                )
                AND (LOWER(a.state) LIKE LOWER(:state_pattern) OR a.state IS NULL)
                ORDER BY area_km2 DESC;
            """)
            
            result = conn.execute(aquifer_query, {
                "state_geojson": state_geojson, 
                "state_pattern": state_pattern
            })
            
            rows = result.fetchall()

            if not rows:
                return f"❌ No aquifer data found for {state}"

            stats = {}
            total_area = 0.0

            for row in rows:
                try:
                    major_type = row[0] if row[0] else "Unclassified"
                    specific_name = row[1] if row[1] else "Unknown"
                    
                    zone_val = parse_numeric(row[2])
                    depth_val = parse_numeric(row[3]) or parse_numeric(row[4])
                    
                    area = float(row[6]) if row[6] else 0.0
                    
                    if area <= 0: continue

                    if major_type not in stats:
                        stats[major_type] = {
                            "area": 0.0,
                            "depths": [],
                            "zones": [],
                            "yields": set(),
                            "subtypes": set()
                        }
                    
                    stats[major_type]["area"] += area
                    stats[major_type]["subtypes"].add(specific_name)
                    total_area += area
                    
                    if depth_val > 0: stats[major_type]["depths"].append(depth_val)
                    if zone_val > 0: stats[major_type]["zones"].append(zone_val)
                    if row[5]: stats[major_type]["yields"].add(str(row[5]).strip())
                    
                except Exception:
                    continue

            if total_area == 0:
                 return f"❌ Data found but area calculations failed"

            location = f"{district}, {state}" if district else state
            response = [f"\n🌊 AQUIFER REPORT: {location.upper()}"]
            response.append(f"Total Area: {total_area:,.2f} km²\n")
            
            sorted_stats = sorted(stats.items(), key=lambda x: x[1]['area'], reverse=True)
            
            for aq_type, data in sorted_stats:
                pct = (data['area'] / total_area) * 100
                avg_depth = np.mean(data['depths']) if data['depths'] else 0
                
                response.append(f"🪨 {aq_type}")
                response.append(f"   Coverage: {data['area']:,.1f} km² ({pct:.1f}%)")
                response.append(f"   Avg Water Depth: {avg_depth:.1f} m bgl")
            
            return "\n".join(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}"


@tool
def get_groundwater_wells_summary(state: str, district: str = None, year: int = None) -> str:
    """
    Get groundwater level statistics and trends.
    
    Args:
        state: State name (required, e.g., "Kerala", "Maharashtra")
        district: District name (optional)
        year: Year to analyze (optional, defaults to all available data)
    """
    # ✅ STEP 1: Parse if input is a string like 'state="Maharashtra", year=2015'
    if isinstance(state, str) and '=' in state:
        parsed = parse_agent_input(state)
        if parsed:
            state = parsed.get('state', '')
            district = parsed.get('district', district)
            year = parsed.get('year', year)
    
    # ✅ STEP 2: Handle dictionary input
    elif isinstance(state, dict):
        district = state.get('district', district)
        year = state.get('year', year)
        state = state.get('state', '')
    
    # ✅ STEP 3: Clean all inputs
    if state:
        state = str(state).strip().strip('"').strip("'").strip()
    if district:
        district = str(district).strip().strip('"').strip("'").strip() if district else None
    if year:
        try:
            year = int(year) if year else None
        except (ValueError, TypeError):
            year = None
    
    if not state or state.lower() in ['none', 'null', '']:
        return "❌ Error: State name is required"
    
    if district and district.lower() in ['none', 'null', '']:
        district = None
    
    location = f"{district}, {state}" if district else state
    print(f"\n💧 Querying wells: {location}, year={year or 'all'}")
    
    try:
        with engine.connect() as conn:
            boundary_geojson = get_boundary_geojson(conn, state, district)
            
            if not boundary_geojson:
                return f"❌ Location not found: '{location}'"
            
            where_clause = """
                WHERE ST_Intersects(
                    ST_GeomFromGeoJSON(:boundary_geojson),
                    ST_SetSRID(ST_MakePoint(
                        COALESCE(NULLIF("LON", 0), "LONGITUD_1"),
                        COALESCE(NULLIF("LAT", 0), "LATITUDE_1")
                    ), 4326)
                )
                AND "GWL" IS NOT NULL
            """
            params = {"boundary_geojson": boundary_geojson}
            
            if year:
                where_clause += ' AND EXTRACT(YEAR FROM "Date") = :year'
                params["year"] = year
            
            query = text(f"""
                SELECT 
                    DATE_TRUNC('month', "Date") as month,
                    AVG("GWL") as avg_gwl,
                    COUNT(*) as n_readings
                FROM groundwater_level
                {where_clause}
                GROUP BY DATE_TRUNC('month', "Date")
                ORDER BY month;
            """)
            
            result = conn.execute(query, params)
            data = [(row[0], float(row[1]), int(row[2])) for row in result]
            
            if len(data) == 0:
                return f"❌ No data available for {location}" + (f" in {year}" if year else "")
            
            dates = [d[0] for d in data]
            gwl_values = np.array([d[1] for d in data])
            n_months = len(data)
            
            response = [f"\n💧 GROUNDWATER: {location.upper()}"]
            response.append(f"Period: {dates[0].date()} to {dates[-1].date()}")
            response.append(f"Months: {n_months}, Readings: {sum(d[2] for d in data):,}\n")
            
            response.append(f"Mean Depth: {np.mean(gwl_values):.2f} m bgl")
            response.append(f"Range: {np.min(gwl_values):.2f} to {np.max(gwl_values):.2f} m")
            
            TREND_THRESHOLD = 3
            
            if n_months >= TREND_THRESHOLD:
                x = np.array([(d - dates[0]).days for d in dates])
                y = gwl_values
                
                slope, _, r_value, p_value, _ = linregress(x, y)
                
                slope_per_year = slope * 365.25
                r_squared = r_value ** 2
                
                response.append(f"\nTrend: {'DECLINING' if slope_per_year > 0 else 'RECOVERING'}")
                response.append(f"Rate: {abs(slope_per_year):.4f} m/year")
                response.append(f"Confidence (R²): {r_squared:.3f}")
                
                if slope_per_year > 1.0:
                    response.append("⚠️  WARNING: Rapid decline!")
                elif slope_per_year < -0.5:
                    response.append("✅ POSITIVE: Water table recovering!")
            else:
                response.append(f"\n⚠️  Trend unavailable - only {n_months} month(s)")
            
            return "\n".join(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}"



@tool
def get_grace_data(state: str, district: str = None, year: int = None, month: int = None) -> str:
    """
    Get GRACE satellite water storage data.
    
    Args:
        state: State name (required)
        district: District name (optional)
        year: Year (optional, defaults to latest available)
        month: Month 1-12 (optional)
    """
    # ✅ STEP 1: Parse if input is a string like 'state="Maharashtra", year=2015'
    if isinstance(state, str) and '=' in state:
        parsed = parse_agent_input(state)
        if parsed:
            state = parsed.get('state', '')
            district = parsed.get('district', district)
            year = parsed.get('year', year)
            month = parsed.get('month', month)
    
    # ✅ STEP 2: Handle dictionary input
    elif isinstance(state, dict):
        district = state.get('district', district)
        year = state.get('year', year)
        month = state.get('month', month)
        state = state.get('state', '')
    
    # ✅ STEP 3: Clean all inputs
    if state:
        state = str(state).strip().strip('"').strip("'").strip()
    if district:
        district = str(district).strip().strip('"').strip("'").strip() if district else None
    if year:
        try:
            year = int(year) if year else None
        except (ValueError, TypeError):
            year = None
    if month:
        try:
            month = int(month) if month else None
        except (ValueError, TypeError):
            month = None
    
    if not state or state.lower() in ['none', 'null', '']:
        return "❌ Error: State name is required"
    
    location = f"{district}, {state}" if district else state
    
    if year is None:
        year = max(y for y, m in GRACE_BAND_MAPPING.keys())
    
    print(f"\n🛰️  Querying GRACE: {location}, {year}-{month or 'annual'}")
    
    try:
        with engine.connect() as conn:
            if month:
                if (year, month) not in GRACE_BAND_MAPPING:
                    return f"❌ No GRACE data for {year}-{month:02d}"
                bands = [GRACE_BAND_MAPPING[(year, month)]]
            else:
                bands = [GRACE_BAND_MAPPING[(y, m)] for y, m in GRACE_BAND_MAPPING.keys() if y == year]
                if not bands:
                    return f"❌ No GRACE data for {year}"
            
            boundary_geojson = get_boundary_geojson(conn, state, district)
            
            if not boundary_geojson:
                return f"❌ Location not found: {location}"
            
            all_values = []
            
            for band in bands:
                query = text(f"""
                    WITH grace_pixels AS (
                        SELECT 
                            ST_Y(ST_SetSRID((ST_PixelAsCentroids(rast, {band})).geom, 4326)) as lat,
                            (ST_PixelAsCentroids(rast, {band})).val as tws,
                            ST_SetSRID((ST_PixelAsCentroids(rast, {band})).geom, 4326) as geom
                        FROM public.grace_lwe
                        WHERE rid = 1
                    )
                    SELECT lat, tws
                    FROM grace_pixels
                    WHERE tws IS NOT NULL
                    AND ST_Intersects(geom, ST_GeomFromGeoJSON(:boundary))
                """)
                
                result = conn.execute(query, {"boundary": boundary_geojson})
                pixels = [(float(row[0]), float(row[1])) for row in result]
                all_values.extend(pixels)
            
            if not all_values:
                return f"❌ No GRACE data retrieved"
            
            lats, values = zip(*all_values)
            weights = np.cos(np.radians(lats))
            weighted_avg = np.sum(np.array(values) * weights) / np.sum(weights)
            
            response = [f"\n🛰️  GRACE: {location.upper()}"]
            response.append(f"Period: {year}-{month or 'annual'}")
            response.append(f"Pixels: {len(all_values):,}\n")
            
            response.append(f"Average TWS: {weighted_avg:.2f} cm")
            response.append(f"Range: {min(values):.2f} to {max(values):.2f} cm")
            
            if weighted_avg > 0:
                response.append("Status: More water than baseline ✅")
            else:
                response.append("Status: Less water than baseline ⚠️")
            
            return "\n".join(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}"


@tool
def get_rainfall_data(state: str, district: str = None, year: int = None, month: int = None) -> str:
    """Get rainfall statistics."""
    
    # ✅ STEP 1: Parse if input is a string
    if isinstance(state, str) and '=' in state:
        parsed = parse_agent_input(state)
        if parsed:
            state = parsed.get('state', '')
            district = parsed.get('district', district)
            year = parsed.get('year', year)
            month = parsed.get('month', month)
    
    # ✅ STEP 2: Handle dictionary input
    elif isinstance(state, dict):
        district = state.get('district', district)
        year = state.get('year', year)
        month = state.get('month', month)
        state = state.get('state', '')
    
    # ✅ STEP 3: Clean inputs
    if state:
        state = str(state).strip().strip('"').strip("'").strip()
    if district:
        district = str(district).strip().strip('"').strip("'").strip() if district else None
    if year:
        try:
            year = int(year) if year else None
        except (ValueError, TypeError):
            year = None
    if month:
        try:
            month = int(month) if month else None
        except (ValueError, TypeError):
            month = None
    
    if not state or state.lower() in ['none', 'null', '']:
        return "❌ Error: State name is required"
    
    location = f"{district}, {state}" if district else state
    
    if year is None or year == 0:
        year = max(RAINFALL_TABLES.keys())
    
    if year not in RAINFALL_TABLES:
        return f"❌ No rainfall data for {year}"
    
    table_name = RAINFALL_TABLES[year]
    
    print(f"\n🌧️  Querying rainfall: {location}, {year}-{month or 'annual'}")
    
    try:
        with engine.connect() as conn:
            if month:
                first_day, last_day = get_month_day_range(year, month)
                bands = list(range(first_day, last_day + 1))
                days_in_period = len(bands)
            else:
                days_in_year = 366 if calendar.isleap(year) else 365
                bands = list(range(1, days_in_year + 1))
                days_in_period = days_in_year
            
            boundary_geojson = get_boundary_geojson(conn, state, district)
            
            if not boundary_geojson:
                return f"❌ Location not found: {location}"
            
            all_values = []
            
            for band in bands:
                query = text(f"""
                    WITH pixels AS (
                        SELECT 
                            ST_Y(ST_SetSRID((ST_PixelAsCentroids(rast, {band})).geom, 4326)) as lat,
                            (ST_PixelAsCentroids(rast, {band})).val as val,
                            ST_SetSRID((ST_PixelAsCentroids(rast, {band})).geom, 4326) as geom
                        FROM {table_name}
                        WHERE rid = 1
                    )
                    SELECT lat, val
                    FROM pixels
                    WHERE val IS NOT NULL 
                    AND val >= 0 
                    AND val <= 500
                    AND ST_Intersects(geom, ST_GeomFromGeoJSON(:boundary))
                """)
                
                result = conn.execute(query, {"boundary": boundary_geojson})
                all_values.extend([(float(row[0]), float(row[1])) for row in result])
            
            if not all_values:
                return f"❌ No rainfall data retrieved"
            
            lats, values = zip(*all_values)
            weights = np.cos(np.radians(lats))
            daily_avg = np.sum(np.array(values) * weights) / np.sum(weights)
            total_rainfall = daily_avg * days_in_period
            
            response = [f"\n🌧️  RAINFALL: {location.upper()}"]
            response.append(f"Period: {year}-{month or 'annual'}\n")
            
            response.append(f"Daily Average: {daily_avg:.2f} mm/day")
            response.append(f"Total: {total_rainfall:.2f} mm")
            response.append(f"Range: {min(values):.2f} to {max(values):.2f} mm")
            
            return "\n".join(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}"

# =====================================================================
# TIMESERIES TOOL & HELPERS
# =====================================================================

def query_wells_monthly_for_tool(conn, boundary_geojson):
    """Query monthly groundwater levels"""
    query = text("""
        SELECT 
            DATE_TRUNC('month', "Date") as period,
            AVG("GWL") as avg_gwl,
            COUNT(*) as count
        FROM groundwater_level
        WHERE ST_Intersects(
            ST_GeomFromGeoJSON(:boundary_geojson),
            ST_SetSRID(ST_MakePoint(
                COALESCE(NULLIF("LON", 0), "LONGITUD_1"),
                COALESCE(NULLIF("LAT", 0), "LATITUDE_1")
            ), 4326)
        )
        AND "GWL" IS NOT NULL
        GROUP BY DATE_TRUNC('month', "Date")
        ORDER BY period;
    """)
    
    result = conn.execute(query, {"boundary_geojson": boundary_geojson})
    data = []
    for row in result:
        data.append({
            "period": row[0],
            "avg_gwl": float(row[1]),
            "count": int(row[2])
        })
    return pd.DataFrame(data)


def query_grace_monthly_for_tool(conn, boundary_geojson):
    """Query monthly GRACE data"""
    grace_data = []
    
    for (year, month), band in sorted(GRACE_BAND_MAPPING.items()):
        try:
            query = text(f"""
                WITH grace_pixels AS (
                    SELECT 
                        ST_Y(ST_SetSRID((ST_PixelAsCentroids(rast, {band})).geom, 4326)) as lat,
                        (ST_PixelAsCentroids(rast, {band})).val as tws,
                        ST_SetSRID((ST_PixelAsCentroids(rast, {band})).geom, 4326) as geom
                    FROM public.grace_lwe
                    WHERE rid = 1
                )
                SELECT 
                    SUM(tws * COS(RADIANS(lat))) / NULLIF(SUM(COS(RADIANS(lat))), 0) as weighted_avg
                FROM grace_pixels
                WHERE tws IS NOT NULL
                AND ST_Intersects(geom, ST_GeomFromGeoJSON(:boundary))
            """)
            
            result = conn.execute(query, {"boundary": boundary_geojson}).fetchone()
            
            if result and result[0] is not None:
                grace_data.append({
                    "period": datetime(year, month, 1),
                    "avg_tws": float(result[0])
                })
        except Exception:
            continue
    
    return pd.DataFrame(grace_data)


def query_rainfall_monthly_for_tool(conn, boundary_geojson):
    """OPTIMIZED: Query monthly rainfall with parallel processing"""
    
    if not RAINFALL_TABLES:
        return pd.DataFrame(columns=['period', 'avg_rainfall'])
    
    print(f"  🌧️  Querying rainfall: {len(RAINFALL_TABLES)} years (parallel)...")
    
    def process_month(args):
        year, month, table_name, boundary_geojson = args
        
        try:
            with engine.connect() as thread_conn:
                first_day, last_day = get_month_day_range(year, month)
                bands = list(range(first_day, last_day + 1))
                bands_str = ','.join(map(str, bands))
                
                query = text(f"""
                    WITH daily_pixels AS (
                        SELECT 
                            ST_Y(ST_SetSRID((ST_PixelAsCentroids(rast, unnest(ARRAY[{bands_str}]))).geom, 4326)) as lat,
                            (ST_PixelAsCentroids(rast, unnest(ARRAY[{bands_str}]))).val as val,
                            ST_SetSRID((ST_PixelAsCentroids(rast, unnest(ARRAY[{bands_str}]))).geom, 4326) as geom
                        FROM {table_name}
                        WHERE rid = 1
                    )
                    SELECT 
                        SUM(val * COS(RADIANS(lat))) / NULLIF(SUM(COS(RADIANS(lat))), 0) as weighted_avg
                    FROM daily_pixels
                    WHERE val IS NOT NULL 
                    AND val >= 0 
                    AND val <= 500
                    AND ST_Intersects(geom, ST_GeomFromGeoJSON(:boundary))
                """)
                
                result = thread_conn.execute(query, {"boundary": boundary_geojson}).fetchone()
                
                if result and result[0] is not None:
                    return {
                        "period": datetime(year, month, 1),
                        "avg_rainfall": float(result[0])
                    }
        except Exception:
            pass
        
        return None
    
    tasks = []
    for year, table_name in sorted(RAINFALL_TABLES.items()):
        for month in range(1, 13):
            tasks.append((year, month, table_name, boundary_geojson))
    
    rainfall_data = []
    futures = [EXECUTOR.submit(process_month, task) for task in tasks]
    
    for future in as_completed(futures):
        result = future.result()
        if result:
            rainfall_data.append(result)
    
    print(f"  ✓ Rainfall: {len(rainfall_data)} months")
    
    return pd.DataFrame(rainfall_data)


def merge_timeseries_data(wells_df, grace_df, rainfall_df):
    """Merge all three datasets"""
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
    
    if not combined.empty:
        combined = combined.set_index("period")
        combined = combined.interpolate(method="time", limit_direction="both")
        combined = combined.reset_index()
    
    return combined


def analyze_simple(df, location):
    """Comprehensive analysis of all three datasets - ENHANCED VERSION"""
    if len(df) < 24:
        return f"❌ Need at least 24 months for analysis (found {len(df)})"
    
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    response = [f"\n{'='*70}"]
    response.append(f"📈 COMPREHENSIVE TIMESERIES ANALYSIS: {location.upper()}")
    response.append(f"{'='*70}")
    response.append(f"• Period: {df['period'].min().date()} to {df['period'].max().date()}")
    response.append(f"• Total Months: {len(df)}\n")
    
    # ===== 💧 GROUNDWATER ANALYSIS =====
    if 'avg_gwl' in df.columns and not df['avg_gwl'].isna().all():
        response.append("💧 GROUNDWATER LEVEL ANALYSIS:")
        response.append("-" * 70)
        
        df_copy = df.copy().set_index("period").sort_index()
        series = df_copy['avg_gwl'].interpolate(limit_direction='both')
        
        # Deseasonalize
        decomp = seasonal_decompose(series, model='additive', period=12)
        deseasonalized = series - decomp.seasonal
        
        # Calculate trend
        valid_y = deseasonalized.dropna()
        if len(valid_y) >= 2:
            x = np.arange(len(valid_y))
            slope, _, r_value, p_value, _ = linregress(x, valid_y.values)
            slope_per_year = slope * 12
            r_squared = r_value ** 2
            
            # Current status
            current_gwl = series.iloc[-1]
            start_gwl = series.iloc[0]
            total_change = current_gwl - start_gwl
            
            response.append(f"   📍 Starting Level ({df['period'].min().year}): {start_gwl:.2f}m below ground")
            response.append(f"   📍 Current Level ({df['period'].max().year}): {current_gwl:.2f}m below ground")
            response.append(f"   📍 Total Change: {abs(total_change):.2f}m {'DEEPER ⬇️' if total_change > 0 else 'SHALLOWER ⬆️'}")
            
            # Trend direction and severity
            if slope_per_year > 0:
                trend_desc = "DECLINING (getting deeper)"
                if slope_per_year > 1.0:
                    severity = "⚠️  CRITICAL"
                    action = "Immediate water conservation measures needed"
                elif slope_per_year > 0.5:
                    severity = "⚠️  HIGH"
                    action = "Monitor closely and plan water management"
                else:
                    severity = "⚠️  MODERATE"
                    action = "Continue monitoring"
            else:
                trend_desc = "RECOVERING (getting shallower)"
                severity = "✅ POSITIVE"
                action = "Current practices working well"
            
            response.append(f"\n   {severity}: {trend_desc}")
            response.append(f"   📉 Rate: {abs(slope_per_year):.4f}m per year")
            response.append(f"   📊 Confidence (R²): {r_squared:.3f} {'(STRONG)' if r_squared > 0.7 else '(MODERATE)' if r_squared > 0.5 else '(WEAK)'}")
            response.append(f"   💡 Action: {action}")
            
            # Year-by-year pattern (top 5 years)
            response.append(f"\n   📅 YEAR-BY-YEAR PATTERN:")
            yearly_data = series.resample('Y').mean()
            
            for i in range(min(5, len(yearly_data))):
                year = yearly_data.index[i].year
                value = yearly_data.iloc[i]
                
                if i > 0:
                    prev_value = yearly_data.iloc[i-1]
                    change = value - prev_value
                    trend_word = "deeper" if change > 0 else "shallower"
                    response.append(f"      • {year}: {value:.2f}m ({abs(change):.2f}m {trend_word} than {year-1})")
                else:
                    response.append(f"      • {year}: {value:.2f}m (baseline)")
            
            if len(yearly_data) > 5:
                response.append(f"      ... ({len(yearly_data)-5} more years)")
        
        response.append("")
    
    # ===== 🛰️ GRACE WATER STORAGE ANALYSIS =====
    if 'avg_tws' in df.columns and not df['avg_tws'].isna().all():
        response.append("🛰️  GRACE WATER STORAGE ANALYSIS:")
        response.append("-" * 70)
        
        df_copy = df.copy().set_index("period").sort_index()
        series = df_copy['avg_tws'].interpolate(limit_direction='both')
        
        # Deseasonalize
        decomp = seasonal_decompose(series, model='additive', period=12)
        deseasonalized = series - decomp.seasonal
        
        valid_y = deseasonalized.dropna()
        if len(valid_y) >= 2:
            x = np.arange(len(valid_y))
            slope, _, r_value, _, _ = linregress(x, valid_y.values)
            slope_per_year = slope * 12
            r_squared = r_value ** 2
            
            current_tws = series.iloc[-1]
            start_tws = series.iloc[0]
            total_change = current_tws - start_tws
            
            response.append(f"   💧 Starting Storage: {start_tws:.2f}cm")
            response.append(f"   💧 Current Storage: {current_tws:.2f}cm")
            response.append(f"   💧 Total Change: {abs(total_change):.2f}cm {'LOSS ⬇️' if total_change < 0 else 'GAIN ⬆️'}")
            
            if slope_per_year > 0:
                response.append(f"\n   ✅ INCREASING at {slope_per_year:.4f}cm/year")
                if slope_per_year > 2:
                    response.append("   Status: Significant water storage recovery")
            else:
                response.append(f"\n   ⚠️  DECREASING at {abs(slope_per_year):.4f}cm/year")
                if abs(slope_per_year) > 2:
                    response.append("   Status: Significant water storage depletion")
            
            response.append(f"   📊 Confidence (R²): {r_squared:.3f}")
        
        response.append("")
    
    # ===== 🌧️ RAINFALL PATTERN ANALYSIS =====
    if 'avg_rainfall' in df.columns and not df['avg_rainfall'].isna().all():
        response.append("🌧️  RAINFALL PATTERN ANALYSIS:")
        response.append("-" * 70)
        
        df_copy = df.copy().set_index("period").sort_index()
        series = df_copy['avg_rainfall']
        
        # Calculate annual totals
        yearly = series.resample('Y').sum()
        
        response.append(f"   🌧️  Average Annual: {yearly.mean():.0f}mm")
        response.append(f"   🌧️  Wettest Year: {yearly.max():.0f}mm ({yearly.idxmax().year})")
        response.append(f"   🌧️  Driest Year: {yearly.min():.0f}mm ({yearly.idxmin().year})")
        response.append(f"   🌧️  Variability: {yearly.std():.0f}mm (std deviation)")
        
        # Monsoon analysis
        monsoon = series[series.index.month.isin([6,7,8,9])]
        if len(monsoon) > 0:
            response.append(f"   🌧️  Monsoon Average: {monsoon.mean():.2f}mm/day")
        
        # Recent trend comparison
        if len(yearly) >= 6:
            recent_avg = yearly.tail(3).mean()
            older_avg = yearly.head(3).mean()
            change_pct = ((recent_avg - older_avg) / older_avg) * 100
            
            if recent_avg > older_avg * 1.1:
                response.append(f"\n   📈 Recent Trend: WETTER ({change_pct:+.1f}% vs early period)")
            elif recent_avg < older_avg * 0.9:
                response.append(f"\n   📉 Recent Trend: DRIER ({change_pct:+.1f}% vs early period)")
            else:
                response.append(f"\n   📊 Recent Trend: STABLE ({change_pct:+.1f}% vs early period)")
        
        response.append("")
    
    # ===== 🔬 INTEGRATED INSIGHTS =====
    response.append("🔬 INTEGRATED INSIGHTS:")
    response.append("=" * 70)
    
    # Check for correlations/patterns
    has_gwl = 'avg_gwl' in df.columns and not df['avg_gwl'].isna().all()
    has_grace = 'avg_tws' in df.columns and not df['avg_tws'].isna().all()
    has_rain = 'avg_rainfall' in df.columns and not df['avg_rainfall'].isna().all()
    
    if has_gwl and has_rain:
        df_copy = df.copy().set_index("period")
        gwl_series = df_copy['avg_gwl'].interpolate()
        rain_series = df_copy['avg_rainfall']
        
        if len(gwl_series) >= 24 and len(rain_series) >= 24:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Groundwater trend
            gwl_decomp = seasonal_decompose(gwl_series, model='additive', period=12)
            gwl_deseason = gwl_series - gwl_decomp.seasonal
            gwl_valid = gwl_deseason.dropna()
            x = np.arange(len(gwl_valid))
            gwl_slope, *_ = linregress(x, gwl_valid.values)
            gwl_slope_per_year = gwl_slope * 12
            
            # Rainfall trend
            yearly_rain = rain_series.resample('Y').sum()
            if len(yearly_rain) >= 6:
                recent_rain = yearly_rain.tail(3).mean()
                older_rain = yearly_rain.head(3).mean()
                
                # Integrated interpretation
                if gwl_slope_per_year > 0.1:  # Declining groundwater
                    if recent_rain < older_rain * 0.9:
                        response.append("⚠️  CLIMATE STRESS: Both groundwater AND rainfall declining")
                        response.append("   → Climate-driven water scarcity likely")
                        response.append("   → Recommend: Rainwater harvesting + conservation")
                    else:
                        response.append("⚠️  OVER-EXTRACTION: Groundwater declining despite stable rainfall")
                        response.append("   → Human usage exceeds natural recharge")
                        response.append("   → Recommend: Reduce extraction + artificial recharge")
                
                elif gwl_slope_per_year < -0.1:  # Recovering groundwater
                    if recent_rain > older_rain * 1.1:
                        response.append("✅ NATURAL RECOVERY: Groundwater recovering with increased rainfall")
                        response.append("   → Climate patterns favorable")
                    else:
                        response.append("✅ MANAGEMENT SUCCESS: Groundwater recovering despite stable rainfall")
                        response.append("   → Water conservation measures effective")
                        response.append("   → Recommend: Continue current practices")
                
                else:  # Stable groundwater
                    response.append("📊 EQUILIBRIUM: Groundwater stable")
                    response.append("   → Extraction balanced with recharge")
                    response.append("   → Recommend: Maintain current management")
    
    # GRACE-Groundwater correlation
    if has_gwl and has_grace:
        df_copy = df.copy().set_index("period")
        gwl_series = df_copy['avg_gwl'].interpolate()
        grace_series = df_copy['avg_tws'].interpolate()
        
        if len(gwl_series) >= 24 and len(grace_series) >= 24:
            # Check if both are changing in same direction
            gwl_start = gwl_series.iloc[0]
            gwl_end = gwl_series.iloc[-1]
            grace_start = grace_series.iloc[0]
            grace_end = grace_series.iloc[-1]
            
            gwl_declining = (gwl_end - gwl_start) > 1.0  # Getting deeper
            grace_declining = (grace_end - grace_start) < -2.0  # Losing storage
            
            if gwl_declining and grace_declining:
                response.append("\n⚠️  CONFIRMED DEPLETION: Both groundwater wells AND satellite data show decline")
                response.append("   → High confidence in water stress assessment")
    
    # Final recommendations
    response.append(f"\n{'='*70}")
    response.append("💡 RECOMMENDATIONS:")
    
    if has_gwl:
        if gwl_slope_per_year > 1.0:
            response.append("   🚨 URGENT: Implement immediate water restrictions")
            response.append("   🚨 URGENT: Promote water-efficient technologies")
            response.append("   🚨 URGENT: Develop artificial recharge projects")
        elif gwl_slope_per_year > 0.5:
            response.append("   ⚠️  Monitor water usage patterns closely")
            response.append("   ⚠️  Plan for water conservation measures")
            response.append("   ⚠️  Consider rainwater harvesting initiatives")
        elif gwl_slope_per_year < -0.5:
            response.append("   ✅ Continue successful water management practices")
            response.append("   ✅ Document and share best practices")
        else:
            response.append("   📊 Maintain current monitoring frequency")
            response.append("   📊 Continue balanced water management")
    
    response.append(f"{'='*70}")
    
    return "\n".join(response)


@tool
def get_timeseries_analysis(state: str, district: str = None) -> str:
    """
    Comprehensive timeseries analysis showing groundwater trends over time.
    Use when user asks about trends, changes over time, or historical patterns.
    
    Args:
        state: State name (required)
        district: District name (optional)
    """
    # ✅ Handle dictionary input from LangChain
   # ✅ Parse string input
    if isinstance(state, str) and '=' in state:
        parsed = parse_agent_input(state)
        if parsed:
            state = parsed.get('state', '')
            district = parsed.get('district', district)
    
    # ✅ Handle dictionary input
    elif isinstance(state, dict):
        district = state.get('district', district)
        state = state.get('state', '')
    
    # Clean inputs
    if state:
        state = str(state).strip().strip('"').strip("'").strip()
    if district:
        district = str(district).strip().strip('"').strip("'").strip() if district else None
    
    if not state or state.lower() in ['none', 'null', '']:
        return "❌ Error: State name is required"
    
    location = f"{district}, {state}" if district else state
    print(f"\n📈 Analyzing timeseries: {location}")
    
    try:
        with engine.connect() as conn:
            boundary_geojson = get_boundary_geojson(conn, state, district)
            
            if not boundary_geojson:
                return f"❌ Location not found: {location}"
            
            print("  📊 Fetching data...")
            wells_data = query_wells_monthly_for_tool(conn, boundary_geojson)
            grace_data = query_grace_monthly_for_tool(conn, boundary_geojson)
            rainfall_data = query_rainfall_monthly_for_tool(conn, boundary_geojson)
            
            combined = merge_timeseries_data(wells_data, grace_data, rainfall_data)
            
            if len(combined) == 0:
                return f"❌ No timeseries data available"
            
            result = analyze_simple(combined, location)
            
            return result
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}"

def build_module_context_string(map_context: dict) -> str:
    """
    Build module context string from frontend's complete map_context object
    
    Args:
        map_context: Complete MapContext from frontend (includes region, temporal, data_summary, etc.)
    
    Returns:
        Formatted context string for agent prompt
    """
    
    if not map_context:
        return """**MODULE CONTEXT:**
No context provided - user may be on homepage.
Use database tools to answer location-specific questions."""
    
    # Extract core context
    region = map_context.get('region', {})
    temporal = map_context.get('temporal', {})
    data_summary = map_context.get('data_summary', {})
    
    # Build location string
    state = region.get('state')
    district = region.get('district')
    
    if district and state:
        location_str = f"{district}, {state}"
    elif state:
        location_str = state
    else:
        location_str = "All India"
    
    # Build time string
    year = temporal.get('year')
    month = temporal.get('month')
    season = temporal.get('season')
    
    time_str = ""
    if year and month:
        time_str = f" ({year}-{month:02d})"
    elif year:
        time_str = f" ({year})"
    if season:
        time_str += f" [{season}]"
    
    # Determine active module from data_summary
    active_module = data_summary.get('active_module')
    
    # Start building context
    context_parts = [
        f"**CURRENT LOCATION:** {location_str}{time_str}",
    ]
    
    # If no active module, check which data is present
    if not active_module:
        active_layers = map_context.get('active_layers', [])
        if active_layers:
            context_parts.insert(0, f"**ACTIVE LAYERS:** {', '.join(active_layers)}")
        else:
            context_parts.insert(0, "**CURRENT MODULE:** None (homepage or map view)")
        
        # Add any available data summaries
        if 'wells' in data_summary:
            wells = data_summary['wells']
            context_parts.append(f"\n**WELLS DATA:**")
            context_parts.append(f"- Data Points: {wells.get('data_points', 'N/A')}")
            context_parts.append(f"- Average GWL: {wells.get('avg_gwl', 'N/A')} m")
            context_parts.append(f"- Unique Sites: {wells.get('unique_sites', 'N/A')}")
        
        if 'grace' in data_summary:
            grace = data_summary['grace']
            context_parts.append(f"\n**GRACE DATA:**")
            context_parts.append(f"- Regional Avg: {grace.get('regional_average_cm', 'N/A')} cm")
            context_parts.append(f"- Data Points: {grace.get('data_points', 'N/A')}")
        
        if 'rainfall' in data_summary:
            rainfall = data_summary['rainfall']
            context_parts.append(f"\n**RAINFALL DATA:**")
            context_parts.append(f"- Regional Avg: {rainfall.get('regional_average_mm_per_day', 'N/A')} mm/day")
        
        return "\n".join(context_parts)
    
    # Format based on active module
    context_parts.insert(0, f"**CURRENT MODULE:** {active_module}")
    context_parts.append("\n**FRONTEND DATA (what user sees on screen):**")
    
    if active_module == 'ASI' and 'asi' in data_summary:
        asi = data_summary['asi']
        stats = asi.get('statistics', {})
        interp = asi.get('interpretation', {})
        context_parts.append(f"""
ASI (Aquifer Suitability Index):
- Mean Score: {stats.get('mean_asi', 'N/A')}/5.0
- Rating: {interp.get('regional_rating', 'N/A')}
- Dominant Aquifer: {stats.get('dominant_aquifer', 'N/A')}
- Average Specific Yield: {stats.get('avg_specific_yield', 'N/A')}
- High-Suitability Areas: {interp.get('high_suitability_percentage', 'N/A')}%
- Total Area: {stats.get('total_area_km2', 'N/A')} km²
Assessment: {interp.get('regional_narrative', 'N/A')}""")
    
    elif active_module == 'SASS' and 'sass' in data_summary:
        sass = data_summary['sass']
        stats = sass.get('statistics', {})
        interp = sass.get('interpretation', {})
        context_parts.append(f"""
SASS (Aquifer Stress Score):
- Mean Score: {stats.get('mean_sass', 'N/A')}
- Status: {interp.get('overall_status', 'N/A')}
- Critical Sites: {stats.get('critical_sites', 'N/A')}
- Stressed Sites: {stats.get('stressed_sites', 'N/A')}
- Total Sites: {stats.get('sites_analyzed', 'N/A')}
Assessment: {interp.get('regional_narrative', 'N/A')}""")
    
    elif active_module == 'NETWORK_DENSITY' and 'network_density' in data_summary:
        nd = data_summary['network_density']
        stats = nd.get('statistics', {})
        interp = nd.get('interpretation', {})
        context_parts.append(f"""
Network Density Analysis:
- Total Sites: {stats.get('total_sites', 'N/A')}
- Avg Signal Strength: {stats.get('avg_strength', 'N/A')}
- Avg Local Density: {stats.get('avg_local_density', 'N/A')} sites/km²
- Grid Cells: {nd.get('grid_count', 'N/A')}
Quality: {interp.get('signal_quality_rating', 'N/A')} signal, {interp.get('coverage_quality_rating', 'N/A')} coverage""")
    
    elif active_module == 'FORECAST' and 'forecast' in data_summary:
        fc = data_summary['forecast']
        stats = fc.get('statistics', {})
        interp = fc.get('interpretation', {})
        context_parts.append(f"""
GWL Forecast:
- Forecast Period: {fc.get('forecast_months', 'N/A')} months
- Mean Change: {stats.get('mean_change_m', 'N/A')} m
- Declining Cells: {stats.get('declining_cells', 'N/A')}
- Recovering Cells: {stats.get('recovering_cells', 'N/A')}
- Mean R²: {stats.get('mean_r_squared', 'N/A')}
- GRACE Contribution: {stats.get('mean_grace_contribution', 'N/A')} m
Confidence: {interp.get('confidence', 'N/A')}""")
    
    elif active_module == 'RECHARGE' and 'recharge' in data_summary:
        rch = data_summary['recharge']
        potential = rch.get('potential', {})
        params = rch.get('analysis_parameters', {})
        context_parts.append(f"""
Recharge Planning:
- Total Potential: {potential.get('total_recharge_potential_mcm', 'N/A')} MCM/year
- Per km²: {potential.get('per_km2_mcm', 'N/A')} MCM
- Dominant Lithology: {params.get('dominant_lithology', 'N/A')}
- Structure Types: {len(rch.get('structure_plan', []))} types planned""")
    
    elif active_module == 'SIGNIFICANT_TRENDS' and 'significant_trends' in data_summary:
        st = data_summary['significant_trends']
        stats = st.get('statistics', {})
        context_parts.append(f"""
Significant Trends:
- Total Significant Sites: {stats.get('total_significant', 'N/A')}
- Declining: {stats.get('declining', 'N/A')}
- Recovering: {stats.get('recovering', 'N/A')}
- Mean Slope: {stats.get('mean_slope', 'N/A')} m/year
- P-value Threshold: {st.get('p_threshold', 'N/A')}""")
    
    elif active_module == 'CHANGEPOINTS' and 'changepoints' in data_summary:
        cp = data_summary['changepoints']
        stats = cp.get('statistics', {})
        context_parts.append(f"""
Changepoint Detection:
- Sites Analyzed: {stats.get('sites_analyzed', 'N/A')}
- Sites with Changepoints: {stats.get('sites_with_changepoints', 'N/A')}
- Detection Rate: {stats.get('detection_rate', 'N/A')}%
- Changepoints Found: {cp.get('changepoints_found', 'N/A')}""")
    
    elif active_module == 'LAG_CORRELATION' and 'lag_correlation' in data_summary:
        lc = data_summary['lag_correlation']
        stats = lc.get('statistics', {})
        context_parts.append(f"""
Rainfall-GWL Lag Analysis:
- Sites Analyzed: {lc.get('sites_analyzed', 'N/A')}
- Mean Lag: {stats.get('mean_lag', 'N/A')} months
- Median Lag: {stats.get('median_lag', 'N/A')} months
- Mean Correlation: {stats.get('mean_abs_correlation', 'N/A')}""")
    
    elif active_module == 'HOTSPOTS' and 'hotspots' in data_summary:
        hs = data_summary['hotspots']
        stats = hs.get('statistics', {})
        context_parts.append(f"""
Decline Hotspots (DBSCAN Clustering):
- Declining Sites: {stats.get('total_declining_sites', 'N/A')}
- Clusters Found: {stats.get('n_clusters', 'N/A')}
- Clustered Points: {stats.get('clustered_points', 'N/A')}
- Clustering Rate: {stats.get('clustering_rate', 'N/A')}%""")
    
    elif active_module == 'GRACE_DIVERGENCE' and 'divergence' in data_summary:
        div = data_summary['divergence']
        stats = div.get('statistics', {})
        interp = div.get('interpretation', {})
        context_parts.append(f"""
GRACE vs Ground Divergence:
- Mean Divergence: {stats.get('mean_divergence', 'N/A')}
- Positive Pixels: {stats.get('positive_divergence_pixels', 'N/A')}
- Negative Pixels: {stats.get('negative_divergence_pixels', 'N/A')}
Agreement: {interp.get('overall_agreement', 'N/A')}""")
    
    else:
        # Fallback: module is set but no data, or unknown module
        context_parts.append(f"No data available for {active_module} module yet.")
    
    return "\n".join(context_parts)


# =====================================================================
# AGENT SETUP
# =====================================================================

print("🤖 Initializing LLM...")

llm = ChatOllama(
    model="gemma3:12b",
    temperature=0,
    keep_alive="10m",
    num_ctx=8192
)

agent_prompt = PromptTemplate.from_template("""
You are a hydrogeology expert assistant integrated with the GeoHydro dashboard.

**CURRENT DASHBOARD CONTEXT:**
{module_context}

**YOUR CAPABILITIES:**
1. **FRONTEND DATA**: User is viewing {current_module} for {current_location}
   - You have access to the statistics currently displayed on screen
   - Use this for questions about "current", "this", "here"

2. **KNOWLEDGE BASE (RAG)**: Definitions, formulas, methodologies
   - Use for "what is X?", "how is X calculated?", "explain X"

3. **DATABASE (SQL)**: Real-time data for ANY Indian state/district
   - Use when user asks about DIFFERENT locations
   - Use for historical data, trends, comparisons

**DECISION TREE:**
==================

STEP 1: Is this a GREETING or GENERAL CHAT?
   Examples: "Hi", "Hello", "How are you?", "Thanks"
   → Answer directly, NO tools needed
   
STEP 2: Is this asking for DEFINITION or METHODOLOGY?
   Triggers: "what is", "define", "explain", "how is...calculated", "formula for"
   Examples: "What is ASI?", "How is SASS calculated?"
   → Use RAG tools: search_definition or search_analysis
   
STEP 3: Is this about the CURRENT MODULE and CURRENT LOCATION?
   Triggers: "here", "this", "current", "what's the", "show me the"
   Check: Does user mention {current_module}? OR use words like "here"/"current"?
   Example: User viewing ASI for Maharashtra, asks "What's the ASI here?"
   → Use frontend_data from module_context
   → DO NOT query database
   
STEP 4: Is this about a DIFFERENT LOCATION?
   Triggers: User mentions a state/district that is NOT {current_location}
   Example: User viewing Maharashtra, asks "What about Kerala?"
   → Use SQL tools to query database
   → Tools: get_aquifer_properties, get_groundwater_wells_summary, get_grace_data, etc.

STEP 5: Is this asking for TRENDS or TIMESERIES?
   Triggers: "trend", "over time", "historical", "changes", "pattern"
   → Use: get_timeseries_analysis


**AVAILABLE TOOLS:**
{tools}

Tool names: [{tool_names}]


**RESPONSE FORMAT:**

For questions about CURRENT module (Step 3):
-------------------------------------------
Thought: User is asking about {current_module} for {current_location}, which is currently open. I should use the frontend data provided.
Final Answer: [Answer using data from module_context, DO NOT call any tools]

For DEFINITION questions (Step 2):
---------------------------------
Thought: User wants to know what X means. I should use the knowledge base.
Action: search_definition
Action Input: term="X"
Observation: [RAG returns definition]
Thought: I now have the definition
Final Answer: [Explain the definition naturally]

For DIFFERENT LOCATION questions (Step 4):
------------------------------------------
Thought: User is asking about [different state/district], which is different from current location {current_location}. I need to query the database.
Action: [appropriate tool]
Action Input: state="StateName", district="DistrictName"
Observation: [Database returns data]
Thought: I now have the data
Final Answer: [Present the data naturally]


**CRITICAL RULES:**
===================
1. **ALWAYS check module_context first** - if data is already there, USE IT
2. **DO NOT query database** for current location if data is in module_context
3. **Use exact format** for tool inputs: state="Kerala", NOT {{'state': 'Kerala'}}
4. **After ONE tool call**, provide Final Answer immediately
5. **Keep answers conversational**, not robotic
6. **Never say "based on the data provided"** - just state the findings


**EXAMPLES:**
=============

Example 1: Current Module Question
User viewing: ASI for Maharashtra
User asks: "What's the ASI score here?"

Thought: User is asking about ASI for Maharashtra, which is the current module open. I should use frontend data.
Final Answer: The Aquifer Suitability Index (ASI) for Maharashtra is 3.2 out of 5.0, indicating good aquifer suitability. About 45% of the region has high-suitability areas, dominated by alluvial aquifers with an average specific yield of 8%.


Example 2: Definition Question  
User asks: "What is ASI?"

Thought: User wants to know what ASI means. I should search the knowledge base.
Action: search_definition
Action Input: term="ASI"
Observation: ASI: Aquifer Suitability Index - A normalized score (0-5) measuring an aquifer's capacity...
Thought: I now have the definition
Final Answer: ASI stands for Aquifer Suitability Index. It's a score from 0 to 5 that measures how suitable an aquifer is for groundwater storage and transmission, based on the rock type and its ability to hold water (specific yield). Higher scores mean better aquifers.


Example 3: Different Location Question
User viewing: ASI for Maharashtra  
User asks: "What about Kerala?"

Thought: User is asking about Kerala, which is different from the current location Maharashtra. I need to query the database.
Action: get_aquifer_properties
Action Input: state="Kerala"
Observation: [Database returns Kerala aquifer data]
Thought: I now have Kerala's data
Final Answer: Kerala has different geology from Maharashtra. It's dominated by hard-rock formations (granite and gneiss) covering about 75% of the state, with coastal alluvial aquifers in the remaining 25%...


Example 4: Methodology Question
User asks: "How is ASI calculated?"

Thought: User wants to understand the calculation methodology. I should search the analysis knowledge base.
Action: search_analysis  
Action Input: name="ASI"
Observation: [RAG returns calculation steps]
Thought: I now have the methodology
Final Answer: ASI is calculated in 4 steps: First, we map each rock type to its specific yield (alluvium gets 0.10, sandstone 0.06, etc.). Then we find the 5th and 95th percentile values across the region. Next, we normalize each value to a 0-5 scale using the formula: ASI = ((Sy - q_low) / (q_high - q_low)) × 5. This stretching ensures the scores use the full range.


**NOW ANSWER THE USER'S QUESTION:**
Question: {input}

{agent_scratchpad}
"""
)

all_tools = [
    search_definition,
    search_analysis,
    get_aquifer_properties,
    get_groundwater_wells_summary,
    get_grace_data,
    get_rainfall_data,
    get_timeseries_analysis
]

agent = create_react_agent(
    llm=llm,
    tools=all_tools,
    prompt=agent_prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=all_tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True,
    early_stopping_method="force"
)

print("✅ Agent ready with database + knowledge base\n")


def invoke_agent_with_context(user_input: str, map_context: dict = None):
    """
    Invoke agent with module context from frontend
    
    Args:
        user_input: User's question
        map_context: Complete map context from frontend
    """
    
    # Build context string
    context_str = build_module_context_string(map_context) if map_context else ""
    
    # Extract current module and location for prompt variables
    if map_context:
        data_summary = map_context.get('data_summary', {})
        region = map_context.get('region', {})
        
        current_module = data_summary.get('active_module', 'None')
        
        state = region.get('state')
        district = region.get('district')
        if district and state:
            current_location = f"{district}, {state}"
        elif state:
            current_location = state
        else:
            current_location = "All India"
    else:
        current_module = "None"
        current_location = "Not specified"
    
    # Invoke agent
    result = agent_executor.invoke({
        "input": user_input,
        "module_context": context_str,
        "current_module": current_module,
        "current_location": current_location
    })
    
    return result


# =====================================================================
# REUSABLE AGENT FUNCTIONS (for FastAPI integration)
# =====================================================================

def get_agent_executor():
    """
    Returns a fresh agent executor instance for use in other modules (e.g., FastAPI).
    Creates a new executor each time to avoid state issues.
    """
    tools = [
        search_definition,
        search_analysis,
        get_aquifer_properties,
        get_groundwater_wells_summary,
        get_grace_data,
        get_rainfall_data,
        get_timeseries_analysis
    ]
    
    llm = ChatOllama(
        model="gemma3:12b",
        temperature=0,
        keep_alive="10m",
        num_ctx=8192
    )
    
    agent_prompt = PromptTemplate.from_template("""
You are a hydrogeology expert assistant with Three types of capabilities:

1. **KNOWLEDGE BASE**: Definitions, formulas, analysis methodologies
2. **DATABASE**: Real-time groundwater, rainfall, GRACE satellite data for Indian states
3. **CHATBOT**: Friendly conversationalist when asked very simple/normal questions

Available tools: {tools}
Tool names: [{tool_names}]

🔍 TOOL SELECTION GUIDE:
=========================

**For SIMPLE GREETINGS & CONVERSATIONAL QUESTIONS:**
- "Hi", "Hello", "Who are you?", "What's your name?" → Answer directly, NO tools needed
- Just respond naturally and conversationally

**For DEFINITIONS & CONCEPTS:**
- "What is X?" → search_definition
- "Define X" → search_definition
- Examples: "What is specific yield?", "Define aquifer"

**For ANALYSIS METHODS:**
- "How is X calculated?" → search_analysis
- "Explain X formula" → search_analysis
- Examples: "How is ASI calculated?", "Explain Mann-Kendall test"

**For REAL DATA:**
- "Groundwater in X state" → get_groundwater_wells_summary
- "GRACE data for X" → get_grace_data
- "Rainfall in X" → get_rainfall_data
- "Aquifers in X" → get_aquifer_properties
- "Groundwater trend in X" → get_timeseries_analysis

📋 RESPONSE RULES:
==================
1. For greetings/basic questions: Skip tools, answer directly in Final Answer
2. For technical questions: Use ONE tool, then give Final Answer
3. Keep answers natural and conversational
4. If you feel you dont have that much Data, just tell the user to refeer that particular module for more information.

📋 FORMAT (when using tools):
=============================
Thought: [reasoning]
Action: [tool name]
Action Input: state="StateName", year=2015

**CRITICAL**: Use this format: state="Maharashtra", year=2015
NOT: {{'state': 'Maharashtra', 'year': 2015}}

After ONE observation, provide Final Answer immediately.

Question: {input}

{agent_scratchpad}
""")
    
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=agent_prompt
    )
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,  # Set to False for production
        max_iterations=5,
        handle_parsing_errors=True,
        early_stopping_method="force"
    )


def query_agent(question: str, map_context: dict = None, verbose: bool = False) -> dict:
    """
    Query the agent with a question and optional map context.
    
    Args:
        question: The user's question
        map_context: Optional map context from frontend (region, temporal, data_summary, etc.)
        verbose: Whether to print debug information
        
    Returns:
        dict with 'output' (answer), 'success' (bool), and optionally 'error' message
    """
    try:
        # Use context-aware invocation if map_context is provided
        if map_context:
            result = invoke_agent_with_context(question, map_context)
        else:
            # Use basic agent without context
            agent_executor = get_agent_executor()
            result = agent_executor.invoke({"input": question})
        
        return {
            "output": result.get("output", "No response generated"),
            "success": True
        }
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        if verbose:
            import traceback
            traceback.print_exc()
        return {
            "output": error_msg,
            "success": False,
            "error": str(e)
        }


# =====================================================================
# MAIN LOOP (UPDATED)
# =====================================================================

def main():
    print("="*70)
    print("🌊 INTEGRATED GROUNDWATER ASSISTANT")
    print("="*70)
    print("\n📚 KNOWLEDGE BASE QUERIES:")
    print("  • What is specific yield?")
    print("  • How is ASI calculated?")
    print("  • Explain Mann-Kendall test")
    
    print("\n📊 DATABASE QUERIES:")
    print("  • Groundwater levels in Kerala")
    print("  • GRACE data for Maharashtra 2023")
    print("  • Rainfall in Tamil Nadu")
    print("  • Show groundwater trends in Punjab")
    
    print("\nType 'exit' to quit\n")
    print("="*70 + "\n")
    
    # Example map context (in production, this comes from frontend)
    example_map_context = {
        "region": {"state": "Maharashtra", "district": None},
        "temporal": {"year": 2023, "month": 6, "season": None},
        "data_summary": {
            "active_module": "ASI",
            "asi": {
                "statistics": {
                    "mean_asi": 3.2,
                    "dominant_aquifer": "Alluvium",
                    "avg_specific_yield": 0.08,
                    "total_area_km2": 307713
                },
                "interpretation": {
                    "regional_rating": "Good",
                    "high_suitability_percentage": 45,
                    "regional_narrative": "Maharashtra exhibits good aquifer suitability with favorable geological conditions."
                }
            }
        }
    }
    
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("\n👋 Goodbye!\n")
            break
        
        try:
            print()
            # Pass map context to agent
            result = invoke_agent_with_context(user_input, example_map_context)
            
            print("\n" + "="*70)
            print("🤖 ANSWER:")
            print("="*70)
            print(result["output"])
            print("="*70 + "\n")
        
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()