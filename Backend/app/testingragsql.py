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
import logging
from logging.handlers import RotatingFileHandler

load_dotenv()

# =====================================================================
# LOGGING CONFIGURATION
# =====================================================================

def setup_logger():
    """Configure comprehensive logging for debugging"""
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('GroundwaterAgent')
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler - INFO level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler - DEBUG level with rotation
    file_handler = RotatingFileHandler(
        'logs/agent_debug.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Initialize logger
logger = setup_logger()
logger.info("="*70)
logger.info("GROUNDWATER AGENT INITIALIZED")
logger.info("="*70)

# =====================================================================
# GLOBAL THREAD POOL
# =====================================================================
EXECUTOR = ThreadPoolExecutor(max_workers=6)
logger.debug("Thread pool executor initialized with 6 workers")

# =====================================================================
# DATABASE CONNECTION
# =====================================================================

DATABASE_URL = os.environ.get("DB_URI")
if not DATABASE_URL:
    logger.error("DB_URI not set in .env file")
    exit(1)

logger.info(f"Database URL configured: {DATABASE_URL[:20]}...")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
logger.debug("SQLAlchemy engine created")

try:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("Database connected successfully")
except Exception as e:
    logger.error(f"Database connection failed: {e}", exc_info=True)
    exit(1)


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def parse_numeric(value):
    """Safely parses messy database strings into a single float."""
    logger.debug(f"parse_numeric called with value: {value}")
    
    if not value:
        logger.debug("Value is None/empty, returning 0.0")
        return 0.0
    
    s = str(value).strip().lower()
    
    if s in ['-', '', ' ', 'na', 'n/a', 'nil']:
        logger.debug(f"Value '{s}' is invalid marker, returning 0.0")
        return 0.0
        
    try:
        if '-' in s:
            parts = [float(p.strip()) for p in s.split('-') if p.strip().replace('.','',1).isdigit()]
            if len(parts) >= 2:
                result = sum(parts) / len(parts)
                logger.debug(f"Parsed range '{s}' to average: {result}")
                return result
        
        if ' to ' in s:
            parts = [float(p.strip()) for p in s.split(' to ') if p.strip().replace('.','',1).isdigit()]
            if len(parts) >= 2:
                result = sum(parts) / len(parts)
                logger.debug(f"Parsed 'to' range '{s}' to average: {result}")
                return result

        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", s)
        if numbers:
            result = float(numbers[0])
            logger.debug(f"Extracted number from '{s}': {result}")
            return result
            
        logger.debug(f"No numbers found in '{s}', returning 0.0")
        return 0.0
        
    except Exception as e:
        logger.warning(f"Error parsing '{value}': {e}")
        return 0.0

def get_boundary_geojson(conn, state: str, district: str = None):
    """Fetch the boundary GeoJSON for a state or district."""
    logger.debug(f"get_boundary_geojson called: state={state}, district={district}")
    
    if district and state:
        query = text("""
            SELECT ST_AsGeoJSON(ST_MakeValid(geometry)) as geojson
            FROM public.district_state
            WHERE UPPER("District") = UPPER(:district)
            AND UPPER("State") = UPPER(:state)
            LIMIT 1;
        """)
        params = {"district": district, "state": state}
        logger.debug(f"Querying district boundary: {district}, {state}")
    elif state:
        query = text("""
            SELECT ST_AsGeoJSON(ST_Union(ST_MakeValid(geometry))) as geojson
            FROM public.district_state
            WHERE UPPER("State") = UPPER(:state);
        """)
        params = {"state": state}
        logger.debug(f"Querying state boundary: {state}")
    else:
        logger.warning("No state provided to get_boundary_geojson")
        return None
    
    result = conn.execute(query, params).fetchone()
    if result:
        logger.debug(f"Boundary GeoJSON retrieved successfully (length: {len(result[0])})")
    else:
        logger.warning(f"No boundary found for state={state}, district={district}")
    return result[0] if result else None

def get_month_day_range(year: int, month: int):
    """Get day-of-year range for a given month"""
    logger.debug(f"get_month_day_range: year={year}, month={month}")
    first_day = datetime(year, month, 1).timetuple().tm_yday
    last_day_of_month = calendar.monthrange(year, month)[1]
    last_day = datetime(year, month, last_day_of_month).timetuple().tm_yday
    logger.debug(f"Day range: {first_day} to {last_day}")
    return first_day, last_day

# =====================================================================
# GRACE BAND MAPPING
# =====================================================================

GRACE_BAND_MAPPING = {}

def auto_detect_grace_bands():
    global GRACE_BAND_MAPPING
    logger.info("Auto-detecting GRACE band mappings...")
    try:
        with engine.connect() as conn:
            total_bands_query = text("SELECT ST_NumBands(rast) FROM public.grace_lwe WHERE rid = 1;")
            total_bands = conn.execute(total_bands_query).fetchone()[0]
            logger.info(f"Total GRACE bands found: {total_bands}")
            
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
                        logger.debug(f"Checked {band}/{total_bands} bands...")
            
            logger.info(f"Detected {len(GRACE_BAND_MAPPING)} available GRACE months")
    except Exception as e:
        logger.error(f"Error auto-detecting GRACE bands: {e}", exc_info=True)

auto_detect_grace_bands()

# =====================================================================
# RAINFALL TABLE DETECTION
# =====================================================================

RAINFALL_TABLES = {}

def detect_rainfall_tables():
    global RAINFALL_TABLES
    logger.info("Detecting rainfall tables...")
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
            
            logger.info(f"Detected {len(RAINFALL_TABLES)} rainfall years: {sorted(RAINFALL_TABLES.keys())}")
    except Exception as e:
        logger.error(f"Error detecting rainfall tables: {e}", exc_info=True)

detect_rainfall_tables()

def parse_agent_input(input_str):
    """
    Parse agent input that might be in format: state="Maharashtra", year=2015
    Returns a dictionary with extracted parameters.
    """
    logger.debug(f"parse_agent_input called with: {input_str}")
    
    if not input_str or not isinstance(input_str, str):
        logger.debug("Input is None or not string, returning empty dict")
        return {}
    
    result = {}
    
    # Pattern to match: key="value" or key=value
    pattern = r'(\w+)\s*=\s*["\']?([^,"\'\n]+)["\']?'
    matches = re.findall(pattern, input_str)
    
    logger.debug(f"Found {len(matches)} parameter matches")
    
    for key, value in matches:
        key = key.strip().lower()
        value = value.strip().strip('"').strip("'")
        
        # Try to convert to int
        if value.isdigit():
            result[key] = int(value)
            logger.debug(f"Parsed parameter: {key}={result[key]} (int)")
        else:
            result[key] = value
            logger.debug(f"Parsed parameter: {key}={result[key]} (str)")
    
    return result

# =====================================================================
# RAG SYSTEM INITIALIZATION
# =====================================================================

class RAGSystem:
    """Handles definitions and analysis knowledge base"""
    
    def __init__(self, definitions_file: str, analyses_file: str):
        logger.info("Initializing RAG Knowledge Base...")
        print("\nINFO: Initializing RAG Knowledge Base...")
        
        # Load data
        self.definitions = self._load_jsonl(definitions_file)
        self.analyses = self._load_jsonl(analyses_file)
        
        logger.info(f"Loaded {len(self.definitions)} definitions")
        logger.info(f"Loaded {len(self.analyses)} analyses")
        print(f"   Loaded {len(self.definitions)} definitions")
        print(f"   Loaded {len(self.analyses)} analyses")
        
        # Setup embeddings
        logger.debug("Initializing HuggingFace embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create vector stores
        logger.debug("Creating Chroma persistent client...")
        client = chromadb.PersistentClient(path="./chroma_groundwater")
        
        # Definitions store
        logger.debug("Building definitions vector store...")
        def_texts = []
        for d in self.definitions:
            if 'term' in d and 'definition' in d:
                text = f"{d['term']}: {d['definition']}"
                if d.get('formula'):
                    text += f"\nFormula: {d['formula']}"
                if d.get('units'):
                    text += f"\nUnits: {d['units']}"
                def_texts.append(text)
        
        logger.debug(f"Creating definitions collection with {len(def_texts)} texts")
        self.def_store = Chroma.from_texts(
            texts=def_texts,
            embedding=embeddings,
            client=client,
            collection_name="definitions"
        )
        
        # Analyses store
        logger.debug("Building analyses vector store...")
        analysis_texts = []
        for a in self.analyses:
            if 'analysis_name' in a and 'description' in a:
                text = f"{a['analysis_name']}: {a['description']}"
                if a.get('formula'):
                    text += f"\nFormula: {json.dumps(a['formula']) if isinstance(a['formula'], dict) else a['formula']}"
                if a.get('steps'):
                    text += f"\nSteps: {json.dumps(a['steps'])}"
                analysis_texts.append(text)
        
        logger.debug(f"Creating analyses collection with {len(analysis_texts)} texts")
        self.analysis_store = Chroma.from_texts(
            texts=analysis_texts,
            embedding=embeddings,
            client=client,
            collection_name="analyses"
        )
        
        logger.info("RAG system ready")
        print("   RAG system ready\n")
    
    def _load_jsonl(self, filepath: str) -> List[Dict]:
        """Load JSONL file"""
        logger.debug(f"Loading JSONL file: {filepath}")
        data = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except Exception as e:
                            logger.warning(f"Failed to parse line {line_num} in {filepath}: {e}")
            logger.debug(f"Successfully loaded {len(data)} entries from {filepath}")
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            print(f"WARNING: File not found: {filepath}")
        return data
    
    def search_definition(self, term: str) -> str:
        """Search for term definition"""
        logger.debug(f"RAG search_definition called for term: '{term}'")
        results = self.def_store.similarity_search(term, k=2)
        
        if not results:
            logger.debug(f"No definition found for '{term}'")
            return f"No definition found for '{term}'"
        
        logger.debug(f"Found {len(results)} definition results for '{term}'")
        output = ["DEFINITION:"]
        for doc in results:
            output.append(doc.page_content)
        return "\n".join(output)
    
    def search_analysis(self, name: str) -> str:
        """Search for analysis methodology"""
        logger.debug(f"RAG search_analysis called for: '{name}'")
        
        # Try exact match first
        for analysis in self.analyses:
            if 'analysis_name' in analysis and name.lower() in analysis['analysis_name'].lower():
                logger.debug(f"Found exact match for analysis: {analysis['analysis_name']}")
                output = [
                    f"ANALYSIS: {analysis['analysis_name']}",
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
        logger.debug(f"No exact match, using vector search for '{name}'")
        results = self.analysis_store.similarity_search(name, k=1)
        if results:
            logger.debug("Found analysis via vector search")
            return f"ANALYSIS:\n{results[0].page_content}"
        
        logger.debug(f"No analysis found for '{name}'")
        return f"No analysis found for '{name}'"


# Initialize RAG (adjust paths as needed)
try:
    logger.info("Attempting to initialize RAG system...")
    RAG = RAGSystem(
        definitions_file="D:/Repos/Work repo/NRSC Internship/dashboard/Backend/app/data/definitions.jsonl",
        analyses_file="D:/Repos/Work repo/NRSC Internship/dashboard/Backend/app/data/analysis_definition.jsonl"
    )
    logger.info("RAG system initialized successfully")
except Exception as e:
    logger.error(f"RAG system initialization failed: {e}", exc_info=True)
    print(f"WARNING: RAG system not available: {e}")
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
    logger.info(f"TOOL CALLED: search_definition(term='{term}')")
    
    if RAG is None:
        logger.warning("RAG system not available")
        return "ERROR: Knowledge base not available"
    
    result = RAG.search_definition(term)
    logger.info(f"search_definition completed, result length: {len(result)}")
    return result

@tool
def search_analysis(name: str) -> str:
    """
    Search for how analyses and calculations work.
    Use when user asks 'how is X calculated?' or 'explain X analysis' or 'what is X formula?'.
    
    Args:
        name: The analysis name (e.g., "ASI", "SASS", "Mann-Kendall", "water balance")
    """
    logger.info(f"TOOL CALLED: search_analysis(name='{name}')")
    
    if RAG is None:
        logger.warning("RAG system not available")
        return "ERROR: Knowledge base not available"
    
    result = RAG.search_analysis(name)
    logger.info(f"search_analysis completed, result length: {len(result)}")
    return result


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
    logger.info(f"TOOL CALLED: get_aquifer_properties(state='{state}', district='{district}')")
    
    # Clean inputs
    if state:
        state = str(state).strip().strip('"').strip("'").strip()
        # Remove parameter prefix if present
        if state.startswith('state='):
            state = state[6:].strip().strip('"').strip("'").strip()
    
    if district:
        district = str(district).strip().strip('"').strip("'").strip()
        if district.startswith('district='):
            district = district[9:].strip().strip('"').strip("'").strip()
    
    if not state or state.lower() in ['none', 'null', '']:
        logger.warning("No valid state provided")
        return "ERROR: State name is required"
    
    logger.info(f"Querying aquifers: {state}, {district or 'all districts'}")
    print(f"\nINFO: Querying aquifers: {state}, {district or 'all districts'}")
    
    try:
        with engine.connect() as conn:
            logger.debug("Getting boundary GeoJSON...")
            state_geojson = get_boundary_geojson(conn, state, district)
            
            if not state_geojson:
                logger.warning(f"Location not found: {state}")
                return f"ERROR: Location not found: {state}"

            normalized = state.replace(',', ' ').replace('&', ' ').strip()
            state_pattern = '%' + '%'.join(normalized.split()) + '%'
            logger.debug(f"Using state pattern: {state_pattern}")

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
            
            logger.debug("Executing aquifer query...")
            result = conn.execute(aquifer_query, {
                "state_geojson": state_geojson, 
                "state_pattern": state_pattern
            })
            
            rows = result.fetchall()
            logger.debug(f"Query returned {len(rows)} rows")

            if not rows:
                logger.warning(f"No aquifer data found for {state}")
                return f"ERROR: No aquifer data found for {state}"

            stats = {}
            total_area = 0.0

            for row_num, row in enumerate(rows, 1):
                try:
                    major_type = row[0] if row[0] else "Unclassified"
                    specific_name = row[1] if row[1] else "Unknown"
                    
                    zone_val = parse_numeric(row[2])
                    depth_val = parse_numeric(row[3]) or parse_numeric(row[4])
                    
                    area = float(row[6]) if row[6] else 0.0
                    
                    if area <= 0: 
                        continue

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
                    
                except Exception as e:
                    logger.warning(f"Error processing row {row_num}: {e}")
                    continue

            if total_area == 0:
                logger.warning("Data found but area calculations failed")
                return f"ERROR: Data found but area calculations failed"

            location = f"{district}, {state}" if district else state
            response = [f"\nAQUIFER REPORT: {location.upper()}"]
            response.append(f"Total Area: {total_area:,.2f} km²\n")
            
            sorted_stats = sorted(stats.items(), key=lambda x: x[1]['area'], reverse=True)
            
            for aq_type, data in sorted_stats:
                pct = (data['area'] / total_area) * 100
                avg_depth = np.mean(data['depths']) if data['depths'] else 0
                
                response.append(f"Type: {aq_type}")
                response.append(f"   Coverage: {data['area']:,.1f} km² ({pct:.1f}%)")
                response.append(f"   Avg Water Depth: {avg_depth:.1f} m bgl")
            
            result_str = "\n".join(response)
            logger.info(f"get_aquifer_properties completed successfully, {len(sorted_stats)} aquifer types")
            return result_str

    except Exception as e:
        logger.error(f"Error in get_aquifer_properties: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return f"ERROR: {str(e)}"


@tool
def get_groundwater_wells_summary(state: str, district: str = None, year: int = None) -> str:
    """
    Get groundwater level statistics and trends.
    
    Args:
        state: State name (required, e.g., "Kerala", "Maharashtra")
        district: District name (optional)
        year: Year to analyze (optional, defaults to all available data)
    """
    logger.info(f"TOOL CALLED: get_groundwater_wells_summary(state='{state}', district='{district}', year={year})")
    
    # ✅ STEP 1: Parse if input is a string like 'state="Maharashtra", year=2015'
    if isinstance(state, str) and '=' in state:
        logger.debug("Input is string with '=', parsing...")
        parsed = parse_agent_input(state)
        if parsed:
            state = parsed.get('state', '')
            district = parsed.get('district', district)
            year = parsed.get('year', year)
            logger.debug(f"Parsed to: state='{state}', district='{district}', year={year}")
    
    # ✅ STEP 2: Handle dictionary input
    elif isinstance(state, dict):
        logger.debug("Input is dictionary, extracting values...")
        district = state.get('district', district)
        year = state.get('year', year)
        state = state.get('state', '')
        logger.debug(f"Extracted: state='{state}', district='{district}', year={year}")
    
    # ✅ STEP 3: Clean all inputs
    if state:
        state = str(state).strip().strip('"').strip("'").strip()
    if district:
        district = str(district).strip().strip('"').strip("'").strip() if district else None
    if year:
        try:
            year = int(year) if year else None
        except (ValueError, TypeError):
            logger.warning(f"Invalid year value: {year}, setting to None")
            year = None
    
    if not state or state.lower() in ['none', 'null', '']:
        logger.warning("No valid state provided")
        return "ERROR: State name is required"
    
    if district and district.lower() in ['none', 'null', '']:
        district = None
    
    location = f"{district}, {state}" if district else state
    logger.info(f"Querying wells: {location}, year={year or 'all'}")
    print(f"\nINFO: Querying wells: {location}, year={year or 'all'}")
    
    try:
        with engine.connect() as conn:
            logger.debug("Getting boundary GeoJSON...")
            boundary_geojson = get_boundary_geojson(conn, state, district)
            
            if not boundary_geojson:
                logger.warning(f"Location not found: '{location}'")
                return f"ERROR: Location not found: '{location}'"
            
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
                logger.debug(f"Added year filter: {year}")
            
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
            
            logger.debug("Executing wells query...")
            result = conn.execute(query, params)
            data = [(row[0], float(row[1]), int(row[2])) for row in result]
            logger.debug(f"Query returned {len(data)} months of data")
            
            if len(data) == 0:
                logger.warning(f"No data available for {location}" + (f" in {year}" if year else ""))
                return f"ERROR: No data available for {location}" + (f" in {year}" if year else "")
            
            dates = [d[0] for d in data]
            gwl_values = np.array([d[1] for d in data])
            n_months = len(data)
            
            response = [f"\nGROUNDWATER REPORT: {location.upper()}"]
            response.append(f"Period: {dates[0].date()} to {dates[-1].date()}")
            response.append(f"Months: {n_months}, Readings: {sum(d[2] for d in data):,}\n")
            
            response.append(f"Mean Depth: {np.mean(gwl_values):.2f} m bgl")
            response.append(f"Range: {np.min(gwl_values):.2f} to {np.max(gwl_values):.2f} m")
            
            TREND_THRESHOLD = 3
            
            if n_months >= TREND_THRESHOLD:
                logger.debug("Calculating trend (sufficient data)...")
                x = np.array([(d - dates[0]).days for d in dates])
                y = gwl_values
                
                slope, _, r_value, p_value, _ = linregress(x, y)
                
                slope_per_year = slope * 365.25
                r_squared = r_value ** 2
                
                logger.debug(f"Trend: slope={slope_per_year:.4f} m/year, R²={r_squared:.3f}")
                
                response.append(f"\nTrend: {'DECLINING' if slope_per_year > 0 else 'RECOVERING'}")
                response.append(f"Rate: {abs(slope_per_year):.4f} m/year")
                response.append(f"Confidence (R²): {r_squared:.3f}")
                
                if slope_per_year > 1.0:
                    response.append("WARNING: Rapid decline!")
                elif slope_per_year < -0.5:
                    response.append("POSITIVE: Water table recovering!")
            else:
                logger.debug(f"Trend unavailable - only {n_months} month(s)")
                response.append(f"\nWARNING: Trend unavailable - only {n_months} month(s)")
            
            result_str = "\n".join(response)
            logger.info("get_groundwater_wells_summary completed successfully")
            return result_str
    
    except Exception as e:
        logger.error(f"Error in get_groundwater_wells_summary: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return f"ERROR: {str(e)}"



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
    logger.info(f"TOOL CALLED: get_grace_data(state='{state}', district='{district}', year={year}, month={month})")
    
    # ✅ STEP 1: Parse if input is a string like 'state="Maharashtra", year=2015'
    if isinstance(state, str) and '=' in state:
        logger.debug("Parsing string input...")
        parsed = parse_agent_input(state)
        if parsed:
            state = parsed.get('state', '')
            district = parsed.get('district', district)
            year = parsed.get('year', year)
            month = parsed.get('month', month)
            logger.debug(f"Parsed: state={state}, district={district}, year={year}, month={month}")
    
    # ✅ STEP 2: Handle dictionary input
    elif isinstance(state, dict):
        logger.debug("Extracting from dictionary...")
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
        logger.warning("No valid state provided")
        return "ERROR: State name is required"
    
    location = f"{district}, {state}" if district else state
    
    if year is None:
        year = max(y for y, m in GRACE_BAND_MAPPING.keys())
        logger.debug(f"Using latest year: {year}")
    
    logger.info(f"Querying GRACE: {location}, {year}-{month or 'annual'}")
    print(f"\nINFO: Querying GRACE: {location}, {year}-{month or 'annual'}")
    
    try:
        with engine.connect() as conn:
            if month:
                if (year, month) not in GRACE_BAND_MAPPING:
                    logger.warning(f"No GRACE data for {year}-{month:02d}")
                    return f"ERROR: No GRACE data for {year}-{month:02d}"
                bands = [GRACE_BAND_MAPPING[(year, month)]]
                logger.debug(f"Using band {bands[0]} for {year}-{month:02d}")
            else:
                bands = [GRACE_BAND_MAPPING[(y, m)] for y, m in GRACE_BAND_MAPPING.keys() if y == year]
                if not bands:
                    logger.warning(f"No GRACE data for {year}")
                    return f"ERROR: No GRACE data for {year}"
                logger.debug(f"Using {len(bands)} bands for year {year}")
            
            boundary_geojson = get_boundary_geojson(conn, state, district)
            
            if not boundary_geojson:
                logger.warning(f"Location not found: {location}")
                return f"ERROR: Location not found: {location}"
            
            all_values = []
            
            for band_num, band in enumerate(bands, 1):
                logger.debug(f"Processing band {band_num}/{len(bands)}")
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
                logger.debug(f"Band {band}: {len(pixels)} pixels")
            
            if not all_values:
                logger.warning("No GRACE data retrieved")
                return f"ERROR: No GRACE data retrieved"
            
            lats, values = zip(*all_values)
            weights = np.cos(np.radians(lats))
            weighted_avg = np.sum(np.array(values) * weights) / np.sum(weights)
            
            logger.debug(f"Total pixels: {len(all_values)}, weighted average: {weighted_avg:.2f}")
            
            response = [f"\nGRACE REPORT: {location.upper()}"]
            response.append(f"Period: {year}-{month or 'annual'}")
            response.append(f"Pixels: {len(all_values):,}\n")
            
            response.append(f"Average TWS: {weighted_avg:.2f} cm")
            response.append(f"Range: {min(values):.2f} to {max(values):.2f} cm")
            
            if weighted_avg > 0:
                response.append("Status: More water than baseline (POSITIVE)")
            else:
                response.append("Status: Less water than baseline (WARNING)")
            
            result_str = "\n".join(response)
            logger.info("get_grace_data completed successfully")
            return result_str
    
    except Exception as e:
        logger.error(f"Error in get_grace_data: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return f"ERROR: {str(e)}"


@tool
def get_rainfall_data(state: str, district: str = None, year: int = None, month: int = None) -> str:
    """Get rainfall statistics."""
    logger.info(f"TOOL CALLED: get_rainfall_data(state='{state}', district='{district}', year={year}, month={month})")
    
    # ✅ STEP 1: Parse if input is a string
    if isinstance(state, str) and '=' in state:
        logger.debug("Parsing string input...")
        parsed = parse_agent_input(state)
        if parsed:
            state = parsed.get('state', '')
            district = parsed.get('district', district)
            year = parsed.get('year', year)
            month = parsed.get('month', month)
    
    # ✅ STEP 2: Handle dictionary input
    elif isinstance(state, dict):
        logger.debug("Extracting from dictionary...")
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
        logger.warning("No valid state provided")
        return "ERROR: State name is required"
    
    location = f"{district}, {state}" if district else state
    
    if year is None or year == 0:
        year = max(RAINFALL_TABLES.keys())
        logger.debug(f"Using latest year: {year}")
    
    if year not in RAINFALL_TABLES:
        logger.warning(f"No rainfall data for {year}")
        return f"ERROR: No rainfall data for {year}"
    
    table_name = RAINFALL_TABLES[year]
    logger.debug(f"Using table: {table_name}")
    
    logger.info(f"Querying rainfall: {location}, {year}-{month or 'annual'}")
    print(f"\nINFO: Querying rainfall: {location}, {year}-{month or 'annual'}")
    
    try:
        with engine.connect() as conn:
            if month:
                first_day, last_day = get_month_day_range(year, month)
                bands = list(range(first_day, last_day + 1))
                days_in_period = len(bands)
                logger.debug(f"Month {month}: bands {first_day}-{last_day} ({days_in_period} days)")
            else:
                days_in_year = 366 if calendar.isleap(year) else 365
                bands = list(range(1, days_in_year + 1))
                days_in_period = days_in_year
                logger.debug(f"Annual: {days_in_period} days")
            
            boundary_geojson = get_boundary_geojson(conn, state, district)
            
            if not boundary_geojson:
                logger.warning(f"Location not found: {location}")
                return f"ERROR: Location not found: {location}"
            
            all_values = []
            
            for band_num, band in enumerate(bands, 1):
                if band_num % 30 == 0:
                    logger.debug(f"Processing band {band_num}/{len(bands)}")
                
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
                logger.warning("No rainfall data retrieved")
                return f"ERROR: No rainfall data retrieved"
            
            logger.debug(f"Total values retrieved: {len(all_values)}")
            
            lats, values = zip(*all_values)
            weights = np.cos(np.radians(lats))
            daily_avg = np.sum(np.array(values) * weights) / np.sum(weights)
            total_rainfall = daily_avg * days_in_period
            
            logger.debug(f"Daily avg: {daily_avg:.2f}, Total: {total_rainfall:.2f}")
            
            response = [f"\nRAINFALL REPORT: {location.upper()}"]
            response.append(f"Period: {year}-{month or 'annual'}\n")
            
            response.append(f"Daily Average: {daily_avg:.2f} mm/day")
            response.append(f"Total: {total_rainfall:.2f} mm")
            response.append(f"Range: {min(values):.2f} to {max(values):.2f} mm")
            
            result_str = "\n".join(response)
            logger.info("get_rainfall_data completed successfully")
            return result_str
    
    except Exception as e:
        logger.error(f"Error in get_rainfall_data: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return f"ERROR: {str(e)}"

# =====================================================================
# TIMESERIES TOOL & HELPERS
# =====================================================================

def query_wells_monthly_for_tool(conn, boundary_geojson):
    """Query monthly groundwater levels"""
    logger.debug("query_wells_monthly_for_tool called")
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
    logger.debug(f"Wells query returned {len(data)} months")
    return pd.DataFrame(data)


def query_grace_monthly_for_tool(conn, boundary_geojson):
    """Query monthly GRACE data"""
    logger.debug("query_grace_monthly_for_tool called")
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
        except Exception as e:
            logger.warning(f"Error querying GRACE for {year}-{month}: {e}")
            continue
    
    logger.debug(f"GRACE query returned {len(grace_data)} months")
    return pd.DataFrame(grace_data)


def query_rainfall_monthly_for_tool(conn, boundary_geojson):
    """OPTIMIZED: Query monthly rainfall with parallel processing"""
    logger.debug("query_rainfall_monthly_for_tool called")
    
    if not RAINFALL_TABLES:
        logger.warning("No rainfall tables available")
        return pd.DataFrame(columns=['period', 'avg_rainfall'])
    
    logger.info(f"Querying rainfall: {len(RAINFALL_TABLES)} years (parallel)...")
    print(f"  INFO: Querying rainfall: {len(RAINFALL_TABLES)} years (parallel)...")
    
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
        except Exception as e:
            logger.warning(f"Error processing {year}-{month}: {e}")
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
    
    logger.info(f"Rainfall: {len(rainfall_data)} months")
    print(f"  SUCCESS: Rainfall: {len(rainfall_data)} months")
    
    return pd.DataFrame(rainfall_data)


def merge_timeseries_data(wells_df, grace_df, rainfall_df):
    """Merge all three datasets"""
    logger.debug("merge_timeseries_data called")
    
    if not wells_df.empty:
        combined = wells_df.copy()
        logger.debug(f"Starting with wells: {len(combined)} rows")
    else:
        combined = pd.DataFrame(columns=["period"])
        logger.debug("Starting with empty dataframe")
    
    if not grace_df.empty:
        combined = combined.merge(grace_df, on="period", how="outer")
        logger.debug(f"After GRACE merge: {len(combined)} rows")
    else:
        combined["avg_tws"] = np.nan
    
    if not rainfall_df.empty:
        combined = combined.merge(rainfall_df, on="period", how="outer")
        logger.debug(f"After rainfall merge: {len(combined)} rows")
    else:
        combined["avg_rainfall"] = np.nan
    
    combined = combined.sort_values("period").reset_index(drop=True)
    
    if not combined.empty:
        combined = combined.set_index("period")
        combined = combined.interpolate(method="time", limit_direction="both")
        combined = combined.reset_index()
        logger.debug("Interpolation completed")
    
    logger.debug(f"Final combined dataframe: {len(combined)} rows")
    return combined


def analyze_simple(df, location):
    """Comprehensive analysis of all three datasets - ENHANCED VERSION"""
    logger.debug(f"analyze_simple called for {location} with {len(df)} rows")
    
    if len(df) < 24:
        logger.warning(f"Need at least 24 months for analysis (found {len(df)})")
        return f"ERROR: Need at least 24 months for analysis (found {len(df)})"
    
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    response = [f"\n{'='*70}"]
    response.append(f"COMPREHENSIVE TIMESERIES ANALYSIS: {location.upper()}")
    response.append(f"{'='*70}")
    response.append(f"• Period: {df['period'].min().date()} to {df['period'].max().date()}")
    response.append(f"• Total Months: {len(df)}\n")
    
    # ===== 💧 GROUNDWATER ANALYSIS =====
    if 'avg_gwl' in df.columns and not df['avg_gwl'].isna().all():
        logger.debug("Analyzing groundwater data...")
        response.append("GROUNDWATER LEVEL ANALYSIS:")
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
            
            logger.debug(f"GWL trend: {slope_per_year:.4f} m/year, R²={r_squared:.3f}")
            
            # Current status
            current_gwl = series.iloc[-1]
            start_gwl = series.iloc[0]
            total_change = current_gwl - start_gwl
            
            response.append(f"   Starting Level ({df['period'].min().year}): {start_gwl:.2f}m below ground")
            response.append(f"   Current Level ({df['period'].max().year}): {current_gwl:.2f}m below ground")
            response.append(f"   Total Change: {abs(total_change):.2f}m {'DEEPER (DECLINING)' if total_change > 0 else 'SHALLOWER (RECOVERING)'}")
            
            # Trend direction and severity
            if slope_per_year > 0:
                trend_desc = "DECLINING (getting deeper)"
                if slope_per_year > 1.0:
                    severity = "WARNING: CRITICAL"
                    action = "Immediate water conservation measures needed"
                elif slope_per_year > 0.5:
                    severity = "WARNING: HIGH"
                    action = "Monitor closely and plan water management"
                else:
                    severity = "WARNING: MODERATE"
                    action = "Continue monitoring"
            else:
                trend_desc = "RECOVERING (getting shallower)"
                severity = "STATUS: POSITIVE"
                action = "Current practices working well"
            
            response.append(f"\n   {severity}: {trend_desc}")
            response.append(f"   Trend Rate: {abs(slope_per_year):.4f}m per year")
            response.append(f"   Confidence (R²): {r_squared:.3f} {'(STRONG)' if r_squared > 0.7 else '(MODERATE)' if r_squared > 0.5 else '(WEAK)'}")
            response.append(f"   Recommendation: {action}")
            
            # Year-by-year pattern (top 5 years)
            response.append(f"\n   YEAR-BY-YEAR PATTERN:")
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
        logger.debug("Analyzing GRACE data...")
        response.append("GRACE WATER STORAGE ANALYSIS:")
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
            
            logger.debug(f"GRACE trend: {slope_per_year:.4f} cm/year, R²={r_squared:.3f}")
            
            current_tws = series.iloc[-1]
            start_tws = series.iloc[0]
            total_change = current_tws - start_tws
            
            response.append(f"   Starting Storage: {start_tws:.2f}cm")
            response.append(f"   Current Storage: {current_tws:.2f}cm")
            response.append(f"   Total Change: {abs(total_change):.2f}cm {'LOSS (DECLINING)' if total_change < 0 else 'GAIN (INCREASING)'}")
            
            if slope_per_year > 0:
                response.append(f"\n   STATUS: INCREASING at {slope_per_year:.4f}cm/year")
                if slope_per_year > 2:
                    response.append("   Status: Significant water storage recovery")
            else:
                response.append(f"\n   WARNING: DECREASING at {abs(slope_per_year):.4f}cm/year")
                if abs(slope_per_year) > 2:
                    response.append("   Status: Significant water storage depletion")
            
            response.append(f"   Confidence (R²): {r_squared:.3f}")
        
        response.append("")
    
    # ===== 🌧️ RAINFALL PATTERN ANALYSIS =====
    if 'avg_rainfall' in df.columns and not df['avg_rainfall'].isna().all():
        logger.debug("Analyzing rainfall data...")
        response.append("RAINFALL PATTERN ANALYSIS:")
        response.append("-" * 70)
        
        df_copy = df.copy().set_index("period").sort_index()
        series = df_copy['avg_rainfall']
        
        # Calculate annual totals
        yearly = series.resample('Y').sum()
        
        response.append(f"   Average Annual: {yearly.mean():.0f}mm")
        response.append(f"   Wettest Year: {yearly.max():.0f}mm ({yearly.idxmax().year})")
        response.append(f"   Driest Year: {yearly.min():.0f}mm ({yearly.idxmin().year})")
        response.append(f"   Variability: {yearly.std():.0f}mm (std deviation)")
        
        # Monsoon analysis
        monsoon = series[series.index.month.isin([6,7,8,9])]
        if len(monsoon) > 0:
            response.append(f"   Monsoon Average: {monsoon.mean():.2f}mm/day")
        
        # Recent trend comparison
        if len(yearly) >= 6:
            recent_avg = yearly.tail(3).mean()
            older_avg = yearly.head(3).mean()
            change_pct = ((recent_avg - older_avg) / older_avg) * 100
            
            if recent_avg > older_avg * 1.1:
                response.append(f"\n   Recent Trend: WETTER ({change_pct:+.1f}% vs early period)")
            elif recent_avg < older_avg * 0.9:
                response.append(f"\n   Recent Trend: DRIER ({change_pct:+.1f}% vs early period)")
            else:
                response.append(f"\n   Recent Trend: STABLE ({change_pct:+.1f}% vs early period)")
        
        response.append("")
    
    # ===== 🔬 INTEGRATED INSIGHTS =====
    logger.debug("Generating integrated insights...")
    response.append("INTEGRATED INSIGHTS:")
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
                        response.append("WARNING: CLIMATE STRESS: Both groundwater AND rainfall declining")
                        response.append("   → Climate-driven water scarcity likely")
                        response.append("   → Recommend: Rainwater harvesting + conservation")
                    else:
                        response.append("WARNING: OVER-EXTRACTION: Groundwater declining despite stable rainfall")
                        response.append("   → Human usage exceeds natural recharge")
                        response.append("   → Recommend: Reduce extraction + artificial recharge")
                
                elif gwl_slope_per_year < -0.1:  # Recovering groundwater
                    if recent_rain > older_rain * 1.1:
                        response.append("STATUS: NATURAL RECOVERY: Groundwater recovering with increased rainfall")
                        response.append("   → Climate patterns favorable")
                    else:
                        response.append("STATUS: MANAGEMENT SUCCESS: Groundwater recovering despite stable rainfall")
                        response.append("   → Water conservation measures effective")
                        response.append("   → Recommend: Continue current practices")
                
                else:  # Stable groundwater
                    response.append("STATUS: EQUILIBRIUM: Groundwater stable")
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
                response.append("\nWARNING: CONFIRMED DEPLETION: Both groundwater wells AND satellite data show decline")
                response.append("   → High confidence in water stress assessment")
    
    # Final recommendations
    response.append(f"\n{'='*70}")
    response.append("RECOMMENDATIONS:")
    
    if has_gwl:
        if gwl_slope_per_year > 1.0:
            response.append("   URGENT: Implement immediate water restrictions")
            response.append("   URGENT: Promote water-efficient technologies")
            response.append("   URGENT: Develop artificial recharge projects")
        elif gwl_slope_per_year > 0.5:
            response.append("   WARNING: Monitor water usage patterns closely")
            response.append("   WARNING: Plan for water conservation measures")
            response.append("   WARNING: Consider rainwater harvesting initiatives")
        elif gwl_slope_per_year < -0.5:
            response.append("   SUCCESS: Continue successful water management practices")
            response.append("   SUCCESS: Document and share best practices")
        else:
            response.append("   INFO: Maintain current monitoring frequency")
            response.append("   INFO: Continue balanced water management")
    
    response.append(f"{'='*70}")
    
    result = "\n".join(response)
    logger.debug(f"Analysis complete, result length: {len(result)}")
    return result


@tool
def get_timeseries_analysis(state: str, district: str = None) -> str:
    """
    Comprehensive timeseries analysis showing groundwater trends over time.
    Use when user asks about trends, changes over time, or historical patterns.
    
    Args:
        state: State name (required)
        district: District name (optional)
    """
    logger.info(f"TOOL CALLED: get_timeseries_analysis(state='{state}', district='{district}')")
    
    # Parse string input
    if isinstance(state, str) and '=' in state:
        logger.debug("Parsing string input...")
        parsed = parse_agent_input(state)
        if parsed:
            state = parsed.get('state', '')
            district = parsed.get('district', district)
    
    # Handle dictionary input
    elif isinstance(state, dict):
        logger.debug("Extracting from dictionary input...")
        district = state.get('district', district)
        state = state.get('state', '')
    
    # Clean inputs
    if state:
        state = str(state).strip().strip('"').strip("'").strip()
    if district:
        district = str(district).strip().strip('"').strip("'").strip() if district else None
    
    if not state or state.lower() in ['none', 'null', '']:
        logger.warning("No valid state provided")
        return "ERROR: State name is required"
    
    location = f"{district}, {state}" if district else state
    logger.info(f"Analyzing timeseries: {location}")
    print(f"\nINFO: Analyzing timeseries: {location}")
    
    try:
        with engine.connect() as conn:
            logger.debug("Getting boundary GeoJSON...")
            boundary_geojson = get_boundary_geojson(conn, state, district)
            
            if not boundary_geojson:
                logger.warning(f"Location not found: {location}")
                return f"ERROR: Location not found: {location}"
            
            logger.info("Fetching data from multiple sources...")
            print("  INFO: Fetching data...")
            
            logger.debug("Querying wells data...")
            wells_data = query_wells_monthly_for_tool(conn, boundary_geojson)
            logger.debug(f"Wells data: {len(wells_data)} months")
            
            logger.debug("Querying GRACE data...")
            grace_data = query_grace_monthly_for_tool(conn, boundary_geojson)
            logger.debug(f"GRACE data: {len(grace_data)} months")
            
            logger.debug("Querying rainfall data...")
            rainfall_data = query_rainfall_monthly_for_tool(conn, boundary_geojson)
            logger.debug(f"Rainfall data: {len(rainfall_data)} months")
            
            logger.debug("Merging timeseries data...")
            combined = merge_timeseries_data(wells_data, grace_data, rainfall_data)
            logger.debug(f"Combined data: {len(combined)} months")
            
            if len(combined) == 0:
                logger.warning("No timeseries data available")
                return f"ERROR: No timeseries data available"
            
            logger.debug("Analyzing combined data...")
            result = analyze_simple(combined, location)
            
            logger.info("get_timeseries_analysis completed successfully")
            return result
    
    except Exception as e:
        logger.error(f"Error in get_timeseries_analysis: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return f"ERROR: {str(e)}"

# =============================================================================
# ADVANCED MODULE TOOLS
# =============================================================================

@tool
def get_asi_analysis(state: str, district: str = None) -> str:
    """
    Get Aquifer Suitability Index (ASI) analysis for a region.
    ASI is a 0-5 score measuring aquifer storage/transmission potential.
    
    Args:
        state: State name (required)
        district: District name (optional)
    """
    logger.info(f" TOOL CALLED: get_asi_analysis(state='{state}', district='{district}')")
    
    # Parse input
    if isinstance(state, str) and '=' in state:
        parsed = parse_agent_input(state)
        if parsed:
            state = parsed.get('state', '')
            district = parsed.get('district', district)
    elif isinstance(state, dict):
        district = state.get('district', district)
        state = state.get('state', '')
    
    # Clean inputs
    if state:
        state = str(state).strip().strip('"').strip("'").strip()
    if district:
        district = str(district).strip().strip('"').strip("'").strip() if district else None
    
    if not state or state.lower() in ['none', 'null', '']:
        logger.warning("No valid state provided")
        return "ERROR: State name is required"
    
    logger.info(f"Querying ASI: {location}")
    
    try:
        with engine.connect() as conn:
            boundary_geojson = get_boundary_geojson(conn, state, district)
            
            if not boundary_geojson:
                logger.warning(f"Location not found: {location}")
                return f"ERROR: Location not found: {location}"
            
            # Get aquifer data
            aquifer_query = text("""
                SELECT 
                    a.aquifers as majoraquif,
                    a.yeild__,
                    ST_Area(ST_Transform(ST_MakeValid(a.geometry), 32643)) / 1000000.0 as area_km2
                FROM public.aquifers a
                WHERE ST_Intersects(
                    ST_MakeValid(a.geometry),
                    ST_GeomFromGeoJSON(:boundary_geojson)
                )
            """)
            
            logger.debug("Executing ASI aquifer query...")
            result = conn.execute(aquifer_query, {"boundary_geojson": boundary_geojson})
            
            # Process data
            sy_map = {
                'alluvium': 0.10, 'sandstone': 0.06, 'limestone': 0.05,
                'basalt': 0.03, 'granite': 0.02
            }
            
            sy_values = []
            total_area = 0.0
            
            for row in result:
                yield_str = str(row[1]).lower() if row[1] else ""
                area = float(row[2]) if row[2] else 0.0
                
                # Parse specific yield
                sy = 0.04  # default
                try:
                    import re
                    numbers = re.findall(r'\d+\.?\d*', yield_str)
                    if numbers:
                        vals = [float(n) for n in numbers]
                        sy = sum(vals) / len(vals) / 100.0
                except:
                    majoraquif = str(row[0]).lower() if row[0] else ""
                    for key, val in sy_map.items():
                        if key in majoraquif:
                            sy = val
                            break
                
                if area > 0:
                    sy_values.append(sy)
                    total_area += area
            
            if not sy_values:
                logger.warning(f"No aquifer data for {location}")
                return f"ERROR: No aquifer data for {location}"
            
            # Calculate ASI
            sy_array = np.array(sy_values)
            q_low, q_high = np.quantile(sy_array, 0.05), np.quantile(sy_array, 0.95)
            
            if q_high - q_low < 1e-6:
                q_low, q_high = 0.01, 0.15
            
            asi_scores = ((sy_array.clip(q_low, q_high) - q_low) / (q_high - q_low) * 5.0).clip(0, 5)
            mean_asi = np.mean(asi_scores)
            
            logger.debug(f"ASI calculated: {mean_asi:.2f}/5.0")
            
            # Interpretation
            if mean_asi > 3.5:
                rating = "Excellent"
            elif mean_asi > 2.5:
                rating = "Good"
            elif mean_asi > 1.5:
                rating = "Moderate"
            else:
                rating = "Poor"
            
            response = [f"\nASI ANALYSIS: {location.upper()}"]
            response.append(f"Mean ASI Score: {mean_asi:.2f}/5.0")
            response.append(f"Rating: {rating}")
            response.append(f"Total Area: {total_area:,.2f} km²")
            response.append(f"High Suitability: {(asi_scores >= 3.5).sum() / len(asi_scores) * 100:.1f}%")
            
            if mean_asi > 3.5:
                response.append("\nSTATUS: Excellent aquifer conditions - highly favorable for groundwater development")
            elif mean_asi > 2.5:
                response.append("\nSTATUS: Good aquifer conditions - suitable for sustainable extraction")
            else:
                response.append("\nWARNING: Challenging conditions - requires careful water management")
            
            result_str = "\n".join(response)
            logger.info("get_asi_analysis completed successfully")
            return result_str
    
    except Exception as e:
        logger.error(f"Error in get_asi_analysis: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return f"ERROR: {str(e)}"


@tool
def get_sass_analysis(state: str, district: str = None, year: int = None, month: int = None) -> str:
    """
    Get Spatio-Temporal Aquifer Stress Score (SASS) - composite stress index.
    Combines groundwater, GRACE satellite, and rainfall data.
    
    Args:
        state: State name (required)
        district: District name (optional)
        year: Year (required)
        month: Month 1-12 (required)
    """
    logger.info(f" TOOL CALLED: get_sass_analysis(state='{state}', district='{district}', year={year}, month={month})")
    
    # Parse input
    if isinstance(state, str) and '=' in state:
        parsed = parse_agent_input(state)
        if parsed:
            state = parsed.get('state', '')
            district = parsed.get('district', district)
            year = parsed.get('year', year)
            month = parsed.get('month', month)
    elif isinstance(state, dict):
        district = state.get('district', district)
        year = state.get('year', year)
        month = state.get('month', month)
        state = state.get('state', '')
    
    # Clean inputs
    if state:
        state = str(state).strip().strip('"').strip("'").strip()
    if district:
        district = str(district).strip().strip('"').strip("'").strip() if district else None
    if year:
        try:
            year = int(year)
        except (ValueError, TypeError):
            return "ERROR: Valid year required"
    if month:
        try:
            month = int(month)
        except (ValueError, TypeError):
            return "ERROR: Valid month (1-12) required"
    
    if not state or not year or not month:
        logger.warning("SASS requires state, year, and month")
        return "ERROR: State, year, and month are required for SASS analysis"
    
    logger.info(f"Querying SASS: {location}, {year}-{month:02d}")
    
    return f"WARNING: SASS analysis requires year={year} and month={month}. For detailed stress assessment, please use the SASS module in the dashboard at {location}."


@tool
def get_forecast_analysis(state: str, district: str = None, forecast_months: int = 12) -> str:
    """
    Get groundwater level forecast analysis.
    Predicts future GWL using trend + GRACE data.
    
    Args:
        state: State name (required)
        district: District name (optional)
        forecast_months: Number of months to forecast (default: 12)
    """
    logger.info(f" TOOL CALLED: get_forecast_analysis(state='{state}', district='{district}', forecast_months={forecast_months})")
    
    # Parse input
    if isinstance(state, str) and '=' in state:
        parsed = parse_agent_input(state)
        if parsed:
            state = parsed.get('state', '')
            district = parsed.get('district', district)
            forecast_months = parsed.get('forecast_months', forecast_months)
    elif isinstance(state, dict):
        district = state.get('district', district)
        forecast_months = state.get('forecast_months', forecast_months)
        state = state.get('state', '')
    
    # Clean inputs
    if state:
        state = str(state).strip().strip('"').strip("'").strip()
    if district:
        district = str(district).strip().strip('"').strip("'").strip() if district else None
    
    if not state:
        logger.warning("No state provided for forecast")
        return "ERROR: State name is required"
    
    location = f"{district}, {state}" if district else state
    
    return f"INFO: For detailed {forecast_months}-month groundwater forecast for {location}, please use the Forecast module in the dashboard. This analysis requires complex spatial modeling with GRACE satellite data."


@tool
def get_recharge_planning(state: str, district: str = None, year: int = None) -> str:
    """
    Get managed aquifer recharge (MAR) planning recommendations.
    Provides recharge potential and structure recommendations.
    
    Args:
        state: State name (required)
        district: District name (optional)
        year: Year for rainfall data (optional)
    """
    logger.info(f" TOOL CALLED: get_recharge_planning(state='{state}', district='{district}', year={year})")
    
    # Parse input
    if isinstance(state, str) and '=' in state:
        parsed = parse_agent_input(state)
        if parsed:
            state = parsed.get('state', '')
            district = parsed.get('district', district)
            year = parsed.get('year', year)
    elif isinstance(state, dict):
        district = state.get('district', district)
        year = state.get('year', year)
        state = state.get('state', '')
    
    # Clean inputs
    if state:
        state = str(state).strip().strip('"').strip("'").strip()
    if district:
        district = str(district).strip().strip('"').strip("'").strip() if district else None
    
    if not state:
        logger.warning("No state provided for recharge planning")
        return "ERROR: State name is required"
    
    location = f"{district}, {state}" if district else state
    
    return f"INFO: For comprehensive recharge planning for {location}, please use the Recharge Planning module. It provides MAR potential calculations, structure recommendations (percolation tanks, check dams, etc.), and site-specific priorities."


@tool
def get_significant_trends(state: str, district: str = None) -> str:
    """
    Get sites with statistically significant groundwater trends.
    Uses Mann-Kendall test or linear regression.
    
    Args:
        state: State name (required)
        district: District name (optional)
    """
    logger.info(f" TOOL CALLED: get_significant_trends(state='{state}', district='{district}')")
    
    # Parse input
    if isinstance(state, str) and '=' in state:
        parsed = parse_agent_input(state)
        if parsed:
            state = parsed.get('state', '')
            district = parsed.get('district', district)
    elif isinstance(state, dict):
        district = state.get('district', district)
        state = state.get('state', '')
    
    # Clean inputs
    if state:
        state = str(state).strip().strip('"').strip("'").strip()
    if district:
        district = str(district).strip().strip('"').strip("'").strip() if district else None
    
    if not state:
        logger.warning("No state provided")
        return "ERROR: State name is required"
    
    location = f"{district}, {state}" if district else state
    
    return f"INFO: For detailed significance-tested trend analysis for {location}, please use the Significant Trends module. It identifies sites with statistically robust declining or recovering trends using Mann-Kendall testing."


@tool
def get_network_density(state: str, district: str = None) -> str:
    """
    Get well network density and data quality analysis.
    Shows monitoring coverage and signal strength.
    
    Args:
        state: State name (required)
        district: District name (optional)
    """
    logger.info(f" TOOL CALLED: get_network_density(state='{state}', district='{district}')")
    
    # Parse input
    if isinstance(state, str) and '=' in state:
        parsed = parse_agent_input(state)
        if parsed:
            state = parsed.get('state', '')
            district = parsed.get('district', district)
    elif isinstance(state, dict):
        district = state.get('district', district)
        state = state.get('state', '')
    
    # Clean inputs
    if state:
        state = str(state).strip().strip('"').strip("'").strip()
    if district:
        district = str(district).strip().strip('"').strip("'").strip() if district else None
    
    if not state:
        logger.warning("No state provided")
        return "ERROR: State name is required"
    
    location = f"{district}, {state}" if district else state
    
    return f"INFO: For network density and data quality assessment for {location}, please use the Network Density module. It shows monitoring coverage, signal strength, and identifies data gaps."


@tool
def get_changepoint_detection(state: str, district: str = None) -> str:
    """
    Detect structural breaks in groundwater time series.
    Identifies regime shifts using PELT algorithm.
    
    Args:
        state: State name (required)
        district: District name (optional)
    """
    logger.info(f"TOOL CALLED: get_changepoint_detection(state='{state}', district='{district}')")
    
    # Parse input
    if isinstance(state, str) and '=' in state:
        parsed = parse_agent_input(state)
        if parsed:
            state = parsed.get('state', '')
            district = parsed.get('district', district)
    elif isinstance(state, dict):
        district = state.get('district', district)
        state = state.get('state', '')
    
    # Clean inputs
    if state:
        state = str(state).strip().strip('"').strip("'").strip()
    if district:
        district = str(district).strip().strip('"').strip("'").strip() if district else None
    
    if not state:
        logger.warning("No state provided")
        return "ERROR: State name is required"
    
    location = f"{district}, {state}" if district else state
    
    return f"INFO: For changepoint detection analysis for {location}, please use the Changepoints module. It identifies dates when groundwater behavior changed abruptly (e.g., due to policy changes, infrastructure, or climate shifts)."


@tool
def get_lag_correlation(state: str, district: str = None) -> str:
    """
    Find optimal lag between rainfall and groundwater response.
    Shows how long rainfall takes to affect GWL.
    
    Args:
        state: State name (required)
        district: District name (optional)
    """
    logger.info(f"TOOL CALLED: get_lag_correlation(state='{state}', district='{district}')")
    
    # Parse input
    if isinstance(state, str) and '=' in state:
        parsed = parse_agent_input(state)
        if parsed:
            state = parsed.get('state', '')
            district = parsed.get('district', district)
    elif isinstance(state, dict):
        district = state.get('district', district)
        state = state.get('state', '')
    
    # Clean inputs
    if state:
        state = str(state).strip().strip('"').strip("'").strip()
    if district:
        district = str(district).strip().strip('"').strip("'").strip() if district else None
    
    if not state:
        logger.warning("No state provided")
        return "ERROR: State name is required"
    
    location = f"{district}, {state}" if district else state
    
    return f"INFO: For rainfall-GWL lag correlation analysis for {location}, please use the Lag Correlation module. It shows the time delay between rainfall events and groundwater level response (typically 1-12 months)."


@tool
def get_decline_hotspots(state: str, district: str = None) -> str:
    """
    Identify spatially clustered declining well sites.
    Uses DBSCAN clustering to find critical zones.
    
    Args:
        state: State name (required)
        district: District name (optional)
    """
    logger.info(f" TOOL CALLED: get_decline_hotspots(state='{state}', district='{district}')")
    
    # Parse input
    if isinstance(state, str) and '=' in state:
        parsed = parse_agent_input(state)
        if parsed:
            state = parsed.get('state', '')  # ✅ FIXED
            district = parsed.get('district', district)
    elif isinstance(state, dict):
        district = state.get('district', district)
        state = state.get('state', '')
    
    # Clean inputs
    if state:
        state = str(state).strip().strip('"').strip("'").strip()
    if district:
        district = str(district).strip().strip('"').strip("'").strip() if district else None
    
    if not state:
        logger.warning("No state provided")
        return "ERROR: State name is required"
    
    location = f"{district}, {state}" if district else state
    
    return f"INFO: For decline hotspot clustering analysis for {location}, please use the Hotspots module. It identifies spatial clusters of declining wells that require urgent intervention using DBSCAN algorithm."


@tool
def get_grace_divergence(state: str, district: str = None, year: int = None, month: int = None) -> str:
    """
    Compare GRACE satellite vs ground well measurements.
    Shows agreement/disagreement between satellite and ground data.
    
    Args:
        state: State name (required)
        district: District name (optional)
        year: Year (required)
        month: Month 1-12 (required)
    """
    logger.info(f" TOOL CALLED: get_grace_divergence(state='{state}', district='{district}', year={year}, month={month})")
    
    # Parse input
    if isinstance(state, str) and '=' in state:
        parsed = parse_agent_input(state)
        if parsed:
            state = parsed.get('state', '')
            district = parsed.get('district', district)
            year = parsed.get('year', year)
            month = parsed.get('month', month)
    elif isinstance(state, dict):
        district = state.get('district', district)
        year = state.get('year', year)
        month = state.get('month', month)
        state = state.get('state', '')
    
    # Clean inputs
    if state:
        state = str(state).strip().strip('"').strip("'").strip()
    if not year or not month:
        logger.warning("Year and month required for GRACE divergence")
        return "ERROR: Year and month required for GRACE divergence analysis"
    
    location = f"{district}, {state}" if district else state
    
    return f"INFO: For GRACE vs ground divergence analysis for {location} ({year}-{month:02d}), please use the GRACE Divergence module. It shows pixel-level differences between satellite and well measurements to identify data quality issues or hydrogeological anomalies."


def build_module_context_string(map_context: dict) -> str:
    """
    Build module context string from frontend's complete map_context object
    
    Args:
        map_context: Complete MapContext from frontend (includes region, temporal, data_summary, etc.)
    
    Returns:
        Formatted context string for agent prompt
    """
    logger.debug(f"build_module_context_string called with context: {bool(map_context)}")
    
    if not map_context:
        logger.debug("No context provided, returning default message")
        return """**MODULE CONTEXT:**
No context provided - user may be on homepage.
Use database tools to answer location-specific questions."""
    
    # Extract core context
    region = map_context.get('region', {})
    temporal = map_context.get('temporal', {})
    data_summary = map_context.get('data_summary', {})
    
    logger.debug(f"Context components - region: {bool(region)}, temporal: {bool(temporal)}, data_summary: {bool(data_summary)}")
    
    # Build location string
    state = region.get('state')
    district = region.get('district')
    
    if district and state:
        location_str = f"{district}, {state}"
    elif state:
        location_str = state
    else:
        location_str = "All India"
    
    logger.debug(f"Location: {location_str}")
    
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
    
    logger.debug(f"Time: {time_str}")
    
    # Determine active module from data_summary
    active_module = data_summary.get('active_module')
    logger.debug(f"Active module: {active_module}")
    
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
        
        if 'aquifers' in data_summary:
            aquifers = data_summary['aquifers']
            context_parts.append(f"\n**AQUIFER DATA:**")
            context_parts.append(f"- Total Aquifers: {aquifers.get('count', 'N/A')}")
            context_parts.append(f"- Dominant Type: {aquifers.get('dominant_type', 'N/A')}")
            
            # Safely format numeric values
            dom_area = aquifers.get('dominant_area_sqkm')
            if dom_area is not None:
                context_parts.append(f"- Dominant Area: {dom_area:.2f} km²")
            
            # Add lithology breakdown if available
            dom_lithos = aquifers.get('dominant_lithologies', [])
            if dom_lithos:
                litho_str = ", ".join([f"{l['name']} ({l['area_km2']} km²)" for l in dom_lithos[:3]])
                context_parts.append(f"- Top Lithologies: {litho_str}")
            
            total_area = aquifers.get('total_area_sqkm')
            avg_zone = aquifers.get('avg_zone_m')
            avg_mbgl = aquifers.get('avg_mbgl')
            
            if total_area is not None:
                context_parts.append(f"- Total Area: {total_area:.2f} km²")
            if avg_zone is not None:
                context_parts.append(f"- Avg Zone Depth: {avg_zone:.2f} m")
            if avg_mbgl is not None:
                context_parts.append(f"- Avg Water Table: {avg_mbgl:.2f} mbgl")
        
        context_str = "\n".join(context_parts)
        logger.debug(f"Built context string (length: {len(context_str)})")
        return context_str
    
    # Format based on active module
    context_parts.insert(0, f"**CURRENT MODULE:** {active_module}")
    context_parts.append("\n**FRONTEND DATA (what user sees on screen):**")
    
    if active_module == 'ASI' and 'asi' in data_summary:
        logger.debug("Building ASI context...")
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
        logger.debug("Building SASS context...")
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
        logger.debug("Building Network Density context...")
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
        logger.debug("Building Forecast context...")
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
        logger.debug("Building Recharge context...")
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
        logger.debug("Building Significant Trends context...")
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
        logger.debug("Building Changepoints context...")
        cp = data_summary['changepoints']
        stats = cp.get('statistics', {})
        context_parts.append(f"""
Changepoint Detection:
- Sites Analyzed: {stats.get('sites_analyzed', 'N/A')}
- Sites with Changepoints: {stats.get('sites_with_changepoints', 'N/A')}
- Detection Rate: {stats.get('detection_rate', 'N/A')}%
- Changepoints Found: {cp.get('changepoints_found', 'N/A')}""")
    
    elif active_module == 'LAG_CORRELATION' and 'lag_correlation' in data_summary:
        logger.debug("Building Lag Correlation context...")
        lc = data_summary['lag_correlation']
        stats = lc.get('statistics', {})
        context_parts.append(f"""
Rainfall-GWL Lag Analysis:
- Sites Analyzed: {lc.get('sites_analyzed', 'N/A')}
- Mean Lag: {stats.get('mean_lag', 'N/A')} months
- Median Lag: {stats.get('median_lag', 'N/A')} months
- Mean Correlation: {stats.get('mean_abs_correlation', 'N/A')}""")
    
    elif active_module == 'HOTSPOTS' and 'hotspots' in data_summary:
        logger.debug("Building Hotspots context...")
        hs = data_summary['hotspots']
        stats = hs.get('statistics', {})
        context_parts.append(f"""
Decline Hotspots (DBSCAN Clustering):
- Declining Sites: {stats.get('total_declining_sites', 'N/A')}
- Clusters Found: {stats.get('n_clusters', 'N/A')}
- Clustered Points: {stats.get('clustered_points', 'N/A')}
- Clustering Rate: {stats.get('clustering_rate', 'N/A')}%""")
    
    elif active_module == 'GRACE_DIVERGENCE' and 'divergence' in data_summary:
        logger.debug("Building GRACE Divergence context...")
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
        logger.debug(f"No data available for {active_module} module")
        context_parts.append(f"No data available for {active_module} module yet.")
    
    context_str = "\n".join(context_parts)
    logger.debug(f"Built context string (length: {len(context_str)})")
    return context_str


# =====================================================================
# AGENT SETUP
# =====================================================================

logger.info(" Initializing LLM...")
print("🤖 Initializing LLM...")

llm = ChatOllama(
    model="gemma3:12b",
    temperature=0,
    keep_alive="10m",
    num_ctx=8192
)
logger.debug("LLM initialized with model=gemma3:12b")

agent_prompt = PromptTemplate.from_template("""
You are a hydrogeology expert assistant for groundwater analysis in India.

**YOUR CAPABILITIES:**
1. **KNOWLEDGE BASE**: Define groundwater terms, explain analysis methodologies and formulas
2. **DASHBOARD CONTEXT**: Interpret current data shown on user's screen (when available)
3. **DATABASE TOOLS**: Query real-time groundwater, rainfall, and satellite data for any location
4. **CONVERSATIONAL**: Answer friendly and explain complex concepts simply

**CURRENT DASHBOARD CONTEXT:**
{module_context}

Available tools: {tools}
Tool names: [{tool_names}]

═══════════════════════════════════════════════════════════════════
🎯 CRITICAL: WHEN TO USE CONTEXT vs TOOLS
═══════════════════════════════════════════════════════════════════

**SCENARIO 1: Answer DIRECTLY from module_context (NO TOOLS)**
When ALL these are true:
✅ Module_context contains data about a specific location/module
✅ User asks about "this", "here", "current", or mentions the SAME location as in context
✅ The context data is sufficient to answer the question
✅ User is NOT asking to "calculate" or "analyze" (they want to understand what's shown)

Examples that should use CONTEXT ONLY:
- "What's the ASI score here?" → Just read the mean_asi from context
- "Tell me about this region" → Summarize the statistics from context
- "Is the groundwater declining?" → Check the trend data in context
- "What's the stress level?" → Read the SASS data from context

**How to answer from context:**
1. Read the statistics from module_context
2. Explain them conversationally and clearly
3. Provide interpretation and significance
4. DO NOT call any tools

**SCENARIO 2: Use TOOLS**
When ANY of these is true:
✅ User asks about a DIFFERENT location than shown in context
✅ User explicitly requests "calculate", "analyze", "compute"
✅ Module_context is empty or missing data needed to answer
✅ User asks "what is X?" (definition) → search_definition
✅ User asks "how is X calculated?" (methodology) → search_analysis
✅ User wants detailed raw data beyond summary statistics

Examples that NEED TOOLS:
- "Calculate ASI for Kerala" (while viewing Maharashtra) → get_asi_analysis
- "What is specific yield?" → search_definition
- "How is SASS calculated?" → search_analysis
- "Show me groundwater trends" (no context) → get_timeseries_analysis

═══════════════════════════════════════════════════════════════════
🔧 TOOL SELECTION GUIDE
═══════════════════════════════════════════════════════════════════

**KNOWLEDGE QUERIES (search in knowledge base):**
- "What is [term]?" → search_definition
- "Define [term]" → search_definition
- "Explain [term]" → search_definition
Examples: "What is aquifer?", "Define specific yield"

**METHODOLOGY QUERIES (how things are calculated):**
- "How is [X] calculated?" → search_analysis
- "Explain [X] methodology" → search_analysis
- "What's the formula for [X]?" → search_analysis
Examples: "How is ASI calculated?", "Explain Mann-Kendall test"

**DATA QUERIES (query database for specific locations):**

Basic Data Tools:
- Groundwater levels → get_groundwater_wells_summary(state, district, year)
- GRACE satellite → get_grace_data(state, district, year, month)
- Rainfall → get_rainfall_data(state, district, year, month)
- Aquifer types → get_aquifer_properties(state, district)
- Time trends → get_timeseries_analysis(state, district)

Advanced Analysis Tools:
- ASI (Aquifer Suitability) → get_asi_analysis(state, district)
- SASS (Stress Score) → get_sass_analysis(state, district, year, month)
- Forecasting → get_forecast_analysis(state, district, forecast_months)
- Recharge Planning → get_recharge_planning(state, district, year)
- Statistical Trends → get_significant_trends(state, district)
- Network Quality → get_network_density(state, district)
- Regime Changes → get_changepoint_detection(state, district)
- Rainfall Lag → get_lag_correlation(state, district)
- Critical Zones → get_decline_hotspots(state, district)
- Satellite vs Ground → get_grace_divergence(state, district, year, month)

═══════════════════════════════════════════════════════════════════
📋 RESPONSE FORMAT
═══════════════════════════════════════════════════════════════════

**FORMAT A: When answering from module_context (no tools needed):**

Thought: The module_context shows [module name] data for [location]. I can answer this directly from the displayed statistics.
Final Answer: [Clear, conversational explanation of the context data with interpretations]

Example:
Thought: The module_context shows ASI data for Maharashtra. The mean ASI score of 3.2 is displayed, so I can explain this directly.
Final Answer: The Aquifer Suitability Index (ASI) for Maharashtra is **3.2 out of 5.0**, which is rated as **Good**. This indicates favorable conditions for groundwater storage and transmission. About 45% of the aquifers show high suitability, with alluvium being the dominant aquifer type (average specific yield of 0.08). This suggests Maharashtra has reasonably good potential for sustainable groundwater development across its 307,713 km² area.

**FORMAT B: When using a tool:**

Thought: [Why I need to use a tool - e.g., different location, need definition, context insufficient, etc.]
Action: [tool_name]
Action Input: state="StateName", district="DistrictName", year=2023

**CRITICAL FORMAT RULE**: 
✅ CORRECT: state="Maharashtra", year=2015, month=6
❌ WRONG: {{'state': 'Maharashtra', 'year': 2015}}

After receiving ONE tool observation, immediately provide Final Answer with interpretation.

**FORMAT C: Simple conversational questions (greetings, etc.):**

Final Answer: [Direct friendly response]

Example:
Question: Hi, who are you?
Final Answer: Hello! I'm your groundwater analysis assistant. I can help you understand aquifer data, groundwater trends, rainfall patterns, and satellite measurements across India. I can also explain technical terms and methodologies. What would you like to know?

═══════════════════════════════════════════════════════════════════
💡 BEST PRACTICES
═══════════════════════════════════════════════════════════════════

1. **Always check module_context FIRST** before deciding to call a tool
2. **Use context data when asking about current location** - don't waste time calling tools
3. **Explain technical numbers** - don't just state them (e.g., "3.2/5.0 means Good aquifer potential")
4. **Be conversational** - avoid jargon, use clear language
5. **Call ONE tool maximum**, then answer immediately
6. **If context is insufficient**, acknowledge it and use the appropriate tool

═══════════════════════════════════════════════════════════════════

Question: {input}

{agent_scratchpad}
""")

all_tools = [
    # Basic tools
    search_definition,
    search_analysis,
    get_aquifer_properties,
    get_groundwater_wells_summary,
    get_grace_data,
    get_rainfall_data,
    get_timeseries_analysis,
    
    # Advanced module tools
    get_asi_analysis,
    get_sass_analysis,
    get_forecast_analysis,
    get_recharge_planning,
    get_significant_trends,
    get_network_density,
    get_changepoint_detection,
    get_lag_correlation,
    get_decline_hotspots,
    get_grace_divergence
]

logger.debug(f"Configured {len(all_tools)} tools for agent")

agent = create_react_agent(
    llm=llm,
    tools=all_tools,
    prompt=agent_prompt
)
logger.debug("Agent created successfully")

agent_executor = AgentExecutor(
    agent=agent,
    tools=all_tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True,
    early_stopping_method="force"
)
logger.debug("Agent executor created")

logger.info(" Agent ready with database + knowledge base")
print(" Agent ready with database + knowledge base\n")


def invoke_agent_with_context(user_input: str, map_context: dict = None):
    """
    Invoke agent with module context from frontend
    
    Args:
        user_input: User's question
        map_context: Complete map context from frontend
    """
    logger.info("="*70)
    logger.info(f"AGENT INVOCATION: '{user_input}'")
    logger.info("="*70)
    
    logger.debug(f"Map context provided: {bool(map_context)}")
    
    # Build context string
    logger.debug("Building context string...")
    context_str = build_module_context_string(map_context) if map_context else ""
    logger.debug(f"Context string built (length: {len(context_str)})")
    
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
    
    logger.info(f"Current module: {current_module}")
    logger.info(f"Current location: {current_location}")
    
    # Invoke agent
    logger.debug("Invoking agent executor...")
    result = agent_executor.invoke({
        "input": user_input,
        "module_context": context_str,
        "current_module": current_module,
        "current_location": current_location
    })
    
    logger.info("Agent execution completed")
    logger.debug(f"Result output length: {len(result.get('output', ''))}")
    
    return result


# =====================================================================
# REUSABLE AGENT FUNCTIONS (for FastAPI integration)
# =====================================================================

def get_agent_executor():
    """
    Returns a fresh agent executor instance for use in other modules (e.g., FastAPI).
    Creates a new executor each time to avoid state issues.
    """
    logger.debug("get_agent_executor called")
    
    tools = [
        search_definition,
        search_analysis,
        get_aquifer_properties,
        get_groundwater_wells_summary,
        get_grace_data,
        get_rainfall_data,
        get_timeseries_analysis,
        get_asi_analysis,
        get_sass_analysis,
        get_forecast_analysis,
        get_recharge_planning,
        get_significant_trends,
        get_network_density,
        get_changepoint_detection,
        get_lag_correlation,
        get_decline_hotspots,
        get_grace_divergence
    ]
    
    llm = ChatOllama(
        model="gemma3:12b",
        temperature=0,
        keep_alive="10m",
        num_ctx=8192
    )
    
    agent_prompt = PromptTemplate.from_template("""
You are a hydrogeology expert assistant for groundwater analysis in India.

**YOUR CAPABILITIES:**
1. **KNOWLEDGE BASE**: Define groundwater terms, explain analysis methodologies and formulas
2. **DASHBOARD CONTEXT**: Interpret current data shown on user's screen (when available)
3. **DATABASE TOOLS**: Query real-time groundwater, rainfall, and satellite data for any location
4. **CONVERSATIONAL**: Answer friendly and explain complex concepts simply

**CURRENT DASHBOARD CONTEXT:**
{module_context}

Available tools: {tools}
Tool names: [{tool_names}]

═══════════════════════════════════════════════════════════════════
🎯 WHEN TO USE CONTEXT vs TOOLS
═══════════════════════════════════════════════════════════════════

**Answer DIRECTLY from module_context when:**
✅ User asks about the CURRENT location/module shown on screen
✅ Context data is sufficient to answer the question
✅ Questions like "What's the score?", "Tell me about this", "Summarize this"

**Use TOOLS when:**
✅ User asks about a DIFFERENT location
✅ User explicitly requests "calculate", "analyze"
✅ Context is empty or missing needed data
✅ User asks "what is X?" (definition) or "how is X calculated?" (methodology)

═══════════════════════════════════════════════════════════════════
🔧 TOOL SELECTION GUIDE
═══════════════════════════════════════════════════════════════════

**KNOWLEDGE QUERIES:**
- "What is [term]?" → search_definition
- "How is [X] calculated?" → search_analysis

**DATA QUERIES:**
- Groundwater levels → get_groundwater_wells_summary
- GRACE satellite → get_grace_data
- Rainfall → get_rainfall_data
- Aquifer types → get_aquifer_properties
- Trends over time → get_timeseries_analysis

**ADVANCED MODULES:**
- ASI → get_asi_analysis
- SASS → get_sass_analysis
- Forecast → get_forecast_analysis
- Recharge → get_recharge_planning
- Trends → get_significant_trends
- Network → get_network_density
- Changepoints → get_changepoint_detection
- Lag → get_lag_correlation
- Hotspots → get_decline_hotspots
- Divergence → get_grace_divergence

═══════════════════════════════════════════════════════════════════
📋 RESPONSE FORMAT
═══════════════════════════════════════════════════════════════════

**When answering from module_context (no tools):**
Thought: The module_context has [description]. I can answer directly.
Final Answer: [Clear explanation of context data]

**When using a tool:**
Thought: [Why I need a tool]
Action: [tool_name]
Action Input: state="StateName", year=2023

**CRITICAL**: Use format state="Maharashtra", year=2015
NOT: {{'state': 'Maharashtra', 'year': 2015}}

**For greetings/conversational:**
Final Answer: [Direct friendly response]

═══════════════════════════════════════════════════════════════════
💡 KEY RULES
═══════════════════════════════════════════════════════════════════

1. Check module_context FIRST before calling tools
2. Use context data when available for current location
3. Explain numbers conversationally (e.g., "3.2/5.0 means Good")
4. Call ONE tool max, then answer
5. Be friendly and avoid jargon
6. If context is insufficient, refer user to the specific dashboard module

═══════════════════════════════════════════════════════════════════

Question: {input}

{agent_scratchpad}
""")
    
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=agent_prompt
    )
    
    logger.debug("Created new agent executor instance")
    
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
    logger.info("="*70)
    logger.info(f"QUERY_AGENT CALLED: '{question}'")
    logger.info(f"Verbose mode: {verbose}")
    logger.info("="*70)
    
    try:
        # Use context-aware invocation if map_context is provided
        if map_context:
            logger.debug("Using context-aware invocation")
            result = invoke_agent_with_context(question, map_context)
        else:
            logger.debug("Using basic agent without context")
            # Use basic agent without context
            agent_executor = get_agent_executor()
            # Build empty context string when no context is provided
            context_str = build_module_context_string(None)
            result = agent_executor.invoke({
                "input": question,
                "module_context": context_str
            })
        
        logger.info("Query successful")
        return {
            "output": result.get("output", "No response generated"),
            "success": True
        }
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        logger.error(f"Query failed: {e}", exc_info=True)
        if verbose:
            import traceback
            traceback.print_exc()
        return {
            "output": error_msg,
            "success": False,
            "error": str(e)
        }


# =====================================================================
# MAIN LOOP
# =====================================================================

def main():
    logger.info("="*70)
    logger.info("MAIN LOOP STARTED")
    logger.info("="*70)
    
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
    
    iteration = 0
    while True:
        iteration += 1
        logger.debug(f"Main loop iteration {iteration}")
        
        user_input = input("You: ").strip()
        
        if not user_input:
            logger.debug("Empty input, continuing...")
            continue
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            logger.info("User requested exit")
            print("\n👋 Goodbye!\n")
            break
        
        try:
            print()
            logger.info(f"Processing user input: '{user_input}'")
            
            # Pass map context to agent
            result = invoke_agent_with_context(user_input, example_map_context)
            
            print("\n" + "="*70)
            print("🤖 ANSWER:")
            print("="*70)
            print(result["output"])
            print("="*70 + "\n")
            
            logger.info("Response delivered successfully")
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            print(f"\n❌ Error: {e}\n")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    logger.info("Script executed as main")
    main()
    logger.info("Script execution completed")