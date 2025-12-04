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

# ============= CHATBOT IMPORTS =============
try:
    from langchain_chroma import Chroma
    from langchain_ollama import ChatOllama
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableLambda, RunnableMap
    import chromadb
    CHATBOT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Chatbot dependencies not available: {e}")
    CHATBOT_AVAILABLE = False

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="GeoHydro API - Wells, GRACE, Rainfall & AI Chatbot",
    version="5.1.0",
    description="Groundwater monitoring API with integrated AI chatbot and Dash-compatible seasonal decomposition"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = os.environ.get("DB_URI")
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)

print("✅ GeoHydro API Started!")

# ============= CHATBOT SETUP =============
CHATBOT_ENABLED = False
chatbot_chain = None

if CHATBOT_AVAILABLE:
    print("🤖 Initializing Chatbot...")
    try:
        # Initialize embeddings
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize ChromaDB
        chroma_client = chromadb.PersistentClient(path="/dashboard/chroma_db")
        
        # Initialize vector stores
        vector_store_analysis = Chroma(
            client=chroma_client,
            collection_name="analysis_collection",
            embedding_function=embedding_function,
        )
        vector_store_definitions = Chroma(
            client=chroma_client,
            collection_name="definitions_collection",
            embedding_function=embedding_function,
        )
        
        # Initialize LLM with stop tokens
        llama_model = ChatOllama(
            model="llama3.1:8b",
            temperature=0.3,
            stop=["\n\nHuman:", "\n\nUser:", "```", "\n\n---", "\n0", "\n1"]
        )
        
        # Helper function for chatbot
        def retrieve_context_fn(query: str):
            """Retrieve information to help answer a query"""
            docs1 = vector_store_definitions.similarity_search(query, k=2)
            docs2 = vector_store_analysis.similarity_search(query, k=2)
            docs = docs1 + docs2
            return "\n\n".join(
                (f"Source: {doc.metadata}\nContent: {doc.page_content}")
                for doc in docs
            )

        # Create retrievable context runnable
        retrieve_context = RunnableLambda(lambda x: retrieve_context_fn(x["question"]))

        # Define prompt template with variable length
        prompt = PromptTemplate(
            template="""
You are a Groundwater and Satellite-based Remote Sensing domain expert specializing in Indian hydrogeology.

CURRENT MAP DISPLAY(if explicitly asked about the map ):
{map_context}

RETRIEVED KNOWLEDGE (use only if needed):
{context}

USER QUESTION:
{question}

Instructions:
- If question uses map-reference words ("this", "here", "current", "showing", "visible", "pattern", "displayed") → analyze map data with specific values
- If question is general ("what is", "explain", "define", "tell me about") → give educational answer using retrieved knowledge, IGNORE map context
  * For map analysis: Provide comprehensive analysis (150-400 words)
  * Include specific numbers, trends, comparisons, and actionable insights
  * Reference the actual data values shown in the map context
  * Explain what the patterns mean for groundwater management
- If the user asks for definitions/concepts (like "what is", "explain", "define") → give brief educational answer (2-3 sentences, 50-100 words)
- If map data is present but question is ambiguous → prioritize analyzing the visible data with detailed explanation
- Use retrieved knowledge only for technical terms or additional context
- Keep responses India-centric
- Be direct - no preamble or meta-commentary
- Never add trailing numbers or artifacts

Answer directly:
""")

        # Create the simplified chain - LLM handles classification
        chatbot_chain = RunnableMap({
            "question": lambda x: x["question"],
            "map_context": lambda x: str(x.get("map_context") or "No map data currently displayed"),
            "context": retrieve_context
        }) | prompt | llama_model
        
        CHATBOT_ENABLED = True
        print("✅ Chatbot initialized successfully!")
        
    except Exception as e:
        CHATBOT_ENABLED = False
        print(f"⚠️  Chatbot initialization failed: {e}")
        print("   API will continue without chatbot functionality")
else:
    print("⚠️  Chatbot dependencies not installed. Chatbot disabled.")

# ============= CHATBOT MODELS =============
class ChatRequest(BaseModel):
    question: str
    map_context: Optional[dict] = None

class ChatResponse(BaseModel):
    answer: str
    sources_used: int

# =============================================================================
# GRACE BAND MAPPING
# =============================================================================

GRACE_BAND_MAPPING = {}

def auto_detect_grace_bands():
    """Auto-detect GRACE bands"""
    global GRACE_BAND_MAPPING
    print("🔍 Auto-detecting GRACE band mappings...")
    try:
        with engine.connect() as conn:
            total_bands_query = text("SELECT ST_NumBands(rast) FROM lwe_thickness_india WHERE rid = 1;")
            total_bands = conn.execute(total_bands_query).fetchone()[0]
            base_date = datetime(2002, 1, 1)
            for band in range(1, total_bands + 1):
                check_query = text(f"""
                    SELECT COUNT(*) FROM (
                        SELECT (ST_PixelAsCentroids(rast, {band})).val as val
                        FROM lwe_thickness_india WHERE rid = 1 LIMIT 10
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
# HELPER FUNCTIONS
# =============================================================================

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
    """Get boundary as GeoJSON string for SQL queries"""
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
        boundary_query = text("""
            SELECT ST_AsGeoJSON(ST_Union(ST_MakeValid(geometry))) as geojson
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
        return boundary_row[0]
    except Exception as e:
        print(f"Error getting boundary GeoJSON: {e}")
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
    """Categorize GWL depth"""
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

# =============================================================================
# ROOT ENDPOINT
# =============================================================================

@app.get("/")
def root():
    return {
        "title": "GeoHydro API - Groundwater Monitoring System with AI Chatbot",
        "version": "5.1.0",
        "description": "API for groundwater data including wells, GRACE, rainfall, and AI chatbot with Dash-compatible seasonal decomposition",
        "chatbot_enabled": CHATBOT_ENABLED,
        "endpoints": {
            "chatbot": [
                "POST /api/chat - Ask questions to AI assistant (if enabled)"
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
                "GET /api/wells/timeseries - Monthly/yearly aggregates with seasonal decomposition",
                "GET /api/wells/summary - Regional statistics",
                "GET /api/wells/storage - Pre/post-monsoon storage",
                "GET /api/wells/years - Available year range (dynamic)"
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
            "grace_months_available": len(GRACE_BAND_MAPPING)
        }
    }

# =============================================================================
# CHATBOT ENDPOINT
# =============================================================================

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint to answer groundwater and remote sensing questions"""
    if not CHATBOT_ENABLED:
        raise HTTPException(
            status_code=503, 
            detail="Chatbot is not available. Please check if ChromaDB and LLaMA are properly configured."
        )
    
    try:
        # Get context from vector store
        context = retrieve_context_fn(request.question)
        sources_count = len(context.split("Source:")) - 1
        
        # Invoke the chain with map context
        result = chatbot_chain.invoke({
            "question": request.question,
            "map_context": request.map_context or {}
        })
        
        # Clean the response - remove trailing artifacts
        answer = result.content.strip()
        
        # Remove trailing single digits that are artifacts
        while answer and answer[-1].isdigit() and (len(answer) < 2 or not answer[-2].isdigit()):
            answer = answer[:-1].strip()
        
        # Remove trailing punctuation artifacts
        answer = answer.rstrip('.,;: ')
        
        return ChatResponse(
            answer=answer,
            sources_used=sources_count
        )
    
    except Exception as e:
        print(f"Chatbot error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

# =============================================================================
# GEOGRAPHY ENDPOINTS
# =============================================================================

@app.get("/api/states")
def get_all_states():
    """Get list of all states"""
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
    """Get all districts in a state"""
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

@app.get("/api/district/{state_name}/{district_name}")
def get_district(state_name: str, district_name: str):
    """Get specific district details"""
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
    """Get all aquifers"""
    query = text("""
        SELECT 
            aquifer, aquifers, zone_m, mbgl, avg_mbgl,
            m2_perday, m3_per_day, yeild__, per_cm, stname,
            ST_AsGeoJSON(geometry) AS geojson,
            ST_Y(ST_Centroid(geometry)) AS center_lat,
            ST_X(ST_Centroid(geometry)) AS center_lng
        FROM public.major_aquifers
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
                    "stname": row[9], "geometry": json.loads(row[10]),
                    "center": [row[11], row[12]] if row[11] and row[12] else None
                }
                for row in result
            ]
            return aquifers
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/aquifers/state/{state_name}")
def get_aquifers_by_state(state_name: str):
    """Get aquifers for a specific state using spatial intersection"""
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
                    a.m2_perday, a.m3_per_day, a.yeild__, a.per_cm, a.stname,
                    ST_AsGeoJSON(ST_Intersection(
                        ST_MakeValid(a.geometry),
                        ST_GeomFromGeoJSON(:state_geojson)
                    )) AS geojson,
                    ST_Y(ST_Centroid(a.geometry)) AS center_lat,
                    ST_X(ST_Centroid(a.geometry)) AS center_lng
                FROM public.major_aquifers a
                WHERE ST_Intersects(
                    ST_MakeValid(a.geometry),
                    ST_GeomFromGeoJSON(:state_geojson)
                )
                ORDER BY a.aquifer;
            """)
            
            result = conn.execute(aquifer_query, {"state_geojson": state_row[0]})
            
            aquifers = []
            for row in result:
                try:
                    geom_json = json.loads(row[10])
                    aquifers.append({
                        "aquifer": row[0], "aquifers": row[1], "zone_m": row[2],
                        "mbgl": row[3], "avg_mbgl": row[4], "m2_perday": row[5],
                        "m3_per_day": row[6], "yeild": row[7], "per_cm": row[8],
                        "stname": row[9], "geometry": geom_json,
                        "center": [row[11], row[12]] if row[11] and row[12] else None
                    })
                except Exception as e:
                    print(f"Error parsing aquifer geometry: {e}")
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
    """Get aquifers for a specific district using spatial intersection"""
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
                    a.m2_perday, a.m3_per_day, a.yeild__, a.per_cm, a.stname,
                    ST_AsGeoJSON(ST_Intersection(
                        ST_MakeValid(a.geometry),
                        ST_GeomFromGeoJSON(:district_geojson)
                    )) AS geojson,
                    ST_Y(ST_Centroid(a.geometry)) AS center_lat,
                    ST_X(ST_Centroid(a.geometry)) AS center_lng
                FROM public.major_aquifers a
                WHERE ST_Intersects(
                    ST_MakeValid(a.geometry),
                    ST_GeomFromGeoJSON(:district_geojson)
                )
                ORDER BY a.aquifer;
            """)
            
            result = conn.execute(aquifer_query, {"district_geojson": district_row[0]})
            
            aquifers = []
            for row in result:
                try:
                    geom_json = json.loads(row[10])
                    aquifers.append({
                        "aquifer": row[0], "aquifers": row[1], "zone_m": row[2],
                        "mbgl": row[3], "avg_mbgl": row[4], "m2_perday": row[5],
                        "m3_per_day": row[6], "yeild": row[7], "per_cm": row[8],
                        "stname": row[9], "geometry": geom_json,
                        "center": [row[11], row[12]] if row[11] and row[12] else None
                    })
                except Exception as e:
                    print(f"Error parsing aquifer geometry: {e}")
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
    """Get groundwater well measurements using SPATIAL FILTERING"""
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
            "STATE_UT" as state,
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
    """
    Get aggregated well timeseries with seasonal decomposition (Dash-compatible)
    
    Parameters:
    - view: 'raw' (original data), 'seasonal' (repeating pattern only), 'deseasonalized' (trend + residual)
    - aggregation: 'monthly' or 'yearly' (forced to monthly for seasonal/deseasonalized)
    """
    
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
    
    # ✅ FORCE MONTHLY for decomposition (DASH LOGIC)
    if view in ["seasonal", "deseasonalized"]:
        aggregation = "monthly"
    
    if aggregation == "monthly":
        query = text(f"""
            SELECT 
                DATE_TRUNC('month', "Date") as period,
                AVG("GWL") as avg_gwl,
                COUNT(*) as count
            FROM groundwater_level
            {where_clause}
            AND "GWL" IS NOT NULL
            GROUP BY DATE_TRUNC('month', "Date")
            ORDER BY period;
        """)
    else:
        query = text(f"""
            SELECT 
                DATE_TRUNC('year', "Date") as period,
                AVG("GWL") as avg_gwl,
                COUNT(*) as count
            FROM groundwater_level
            {where_clause}
            AND "GWL" IS NOT NULL
            GROUP BY DATE_TRUNC('year', "Date")
            ORDER BY period;
        """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(query, params)
            
            timeseries = []
            for row in result:
                timeseries.append({
                    "date": row[0].isoformat(),
                    "avg_gwl": round(float(row[1]), 2),
                    "count": int(row[2])
                })
            
            if not timeseries:
                return {
                    "view": view,
                    "aggregation": aggregation,
                    "filters": {"state": state, "district": district},
                    "count": 0,
                    "timeseries": [],
                    "statistics": None
                }
            
            # ============= SEASONAL DECOMPOSITION (DASH LOGIC) =============
            if view in ["seasonal", "deseasonalized"] and len(timeseries) >= 24:
                df = pd.DataFrame(timeseries)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                
                # Perform decomposition
                decomp = seasonal_decompose(df['avg_gwl'], model='additive', period=12)
                
                # Prepare output based on view
                output_timeseries = []
                
                if view == "seasonal":
                    # Return ONLY seasonal component
                    for idx in df.index:
                        if pd.notna(decomp.seasonal[idx]):
                            output_timeseries.append({
                                "date": idx.isoformat(),
                                "value": round(float(decomp.seasonal[idx]), 2),
                                "component": "seasonal"
                            })
                    
                    # Calculate seasonal amplitude
                    seasonal_vals = decomp.seasonal.dropna()
                    statistics = {
                        "seasonal_amplitude": round(float(seasonal_vals.max() - seasonal_vals.min()), 2),
                        "seasonal_mean": round(float(seasonal_vals.mean()), 2),
                        "view": "seasonal",
                        "note": "Shows repeating 12-month pattern only"
                    }
                
                elif view == "deseasonalized":
                    # Return trend + residual (no seasonality)
                    deseasonalized = decomp.trend + decomp.resid
                    
                    for idx in df.index:
                        if pd.notna(deseasonalized[idx]):
                            output_timeseries.append({
                                "date": idx.isoformat(),
                                "value": round(float(deseasonalized[idx]), 2),
                                "component": "deseasonalized"
                            })
                    
                    # Calculate trend statistics (DASH LOGIC)
                    deseas_clean = deseasonalized.dropna()
                    if len(deseas_clean) >= 2:
                        x = np.arange(len(deseas_clean))
                        y = deseas_clean.values
                        slope, intercept, r_value, p_value, std_err = linregress(x, y)
                        
                        # Calculate trendline values for plotting
                        trendline = []
                        for i, idx in enumerate(deseas_clean.index):
                            trendline.append({
                                "date": idx.isoformat(),
                                "trendline_value": round(float(slope * i + intercept), 2)
                            })
                        
                        statistics = {
                            "trend_slope_m_per_month": round(float(slope), 4),
                            "trend_slope_m_per_year": round(float(slope * 12), 4),
                            "r_squared": round(float(r_value ** 2), 3),
                            "p_value": round(float(p_value), 4),
                            "trend_direction": "declining" if slope > 0 else "recovering",
                            "significance": "significant" if p_value < 0.05 else "not_significant",
                            "trendline": trendline,
                            "view": "deseasonalized",
                            "note": "Seasonal pattern removed, showing long-term trend"
                        }
                    else:
                        statistics = {
                            "error": "Insufficient data for trend calculation",
                            "view": "deseasonalized"
                        }
                
                return {
                    "view": view,
                    "aggregation": "monthly",
                    "filters": {"state": state, "district": district},
                    "count": len(output_timeseries),
                    "chart_config": {
                        "gwl_chart_type": "line",
                        "rainfall_chart_type": "line",
                        "rainfall_field": "value",
                        "rainfall_unit": "mm/day (component)",
                        "gwl_y_axis_reversed": True
                    },
                    "timeseries": output_timeseries,
                    "statistics": statistics
                }
            
            # ============= RAW VIEW (DASH-COMPATIBLE) =============
            elif view == "raw":
                # ✅ ENHANCEMENT: Add monthly rainfall totals (DASH LOGIC)
                enhanced_timeseries = []
                for item in timeseries:
                    new_item = item.copy()
                    
                    # Calculate monthly rainfall total (mm/day × days in month)
                    # Note: avg_rainfall not in current query, but keeping structure ready
                    date = pd.to_datetime(item["date"])
                    days_in_month = date.days_in_month
                    new_item["days_in_month"] = days_in_month
                    
                    # If rainfall data exists, convert to monthly total
                    if "avg_rainfall" in new_item and new_item.get("avg_rainfall") is not None:
                        new_item["monthly_rainfall_mm"] = round(
                            new_item["avg_rainfall"] * days_in_month, 2
                        )
                    
                    enhanced_timeseries.append(new_item)
                
                # Calculate basic statistics for raw data
                df = pd.DataFrame(timeseries)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                
                if len(df) >= 2:
                    x = np.arange(len(df))
                    y = df['avg_gwl'].values
                    slope, intercept, r_value, p_value, std_err = linregress(x, y)
                    
                    # Calculate trendline
                    trendline = []
                    for i, idx in enumerate(df.index):
                        trendline.append({
                            "date": idx.isoformat(),
                            "trendline_value": round(float(slope * i + intercept), 2)
                        })
                    
                    multiplier = 12 if aggregation == "monthly" else 1
                    
                    statistics = {
                        "mean_gwl": round(float(np.mean(y)), 2),
                        "min_gwl": round(float(np.min(y)), 2),
                        "max_gwl": round(float(np.max(y)), 2),
                        "std_gwl": round(float(np.std(y)), 2),
                        "trend_slope_m_per_year": round(float(slope * multiplier), 4),
                        "r_squared": round(float(r_value ** 2), 3),
                        "p_value": round(float(p_value), 4),
                        "trend_direction": "declining" if slope > 0 else "recovering",
                        "significance": "significant" if p_value < 0.05 else "not_significant",
                        "trendline": trendline,
                        "view": "raw",
                        "note": "Original data with seasonality intact. Rainfall shown as monthly totals (when available)."
                    }
                else:
                    statistics = None
                
                return {
                    "view": view,
                    "aggregation": aggregation,
                    "filters": {"state": state, "district": district},
                    "count": len(timeseries),
                    "chart_config": {  # ✅ DASH-COMPATIBLE CONFIG
                        "gwl_chart_type": "line",
                        "rainfall_chart_type": "bar",
                        "rainfall_field": "monthly_rainfall_mm",
                        "rainfall_unit": "mm/month",
                        "gwl_y_axis_reversed": True
                    },
                    "timeseries": enhanced_timeseries,  # ✅ With monthly totals
                    "statistics": statistics
                }
            
            else:
                # Not enough data for decomposition
                return {
                    "view": view,
                    "aggregation": aggregation,
                    "filters": {"state": state, "district": district},
                    "count": len(timeseries),
                    "timeseries": timeseries,
                    "statistics": None,
                    "error": f"Need at least 24 months of data for {view} view (have {len(timeseries)} months)"
                }
    
    except Exception as e:
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/wells/summary")
def get_wells_summary(
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None)
):
    """Get regional statistics including trends"""
    
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

@app.get("/api/wells/storage")
def get_wells_storage(
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    start_year: Optional[int] = Query(None),
    end_year: Optional[int] = Query(None)
):
    """Calculate pre/post-monsoon storage fluctuation"""
    
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
    
    where_aquifer = "WHERE ST_Intersects(ST_MakeValid(geometry), ST_GeomFromGeoJSON(:boundary_geojson))" if boundary_geojson else ""
    aquifer_query = text(f"""
        SELECT 
            ST_Area(ST_Transform(ST_MakeValid(geometry), 32643)) as area_m2,
            yeild__
        FROM major_aquifers
        {where_aquifer}
    """)

    try:
        with engine.connect() as conn:
            result = conn.execute(aquifer_query, params if boundary_geojson else {})
            
            total_area_m2 = 0.0
            weighted_sy_sum = 0.0
            
            for row in result:
                area = float(row[0]) if row[0] else 0
                yield_str = str(row[1]).lower() if row[1] else ""
                
                sy = 0.05
                try:
                    import re
                    numbers = re.findall(r'\d+\.?\d*', yield_str)
                    if numbers:
                        vals = [float(n) for n in numbers]
                        sy = sum(vals) / len(vals) / 100.0
                except:
                    sy = 0.05
                
                total_area_m2 += area
                weighted_sy_sum += (area * sy)
            
            if total_area_m2 == 0:
                return {"error": "No aquifer data available for this region"}
            
            specific_yield = weighted_sy_sum / total_area_m2
            
            year_filter_start = 'AND EXTRACT(YEAR FROM "Date") >= :start_year' if start_year else ""
            year_filter_end = 'AND EXTRACT(YEAR FROM "Date") <= :end_year' if end_year else ""

            storage_query = text(f"""
                SELECT 
                    EXTRACT(YEAR FROM "Date") as year,
                    AVG(CASE WHEN EXTRACT(MONTH FROM "Date") IN (4,5,6) THEN "GWL" END) as pre_gwl,
                    AVG(CASE WHEN EXTRACT(MONTH FROM "Date") IN (10,11,12) THEN "GWL" END) as post_gwl
                FROM groundwater_level
                {where_clause}
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
            
            result = conn.execute(storage_query, params)
            
            storage_years = []
            for row in result:
                year = int(row[0])
                pre_gwl = float(row[1])
                post_gwl = float(row[2])
                
                fluctuation_m = pre_gwl - post_gwl
                storage_change_mcm = (fluctuation_m * total_area_m2 * specific_yield) / 1_000_000
                
                storage_years.append({
                    "year": year,
                    "pre_monsoon_gwl": round(pre_gwl, 2),
                    "post_monsoon_gwl": round(post_gwl, 2),
                    "fluctuation_m": round(fluctuation_m, 2),
                    "storage_change_mcm": round(storage_change_mcm, 2)
                })
            
            if not storage_years:
                return {
                    "error": "Insufficient seasonal data for storage calculation"
                }
            
            avg_storage_change = np.mean([s["storage_change_mcm"] for s in storage_years])
            
            return {
                "filters": {"state": state, "district": district},
                "aquifer_properties": {
                    "total_area_km2": round(total_area_m2 / 1_000_000, 2),
                    "area_weighted_specific_yield": round(specific_yield, 4)
                },
                "summary": {
                    "avg_annual_storage_change_mcm": round(avg_storage_change, 2),
                    "years_analyzed": len(storage_years)
                },
                "yearly_storage": storage_years
            }
    
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/wells/years")
def get_wells_years():
    """Get available year range for wells data"""
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
    """Get available GRACE months"""
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
    """Get GRACE data with proper missing month handling"""
    
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
                            FROM lwe_thickness_india WHERE rid = 1 LIMIT 10
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
                    FROM lwe_thickness_india
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
    """Get Rainfall data - Returns AVERAGE daily rainfall (mm/day)"""
    
    table_name = f"rf25_ind{year}_rfp25"
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

@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        chatbot_status = "enabled" if CHATBOT_ENABLED else "disabled"
        
        return {
            "status": "healthy",
            "database": "connected",
            "grace_bands": len(GRACE_BAND_MAPPING),
            "chatbot": chatbot_status,
            "version": "5.1.0",
            "features": [
                "Dash-compatible seasonal decomposition",
                "Monthly rainfall totals for raw view",
                "Frontend chart configuration guidance"
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*80)
    print("🚀 GeoHydro API v5.1.0 - Dash-Compatible Edition")
    print("="*80)
    print(f"📊 GRACE Months Available: {len(GRACE_BAND_MAPPING)}")
    print(f"🤖 Chatbot Status: {'✅ Enabled' if CHATBOT_ENABLED else '⚠️  Disabled'}")
    print(f"🌧️  Rainfall: Monthly totals for Raw view")
    print(f"📈 Decomposition: Forced monthly for seasonal/deseasonalized")
    print("="*80)
    print("\n🌐 API Documentation: http://localhost:8000/docs")
    print("🏥 Health Check: http://localhost:8000/health")
    print("💬 Chatbot: POST http://localhost:8000/api/chat")
    print("📊 Timeseries: GET http://localhost:8000/api/wells/timeseries")
    print("\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)