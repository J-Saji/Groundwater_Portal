"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import axios from "axios";
import dynamic from "next/dynamic";
import "leaflet/dist/leaflet.css";

// Add Plotly import
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

// ✅ Lazy load React Leaflet components
const MapContainer = dynamic(() => import("react-leaflet").then(m => m.MapContainer), { ssr: false });
const TileLayer = dynamic(() => import("react-leaflet").then(m => m.TileLayer), { ssr: false });
const GeoJSON = dynamic(() => import("react-leaflet").then(m => m.GeoJSON), { ssr: false });
const CircleMarker = dynamic(() => import("react-leaflet").then(m => m.CircleMarker), { ssr: false });
const Popup = dynamic(() => import("react-leaflet").then(m => m.Popup), { ssr: false });

type Geometry = GeoJSON.FeatureCollection<GeoJSON.Geometry, GeoJSON.GeoJsonProperties>;

interface StateType {
  State: string;
}

interface DistrictType {
  district_name: string;
  geometry: Geometry;
  center?: [number, number];
}

interface AquiferType {
  aquifer: string;
  aquifers: string;
  zone_m: number;
  mbgl: number;
  avg_mbgl: number;
  m2_perday: number;
  m3_per_day: number;
  yeild: number;
  per_cm: number;
  stname: string;
  geometry: Geometry;
  center: [number, number] | null;
}

interface GracePoint {
  longitude: number;
  latitude: number;
  lwe_cm: number;
  cell_area_km2?: number;
}

interface RainfallPoint {
  longitude: number;
  latitude: number;
  rainfall_mm: number;
  days_averaged?: number;
}

interface WellPoint {
  site_id: string;
  date: string;
  year: number;
  month: number;
  gwl: number;
  gwl_category: string;
  latitude: number;
  longitude: number;
  state: string;
  district: string;
  season?: string;
  site_name?: string;
  site_type?: string;
  aquifer?: string;
  hovertext: string;
}

// ============= UPDATED: Timeseries Interfaces with Rainfall =============
interface TimeseriesPoint {
  date: string;
  avg_gwl?: number;        // For raw view GWL
  value?: number;          // For seasonal/deseasonalized GWL
  avg_rainfall?: number;   // Raw rainfall (mm/day) - kept for reference
  monthly_rainfall_mm?: number;  // ✅ NEW: Monthly rainfall total (mm/month)
  days_in_month?: number;  // ✅ NEW: Days in month
  count?: number;
  component?: string;
}

interface TrendlinePoint {
  date: string;
  trendline_value: number;
}

interface ChartConfig {
  gwl_chart_type: string;
  rainfall_chart_type: string;
  rainfall_field: string;
  rainfall_unit: string;
  gwl_y_axis_reversed: boolean;
}

interface TimeseriesStatistics {
  mean_gwl?: number;
  min_gwl?: number;
  max_gwl?: number;
  std_gwl?: number;
  trend_slope_m_per_month?: number;
  trend_slope_m_per_year: number;
  r_squared: number;
  p_value: number;
  trend_direction: string;
  significance: string;
  trendline: TrendlinePoint[];
  view: string;
  note?: string;
  seasonal_amplitude?: number;
  seasonal_mean?: number;
}

interface TimeseriesResponse {
  view: string;
  aggregation: string;
  filters: {
    state: string | null;
    district: string | null;
  };
  count: number;
  chart_config?: ChartConfig;  // ✅ NEW: Chart configuration from backend
  timeseries: TimeseriesPoint[];
  statistics: TimeseriesStatistics | null;
  error?: string;
}

interface WellsSummary {
  filters: {
    state: string | null;
    district: string | null;
  };
  statistics: {
    mean_gwl: number;
    min_gwl: number;
    max_gwl: number;
    std_gwl: number;
  };
  trend: {
    slope_m_per_year: number;
    r_squared: number;
    p_value: number;
    trend_direction: string;
    significance: string;
  };
  temporal_coverage: {
    start_date: string;
    end_date: string;
    months_of_data: number;
    span_years: number;
  };
  error?: string;
}

interface StorageYear {
  year: number;
  pre_monsoon_gwl: number;
  post_monsoon_gwl: number;
  fluctuation_m: number;
  storage_change_mcm: number;
}

interface StorageResponse {
  filters: {
    state: string | null;
    district: string | null;
  };
  aquifer_properties: {
    total_area_km2: number;
    area_weighted_specific_yield: number;
  };
  summary: {
    avg_annual_storage_change_mcm: number;
    years_analyzed: number;
  };
  yearly_storage: StorageYear[];
  error?: string;
}

interface GraceResponse {
  data_type: string;
  year: number;
  month: number | null;
  description: string;
  status?: string;
  message?: string;
  fallback_used?: boolean;
  fallback_message?: string;
  calculation_method?: string;
  months_available?: number;
  available_month_list?: number[];
  total_months?: number;
  regional_average_cm?: number;
  total_area_km2?: number;
  count: number;
  points: GracePoint[];
}

interface RainfallResponse {
  data_type: string;
  year: number;
  month: number | null;
  day: number | null;
  description: string;
  status?: string;
  fallback_used?: boolean;
  fallback_message?: string;
  calculation_method?: string;
  unit?: string;
  days_included?: number;
  regional_average_mm_per_day?: number;
  count: number;
  points: RainfallPoint[];
}

interface WellsResponse {
  data_type: string;
  filters: {
    state: string | null;
    district: string | null;
    year: number | null;
    month: number | null;
    season: string | null;
    start_date: string | null;
    end_date: string | null;
  };
  spatial_filter: string;
  count: number;
  wells: WellPoint[];
}

interface YearRangeResponse {
  min_year: number;
  max_year: number;
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  sourcesUsed?: number;
  timestamp: Date;
}

interface MapContext {
  active_layers: string[];
  region: {
    state: string | null;
    district: string | null;
  };
  temporal: {
    year: number;
    month: number | null;
    season: string | null;
  };
  data_summary: {
    [key: string]: any;
  };
}

const COLOR_PALETTE = [
  "#E91E63", "#9C27B0", "#673AB7", "#3F51B5", "#2196F3", "#03A9F4",
  "#00BCD4", "#009688", "#4CAF50", "#8BC34A", "#CDDC39", "#FFEB3B",
  "#FFC107", "#FF9800", "#FF5722", "#795548", "#607D88", "#F44336",
];

// ============= DASH-COMPATIBLE COLORS =============
const GWL_COLOR = "#8D6E63";
const RAIN_COLOR = "#1E88E5";
const TREND_DECLINE_COLOR = "#DC2626";
const TREND_RECOVER_COLOR = "#16A34A";

const getAquiferColor = (aquiferType: string, index: number): string => {
  const type = aquiferType?.toLowerCase() || "";
  
  if (type.includes("alluvial") || type.includes("unconsolidated")) return "#4CAF50";
  if (type.includes("hard rock") || type.includes("crystalline") || type.includes("granite")) return "#E91E63";
  if (type.includes("sandstone") || type.includes("consolidated")) return "#2196F3";
  if (type.includes("limestone") || type.includes("carbonate")) return "#9C27B0";
  if (type.includes("volcanic") || type.includes("basalt")) return "#FF5722";
  if (type.includes("shale")) return "#795548";
  if (type.includes("quartzite")) return "#00BCD4";
  if (type.includes("phyllite") || type.includes("schist")) return "#673AB7";
  return COLOR_PALETTE[index % COLOR_PALETTE.length];
};

const getGraceColor = (value: number): string => {
  if (value < -10) return "#8B0000";
  if (value < -5) return "#DC143C";
  if (value < 0) return "#FF6347";
  if (value < 5) return "#32CD32";
  if (value < 10) return "#1E90FF";
  return "#0000CD";
};

const getRainfallColor = (value: number): string => {
  if (value < 1) return "#F0F0F0";
  if (value < 10) return "#B3E5FC";
  if (value < 25) return "#4FC3F7";
  if (value < 50) return "#2196F3";
  if (value < 100) return "#1976D2";
  return "#0D47A1";
};

const getWellColor = (category: string): string => {
  switch (category) {
    case "Recharge": return "#1E88E5";
    case "Shallow (0-30m)": return "#43A047";
    case "Moderate (30-60m)": return "#FB8C00";
    case "Deep (60-100m)": return "#E53935";
    case "Very Deep (>100m)": return "#B71C1C";
    default: return "#9E9E9E";
  }
};

export default function Home() {
  const [states, setStates] = useState<StateType[]>([]);
  const [districts, setDistricts] = useState<DistrictType[]>([]);
  const [aquifers, setAquifers] = useState<AquiferType[]>([]);
  const [graceData, setGraceData] = useState<GracePoint[]>([]);
  const [rainfallData, setRainfallData] = useState<RainfallPoint[]>([]);
  const [wellsData, setWellsData] = useState<WellPoint[]>([]);
  
  const [graceResponse, setGraceResponse] = useState<GraceResponse | null>(null);
  const [rainfallResponse, setRainfallResponse] = useState<RainfallResponse | null>(null);
  const [wellsResponse, setWellsResponse] = useState<WellsResponse | null>(null);
  
  // ============= UPDATED: Timeseries State =============
  const [timeseriesResponse, setTimeseriesResponse] = useState<TimeseriesResponse | null>(null);
  const [summaryData, setSummaryData] = useState<WellsSummary | null>(null);
  const [storageData, setStorageData] = useState<StorageResponse | null>(null);
  
  const [wellYearRange, setWellYearRange] = useState({ min: 1994, max: 2024 });
  
  const [selectedState, setSelectedState] = useState<string>("");
  const [selectedDistrict, setSelectedDistrict] = useState<string>("");
  const [showAquifers, setShowAquifers] = useState<boolean>(false);
  const [showGrace, setShowGrace] = useState<boolean>(false);
  const [showRainfall, setShowRainfall] = useState<boolean>(false);
  const [showWells, setShowWells] = useState<boolean>(false);
  
  const [showTimeseries, setShowTimeseries] = useState<boolean>(false);
  const [showSummary, setShowSummary] = useState<boolean>(false);
  const [showStorage, setShowStorage] = useState<boolean>(false);
  
  const [districtGeo, setDistrictGeo] = useState<Geometry | null>(null);
  const [center, setCenter] = useState<[number, number]>([22.9734, 78.6569]);
  const [zoom, setZoom] = useState(5);
  const [mapKey, setMapKey] = useState<number>(0);

  const [selectedYear, setSelectedYear] = useState<number>(2011);
  const [selectedMonth, setSelectedMonth] = useState<number | null>(null);
  const [selectedDay, setSelectedDay] = useState<number | null>(null);
  const [selectedSeason, setSelectedSeason] = useState<string>("");
  
  // ============= UPDATED: View Selection =============
  const [timeseriesView, setTimeseriesView] = useState<"raw" | "seasonal" | "deseasonalized">("raw");

  const [alertMessage, setAlertMessage] = useState<string>("");
  const [alertType, setAlertType] = useState<"info" | "warning" | "error" | "success">("info");
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const [isChatOpen, setIsChatOpen] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content: "Hello! I'm your Groundwater and Remote Sensing expert. Ask me anything about GRACE data, rainfall patterns, groundwater levels, or aquifer systems!",
      timestamp: new Date()
    }
  ]);
  const [chatInput, setChatInput] = useState("");
  const [isChatLoading, setIsChatLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const backendURL = "http://127.0.0.1:8000";

  const showAlert = (message: string, type: "info" | "warning" | "error" | "success" = "info") => {
    setAlertMessage(message);
    setAlertType(type);
    setTimeout(() => setAlertMessage(""), 5000);
  };

  const buildMapContext = useCallback((): MapContext => {
    const activeLayers: string[] = [];
    if (showAquifers) activeLayers.push("aquifers");
    if (showGrace) activeLayers.push("grace");
    if (showRainfall) activeLayers.push("rainfall");
    if (showWells) activeLayers.push("wells");

    const dataSummary: any = {};

    if (showAquifers && aquifers.length > 0) {
      dataSummary.aquifers = {
        count: aquifers.length,
        types: Array.from(new Set(aquifers.map(a => a.aquifer))).slice(0, 3)
      };
    }

    if (showGrace && graceResponse) {
      dataSummary.grace = {
        year: graceResponse.year,
        month: graceResponse.month,
        description: graceResponse.description,
        regional_average_cm: graceResponse.regional_average_cm,
        data_points: graceResponse.count,
        status: graceResponse.status
      };
    }

    if (showRainfall && rainfallResponse) {
      dataSummary.rainfall = {
        year: rainfallResponse.year,
        month: rainfallResponse.month,
        day: rainfallResponse.day,
        description: rainfallResponse.description,
        regional_average_mm_per_day: rainfallResponse.regional_average_mm_per_day,
        data_points: rainfallResponse.count,
        unit: rainfallResponse.unit
      };
    }

    if (showWells && wellsResponse) {
      const categories = wellsData.reduce((acc, w) => {
        acc[w.gwl_category] = (acc[w.gwl_category] || 0) + 1;
        return acc;
      }, {} as Record<string, number>);

      dataSummary.wells = {
        year: wellsResponse.filters.year,
        month: wellsResponse.filters.month,
        season: wellsResponse.filters.season,
        data_points: wellsResponse.count,
        categories: categories,
        avg_gwl: wellsData.length > 0 
          ? (wellsData.reduce((sum, w) => sum + w.gwl, 0) / wellsData.length).toFixed(2) 
          : null
      };
    }

    if (summaryData && !summaryData.error) {
      dataSummary.summary = {
        mean_gwl: summaryData.statistics.mean_gwl,
        trend_direction: summaryData.trend.trend_direction,
        slope_m_per_year: summaryData.trend.slope_m_per_year
      };
    }

    return {
      active_layers: activeLayers,
      region: {
        state: selectedState || null,
        district: selectedDistrict || null
      },
      temporal: {
        year: selectedYear,
        month: selectedMonth,
        season: selectedSeason || null
      },
      data_summary: dataSummary
    };
  }, [
    showAquifers, showGrace, showRainfall, showWells,
    aquifers, graceResponse, rainfallResponse, wellsResponse, wellsData,
    summaryData, selectedState, selectedDistrict, selectedYear, selectedMonth, selectedSeason
  ]);

  const scrollChatToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollChatToBottom();
  }, [chatMessages]);

  const handleSendMessage = async () => {
    if (!chatInput.trim() || isChatLoading) return;

    const userMessage: ChatMessage = {
      role: "user",
      content: chatInput,
      timestamp: new Date()
    };

    setChatMessages(prev => [...prev, userMessage]);
    setChatInput("");
    setIsChatLoading(true);

    try {
      const mapContext = buildMapContext();
      
      const response = await axios.post(`${backendURL}/api/chat`, {
        question: chatInput,
        map_context: mapContext
      });

      const assistantMessage: ChatMessage = {
        role: "assistant",
        content: response.data.answer,
        sourcesUsed: response.data.sources_used,
        timestamp: new Date()
      };

      setChatMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Error sending message:", error);
      const errorMessage: ChatMessage = {
        role: "assistant",
        content: "Sorry, I encountered an error. Please make sure the chatbot is enabled in the backend.",
        timestamp: new Date()
      };
      setChatMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsChatLoading(false);
    }
  };

  const handleChatKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const suggestedQuestions = [
    "What is GRACE satellite data?",
    "Explain groundwater depletion in India",
    "What patterns do you see in this map?",
    "Analyze the current data displayed"
  ];

  const activeLayers = [
    { id: 'aquifers', name: 'Aquifers', icon: '🔷', show: showAquifers, color: 'purple' },
    { id: 'grace', name: 'GRACE', icon: '🌊', show: showGrace, color: 'green' },
    { id: 'rainfall', name: 'Rainfall', icon: '🌧️', show: showRainfall, color: 'blue' },
    { id: 'wells', name: 'Wells', icon: '💧', show: showWells, color: 'red' }
  ].filter(layer => layer.show);

  useEffect(() => {
    axios.get<YearRangeResponse>(`${backendURL}/api/wells/years`)
      .then(res => {
        setWellYearRange({ min: res.data.min_year, max: res.data.max_year });
        setSelectedYear(res.data.max_year);
      })
      .catch(() => {
        setWellYearRange({ min: 1994, max: 2024 });
      });
  }, []);

  useEffect(() => {
    axios.get<StateType[]>(`${backendURL}/api/states`)
      .then(res => setStates(res.data))
      .catch(err => showAlert("Error loading states", "error"));
  }, []);

  useEffect(() => {
    if (!selectedState) {
      setDistricts([]);
      setAquifers([]);
      setDistrictGeo(null);
      setCenter([22.9734, 78.6569]);
      setZoom(5);
      setMapKey(prev => prev + 1);
      return;
    }
    
    setIsLoading(true);
    axios.get<DistrictType[]>(`${backendURL}/api/districts/${selectedState}`)
      .then(res => {
        setDistricts(res.data);
        if (res.data.length > 0) {
          const centers = res.data.filter(d => d.center).map(d => d.center!);
          if (centers.length > 0) {
            const avgLat = centers.reduce((sum, c) => sum + c[0], 0) / centers.length;
            const avgLon = centers.reduce((sum, c) => sum + c[1], 0) / centers.length;
            setCenter([avgLat, avgLon]);
            setZoom(7);
            setMapKey(prev => prev + 1);
          }
        }
      })
      .catch(err => showAlert(`Error loading districts`, "error"))
      .finally(() => setIsLoading(false));
  }, [selectedState]);

  useEffect(() => {
    if (!selectedDistrict || !selectedState) {
      if (selectedState && !selectedDistrict) {
        setDistrictGeo(null);
      }
      return;
    }
    
    setIsLoading(true);
    axios.get(`${backendURL}/api/district/${selectedState}/${selectedDistrict}`)
      .then(res => {
        setDistrictGeo(res.data.geometry);
        if (res.data.center) {
          setCenter(res.data.center);
          setZoom(9);
          setMapKey(prev => prev + 1);
        }
      })
      .catch(err => showAlert(`Error loading boundary`, "error"))
      .finally(() => setIsLoading(false));
  }, [selectedDistrict, selectedState]);

  useEffect(() => {
    if (!showAquifers) {
      setAquifers([]);
      return;
    }
    if (selectedDistrict && selectedState) {
      setIsLoading(true);
      axios.get<AquiferType[]>(`${backendURL}/api/aquifers/district/${selectedState}/${selectedDistrict}`)
        .then(res => {
          setAquifers(res.data);
          showAlert(`Loaded ${res.data.length} aquifers`, "success");
        })
        .catch(err => showAlert("Error loading aquifers", "error"))
        .finally(() => setIsLoading(false));
    } else if (selectedState) {
      setIsLoading(true);
      axios.get<AquiferType[]>(`${backendURL}/api/aquifers/state/${selectedState}`)
        .then(res => {
          setAquifers(res.data);
          showAlert(`Loaded ${res.data.length} aquifers`, "success");
        })
        .catch(err => showAlert("Error loading aquifers", "error"))
        .finally(() => setIsLoading(false));
    }
  }, [showAquifers, selectedState, selectedDistrict]);

  useEffect(() => {
    if (!showGrace) {
      setGraceData([]);
      setGraceResponse(null);
      return;
    }
    setIsLoading(true);
    const params: any = { year: selectedYear };
    if (selectedMonth) params.month = selectedMonth;
    if (selectedState) params.state = selectedState;
    if (selectedDistrict) params.district = selectedDistrict;

    axios.get<GraceResponse>(`${backendURL}/api/grace`, { params })
      .then(res => {
        setGraceResponse(res.data);
        if (res.data.status === "no_data") {
          setGraceData([]);
          showAlert("No GRACE data available", "warning");
          return;
        }
        setGraceData(res.data.points);
        if (!res.data.fallback_used) {
          showAlert(`Loaded ${res.data.count} GRACE points`, "success");
        }
      })
      .catch(err => showAlert("Error loading GRACE data", "error"))
      .finally(() => setIsLoading(false));
  }, [showGrace, selectedYear, selectedMonth, selectedState, selectedDistrict]);

  useEffect(() => {
    if (!showRainfall) {
      setRainfallData([]);
      setRainfallResponse(null);
      return;
    }
    setIsLoading(true);
    const params: any = { year: selectedYear };
    if (selectedMonth) params.month = selectedMonth;
    if (selectedDay) params.day = selectedDay;
    if (selectedState) params.state = selectedState;
    if (selectedDistrict) params.district = selectedDistrict;

    axios.get<RainfallResponse>(`${backendURL}/api/rainfall`, { params })
      .then(res => {
        setRainfallResponse(res.data);
        setRainfallData(res.data.points);
        if (!res.data.fallback_used) {
          showAlert(`Loaded ${res.data.count} rainfall points`, "success");
        }
      })
      .catch(err => showAlert("Error loading rainfall data", "error"))
      .finally(() => setIsLoading(false));
  }, [showRainfall, selectedYear, selectedMonth, selectedDay, selectedState, selectedDistrict]);

  useEffect(() => {
    if (!showWells) {
      setWellsData([]);
      setWellsResponse(null);
      return;
    }
    setIsLoading(true);
    const params: any = { year: selectedYear, max_points: 5000 };
    if (selectedMonth) params.month = selectedMonth;
    if (selectedSeason) params.season = selectedSeason;
    if (selectedState) params.state = selectedState;
    if (selectedDistrict) params.district = selectedDistrict;

    axios.get<WellsResponse>(`${backendURL}/api/wells`, { params })
      .then(res => {
        setWellsResponse(res.data);
        setWellsData(res.data.wells);
        showAlert(`Loaded ${res.data.count} wells`, "success");
      })
      .catch(err => showAlert("Error loading wells data", "error"))
      .finally(() => setIsLoading(false));
  }, [showWells, selectedYear, selectedMonth, selectedSeason, selectedState, selectedDistrict]);

  // ============= UPDATED: Timeseries Effect =============
  useEffect(() => {
    if (!showTimeseries) {
      setTimeseriesResponse(null);
      return;
    }
    setIsLoading(true);
    const params: any = { view: timeseriesView };
    if (selectedState) params.state = selectedState;
    if (selectedDistrict) params.district = selectedDistrict;

    axios.get<TimeseriesResponse>(`${backendURL}/api/wells/timeseries`, { params })
      .then(res => {
        setTimeseriesResponse(res.data);
        showAlert(`Loaded ${res.data.count} time points (${res.data.view} view)`, "success");
      })
      .catch(err => showAlert("Error loading timeseries", "error"))
      .finally(() => setIsLoading(false));
  }, [showTimeseries, timeseriesView, selectedState, selectedDistrict]);

  useEffect(() => {
    if (!showSummary) {
      setSummaryData(null);
      return;
    }
    setIsLoading(true);
    const params: any = {};
    if (selectedState) params.state = selectedState;
    if (selectedDistrict) params.district = selectedDistrict;

    axios.get(`${backendURL}/api/wells/summary`, { params })
      .then(res => {
        setSummaryData(res.data);
        showAlert("Summary loaded", "success");
      })
      .catch(err => showAlert("Error loading summary", "error"))
      .finally(() => setIsLoading(false));
  }, [showSummary, selectedState, selectedDistrict]);

  useEffect(() => {
    if (!showStorage) {
      setStorageData(null);
      return;
    }
    setIsLoading(true);
    const params: any = {};
    if (selectedState) params.state = selectedState;
    if (selectedDistrict) params.district = selectedDistrict;

    axios.get(`${backendURL}/api/wells/storage`, { params })
      .then(res => {
        setStorageData(res.data);
        showAlert("Storage data loaded", "success");
      })
      .catch(err => showAlert("Error loading storage", "error"))
      .finally(() => setIsLoading(false));
  }, [showStorage, selectedState, selectedDistrict]);

  const createAquiferPopup = useCallback((aquifer: AquiferType, colorIndex: number) => {
    return (feature: unknown, layer: L.Layer) => {
      const aquiferColor = getAquiferColor(aquifer.aquifer, colorIndex);
      const popupContent = `
        <div style="font-family: sans-serif; min-width: 200px;">
          <strong style="font-size: 15px; color: ${aquiferColor};">${aquifer.aquifer || 'N/A'}</strong><br/>
          <hr style="margin: 5px 0; border: 1px solid #ddd;"/>
          <table style="width: 100%; font-size: 12px; margin-top: 5px;">
            <tr><td><strong>Type:</strong></td><td>${aquifer.aquifers || 'N/A'}</td></tr>
            <tr><td><strong>Zone (m):</strong></td><td>${aquifer.zone_m || 'N/A'}</td></tr>
            <tr><td><strong>Avg MBGL:</strong></td><td>${aquifer.avg_mbgl || 'N/A'}</td></tr>
          </table>
        </div>
      `;
      if (layer && typeof (layer as any).bindPopup === 'function') {
        (layer as any).bindPopup(popupContent);
      }
    };
  }, []);

  const uniqueAquiferTypesWithColors = Array.from(
    new Map(aquifers.filter(a => a.aquifer).map((a, idx) => [a.aquifer, { type: a.aquifer, color: getAquiferColor(a.aquifer, idx) }])).values()
  );

  const uniqueWellCategories = Array.from(new Set(wellsData.map(w => w.gwl_category))).map(cat => ({ category: cat, color: getWellColor(cat) }));

  const graceYears = Array.from({ length: 24 }, (_, i) => 2002 + i);
  const rainfallYears = Array.from({ length: 31 }, (_, i) => 1994 + i);
  const wellYears = Array.from(
    { length: wellYearRange.max - wellYearRange.min + 1 }, 
    (_, i) => wellYearRange.min + i
  );
  
  const availableYears = showGrace ? graceYears : (showWells ? wellYears : rainfallYears);

  const formatWellPopup = (well: WellPoint): string => {
    const parts = [
      `<div style="font-family: sans-serif; min-width: 200px;">`,
      `<strong style="font-size: 14px; color: ${getWellColor(well.gwl_category)};">Groundwater Level</strong><br/>`,
      `<hr style="margin: 5px 0; border: 1px solid #ddd;"/>`,
      `<table style="width: 100%; font-size: 12px; margin-top: 5px;">`,
      `<tr><td><strong>Date:</strong></td><td>${new Date(well.date).toLocaleDateString()}</td></tr>`,
      `<tr><td><strong>GWL:</strong></td><td>${well.gwl.toFixed(2)} m bgl</td></tr>`,
      `<tr><td><strong>Category:</strong></td><td>${well.gwl_category}</td></tr>`,
      `<tr><td><strong>Site ID:</strong></td><td>${well.site_id}</td></tr>`,
    ];
    if (well.site_name) parts.push(`<tr><td><strong>Site:</strong></td><td>${well.site_name}</td></tr>`);
    if (well.site_type) parts.push(`<tr><td><strong>Type:</strong></td><td>${well.site_type}</td></tr>`);
    if (well.aquifer) parts.push(`<tr><td><strong>Aquifer:</strong></td><td>${well.aquifer}</td></tr>`);
    if (well.season) parts.push(`<tr><td><strong>Season:</strong></td><td>${well.season}</td></tr>`);
    parts.push(`</table></div>`);
    return parts.join('');
  };

  const renderMap = (layerId: string, layerName: string, layerIcon: string, borderColor: string) => {
    return (
      <div className="relative h-full rounded-xl overflow-hidden shadow-2xl bg-white" style={{ borderWidth: '3px', borderColor: borderColor, borderStyle: 'solid' }}>
        <div className="absolute top-4 left-4 z-[1000] bg-white/95 backdrop-blur-sm px-4 py-2 rounded-lg shadow-lg border-2" style={{ borderColor: borderColor }}>
          <div className="flex items-center gap-2">
            <span className="text-2xl">{layerIcon}</span>
            <span className="font-bold text-gray-800">{layerName}</span>
          </div>
        </div>
        
        <MapContainer 
          center={center} 
          zoom={zoom} 
          style={{ height: "100%", width: "100%" }} 
          key={`${layerId}-${mapKey}`}
        >
          <TileLayer 
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          />
          
          {!selectedDistrict && districts.map((d, i) => (
            <GeoJSON key={`dist_${i}`} data={d.geometry} style={{ color: "#FF9800", weight: 1, fillOpacity: 0.05 }} />
          ))}
          
          {districtGeo && <GeoJSON data={districtGeo} style={{ color: borderColor, weight: 3, fillOpacity: 0.15 }} />}
          
          {layerId === 'aquifers' && aquifers.map((aquifer, i) => (
            <GeoJSON
              key={`aq_${i}`}
              data={aquifer.geometry}
              style={{ 
                color: getAquiferColor(aquifer.aquifer, i), 
                weight: 2.5, 
                fillColor: getAquiferColor(aquifer.aquifer, i),
                fillOpacity: 0.5,
                opacity: 0.8 
              }}
              onEachFeature={createAquiferPopup(aquifer, i)}
            />
          ))}
          
          {layerId === 'grace' && graceData.map((point, i) => (
            <CircleMarker
              key={`grace_${i}`}
              center={[point.latitude, point.longitude]}
              radius={5}
              fillColor={getGraceColor(point.lwe_cm)}
              color="white"
              weight={1}
              fillOpacity={0.8}
            >
              <Popup>
                <strong>GRACE LWE</strong><br/>
                Value: {point.lwe_cm.toFixed(2)} cm<br/>
                {point.cell_area_km2 && <>Cell Area: {point.cell_area_km2.toFixed(0)} km²</>}
              </Popup>
            </CircleMarker>
          ))}
          
          {layerId === 'rainfall' && rainfallData.map((point, i) => (
            <CircleMarker
              key={`rain_${i}`}
              center={[point.latitude, point.longitude]}
              radius={5}
              fillColor={getRainfallColor(point.rainfall_mm)}
              color="white"
              weight={1}
              fillOpacity={0.8}
            >
              <Popup>
                <strong>Rainfall</strong><br/>
                Value: {point.rainfall_mm.toFixed(2)} mm/day<br/>
                {point.days_averaged && point.days_averaged > 1 && <>Averaged over: {point.days_averaged} days</>}
              </Popup>
            </CircleMarker>
          ))}
          
          {layerId === 'wells' && wellsData.map((well, i) => (
            <CircleMarker
              key={`well_${well.site_id}_${i}`}
              center={[well.latitude, well.longitude]}
              radius={6}
              fillColor={getWellColor(well.gwl_category)}
              color="white"
              weight={1}
              fillOpacity={0.8}
            >
              <Popup>
                <div dangerouslySetInnerHTML={{ __html: formatWellPopup(well) }} />
              </Popup>
            </CircleMarker>
          ))}
        </MapContainer>

        <div className="absolute bottom-4 right-4 z-[1000] bg-white/95 backdrop-blur-sm p-3 rounded-lg shadow-lg max-w-[200px] max-h-[300px] overflow-y-auto border-2" style={{ borderColor: borderColor }}>
          <div className="text-xs font-bold mb-2 text-gray-700">{layerName} Legend</div>
          
          {layerId === 'aquifers' && uniqueAquiferTypesWithColors.slice(0, 5).map((item, idx) => (
            <div key={idx} className="flex items-center gap-2 mb-1">
              <div className="w-4 h-4 flex-shrink-0 rounded" style={{ backgroundColor: item.color }}></div>
              <span className="text-xs truncate">{item.type}</span>
            </div>
          ))}
          
          {layerId === 'wells' && uniqueWellCategories.map((item, idx) => (
            <div key={idx} className="flex items-center gap-2 mb-1">
              <div className="w-4 h-4 rounded-full flex-shrink-0" style={{ backgroundColor: item.color }}></div>
              <span className="text-xs">{item.category}</span>
            </div>
          ))}
          
          {layerId === 'grace' && (
            <>
              <div className="flex items-center gap-2 mb-1">
                <div className="w-4 h-4 flex-shrink-0" style={{ backgroundColor: "#8B0000" }}></div>
                <span className="text-xs">&lt; -10 cm</span>
              </div>
              <div className="flex items-center gap-2 mb-1">
                <div className="w-4 h-4 flex-shrink-0" style={{ backgroundColor: "#FF6347" }}></div>
                <span className="text-xs">-5 to 0</span>
              </div>
              <div className="flex items-center gap-2 mb-1">
                <div className="w-4 h-4 flex-shrink-0" style={{ backgroundColor: "#32CD32" }}></div>
                <span className="text-xs">0 to 5</span>
              </div>
              <div className="flex items-center gap-2 mb-1">
                <div className="w-4 h-4 flex-shrink-0" style={{ backgroundColor: "#0000CD" }}></div>
                <span className="text-xs">&gt; 10 cm</span>
              </div>
            </>
          )}
          
          {layerId === 'rainfall' && (
            <>
              <div className="flex items-center gap-2 mb-1">
                <div className="w-4 h-4 flex-shrink-0" style={{ backgroundColor: "#F0F0F0" }}></div>
                <span className="text-xs">&lt; 1 mm/day</span>
              </div>
              <div className="flex items-center gap-2 mb-1">
                <div className="w-4 h-4 flex-shrink-0" style={{ backgroundColor: "#4FC3F7" }}></div>
                <span className="text-xs">10-25</span>
              </div>
              <div className="flex items-center gap-2 mb-1">
                <div className="w-4 h-4 flex-shrink-0" style={{ backgroundColor: "#2196F3" }}></div>
                <span className="text-xs">25-50</span>
              </div>
              <div className="flex items-center gap-2 mb-1">
                <div className="w-4 h-4 flex-shrink-0" style={{ backgroundColor: "#0D47A1" }}></div>
                <span className="text-xs">&gt; 100 mm/day</span>
              </div>
            </>
          )}
        </div>
      </div>
    );
  };

  return (
    <main className="flex flex-col h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {isLoading && (
        <div className="absolute top-0 left-0 right-0 z-[2000] bg-gradient-to-r from-blue-600 to-blue-500 text-white text-center py-3 text-sm font-medium shadow-lg">
          <div className="flex items-center justify-center gap-2">
            <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
            Loading data...
          </div>
        </div>
      )}

      {alertMessage && (
        <div className={`
          px-4 py-3 text-sm font-medium text-center z-[1999] shadow-md
          ${alertType === "error" ? "bg-red-50 text-red-800 border-b-2 border-red-200" : ""}
          ${alertType === "warning" ? "bg-yellow-50 text-yellow-800 border-b-2 border-yellow-200" : ""}
          ${alertType === "success" ? "bg-green-50 text-green-800 border-b-2 border-green-200" : ""}
          ${alertType === "info" ? "bg-blue-50 text-blue-800 border-b-2 border-blue-200" : ""}
        `}>
          {alertMessage}
        </div>
      )}

      <div className="bg-white shadow-md border-b-2 border-gray-200">
        <div className="px-6 py-4">
          <div className="flex items-center gap-3 mb-4">
            <div className="bg-gradient-to-r from-blue-600 to-blue-500 text-white p-2 rounded-lg">
              <span className="text-2xl">🗺️</span>
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-800">GeoHydro Dashboard</h1>
              <p className="text-sm text-gray-500">Multi-layer Hydrogeological Analysis Platform with AI Assistant</p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <select
              value={selectedState}
              onChange={(e) => {
                const newState = e.target.value;
                setSelectedState(newState);
                setSelectedDistrict("");
                
                if (!newState) {
                  setCenter([22.9734, 78.6569]);
                  setZoom(5);
                  setDistrictGeo(null);
                  setMapKey(prev => prev + 1);
                }
              }}
              className="border-2 border-gray-300 px-4 py-2 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"
            >
              <option value="">🌍 Select State</option>
              {states.map((state, i) => (
                <option key={i} value={state.State}>{state.State}</option>
              ))}
            </select>

            <select
              value={selectedDistrict}
              onChange={(e) => setSelectedDistrict(e.target.value)}
              disabled={!districts.length}
              className="border-2 border-gray-300 px-4 py-2 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed bg-white"
            >
              <option value="">📍 Select District</option>
              {districts.map((d, i) => (
                <option key={i} value={d.district_name}>{d.district_name}</option>
              ))}
            </select>

            <select
              value={selectedYear}
              onChange={(e) => setSelectedYear(Number(e.target.value))}
              className="border-2 border-gray-300 px-4 py-2 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"
            >
              {availableYears.map(year => (
                <option key={year} value={year}>📅 {year}</option>
              ))}
            </select>

            <select
              value={selectedMonth || ""}
              onChange={(e) => setSelectedMonth(e.target.value ? Number(e.target.value) : null)}
              className="border-2 border-gray-300 px-4 py-2 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"
            >
              <option value="">All Months</option>
              {Array.from({ length: 12 }, (_, i) => i + 1).map(month => (
                <option key={month} value={month}>
                  {new Date(2000, month - 1).toLocaleString('default', { month: 'long' })}
                </option>
              ))}
            </select>

            {showRainfall && selectedMonth && (
              <select
                value={selectedDay || ""}
                onChange={(e) => setSelectedDay(e.target.value ? Number(e.target.value) : null)}
                className="border-2 border-gray-300 px-4 py-2 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"
              >
                <option value="">All Days</option>
                {Array.from({ length: 31 }, (_, i) => i + 1).map(day => (
                  <option key={day} value={day}>Day {day}</option>
                ))}
              </select>
            )}

            {showWells && (
              <select
                value={selectedSeason}
                onChange={(e) => setSelectedSeason(e.target.value)}
                className="border-2 border-gray-300 px-4 py-2 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"
              >
                <option value="">🌦️ All Seasons</option>
                <option value="PREMONSOON">Pre-Monsoon</option>
                <option value="MONSOON">Monsoon</option>
                <option value="POSTMONS_1">Post-Monsoon</option>
              </select>
            )}

            <div className="flex gap-2 ml-auto border-l-2 border-gray-200 pl-4">
              <button
                onClick={() => setShowAquifers(!showAquifers)}
                disabled={!selectedState}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed
                  ${showAquifers 
                    ? 'bg-purple-600 text-white shadow-md' 
                    : 'bg-white border-2 border-purple-300 text-purple-600 hover:bg-purple-50'}`}
              >
                🔷 Aquifers
              </button>

              <button
                onClick={() => setShowWells(!showWells)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200
                  ${showWells 
                    ? 'bg-red-600 text-white shadow-md' 
                    : 'bg-white border-2 border-red-300 text-red-600 hover:bg-red-50'}`}
              >
                💧 Wells
              </button>

              <button
                onClick={() => setShowGrace(!showGrace)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200
                  ${showGrace 
                    ? 'bg-green-600 text-white shadow-md' 
                    : 'bg-white border-2 border-green-300 text-green-600 hover:bg-green-50'}`}
              >
                🌊 GRACE
              </button>

              <button
                onClick={() => setShowRainfall(!showRainfall)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200
                  ${showRainfall 
                    ? 'bg-blue-600 text-white shadow-md' 
                    : 'bg-white border-2 border-blue-300 text-blue-600 hover:bg-blue-50'}`}
              >
                🌧️ Rainfall
              </button>
            </div>
          </div>

          <div className="flex gap-2 mt-3 pt-3 border-t-2 border-gray-200">
            <button
              onClick={() => setShowTimeseries(!showTimeseries)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200
                ${showTimeseries 
                  ? 'bg-indigo-600 text-white shadow-md' 
                  : 'bg-white border-2 border-indigo-300 text-indigo-600 hover:bg-indigo-50'}`}
            >
              📈 Timeseries
            </button>

            <button
              onClick={() => setShowSummary(!showSummary)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200
                ${showSummary 
                  ? 'bg-orange-600 text-white shadow-md' 
                  : 'bg-white border-2 border-orange-300 text-orange-600 hover:bg-orange-50'}`}
            >
              📊 Summary
            </button>

            <button
              onClick={() => setShowStorage(!showStorage)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200
                ${showStorage 
                  ? 'bg-teal-600 text-white shadow-md' 
                  : 'bg-white border-2 border-teal-300 text-teal-600 hover:bg-teal-50'}`}
            >
              💾 Storage
            </button>

            {/* ============= View Selector Dropdown ============= */}
            {showTimeseries && (
              <select
                value={timeseriesView}
                onChange={(e) => setTimeseriesView(e.target.value as "raw" | "seasonal" | "deseasonalized")}
                className="border-2 border-indigo-300 px-4 py-2 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent bg-white"
              >
                <option value="raw">📊 Raw Data</option>
                <option value="seasonal">🔄 Seasonal Pattern</option>
                <option value="deseasonalized">📈 Deseasonalized Trend</option>
              </select>
            )}
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 p-6 overflow-y-auto">
        {/* Maps Grid */}
        {activeLayers.length > 0 && (
          <div className={`grid gap-6 mb-6 ${activeLayers.length === 1 ? 'grid-cols-1' : activeLayers.length === 2 ? 'grid-cols-2' : activeLayers.length === 3 ? 'grid-cols-3' : 'grid-cols-2'}`}>
            {activeLayers.map((layer) => (
              <div key={layer.id} className="h-[600px]">
                {renderMap(layer.id, layer.name, layer.icon, layer.color)}
              </div>
            ))}
          </div>
        )}

        {/* Analytics Panels Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
          {/* ============= TIMESERIES CHART SECTION (WITH RAINFALL) ============= */}
          {showTimeseries && (
            <div className="bg-white rounded-xl shadow-2xl p-4 overflow-y-auto border-2 border-indigo-400">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <span className="text-2xl">📈</span>
                  <h3 className="text-lg font-bold text-gray-800">
                    Timeseries Analysis
                    {timeseriesResponse && (
                      <span className="text-sm font-normal text-gray-500 ml-2">
                        ({timeseriesResponse.view} view)
                      </span>
                    )}
                  </h3>
                </div>
              </div>
              
              {!timeseriesResponse ? (
                <p className="text-gray-500 text-center py-8">No timeseries data available</p>
              ) : timeseriesResponse.error ? (
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 text-sm text-yellow-800">
                  {timeseriesResponse.error}
                </div>
              ) : (
                <div className="space-y-4">
                  {/* ============= DUAL-AXIS PLOTLY CHART ============= */}
                  <div className="bg-gray-50 rounded-lg p-3 border border-gray-200">
                    <Plot
                      data={[
                        // ✅ GWL Trace (Y1 - Left axis, REVERSED for raw view)
                        {
                          x: timeseriesResponse.timeseries.map(p => p.date),
                          y: timeseriesResponse.timeseries.map(p => p.value || p.avg_gwl),
                          type: 'scatter',
                          mode: 'lines+markers',
                          name: timeseriesResponse.view === 'raw' ? 'GWL' : 
                                timeseriesResponse.view === 'seasonal' ? 'Seasonal Pattern' : 
                                'Deseasonalized',
                          line: { color: GWL_COLOR, width: 3 },
                          marker: { size: 5 },
                          yaxis: 'y'
                        },
                        // ✅ Trendline (only for raw and deseasonalized)
                        ...(timeseriesResponse.statistics?.trendline && 
                            (timeseriesResponse.view === 'raw' || timeseriesResponse.view === 'deseasonalized') ? [{
                          x: timeseriesResponse.statistics.trendline.map(t => t.date),
                          y: timeseriesResponse.statistics.trendline.map(t => t.trendline_value),
                          type: 'scatter' as const,
                          mode: 'lines' as const,
                          name: `Trend (R²=${timeseriesResponse.statistics.r_squared.toFixed(3)})`,
                          line: { 
                            color: timeseriesResponse.statistics.trend_direction === 'declining' 
                              ? TREND_DECLINE_COLOR 
                              : TREND_RECOVER_COLOR,
                            width: 2,
                            dash: 'dash' 
                          },
                          yaxis: 'y'
                        }] : []),
                        // ✅ RAINFALL Trace (Y2 - Right axis, BAR for raw, LINE for others)
                        ...(timeseriesResponse.chart_config?.rainfall_field && timeseriesResponse.timeseries.some(p => 
                          (p as any)[timeseriesResponse.chart_config!.rainfall_field] !== undefined
                        ) ? [{
                          x: timeseriesResponse.timeseries.map(p => p.date),
                          y: timeseriesResponse.timeseries.map(p => 
                            (p as any)[timeseriesResponse.chart_config!.rainfall_field] || 0
                          ),
                          type: timeseriesResponse.chart_config.rainfall_chart_type === 'bar' ? 'bar' as const : 'scatter' as const,
                          mode: timeseriesResponse.chart_config.rainfall_chart_type === 'line' ? 'lines' as const : undefined,
                          name: `Rainfall (${timeseriesResponse.chart_config.rainfall_unit})`,
                          marker: { color: RAIN_COLOR, opacity: 0.6 },
                          line: timeseriesResponse.chart_config.rainfall_chart_type === 'line' ? 
                            { color: RAIN_COLOR, width: 2 } : undefined,
                          yaxis: 'y2'
                        }] : [])
                      ]}
                      layout={{
                        autosize: true,
                        height: 450,
                        margin: { l: 60, r: 60, t: 40, b: 60 },
                        xaxis: { 
                          title: 'Date',
                          gridcolor: 'rgba(0,0,0,0.1)',
                          showgrid: true,
                          domain: [0, 1]
                        },
                        // Y1 - GWL axis (LEFT, reversed for raw view)
                        yaxis: { 
                          title: timeseriesResponse.view === 'raw' ? 'GWL (m bgl)' : 
                                 timeseriesResponse.view === 'seasonal' ? 'Seasonal Component (m)' :
                                 'Deseasonalized GWL (m)',
                          gridcolor: 'rgba(0,0,0,0.1)',
                          showgrid: true,
                          autorange: (timeseriesResponse.chart_config?.gwl_y_axis_reversed && 
                                     timeseriesResponse.view === 'raw') ? 'reversed' : true,
                          side: 'left',
                          titlefont: { color: GWL_COLOR },
                          tickfont: { color: GWL_COLOR }
                        },
                        // Y2 - Rainfall axis (RIGHT)
                        ...(timeseriesResponse.chart_config?.rainfall_field ? {
                          yaxis2: {
                            title: `Rainfall (${timeseriesResponse.chart_config.rainfall_unit})`,
                            overlaying: 'y',
                            side: 'right',
                            showgrid: false,
                            titlefont: { color: RAIN_COLOR },
                            tickfont: { color: RAIN_COLOR },
                            rangemode: 'tozero'
                          }
                        } : {}),
                        legend: {
                          orientation: 'h',
                          x: 0,
                          y: -0.2,
                          xanchor: 'left',
                          yanchor: 'top',
                          bgcolor: 'rgba(255,255,255,0.9)',
                          bordercolor: 'rgba(0,0,0,0.1)',
                          borderwidth: 1
                        },
                        hovermode: 'x unified',
                        plot_bgcolor: 'white',
                        paper_bgcolor: 'white',
                        font: { family: 'Arial, sans-serif', size: 12 }
                      }}
                      config={{
                        displayModeBar: true,
                        displaylogo: false,
                        modeBarButtonsToRemove: ['pan2d', 'select2d', 'lasso2d', 'resetScale2d', 'toggleSpikelines'],
                        toImageButtonOptions: {
                          format: 'png',
                          filename: `timeseries_${timeseriesResponse.view}_${new Date().toISOString().split('T')[0]}`,
                          height: 800,
                          width: 1200,
                          scale: 2
                        }
                      }}
                      style={{ width: '100%' }}
                      useResizeHandler={true}
                    />
                  </div>

                  {/* Statistics Card */}
                  {timeseriesResponse.statistics && (
                    <div className={`p-3 rounded-lg border ${
                      timeseriesResponse.statistics.trend_direction === 'declining' 
                        ? 'bg-red-50 border-red-200' 
                        : 'bg-green-50 border-green-200'
                    }`}>
                      <h4 className="font-bold text-sm mb-2">
                        {timeseriesResponse.view === 'raw' && '📊 Raw Data Statistics'}
                        {timeseriesResponse.view === 'seasonal' && '🔄 Seasonal Component'}
                        {timeseriesResponse.view === 'deseasonalized' && '📈 Trend Analysis'}
                      </h4>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        {timeseriesResponse.view === 'raw' && timeseriesResponse.statistics.mean_gwl && (
                          <>
                            <div><strong>Mean GWL:</strong> {timeseriesResponse.statistics.mean_gwl.toFixed(2)} m</div>
                            <div><strong>Min/Max:</strong> {timeseriesResponse.statistics.min_gwl?.toFixed(2)} / {timeseriesResponse.statistics.max_gwl?.toFixed(2)} m</div>
                          </>
                        )}
                        
                        {timeseriesResponse.view === 'seasonal' && (
                          <>
                            <div><strong>Amplitude:</strong> {timeseriesResponse.statistics.seasonal_amplitude?.toFixed(2)} m</div>
                            <div><strong>Mean:</strong> {timeseriesResponse.statistics.seasonal_mean?.toFixed(2)} m</div>
                          </>
                        )}
                        
                        {(timeseriesResponse.view === 'raw' || timeseriesResponse.view === 'deseasonalized') && (
                          <>
                            <div><strong>Slope:</strong> {timeseriesResponse.statistics.trend_slope_m_per_year.toFixed(4)} m/yr</div>
                            <div><strong>R²:</strong> {timeseriesResponse.statistics.r_squared.toFixed(3)}</div>
                            <div><strong>P-value:</strong> {timeseriesResponse.statistics.p_value.toFixed(4)}</div>
                            <div>
                              <strong>Significance:</strong> 
                              <span className={timeseriesResponse.statistics.significance === 'significant' ? 'text-green-600 font-semibold' : 'text-gray-500'}>
                                {' '}{timeseriesResponse.statistics.significance === 'significant' ? '✓ Yes (p<0.05)' : '✗ No'}
                              </span>
                            </div>
                          </>
                        )}
                      </div>
                      {timeseriesResponse.statistics.note && (
                        <div className="mt-2 pt-2 border-t border-gray-300">
                          <p className="text-xs text-gray-700 italic">
                            <strong>Note:</strong> {timeseriesResponse.statistics.note}
                          </p>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Expandable Data Table */}
                  <details className="bg-gray-50 rounded-lg border border-gray-200">
                    <summary className="px-3 py-2 cursor-pointer font-semibold text-sm text-gray-700 hover:bg-gray-100">
                      📋 View Data Table ({timeseriesResponse.count} points)
                    </summary>
                    <div className="max-h-[300px] overflow-y-auto p-3">
                      <table className="w-full text-sm">
                        <thead className="bg-gray-100 sticky top-0">
                          <tr>
                            <th className="px-2 py-1 text-left">Date</th>
                            <th className="px-2 py-1 text-right">
                              {timeseriesResponse.view === 'raw' ? 'Avg GWL (m)' : 'Value (m)'}
                            </th>
                            {timeseriesResponse.view === 'raw' && (
                              <>
                                <th className="px-2 py-1 text-right">Count</th>
                                {timeseriesResponse.chart_config?.rainfall_field && (
                                  <th className="px-2 py-1 text-right">
                                    Rainfall ({timeseriesResponse.chart_config.rainfall_unit})
                                  </th>
                                )}
                              </>
                            )}
                          </tr>
                        </thead>
                        <tbody>
                          {timeseriesResponse.timeseries.map((point, idx) => (
                            <tr key={idx} className="border-b hover:bg-gray-50">
                              <td className="px-2 py-1">{new Date(point.date).toLocaleDateString()}</td>
                              <td className="px-2 py-1 text-right font-medium">
                                {point.value?.toFixed(2) || point.avg_gwl?.toFixed(2) || 'N/A'}
                              </td>
                              {timeseriesResponse.view === 'raw' && (
                                <>
                                  {point.count && (
                                    <td className="px-2 py-1 text-right text-gray-500">{point.count}</td>
                                  )}
                                  {timeseriesResponse.chart_config?.rainfall_field && (
                                    <td className="px-2 py-1 text-right text-blue-600">
                                      {((point as any)[timeseriesResponse.chart_config.rainfall_field])?.toFixed(2) || 'N/A'}
                                    </td>
                                  )}
                                </>
                              )}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </details>

                  {/* Download Button */}
                  <div className="flex justify-center">
                    <button
                      onClick={() => {
                        const headers = [
                          'Date', 
                          timeseriesResponse.view === 'raw' ? 'Avg GWL (m)' : 'Value (m)', 
                          ...(timeseriesResponse.view === 'raw' ? ['Count'] : []),
                          ...(timeseriesResponse.chart_config?.rainfall_field ? 
                            [`Rainfall (${timeseriesResponse.chart_config.rainfall_unit})`] : [])
                        ];
                        
                        const csv = [
                          headers.join(','),
                          ...timeseriesResponse.timeseries.map(p => 
                            [
                              p.date,
                              p.value?.toFixed(2) || p.avg_gwl?.toFixed(2),
                              ...(timeseriesResponse.view === 'raw' && p.count ? [p.count] : []),
                              ...(timeseriesResponse.chart_config?.rainfall_field ? 
                                [((p as any)[timeseriesResponse.chart_config.rainfall_field])?.toFixed(2) || ''] : [])
                            ].join(',')
                          )
                        ].join('\n');
                        
                        const blob = new Blob([csv], { type: 'text/csv' });
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `timeseries_${timeseriesResponse.view}_${new Date().toISOString().split('T')[0]}.csv`;
                        a.click();
                      }}
                      className="px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium hover:bg-indigo-700 transition-all flex items-center gap-2"
                    >
                      <span>📥</span>
                      Download CSV
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ============= SUMMARY SECTION (unchanged) ============= */}
          {showSummary && (
            <div className="bg-white rounded-xl shadow-2xl p-4 overflow-y-auto border-2 border-orange-400">
              <div className="flex items-center gap-2 mb-4">
                <span className="text-2xl">📊</span>
                <h3 className="text-lg font-bold text-gray-800">Regional Summary</h3>
              </div>
              
              {!summaryData ? (
                <p className="text-gray-500 text-center py-8">No summary data available</p>
              ) : summaryData.error ? (
                <p className="text-red-500 text-center py-8">{summaryData.error}</p>
              ) : (
                <div className="space-y-4">
                  <div className="bg-blue-50 p-3 rounded-lg">
                    <h4 className="font-bold text-sm mb-2 text-blue-900">Statistics</h4>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div><strong>Mean GWL:</strong> {summaryData.statistics.mean_gwl.toFixed(2)} m</div>
                      <div><strong>Min GWL:</strong> {summaryData.statistics.min_gwl.toFixed(2)} m</div>
                      <div><strong>Max GWL:</strong> {summaryData.statistics.max_gwl.toFixed(2)} m</div>
                      <div><strong>Std Dev:</strong> {summaryData.statistics.std_gwl.toFixed(2)} m</div>
                    </div>
                  </div>

                  <div className={`p-3 rounded-lg ${summaryData.trend.trend_direction === 'declining' ? 'bg-red-50' : 'bg-green-50'}`}>
                    <h4 className={`font-bold text-sm mb-2 ${summaryData.trend.trend_direction === 'declining' ? 'text-red-900' : 'text-green-900'}`}>
                      Trend Analysis
                    </h4>
                    <div className="space-y-1 text-sm">
                      <div><strong>Slope:</strong> {summaryData.trend.slope_m_per_year.toFixed(4)} m/year</div>
                      <div><strong>Direction:</strong> {summaryData.trend.trend_direction === 'declining' ? '📉 Declining' : '📈 Recovering'}</div>
                      <div><strong>R²:</strong> {summaryData.trend.r_squared.toFixed(3)}</div>
                      <div><strong>P-value:</strong> {summaryData.trend.p_value.toFixed(4)}</div>
                      <div><strong>Significance:</strong> {summaryData.trend.significance === 'significant' ? '✓ Significant' : '✗ Not Significant'}</div>
                    </div>
                  </div>

                  <div className="bg-gray-50 p-3 rounded-lg">
                    <h4 className="font-bold text-sm mb-2 text-gray-900">Temporal Coverage</h4>
                    <div className="space-y-1 text-sm">
                      <div><strong>Period:</strong> {new Date(summaryData.temporal_coverage.start_date).toLocaleDateString()} to {new Date(summaryData.temporal_coverage.end_date).toLocaleDateString()}</div>
                      <div><strong>Span:</strong> {summaryData.temporal_coverage.span_years} years</div>
                      <div><strong>Months of Data:</strong> {summaryData.temporal_coverage.months_of_data}</div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ============= STORAGE SECTION (unchanged) ============= */}
          {showStorage && (
            <div className="bg-white rounded-xl shadow-2xl p-4 overflow-y-auto border-2 border-teal-400">
              <div className="flex items-center gap-2 mb-4">
                <span className="text-2xl">💾</span>
                <h3 className="text-lg font-bold text-gray-800">Storage Analysis</h3>
              </div>
              
              {!storageData ? (
                <p className="text-gray-500 text-center py-8">No storage data available</p>
              ) : storageData.error ? (
                <p className="text-red-500 text-center py-8">{storageData.error}</p>
              ) : (
                <div className="space-y-4">
                  <div className="bg-teal-50 p-3 rounded-lg">
                    <h4 className="font-bold text-sm mb-2 text-teal-900">Aquifer Properties</h4>
                    <div className="space-y-1 text-sm">
                      <div><strong>Total Area:</strong> {storageData.aquifer_properties.total_area_km2.toFixed(2)} km²</div>
                      <div><strong>Specific Yield:</strong> {storageData.aquifer_properties.area_weighted_specific_yield.toFixed(4)}</div>
                    </div>
                  </div>

                  <div className="bg-blue-50 p-3 rounded-lg">
                    <h4 className="font-bold text-sm mb-2 text-blue-900">Summary</h4>
                    <div className="space-y-1 text-sm">
                      <div><strong>Avg Annual Change:</strong> {storageData.summary.avg_annual_storage_change_mcm.toFixed(2)} MCM</div>
                      <div><strong>Years Analyzed:</strong> {storageData.summary.years_analyzed}</div>
                    </div>
                  </div>

                  <div className="bg-gray-50 p-3 rounded-lg">
                    <h4 className="font-bold text-sm mb-2 text-gray-900">Yearly Storage</h4>
                    <div className="max-h-[250px] overflow-y-auto">
                      <table className="w-full text-xs">
                        <thead className="bg-gray-200 sticky top-0">
                          <tr>
                            <th className="px-2 py-1">Year</th>
                            <th className="px-2 py-1">Pre (m)</th>
                            <th className="px-2 py-1">Post (m)</th>
                            <th className="px-2 py-1">Δ (MCM)</th>
                          </tr>
                        </thead>
                        <tbody>
                          {storageData.yearly_storage.map((year, idx) => (
                            <tr key={idx} className="border-b">
                              <td className="px-2 py-1">{year.year}</td>
                              <td className="px-2 py-1">{year.pre_monsoon_gwl.toFixed(2)}</td>
                              <td className="px-2 py-1">{year.post_monsoon_gwl.toFixed(2)}</td>
                              <td className={`px-2 py-1 font-bold ${year.storage_change_mcm > 0 ? 'text-green-600' : 'text-red-600'}`}>
                                {year.storage_change_mcm.toFixed(2)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Chatbot (unchanged) */}
      <div className="fixed right-6 bottom-6 z-[2000]">
        {isChatOpen && (
          <div className="mb-4 w-96 h-[600px] bg-white rounded-2xl shadow-2xl flex flex-col border-2 border-blue-200">
            <div className="bg-gradient-to-r from-blue-600 to-blue-500 text-white px-6 py-4 rounded-t-2xl flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="bg-white/20 p-2 rounded-lg">
                  <span className="text-2xl">🤖</span>
                </div>
                <div>
                  <h3 className="font-bold text-lg">GeoHydro Assistant</h3>
                  <p className="text-xs text-blue-100">
                    {activeLayers.length > 0 
                      ? `Analyzing: ${activeLayers.map(l => l.name).join(', ')}` 
                      : 'Groundwater Expert'}
                  </p>
                </div>
              </div>
              <button
                onClick={() => setIsChatOpen(false)}
                className="text-white hover:bg-white/20 rounded-lg p-2 transition-all"
              >
                ✕
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
              {chatMessages.length === 1 && (
                <div className="mb-4">
                  <p className="text-xs text-gray-500 mb-2 font-medium">Try asking:</p>
                  <div className="space-y-2">
                    {activeLayers.length > 0 ? (
                      <>
                        <button
                          onClick={() => setChatInput("What patterns do you see in the current map?")}
                          className="block w-full text-left text-xs bg-white border border-gray-200 rounded-lg px-3 py-2 hover:bg-blue-50 hover:border-blue-300 transition-all"
                        >
                          💡 What patterns do you see in the current map?
                        </button>
                        <button
                          onClick={() => setChatInput("Analyze the displayed data")}
                          className="block w-full text-left text-xs bg-white border border-gray-200 rounded-lg px-3 py-2 hover:bg-blue-50 hover:border-blue-300 transition-all"
                        >
                          💡 Analyze the displayed data
                        </button>
                        <button
                          onClick={() => setChatInput("What does this tell us about groundwater?")}
                          className="block w-full text-left text-xs bg-white border border-gray-200 rounded-lg px-3 py-2 hover:bg-blue-50 hover:border-blue-300 transition-all"
                        >
                          💡 What does this tell us about groundwater?
                        </button>
                      </>
                    ) : (
                      suggestedQuestions.map((question, idx) => (
                        <button
                          key={idx}
                          onClick={() => setChatInput(question)}
                          className="block w-full text-left text-xs bg-white border border-gray-200 rounded-lg px-3 py-2 hover:bg-blue-50 hover:border-blue-300 transition-all"
                        >
                          💡 {question}
                        </button>
                      ))
                    )}
                  </div>
                </div>
              )}

              {chatMessages.map((message, idx) => (
                <div
                  key={idx}
                  className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                      message.role === "user"
                        ? "bg-blue-600 text-white"
                        : "bg-white border border-gray-200 text-gray-800"
                    }`}
                  >
                    <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                    {message.role === "assistant" && message.sourcesUsed && message.sourcesUsed > 0 && (
                      <div className="mt-2 pt-2 border-t border-gray-200 flex items-center gap-2 text-xs text-gray-500">
                        <span className="bg-green-100 text-green-700 px-2 py-1 rounded">
                          📚 {message.sourcesUsed} sources
                        </span>
                      </div>
                    )}
                    <p className="text-xs mt-1 opacity-60">
                      {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </p>
                  </div>
                </div>
              ))}

              {isChatLoading && (
                <div className="flex justify-start">
                  <div className="bg-white border border-gray-200 rounded-2xl px-4 py-3">
                    <div className="flex items-center gap-2">
                      <div className="animate-spin rounded-full h-4 w-4 border-2 border-blue-600 border-t-transparent"></div>
                      <span className="text-sm text-gray-600">Analyzing map context...</span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            <div className="p-4 border-t border-gray-200 bg-white rounded-b-2xl">
              <div className="flex gap-2">
                <textarea
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyPress={handleChatKeyPress}
                  placeholder={
                    activeLayers.length > 0 
                      ? "Ask about the displayed data..." 
                      : "Ask about groundwater, GRACE, rainfall..."
                  }
                  className="flex-1 border-2 border-gray-300 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                  rows={2}
                  disabled={isChatLoading}
                />
                <button
                  onClick={handleSendMessage}
                  disabled={isChatLoading || !chatInput.trim()}
                  className="bg-blue-600 text-white px-4 rounded-lg hover:bg-blue-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
                >
                  <span className="text-xl">➤</span>
                </button>
              </div>
              <p className="text-xs text-gray-400 mt-2 text-center">
                {activeLayers.length > 0 && (
                  <span className="text-blue-600 font-medium">Context-aware • </span>
                )}
                Powered by LLaMA 3.1 • Press Enter to send
              </p>
            </div>
          </div>
        )}

        <button
          onClick={() => setIsChatOpen(!isChatOpen)}
          className={`bg-gradient-to-r from-blue-600 to-blue-500 text-white rounded-full w-16 h-16 flex items-center justify-center shadow-2xl hover:scale-110 transition-all duration-200 ${
            isChatOpen ? "rotate-0" : "animate-bounce"
          }`}
        >
          {isChatOpen ? (
            <span className="text-2xl">✕</span>
          ) : (
            <span className="text-3xl">💬</span>
          )}
        </button>
      </div>
    </main>
  );
}