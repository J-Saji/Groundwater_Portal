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

interface TimeseriesPoint {
  date: string;
  avg_gwl?: number;
  avg_tws?: number;
  avg_rainfall?: number;
  monthly_rainfall_mm?: number;
  days_in_month?: number;
  count?: number;
  gwl_seasonal?: number;
  grace_seasonal?: number;
  rainfall_seasonal?: number;
  gwl_deseasonalized?: number;
  grace_deseasonalized?: number;
  rainfall_deseasonalized?: number;
}

interface TrendStatistics {
  slope_per_month?: number;
  slope_per_year: number;
  r_squared: number;
  p_value: number;
  direction: string;
  significance: string;
  mean?: number;
  min?: number;
  max?: number;
  std?: number;
}

interface TimeseriesStatistics {
  gwl_trend?: TrendStatistics;
  grace_trend?: TrendStatistics;
  rainfall_trend?: TrendStatistics;
  seasonal_amplitude?: number;
  seasonal_mean?: number;
  view?: string;
  note?: string;
}

interface ChartConfig {
  gwl_chart_type: string;
  grace_chart_type?: string;
  rainfall_chart_type: string;
  rainfall_field: string;
  rainfall_unit: string;
  gwl_y_axis_reversed: boolean;
}

interface TimeseriesResponse {
  view: string;
  aggregation: string;
  filters: {
    state: string | null;
    district: string | null;
  };
  count: number;
  chart_config?: ChartConfig;
  timeseries: TimeseriesPoint[];
  statistics: TimeseriesStatistics | null;
  error?: string;
  message?: string;
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
    [key: string]: any; // Allows dynamic module data
    active_module?: string; // NEW: Track which advanced module is active
    asi?: any;
    network_density?: any;
    sass?: any;
    divergence?: any;
    forecast?: any;
    recharge?: any;
    significant_trends?: any;
    changepoints?: any;
    lag_correlation?: any;
    hotspots?: any;
    timeseries?: any;
  };
}

// ============= NEW: Advanced Module Interfaces =============
interface ASIFeature {
  type: "Feature";
  id: string;
  properties: {
    aquifer: string;
    majoraquif: string;
    asi_score: number;
    specific_yield: number;
    area_m2: number;
  };
  geometry: GeoJSON.Geometry;
}

interface ASIResponse {
  module: string;
  description: string;
  filters: { state: string | null; district: string | null };
  statistics: {
    mean_asi: number;
    median_asi: number;
    std_asi: number;
    min_asi: number;
    max_asi: number;
    dominant_aquifer: string;
    avg_specific_yield: number;
    total_area_km2: number;
  };
  count: number;
  geojson: {
    type: "FeatureCollection";
    features: ASIFeature[];
  };
  methodology: {
    approach: string;
    quantile_stretch: { low: number; high: number };
    interpretation: string;
  };
}

interface NetworkDensitySitePoint {
  site_id: string;
  latitude: number;
  longitude: number;
  slope_m_per_year: number;
  gwl_std: number;
  strength: number;
  n_observations: number;
  local_density_per_km2: number;
  neighbors_within_radius: number;
}

interface NetworkDensityGridPoint {
  latitude: number;
  longitude: number;
  density_per_1000km2: number;
}

interface NetworkDensityResponse {
  module: string;
  description: string;
  filters: { state: string | null; district: string | null };
  parameters: { radius_km: number };
  statistics: {
    total_sites: number;
    avg_strength: number;
    avg_density: number;
    median_observations: number;
    mean_gridded_density: number;
    max_gridded_density: number;
  };
  map1_site_level: {
    count: number;
    data: NetworkDensitySitePoint[];
  };
  map2_gridded: {
    count: number;
    grid_resolution: string;
    data: NetworkDensityGridPoint[];
  };
}

interface SASSPoint {
  site_id: string;
  latitude: number;
  longitude: number;
  sass_score: number;
  gwl_stress: number;
  grace_z: number;
  rain_z: number;
  gwl: number;
}

interface SASSResponse {
  module: string;
  description: string;
  filters: { state: string | null; district: string | null; year: number; month: number };
  formula: string;
  statistics: {
    mean_sass: number;
    max_sass: number;
    min_sass: number;
    stressed_sites: number;
  };
  count: number;
  data: SASSPoint[];
}

interface DivergencePoint {
  latitude: number;
  longitude: number;
  divergence: number;
  grace_z: number;
  well_z_interpolated: number;
  tws: number;
}

interface DivergenceResponse {
  module: string;
  description: string;
  filters: { state: string | null; district: string | null; year: number; month: number };
  statistics: {
    mean_divergence: number;
    positive_divergence_pixels: number;
    negative_divergence_pixels: number;
    max_divergence: number;
    min_divergence: number;
  };
  count: number;
  data: DivergencePoint[];
}

interface ForecastPoint {
  longitude: number;
  latitude: number;
  pred_delta_m: number;           // ← Main forecast value (12-month change)
  current_gwl: number;
  forecast_gwl: number;
  r_squared: number;
  trend_component: number;
  grace_component: number;
  n_months_training: number;
  mean_grace_contribution: number;
}

interface ForecastResponse {
  module: string;
  description: string;
  method: string;
  filters: { state: string | null; district: string | null };
  parameters: {
    forecast_months: number;
    k_neighbors: number;
    grid_resolution: number;
    grace_used: boolean;
  };
  statistics: {
    mean_change_m: number;
    median_change_m: number;
    declining_cells: number;
    recovering_cells: number;
    mean_r_squared: number;
    success_rate: number;
  };
  count: number;
  data: ForecastPoint[];
}

interface RechargeStructure {
  structure_type: string;
  recommended_units: number;
  total_capacity_mcm: number;
  allocation_fraction: number;
}

interface RechargeSiteRecommendation {
  site_id: string;
  latitude: number;
  longitude: number;
  stress_category: string;
  recommended_structure: string;
  current_gwl: number;
}

interface RechargeResponse {
  module: string;
  description: string;
  filters: { state: string | null; district: string | null };
  analysis_parameters: {
    area_km2: number;
    dominant_lithology: string;
    runoff_coefficient: number;
    monsoon_rainfall_m: number;
    capture_fraction: number;
    year_analyzed: number;
  };
  potential: {
    total_recharge_potential_mcm: number;
    per_km2_mcm: number;
  };
  structure_plan: RechargeStructure[];
  site_recommendations: RechargeSiteRecommendation[];
  count: number;
}

interface SignificantTrendPoint {
  site_id: string;
  latitude: number;
  longitude: number;
  slope_m_per_year: number;
  p_value: number;
  trend_direction: string;
  significance_level: string;
  n_months: number;
  date_range: string;
}

interface SignificantTrendsResponse {
  module: string;
  description: string;
  filters: { state: string | null; district: string | null };
  parameters: { p_threshold: number; method: string };
  statistics: {
    total_significant: number;
    declining: number;
    recovering: number;
    mean_slope: number;
    high_significance: number;
  };
  count: number;
  data: SignificantTrendPoint[];
}
interface ChangepointCoverageSite {
  site_id: string;
  n_months: number;
  date_start: string;
  date_end: string;
  span_years: number;
}
interface ChangepointSite {
  site_id: string;
  latitude: number;
  longitude: number;
  changepoint_date: string;
  changepoint_year: number;
  changepoint_month: number;
  n_breakpoints: number;
  all_breakpoints: string[];
  series_length: number;
}

interface ChangepointResponse {
  module: string;
  description: string;
  filters: { state: string | null; district: string | null };
  parameters: { penalty: number; algorithm: string; model: string };
  statistics: {
    total_sites_analyzed: number;
    sites_with_changepoints: number;
    detection_rate: number;
    avg_series_length: number;
    avg_span_years: number;
  };
  changepoints: {
    count: number;
    data: ChangepointSite[];
  };
  coverage: {
    count: number;
    data: ChangepointCoverageSite[];
  };
}

interface LagCorrelationPoint {
  site_id: string;
  latitude: number;
  longitude: number;
  best_lag_months: number;
  correlation: number;
  abs_correlation: number;
  relationship: string;
  n_months_analyzed: number;
}

interface LagCorrelationResponse {
  module: string;
  description: string;
  filters: { state: string | null; district: string | null };
  parameters: { max_lag_months: number };
  statistics: {
    total_sites: number;
    mean_lag: number;
    median_lag: number;
    mean_abs_correlation: number;
    lag_distribution: Record<number, number>;
  };
  count: number;
  data: LagCorrelationPoint[];
}

interface HotspotPoint {
  site_id: string;
  latitude: number;
  longitude: number;
  slope_m_per_year: number;
  cluster: number;
}

interface HotspotCluster {
  cluster_id: number;
  n_sites: number;
  mean_slope: number;
  max_slope: number;
  centroid_lat: number;
  centroid_lon: number;
}

interface HotspotsResponse {
  module: string;
  description: string;
  filters: { state: string | null; district: string | null };
  parameters: {
    eps_km: number;
    min_samples: number;
    algorithm: string;
    metric: string;
  };
  statistics: {
    total_declining_sites: number;
    n_clusters: number;
    noise_points: number;
    clustered_points: number;
    clustering_rate: number;
  };
  clusters: HotspotCluster[];
  count: number;
  data: HotspotPoint[];
}

const COLOR_PALETTE = [
  "#E91E63", "#9C27B0", "#673AB7", "#3F51B5", "#2196F3", "#03A9F4",
  "#00BCD4", "#009688", "#4CAF50", "#8BC34A", "#CDDC39", "#FFEB3B",
  "#FFC107", "#FF9800", "#FF5722", "#795548", "#607D88", "#F44336",
];

// ============= DASH-COMPATIBLE COLORS =============
const GWL_COLOR = "#8D6E63";
const GRACE_COLOR = "#00695C";
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

// ============= NEW: Advanced Module Color Helpers =============
const getStressColor = (category: string): string => {
  switch (category) {
    case 'Critical': return '#DC2626';
    case 'Stressed': return '#F59E0B';
    case 'Moderate': return '#FCD34D';
    case 'Healthy': return '#22C55E';
    default: return '#9CA3AF';
  }
};

const getASIColor = (score: number): string => {
  // Match backend YlGn colorscale (Yellow-Green)
  if (score < 1) return "#FEE5D9";      // Very light yellow
  if (score < 2) return "#FCBBA1";      // Light orange-yellow
  if (score < 3) return "#FB6A4A";      // Orange
  if (score < 4) return "#CB181D";      // Red-orange
  return "#67000D";                      // Dark red
  
  // Alternative: Green scale (higher = better)
  // if (score < 1) return "#FFFFCC";   // Light yellow
  // if (score < 2) return "#C2E699";   // Yellow-green
  // if (score < 3) return "#78C679";   // Light green
  // if (score < 4) return "#31A354";   // Green
  // return "#006837";                   // Dark green
};
const getSASSColor = (score: number): string => {
  if (score < -1) return "#22C55E";
  if (score < 0) return "#84CC16";
  if (score < 1) return "#FCD34D";
  if (score < 2) return "#F59E0B";
  return "#DC2626";
};

const getDensityColor = (density: number): string => {
  // density in sites per 1000 km²
  if (density < 5) return "#FEF0D9";     // Very sparse
  if (density < 10) return "#FDCC8A";    // Sparse
  if (density < 20) return "#FC8D59";    // Moderate
  if (density < 40) return "#E34A33";    // Dense
  return "#B30000";                       // Very dense
};

const getDivergenceColor = (value: number): string => {
  const numValue = Number(value);  // ← FORCE NUMBER CONVERSION
  
  if (numValue < -2) return "#DC2626";
  if (numValue < -1) return "#F87171";
  if (numValue < 0) return "#FCA5A5";
  if (numValue < 1) return "#93C5FD";
  if (numValue < 2) return "#3B82F6";
  return "#1D4ED8";
};

const getHotspotColor = (cluster: number): string => {
  if (cluster === -1) return "#9CA3AF"; // Noise
  return COLOR_PALETTE[cluster % COLOR_PALETTE.length];
};

const getTrendColor = (slope: number): string => {
  if (slope > 0) return "#DC2626"; // Declining (deeper = red)
  return "#16A34A"; // Recovering (shallower = green)
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
  
  const [timeseriesResponse, setTimeseriesResponse] = useState<TimeseriesResponse | null>(null);
  const [summaryData, setSummaryData] = useState<WellsSummary | null>(null);
  const [storageData, setStorageData] = useState<StorageResponse | null>(null);
  
  // ============= NEW: Advanced Module States =============
  const [asiData, setAsiData] = useState<ASIResponse | null>(null);
  const [networkDensityData, setNetworkDensityData] = useState<NetworkDensityResponse | null>(null);
  const [sassData, setSassData] = useState<SASSResponse | null>(null);
  const [divergenceData, setDivergenceData] = useState<DivergenceResponse | null>(null);
  const [forecastData, setForecastData] = useState<ForecastResponse | null>(null);
  const [rechargeData, setRechargeData] = useState<RechargeResponse | null>(null);
  const [significantTrendsData, setSignificantTrendsData] = useState<SignificantTrendsResponse | null>(null);
  const [changepointsData, setChangepointsData] = useState<ChangepointResponse | null>(null);
  const [lagCorrelationData, setLagCorrelationData] = useState<LagCorrelationResponse | null>(null);
  const [hotspotsData, setHotspotsData] = useState<HotspotsResponse | null>(null);
  
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
  
  // ============= NEW: Advanced Module Toggle States =============
  const [showAdvancedMenu, setShowAdvancedMenu] = useState<boolean>(false);
  const [selectedAdvancedModule, setSelectedAdvancedModule] = useState<string>("");
  
  const [districtGeo, setDistrictGeo] = useState<Geometry | null>(null);
  const [center, setCenter] = useState<[number, number]>([22.9734, 78.6569]);
  const [zoom, setZoom] = useState(5);
  const [mapKey, setMapKey] = useState<number>(0);

  const [selectedYear, setSelectedYear] = useState<number>(2011);
  const [selectedMonth, setSelectedMonth] = useState<number | null>(null);
  const [selectedDay, setSelectedDay] = useState<number | null>(null);
  const [selectedSeason, setSelectedSeason] = useState<string>("");
  
  const [timeseriesView, setTimeseriesView] = useState<"raw" | "seasonal" | "deseasonalized">("raw");

  const [alertMessage, setAlertMessage] = useState<string>("");
  const [alertType, setAlertType] = useState<"info" | "warning" | "error" | "success">("info");
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const [isChatOpen, setIsChatOpen] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content: "Hello! I'm your Groundwater and Remote Sensing expert. Ask me anything about GRACE data, rainfall patterns, groundwater levels, aquifer systems, or advanced modules!",
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

    // ============= EXISTING MAP LAYERS CONTEXT =============
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

    // ============= NEW: ADVANCED MODULES CONTEXT =============
    
    // Add active advanced module info
    if (selectedAdvancedModule) {
      dataSummary.active_module = selectedAdvancedModule;
    }

    // ASI Context
    if (selectedAdvancedModule === 'ASI' && asiData) {
      dataSummary.asi = {
        module: asiData.module,
        description: asiData.description,
        statistics: {
          mean_asi: asiData.statistics.mean_asi,
          median_asi: asiData.statistics.median_asi,
          min_asi: asiData.statistics.min_asi,
          max_asi: asiData.statistics.max_asi,
          dominant_aquifer: asiData.statistics.dominant_aquifer,
          total_area_km2: asiData.statistics.total_area_km2
        },
        polygons_analyzed: asiData.count,
        high_quality_count: asiData.geojson.features.filter(f => f.properties.asi_score > 3).length,
        low_quality_count: asiData.geojson.features.filter(f => f.properties.asi_score < 2).length
      };
    }

    // Network Density Context
    if (selectedAdvancedModule === 'NETWORK_DENSITY' && networkDensityData) {
      dataSummary.network_density = {
        module: networkDensityData.module,
        description: networkDensityData.description,
        statistics: {
          total_sites: networkDensityData.statistics.total_sites,
          avg_strength: networkDensityData.statistics.avg_strength,
          avg_local_density: networkDensityData.statistics.avg_local_density
        },
        site_level_count: networkDensityData.map1_site_level.count,
        grid_count: networkDensityData.map2_gridded.count,
        radius_km: networkDensityData.parameters.radius_km
      };
    }

    // SASS Context
    if (selectedAdvancedModule === 'SASS' && sassData) {
      dataSummary.sass = {
        module: sassData.module,
        description: sassData.description,
        formula: sassData.formula,
        statistics: {
          mean_sass: sassData.statistics.mean_sass,
          max_sass: sassData.statistics.max_sass,
          stressed_sites: sassData.statistics.stressed_sites
        },
        sites_analyzed: sassData.count,
        year: sassData.filters.year,
        month: sassData.filters.month
      };
    }

    // Divergence Context
    if (selectedAdvancedModule === 'GRACE_DIVERGENCE' && divergenceData) {
      dataSummary.divergence = {
        module: divergenceData.module,
        description: divergenceData.description,
        statistics: {
          mean_divergence: divergenceData.statistics.mean_divergence,
          positive_divergence_pixels: divergenceData.statistics.positive_divergence_pixels,
          negative_divergence_pixels: divergenceData.statistics.negative_divergence_pixels
        },
        pixels_analyzed: divergenceData.count,
        interpretation: divergenceData.statistics.mean_divergence > 0 
          ? "GRACE shows MORE water than wells suggest" 
          : "GRACE shows LESS water than wells suggest"
      };
    }

    // Forecast Context
    if (selectedAdvancedModule === 'FORECAST' && forecastData) {
      dataSummary.forecast = {
        module: forecastData.module,
        description: forecastData.description,
        method: forecastData.method,
        statistics: {
          mean_change_m: forecastData.statistics.mean_change_m,
          median_change_m: forecastData.statistics.median_change_m,
          declining_cells: forecastData.statistics.declining_cells,
          recovering_cells: forecastData.statistics.recovering_cells,
          mean_r_squared: forecastData.statistics.mean_r_squared,
          mean_grace_contribution: forecastData.statistics.mean_grace_contribution
        },
        forecast_months: forecastData.parameters.forecast_months,
        grid_cells: forecastData.count,
        grace_used: forecastData.parameters.grace_used,
        overall_trend: forecastData.statistics.mean_change_m > 0 ? "Declining" : "Recovering"
      };
    }

    // Recharge Planning Context
    if (selectedAdvancedModule === 'RECHARGE' && rechargeData) {
      dataSummary.recharge = {
        module: rechargeData.module,
        description: rechargeData.description,
        potential: {
          total_recharge_potential_mcm: rechargeData.potential.total_recharge_potential_mcm,
          per_km2_mcm: rechargeData.potential.per_km2_mcm
        },
        analysis_parameters: {
          area_km2: rechargeData.analysis_parameters.area_km2,
          dominant_lithology: rechargeData.analysis_parameters.dominant_lithology,
          monsoon_rainfall_m: rechargeData.analysis_parameters.monsoon_rainfall_m
        },
        structure_plan_summary: rechargeData.structure_plan.map(s => ({
          type: s.structure_type,
          units: s.recommended_units,
          capacity_mcm: s.total_capacity_mcm
        })),
        site_recommendations_count: rechargeData.count
      };
    }

    // Significant Trends Context
    if (selectedAdvancedModule === 'SIGNIFICANT_TRENDS' && significantTrendsData) {
      dataSummary.significant_trends = {
        module: significantTrendsData.module,
        description: significantTrendsData.description,
        statistics: {
          total_significant: significantTrendsData.statistics.total_significant,
          declining: significantTrendsData.statistics.declining,
          recovering: significantTrendsData.statistics.recovering,
          mean_slope: significantTrendsData.statistics.mean_slope
        },
        p_threshold: significantTrendsData.parameters.p_threshold,
        method: significantTrendsData.parameters.method
      };
    }

    // Changepoints Context
    if (selectedAdvancedModule === 'CHANGEPOINTS' && changepointsData) {
      dataSummary.changepoints = {
        module: changepointsData.module,
        description: changepointsData.description,
        statistics: {
          total_sites_analyzed: changepointsData.statistics.total_sites_analyzed,
          sites_with_changepoints: changepointsData.statistics.sites_with_changepoints,
          detection_rate: changepointsData.statistics.detection_rate
        },
        changepoints_found: changepointsData.changepoints.count,
        algorithm: changepointsData.parameters.algorithm
      };
    }

    // Lag Correlation Context
    if (selectedAdvancedModule === 'LAG_CORRELATION' && lagCorrelationData) {
      dataSummary.lag_correlation = {
        module: lagCorrelationData.module,
        description: lagCorrelationData.description,
        statistics: {
          mean_lag: lagCorrelationData.statistics.mean_lag,
          median_lag: lagCorrelationData.statistics.median_lag,
          mean_abs_correlation: lagCorrelationData.statistics.mean_abs_correlation
        },
        sites_analyzed: lagCorrelationData.count,
        max_lag_tested: lagCorrelationData.parameters.max_lag_months,
        lag_distribution: lagCorrelationData.statistics.lag_distribution
      };
    }

    // Hotspots Context
    if (selectedAdvancedModule === 'HOTSPOTS' && hotspotsData) {
      dataSummary.hotspots = {
        module: hotspotsData.module,
        description: hotspotsData.description,
        statistics: {
          total_declining_sites: hotspotsData.statistics.total_declining_sites,
          n_clusters: hotspotsData.statistics.n_clusters,
          clustered_points: hotspotsData.statistics.clustered_points,
          noise_points: hotspotsData.statistics.noise_points
        },
        clusters_detail: hotspotsData.clusters.map(c => ({
          cluster_id: c.cluster_id,
          n_sites: c.n_sites,
          mean_slope: c.mean_slope
        })),
        eps_km: hotspotsData.parameters.eps_km
      };
    }

    // ============= TIMESERIES CONTEXT =============
    if (showTimeseries && timeseriesResponse) {
      dataSummary.timeseries = {
        view: timeseriesResponse.view,
        aggregation: timeseriesResponse.aggregation,
        data_points: timeseriesResponse.count,
        statistics: timeseriesResponse.statistics ? {
          gwl_trend: timeseriesResponse.statistics.gwl_trend,
          grace_trend: timeseriesResponse.statistics.grace_trend,
          rainfall_trend: timeseriesResponse.statistics.rainfall_trend
        } : null
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
    showAquifers, showGrace, showRainfall, showWells, showTimeseries,
    aquifers, graceResponse, rainfallResponse, wellsResponse, wellsData,
    summaryData, timeseriesResponse,
    selectedAdvancedModule, asiData, networkDensityData, sassData, 
    divergenceData, forecastData, rechargeData, significantTrendsData,
    changepointsData, lagCorrelationData, hotspotsData,
    selectedState, selectedDistrict, selectedYear, selectedMonth, selectedSeason
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

  const getSuggestedQuestions = (): string[] => {
    // Base questions
    const baseQuestions = [
      "What is GRACE satellite data?",
      "Explain groundwater depletion in India"
    ];

    // Module-specific questions
    const moduleQuestions: Record<string, string[]> = {
      'ASI': [
        "Which aquifer zones have the best storage potential?",
        "What does the ASI score mean for recharge suitability?",
        "How reliable is this ASI assessment?"
      ],
      'NETWORK_DENSITY': [
        "Where are the monitoring coverage gaps?",
        "Which areas have the strongest GWL signals?",
        "What does local density tell us about data quality?"
      ],
      'SASS': [
        "Which sites are most stressed right now?",
        "How does GRACE data relate to ground measurements here?",
        "What is causing the high stress scores?"
      ],
      'GRACE_DIVERGENCE': [
        "Why is GRACE diverging from ground measurements?",
        "What does positive/negative divergence indicate?",
        "Should we trust GRACE or wells more in this region?"
      ],
      'FORECAST': [
        "What does this forecast indicate for water security?",
        "How much is the GRACE contribution to the prediction?",
        "Which areas should prioritize intervention?"
      ],
      'RECHARGE': [
        "What structures are best for this region?",
        "How was the recharge potential calculated?",
        "Why are different structures recommended for different sites?"
      ],
      'SIGNIFICANT_TRENDS': [
        "Which sites have the most reliable declining trends?",
        "What does the p-value tell us about significance?",
        "Are these trends accelerating or stable?"
      ],
      'CHANGEPOINTS': [
        "What caused these structural breaks in GWL?",
        "Do changepoints align with policy or climate events?",
        "How should we interpret regime shifts?"
      ],
      'LAG_CORRELATION': [
        "What does the rainfall-GWL lag reveal about aquifer type?",
        "Why do some sites respond faster than others?",
        "How can lag information guide irrigation timing?"
      ],
      'HOTSPOTS': [
        "Where are the priority intervention zones?",
        "What connects sites in the same cluster?",
        "How should we address hotspot clusters?"
      ]
    };

    // Return module-specific questions if active, otherwise base questions
    if (selectedAdvancedModule && moduleQuestions[selectedAdvancedModule]) {
      return [...moduleQuestions[selectedAdvancedModule], "Explain this analysis in simple terms"];
    }

    // Default questions when viewing maps
    if (showGrace || showRainfall || showWells) {
      return [
        ...baseQuestions,
        "What patterns do you see in this map?",
        "Analyze the current data displayed"
      ];
    }

    return baseQuestions;
  };

  const activeLayers = [
    { id: 'aquifers', name: 'Aquifers', icon: '🔷', show: showAquifers, color: 'purple' },
    { id: 'grace', name: 'GRACE', icon: '🌊', show: showGrace, color: 'green' },
    { id: 'rainfall', name: 'Rainfall', icon: '🌧️', show: showRainfall, color: 'blue' },
    { id: 'wells', name: 'Wells', icon: '💧', show: showWells, color: 'red' }
  ].filter(layer => layer.show);

  // ============= NEW: Advanced Module Loading Functions =============
  const loadAdvancedModule = async (moduleName: string) => {
    setIsLoading(true);
    const params: any = {};
    if (selectedState) params.state = selectedState;
    if (selectedDistrict) params.district = selectedDistrict;
    
    try {
      switch (moduleName) {
        case 'ASI':
          const asiRes = await axios.get<ASIResponse>(`${backendURL}/api/advanced/asi`, { params });
          setAsiData(asiRes.data);
          showAlert(`ASI Analysis: ${asiRes.data.count} aquifer polygons analyzed`, "success");
          break;
          
        case 'NETWORK_DENSITY':
          const netRes = await axios.get<NetworkDensityResponse>(`${backendURL}/api/advanced/network-density`, { params });
          setNetworkDensityData(netRes.data);
          showAlert(`Network Density: Map1=${netRes.data.map1_site_level.count} sites, Map2=${netRes.data.map2_gridded.count} grid cells`, "success");
          break;
          
        case 'SASS':
          if (!selectedMonth) {
            showAlert("Please select a month for SASS analysis", "warning");
            setIsLoading(false);
            return;
          }
          params.year = selectedYear;
          params.month = selectedMonth;
          const sassRes = await axios.get<SASSResponse>(`${backendURL}/api/advanced/sass`, { params });
          setSassData(sassRes.data);
          showAlert(`SASS: ${sassRes.data.count} sites with stress scores`, "success");
          break;
          
        case 'GRACE_DIVERGENCE':
          if (!selectedMonth) {
            showAlert("Please select a month for divergence analysis", "warning");
            setIsLoading(false);
            return;
          }
          params.year = selectedYear;
          params.month = selectedMonth;
          const divRes = await axios.get<DivergenceResponse>(`${backendURL}/api/advanced/grace-divergence`, { params });
          setDivergenceData(divRes.data);
          showAlert(`Divergence: ${divRes.data.count} pixels analyzed`, "success");
          break;
          
        case 'FORECAST':
          const foreRes = await axios.get<ForecastResponse>(`${backendURL}/api/advanced/forecast`, { params });
          setForecastData(foreRes.data);
          showAlert(`Forecast: ${foreRes.data.count} grid cells predicted (GRACE contribution: ${foreRes.data.statistics.mean_grace_contribution.toFixed(3)} m)`, "success");
          break;
          

      case 'RECHARGE':
        // Recharge planning requires year and month for site recommendations
        if (!selectedMonth) {
          showAlert("Please select a month for site-specific recharge recommendations", "warning");
          // Still load without month for regional analysis
          params.year = selectedYear;
        } else {
          params.year = selectedYear;
          params.month = selectedMonth;
        }
        const rechRes = await axios.get<RechargeResponse>(`${backendURL}/api/advanced/recharge-planning`, { params });
        setRechargeData(rechRes.data);
        showAlert(`Recharge: ${rechRes.data.potential.total_recharge_potential_mcm.toFixed(2)} MCM potential`, "success");
        break;

          
        case 'SIGNIFICANT_TRENDS':
          const trendRes = await axios.get<SignificantTrendsResponse>(`${backendURL}/api/advanced/significant-trends`, { params });
          setSignificantTrendsData(trendRes.data);
          showAlert(`Significant Trends: ${trendRes.data.count} sites found`, "success");
          break;
          
        case 'CHANGEPOINTS':
          const cpRes = await axios.get<ChangepointResponse>(`${backendURL}/api/advanced/changepoints`, { params });
          setChangepointsData(cpRes.data);
          showAlert(`Changepoints: ${cpRes.data.changepoints.count} sites with breaks, ${cpRes.data.coverage.count} total sites analyzed`, "success");
          break;
          
        case 'LAG_CORRELATION':
          const lagRes = await axios.get<LagCorrelationResponse>(`${backendURL}/api/advanced/lag-correlation`, { params });
          setLagCorrelationData(lagRes.data);
          showAlert(`Lag Correlation: ${lagRes.data.count} sites analyzed`, "success");
          break;
          
        case 'HOTSPOTS':
          const hotRes = await axios.get<HotspotsResponse>(`${backendURL}/api/advanced/hotspots`, { params });
          setHotspotsData(hotRes.data);
          showAlert(`Hotspots: ${hotRes.data.statistics.n_clusters} clusters found`, "success");
          break;
          
        default:
          showAlert("Unknown module selected", "warning");
      }
    } catch (error: any) {
      console.error(`Error loading ${moduleName}:`, error);
      showAlert(error.response?.data?.detail || `Error loading ${moduleName}`, "error");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (selectedAdvancedModule) {
      loadAdvancedModule(selectedAdvancedModule);
    }
  }, [selectedAdvancedModule, selectedState, selectedDistrict, selectedYear, selectedMonth]);

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
        showAlert(`Loaded unified timeseries: ${res.data.count} points (${res.data.view} view)`, "success");
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

  // ============= NEW: Render Advanced Module Visualizations =============
  const renderAdvancedModuleContent = () => {
    if (!selectedAdvancedModule) return null;

    // ASI Visualization
    if (selectedAdvancedModule === 'ASI' && asiData) {
      return (
        <div className="lg:col-span-2 xl:col-span-3 bg-white rounded-xl shadow-2xl p-6 border-2 border-purple-400">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <span className="text-2xl">🔷</span>
              <h3 className="text-lg font-bold text-gray-800">Aquifer Suitability Index (ASI)</h3>
            </div>
            <span className="text-sm bg-purple-100 text-purple-800 px-3 py-1 rounded-full font-semibold">
              {asiData.count} aquifer polygons
            </span>
          </div>

          {/* Statistics Cards */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-4">
            <div className="bg-purple-50 p-3 rounded-lg border border-purple-200">
              <div className="text-xs text-gray-600 mb-1">Mean ASI</div>
              <div className="text-2xl font-bold text-purple-600">{asiData.statistics.mean_asi.toFixed(2)}</div>
            </div>
            <div className="bg-green-50 p-3 rounded-lg border border-green-200">
              <div className="text-xs text-gray-600 mb-1">Median</div>
              <div className="text-2xl font-bold text-green-600">{asiData.statistics.median_asi.toFixed(2)}</div>
            </div>
            <div className="bg-blue-50 p-3 rounded-lg border border-blue-200">
              <div className="text-xs text-gray-600 mb-1">Range</div>
              <div className="text-sm font-bold text-blue-600">
                {asiData.statistics.min_asi.toFixed(1)}–{asiData.statistics.max_asi.toFixed(1)}
              </div>
            </div>
            <div className="bg-orange-50 p-3 rounded-lg border border-orange-200">
              <div className="text-xs text-gray-600 mb-1">Avg Sy</div>
              <div className="text-xl font-bold text-orange-600">{asiData.statistics.avg_specific_yield.toFixed(4)}</div>
            </div>
            <div className="bg-teal-50 p-3 rounded-lg border border-teal-200">
              <div className="text-xs text-gray-600 mb-1">Total Area</div>
              <div className="text-lg font-bold text-teal-600">{asiData.statistics.total_area_km2.toFixed(0)} km²</div>
            </div>
          </div>

          {/* Dominant Aquifer Info */}
          <div className="bg-gradient-to-r from-purple-50 to-blue-50 p-3 rounded-lg border border-purple-200 mb-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xs text-gray-600">Dominant Aquifer Type</div>
                <div className="font-bold text-lg text-purple-700">{asiData.statistics.dominant_aquifer}</div>
              </div>
              <div className="text-right">
                <div className="text-xs text-gray-600">Normalization Range</div>
                <div className="text-sm font-semibold text-gray-700">
                  Sy: {asiData.methodology.quantile_stretch.low.toFixed(4)} – {asiData.methodology.quantile_stretch.high.toFixed(4)}
                </div>
              </div>
            </div>
          </div>

          {/* Map with Choropleth Polygons */}
          <div className="h-[500px] relative rounded-lg overflow-hidden">
            <MapContainer 
              center={center} 
              zoom={zoom} 
              style={{ height: "100%", width: "100%" }} 
              key={`asi-${mapKey}`}
            >
              <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
              
              {/* District boundary */}
              {districtGeo && (
                <GeoJSON 
                  data={districtGeo} 
                  style={{ color: "#9333EA", weight: 3, fillOpacity: 0.05 }} 
                />
              )}
              
              {/* ASI Polygons (Choropleth) */}
              {asiData.geojson.features.map((feature, i) => {
                const asiScore = feature.properties.asi_score;
                const fillColor = getASIColor(asiScore);
                
                return (
                  <GeoJSON
                    key={`asi_polygon_${feature.id}_${i}`}
                    data={feature}
                    style={{
                      fillColor: fillColor,
                      fillOpacity: 0.7,
                      color: '#666',
                      weight: 1,
                      opacity: 0.8
                    }}
                    onEachFeature={(feature: any, layer: any) => {
                      const props = feature.properties;
                      const popupContent = `
                        <div style="font-family: sans-serif; min-width: 220px;">
                          <strong style="font-size: 15px; color: ${fillColor};">
                            ASI Score: ${props.asi_score.toFixed(2)}/5
                          </strong><br/>
                          <hr style="margin: 5px 0; border: 1px solid #ddd;"/>
                          <table style="width: 100%; font-size: 12px; margin-top: 5px;">
                            <tbody>
                              <tr><td><strong>Aquifer:</strong></td><td>${props.majoraquif || 'N/A'}</td></tr>
                              <tr><td><strong>Specific Yield:</strong></td><td>${props.specific_yield.toFixed(4)}</td></tr>
                              <tr><td><strong>Area:</strong></td><td>${(props.area_m2 / 1_000_000).toFixed(2)} km²</td></tr>
                              <tr>
                                <td><strong>Quality:</strong></td>
                                <td style="font-weight: bold; color: ${fillColor};">
                                  ${asiScore >= 4 ? 'Excellent' : 
                                    asiScore >= 3 ? 'Good' : 
                                    asiScore >= 2 ? 'Moderate' : 
                                    asiScore >= 1 ? 'Fair' : 'Poor'}
                                </td>
                              </tr>
                            </tbody>
                          </table>
                        </div>
                      `;
                      
                      layer.bindPopup(popupContent);
                      
                      // Highlight on hover
                      layer.on('mouseover', function() {
                        this.setStyle({
                          weight: 3,
                          color: '#333',
                          fillOpacity: 0.9
                        });
                      });
                      
                      layer.on('mouseout', function() {
                        this.setStyle({
                          weight: 1,
                          color: '#666',
                          fillOpacity: 0.7
                        });
                      });
                    }}
                  />
                );
              })}
            </MapContainer>

            {/* Legend */}
            <div className="absolute bottom-4 right-4 z-[1000] bg-white/95 backdrop-blur-sm p-3 rounded-lg shadow-lg border-2 border-purple-300">
              <div className="text-xs font-bold mb-2 text-purple-900">ASI Score (Storage Potential)</div>
              <div className="space-y-1">
                {[
                  { label: 'Excellent (4-5)', score: 4.5, desc: 'Highest potential' },
                  { label: 'Good (3-4)', score: 3.5, desc: 'High potential' },
                  { label: 'Moderate (2-3)', score: 2.5, desc: 'Medium potential' },
                  { label: 'Fair (1-2)', score: 1.5, desc: 'Low potential' },
                  { label: 'Poor (0-1)', score: 0.5, desc: 'Very low potential' }
                ].map((item, idx) => (
                  <div key={idx} className="flex items-center gap-2">
                    <div 
                      className="w-5 h-5 rounded border border-gray-400" 
                      style={{ backgroundColor: getASIColor(item.score) }}
                    ></div>
                    <div className="flex flex-col">
                      <span className="text-xs font-semibold">{item.label}</span>
                      <span className="text-[10px] text-gray-500">{item.desc}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Methodology and Interpretation */}
          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <h4 className="font-bold text-sm mb-2 text-blue-900">📐 Methodology</h4>
              <p className="text-xs text-blue-800">
                <strong>Approach:</strong> {asiData.methodology.approach}
                <br/><strong>Normalization:</strong> Quantile-based (5th-95th percentile) stretching to 0-5 scale
                <br/><strong>Data Source:</strong> Specific yield from lithology or field measurements
              </p>
            </div>
            
            <div className="p-3 bg-green-50 border border-green-200 rounded-lg">
              <h4 className="font-bold text-sm mb-2 text-green-900">💡 Interpretation</h4>
              <p className="text-xs text-green-800">
                {asiData.methodology.interpretation}
                <br/><br/><strong>⚠️ Note:</strong> ASI is a screening-level indicator. Combine with stress maps and field data for final decisions.
              </p>
            </div>
          </div>

          {/* Quick Stats Summary */}
          <div className="mt-4 p-4 bg-gradient-to-r from-purple-50 to-pink-50 border border-purple-200 rounded-lg">
            <h4 className="font-bold text-sm mb-2 text-purple-900">📊 Statistical Summary</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
              <div>
                <div className="text-gray-600">Polygons Analyzed</div>
                <div className="font-bold text-lg text-purple-700">{asiData.count}</div>
              </div>
              <div>
                <div className="text-gray-600">Mean ± Std</div>
                <div className="font-bold text-lg text-purple-700">
                  {asiData.statistics.mean_asi.toFixed(2)} ± {asiData.statistics.std_asi.toFixed(2)}
                </div>
              </div>
              <div>
                <div className="text-gray-600">High Quality (&gt;3)</div>
                <div className="font-bold text-lg text-green-700">
                  {asiData.geojson.features.filter(f => f.properties.asi_score > 3).length}
                </div>
              </div>
              <div>
                <div className="text-gray-600">Low Quality (&lt;2)</div>
                <div className="font-bold text-lg text-red-700">
                  {asiData.geojson.features.filter(f => f.properties.asi_score < 2).length}
                </div>
              </div>
            </div>
          </div>
        </div>
      );
    }
    // Network Density Visualization
// Network Density Visualization
    if (selectedAdvancedModule === 'NETWORK_DENSITY' && networkDensityData) {
      return (
        <div className="lg:col-span-2 xl:col-span-3 bg-white rounded-xl shadow-2xl p-6 border-2 border-indigo-400">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <span className="text-2xl">📊</span>
              <h3 className="text-lg font-bold text-gray-800">Well Network Density Analysis - Dual Map View</h3>
            </div>
            <span className="text-sm bg-indigo-100 text-indigo-800 px-3 py-1 rounded-full font-semibold">
              Map1: {networkDensityData.map1_site_level?.count || 0} sites | Map2: {networkDensityData.map2_gridded?.count || 0} grid cells
            </span>
          </div>

          {/* Statistics Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
            <div className="bg-indigo-50 p-3 rounded-lg border border-indigo-200">
              <div className="text-xs text-gray-600 mb-1">Total Sites</div>
              <div className="text-2xl font-bold text-indigo-600">
                {networkDensityData.statistics?.total_sites ?? 0}
              </div>
            </div>
            <div className="bg-purple-50 p-3 rounded-lg border border-purple-200">
              <div className="text-xs text-gray-600 mb-1">Avg Strength</div>
              <div className="text-2xl font-bold text-purple-600">
                {networkDensityData.statistics?.avg_strength?.toFixed(3) ?? 'N/A'}
              </div>
            </div>
            <div className="bg-blue-50 p-3 rounded-lg border border-blue-200">
              <div className="text-xs text-gray-600 mb-1">Avg Local Density</div>
              <div className="text-xl font-bold text-blue-600">
                {networkDensityData.statistics?.avg_local_density?.toFixed(4) ?? 'N/A'} /km²
              </div>
            </div>
            <div className="bg-green-50 p-3 rounded-lg border border-green-200">
              <div className="text-xs text-gray-600 mb-1">Median Obs</div>
              <div className="text-2xl font-bold text-green-600">
                {networkDensityData.statistics?.median_observations ?? 0}
              </div>
            </div>
          </div>

          {/* Dual Map Layout */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
            
            {/* MAP 1: Site-Level Strength */}
            <div className="bg-gradient-to-br from-blue-50 to-indigo-50 p-4 rounded-lg border-2 border-indigo-300">
              <h4 className="font-bold text-sm mb-2 text-indigo-900">
                Map 1: Site-Level Strength (Symbol Size = Local Density)
              </h4>
              <div className="h-[400px] relative rounded-lg overflow-hidden">
                <MapContainer 
                  center={center} 
                  zoom={zoom} 
                  style={{ height: "100%", width: "100%" }} 
                  key={`network-map1-${mapKey}`}
                >
                  <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                  
                  {districtGeo && (
                    <GeoJSON 
                      data={districtGeo} 
                      style={{ color: "#6366F1", weight: 3, fillOpacity: 0.1 }} 
                    />
                  )}
                  
                  {networkDensityData.map1_site_level?.data?.map((point, i) => {
                    const size = 4 + (point.local_density_per_km2 * 200);
                    return (
                      <CircleMarker
                        key={`net1_${i}`}
                        center={[point.latitude, point.longitude]}
                        radius={Math.min(size, 20)}
                        fillColor="#6366F1"
                        color="white"
                        weight={1}
                        fillOpacity={0.7}
                      >
                        <Popup>
                          <div style={{ fontFamily: 'sans-serif', minWidth: '220px' }}>
                            <strong style={{ fontSize: '14px', color: '#6366F1' }}>
                              Site: {point.site_id}
                            </strong><br/>
                            <hr style={{ margin: '5px 0' }}/>
                            <table style={{ width: '100%', fontSize: '12px' }}>
                              <tbody>
                                <tr><td><strong>Strength:</strong></td><td>{point.strength.toFixed(3)}</td></tr>
                                <tr><td><strong>Local Density:</strong></td><td>{point.local_density_per_km2.toFixed(4)} /km²</td></tr>
                                <tr><td><strong>Neighbors:</strong></td><td>{point.neighbors_within_radius} within {networkDensityData.parameters?.radius_km}km</td></tr>
                                <tr><td><strong>Observations:</strong></td><td>{point.n_observations}</td></tr>
                                <tr><td><strong>Trend:</strong></td><td>{point.slope_m_per_year.toFixed(4)} m/yr</td></tr>
                                <tr><td><strong>GWL Std:</strong></td><td>{point.gwl_std.toFixed(2)} m</td></tr>
                              </tbody>
                            </table>
                          </div>
                        </Popup>
                      </CircleMarker>
                    );
                  })}
                </MapContainer>

                {/* Map 1 Legend */}
                <div className="absolute bottom-4 right-4 z-[1000] bg-white/95 backdrop-blur-sm p-3 rounded-lg shadow-lg border-2 border-indigo-300">
                  <div className="text-xs font-bold mb-2 text-indigo-900">Symbol Size = Density</div>
                  <p className="text-xs text-gray-600">
                    Larger circles = higher local well density within {networkDensityData.parameters?.radius_km}km radius
                  </p>
                </div>
              </div>
            </div>

            {/* MAP 2: Gridded Density Heatmap */}
            <div className="bg-gradient-to-br from-orange-50 to-red-50 p-4 rounded-lg border-2 border-orange-300">
              <h4 className="font-bold text-sm mb-2 text-orange-900">
                Map 2: Gridded Density Heatmap (sites per 1000 km²)
              </h4>
              <div className="h-[400px] relative rounded-lg overflow-hidden">
                <MapContainer 
                  center={center} 
                  zoom={zoom} 
                  style={{ height: "100%", width: "100%" }} 
                  key={`network-map2-${mapKey}`}
                >
                  <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                  
                  {districtGeo && (
                    <GeoJSON 
                      data={districtGeo} 
                      style={{ color: "#F97316", weight: 3, fillOpacity: 0.1 }} 
                    />
                  )}
                  
                  {networkDensityData.map2_gridded?.data?.map((point, i) => (
                    <CircleMarker
                      key={`net2_${i}`}
                      center={[point.y, point.x]}
                      radius={8}
                      fillColor={getDensityColor(point.density_per_1000km2)}
                      color="white"
                      weight={1}
                      fillOpacity={0.8}
                    >
                      <Popup>
                        <div style={{ fontFamily: 'sans-serif', minWidth: '180px' }}>
                          <strong style={{ fontSize: '14px' }}>Grid Cell Density</strong><br/>
                          <hr style={{ margin: '5px 0' }}/>
                          <div style={{ fontSize: '12px' }}>
                            <div><strong>Density:</strong> {point.density_per_1000km2.toFixed(2)} sites/1000km²</div>
                            <div><strong>Location:</strong> {point.y.toFixed(4)}°N, {point.x.toFixed(4)}°E</div>
                          </div>
                        </div>
                      </Popup>
                    </CircleMarker>
                  ))}
                </MapContainer>

                {/* Map 2 Legend */}
                <div className="absolute bottom-4 right-4 z-[1000] bg-white/95 backdrop-blur-sm p-3 rounded-lg shadow-lg border-2 border-orange-300">
                  <div className="text-xs font-bold mb-2 text-orange-900">Density Scale (sites/1000km²)</div>
                  <div className="space-y-1">
                    {[
                      { label: 'Very Dense (>40)', density: 45 },
                      { label: 'Dense (20-40)', density: 30 },
                      { label: 'Moderate (10-20)', density: 15 },
                      { label: 'Sparse (5-10)', density: 7 },
                      { label: 'Very Sparse (<5)', density: 3 }
                    ].map((item, idx) => (
                      <div key={idx} className="flex items-center gap-2">
                        <div 
                          className="w-4 h-4 rounded border border-gray-400" 
                          style={{ backgroundColor: getDensityColor(item.density) }}
                        ></div>
                        <span className="text-xs">{item.label}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Methodology Explanation */}
          <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-sm text-blue-800">
              <strong>ℹ️ What:</strong> Dual-map network analysis showing (1) site-level signal strength with local density and (2) absolute density grid
              <br/><strong>How:</strong> Map 1: Annualized GWL slope normalized by site variability (strength = |slope|/σ), symbol size = local density within {networkDensityData.parameters?.radius_km}km. Map 2: Absolute density per 1000km² on regular grid clipped to AOI.
              <br/><strong>Significance:</strong> Find robust signal corridors (strong + dense in Map 1) and identify sparse coverage zones (low values in Map 2).
            </p>
          </div>
        </div>
      );
    }
    // SASS Visualization
    if (selectedAdvancedModule === 'SASS' && sassData) {
      return (
        <div className="lg:col-span-2 xl:col-span-3 bg-white rounded-xl shadow-2xl p-6 border-2 border-red-400">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <span className="text-2xl">⚠️</span>
              <h3 className="text-lg font-bold text-gray-800">Spatio-Temporal Aquifer Stress Score (SASS)</h3>
            </div>
            <span className="text-sm bg-red-100 text-red-800 px-3 py-1 rounded-full font-semibold">
              {sassData.statistics.stressed_sites} stressed sites
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
            <div className="bg-red-50 p-3 rounded-lg border border-red-200">
              <div className="text-xs text-gray-600 mb-1">Mean SASS</div>
              <div className="text-2xl font-bold text-red-600">{sassData.statistics.mean_sass.toFixed(3)}</div>
            </div>
            <div className="bg-orange-50 p-3 rounded-lg border border-orange-200">
              <div className="text-xs text-gray-600 mb-1">Max Stress</div>
              <div className="text-2xl font-bold text-orange-600">{sassData.statistics.max_sass.toFixed(3)}</div>
            </div>
            <div className="bg-green-50 p-3 rounded-lg border border-green-200">
              <div className="text-xs text-gray-600 mb-1">Min Stress</div>
              <div className="text-2xl font-bold text-green-600">{sassData.statistics.min_sass.toFixed(3)}</div>
            </div>
            <div className="bg-yellow-50 p-3 rounded-lg border border-yellow-200">
              <div className="text-xs text-gray-600 mb-1">Stressed Sites</div>
              <div className="text-2xl font-bold text-yellow-600">{sassData.statistics.stressed_sites}</div>
            </div>
          </div>

          <div className="h-[500px] relative rounded-lg overflow-hidden">
            <MapContainer center={center} zoom={zoom} style={{ height: "100%", width: "100%" }} key={`sass-${mapKey}`}>
              <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
              {districtGeo && <GeoJSON data={districtGeo} style={{ color: "#DC2626", weight: 3, fillOpacity: 0.1 }} />}
              
              {sassData.data.map((point, i) => (
                <CircleMarker
                  key={`sass_${i}`}
                  center={[point.latitude, point.longitude]}
                  radius={8}
                  fillColor={getSASSColor(point.sass_score)}
                  color="white"
                  weight={2}
                  fillOpacity={0.8}
                >
                  <Popup>
                    <div style={{ fontFamily: 'sans-serif', minWidth: '200px' }}>
                      <strong style={{ fontSize: '14px', color: getSASSColor(point.sass_score) }}>SASS: {point.sass_score.toFixed(3)}</strong><br/>
                      <hr style={{ margin: '5px 0' }}/>
                      <table style={{ width: '100%', fontSize: '12px' }}>
                        <tbody>
                          <tr><td><strong>GWL Stress:</strong></td><td>{point.gwl_stress.toFixed(3)}</td></tr>
                          <tr><td><strong>GRACE z:</strong></td><td>{point.grace_z.toFixed(3)}</td></tr>
                          <tr><td><strong>Rain z:</strong></td><td>{point.rain_z.toFixed(3)}</td></tr>
                          <tr><td><strong>GWL:</strong></td><td>{point.gwl.toFixed(2)} m</td></tr>
                        </tbody>
                      </table>
                    </div>
                  </Popup>
                </CircleMarker>
              ))}
            </MapContainer>

            <div className="absolute bottom-4 right-4 z-[1000] bg-white/95 backdrop-blur-sm p-3 rounded-lg shadow-lg border-2 border-red-300">
              <div className="text-xs font-bold mb-2">SASS Score</div>
              <div className="space-y-1">
                {[
                  { label: 'Critical (>2)', score: 2.5 },
                  { label: 'High (1-2)', score: 1.5 },
                  { label: 'Moderate (0-1)', score: 0.5 },
                  { label: 'Low (-1-0)', score: -0.5 },
                  { label: 'Minimal (<-1)', score: -1.5 }
                ].map((item, idx) => (
                  <div key={idx} className="flex items-center gap-2">
                    <div className="w-4 h-4 rounded" style={{ backgroundColor: getSASSColor(item.score) }}></div>
                    <span className="text-xs">{item.label}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-sm text-blue-800">
              <strong>Formula:</strong> {sassData.formula}
              <br/><strong>Significance:</strong> Composite stress index using wells + GRACE + rainfall. Higher values indicate more stress.
            </p>
          </div>
        </div>
      );
    }

    // Forecast Visualization (with Plotly chart)
    if (selectedAdvancedModule === 'FORECAST' && forecastData) {
      return (
        <div className="lg:col-span-2 xl:col-span-3 bg-white rounded-xl shadow-2xl p-6 border-2 border-teal-400">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <span className="text-2xl">📈</span>
              <h3 className="text-lg font-bold text-gray-800">GWL Forecasting (Grid-based)</h3>
            </div>
            <span className="text-sm bg-teal-100 text-teal-800 px-3 py-1 rounded-full font-semibold">
              {forecastData.count} grid cells
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
            <div className="bg-teal-50 p-3 rounded-lg border border-teal-200">
              <div className="text-xs text-gray-600 mb-1">Mean Change (12mo)</div>
              <div className={`text-2xl font-bold ${forecastData.statistics.mean_change_m > 0 ? 'text-red-600' : 'text-green-600'}`}>
                {forecastData.statistics.mean_change_m > 0 ? '+' : ''}{forecastData.statistics.mean_change_m.toFixed(3)} m
              </div>
            </div>
            <div className="bg-red-50 p-3 rounded-lg border border-red-200">
              <div className="text-xs text-gray-600 mb-1">Declining Cells</div>
              <div className="text-2xl font-bold text-red-600">{forecastData.statistics.declining_cells}</div>
            </div>
            <div className="bg-green-50 p-3 rounded-lg border border-green-200">
              <div className="text-xs text-gray-600 mb-1">Recovering Cells</div>
              <div className="text-2xl font-bold text-green-600">{forecastData.statistics.recovering_cells}</div>
            </div>
            <div className="bg-blue-50 p-3 rounded-lg border border-blue-200">
              <div className="text-xs text-gray-600 mb-1">Mean R²</div>
              <div className="text-2xl font-bold text-blue-600">{forecastData.statistics.mean_r_squared.toFixed(3)}</div>
            </div>
          </div>

          <div className="bg-indigo-50 rounded-lg p-3 border border-indigo-200 mb-4">
            <h4 className="font-bold text-sm mb-2 text-indigo-900">Forecast Parameters</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
              <div><strong>Horizon:</strong> {forecastData.parameters.forecast_months} months</div>
              <div><strong>Neighbors (k):</strong> {forecastData.parameters.k_neighbors}</div>
              <div><strong>Grid Resolution:</strong> {forecastData.parameters.grid_resolution}×{forecastData.parameters.grid_resolution}</div>
              <div><strong>GRACE Used:</strong> {forecastData.parameters.grace_used ? '✅ Yes' : '❌ No'}</div>
              <div><strong>Method:</strong> {forecastData.method}</div>
              <div><strong>Success Rate:</strong> {forecastData.statistics.success_rate.toFixed(1)}%</div>
            </div>
          </div>

          {/* Map Visualization */}
          <div className="h-[500px] relative rounded-lg overflow-hidden mb-4">
            <MapContainer center={center} zoom={zoom} style={{ height: "100%", width: "100%" }} key={`forecast-${mapKey}`}>
              <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
              {districtGeo && <GeoJSON data={districtGeo} style={{ color: "#14B8A6", weight: 3, fillOpacity: 0.1 }} />}
              
              {forecastData.data.map((point, i) => {
                // Color: Red = deeper (decline), Green = shallower (recovery)
                const getColor = (change: number) => {
                  if (change > 2) return '#7F1D1D';      // Very deep decline
                  if (change > 1) return '#DC2626';      // Strong decline
                  if (change > 0.5) return '#F87171';    // Moderate decline
                  if (change > 0) return '#FCA5A5';      // Slight decline
                  if (change > -0.5) return '#BBF7D0';   // Slight recovery
                  if (change > -1) return '#86EFAC';     // Moderate recovery
                  if (change > -2) return '#22C55E';     // Strong recovery
                  return '#15803D';                       // Very strong recovery
                };

                return (
                  <CircleMarker
                    key={`forecast_${i}`}
                    center={[point.latitude, point.longitude]}
                    radius={5}
                    fillColor={getColor(point.pred_delta_m)}
                    color="white"
                    weight={1}
                    fillOpacity={0.8}
                  >
                    <Popup>
                      <div style={{ fontFamily: 'sans-serif', minWidth: '220px' }}>
                        <strong style={{ fontSize: '14px', color: getColor(point.pred_delta_m) }}>
                          Forecast: {point.pred_delta_m > 0 ? '+' : ''}{point.pred_delta_m.toFixed(3)} m
                        </strong><br/>
                        <hr style={{ margin: '5px 0' }}/>
                        <table style={{ width: '100%', fontSize: '12px' }}>
                          <tbody>
                            <tr><td><strong>Current GWL:</strong></td><td>{point.current_gwl.toFixed(2)} m bgl</td></tr>
                            <tr><td><strong>Forecast GWL:</strong></td><td>{point.forecast_gwl.toFixed(2)} m bgl</td></tr>
                            <tr><td><strong>Trend Component:</strong></td><td>{point.trend_component.toFixed(3)} m</td></tr>
                            <tr><td><strong>GRACE Component:</strong></td><td>{point.grace_component.toFixed(3)} m</td></tr>
                            <tr><td><strong>Model R²:</strong></td><td>{point.r_squared.toFixed(3)}</td></tr>
                            <tr><td><strong>Training Data:</strong></td><td>{point.n_months_training} months</td></tr>
                          </tbody>
                        </table>
                      </div>
                    </Popup>
                  </CircleMarker>
                );
              })}
            </MapContainer>

            <div className="absolute bottom-4 right-4 z-[1000] bg-white/95 backdrop-blur-sm p-3 rounded-lg shadow-lg border-2 border-teal-300">
              <div className="text-xs font-bold mb-2">12-Month Change (m)</div>
              <div className="space-y-1">
                {[
                  { label: 'Strong Decline (>2m)', change: 2.5 },
                  { label: 'Moderate Decline (1-2m)', change: 1.5 },
                  { label: 'Slight Decline (0-1m)', change: 0.5 },
                  { label: 'Slight Recovery (0 to -1m)', change: -0.5 },
                  { label: 'Moderate Recovery (-1 to -2m)', change: -1.5 },
                  { label: 'Strong Recovery (<-2m)', change: -2.5 }
                ].map((item, idx) => {
                  const getColor = (change: number) => {
                    if (change > 2) return '#7F1D1D';
                    if (change > 1) return '#DC2626';
                    if (change > 0.5) return '#F87171';
                    if (change > 0) return '#FCA5A5';
                    if (change > -0.5) return '#BBF7D0';
                    if (change > -1) return '#86EFAC';
                    if (change > -2) return '#22C55E';
                    return '#15803D';
                  };
                  return (
                    <div key={idx} className="flex items-center gap-2">
                      <div className="w-4 h-4 rounded" style={{ backgroundColor: getColor(item.change) }}></div>
                      <span className="text-xs">{item.label}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Summary Statistics Table */}
          <div className="bg-gradient-to-r from-teal-50 to-green-50 p-4 rounded-lg border border-teal-200">
            <h4 className="font-bold text-sm mb-3 text-teal-900">Grid Statistics</h4>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              <div className="bg-white p-2 rounded border border-gray-200">
                <div className="text-xs text-gray-600">Total Cells</div>
                <div className="text-lg font-bold text-gray-800">{forecastData.count}</div>
              </div>
              <div className="bg-white p-2 rounded border border-gray-200">
                <div className="text-xs text-gray-600">Median Change</div>
                <div className={`text-lg font-bold ${forecastData.statistics.median_change_m > 0 ? 'text-red-600' : 'text-green-600'}`}>
                  {forecastData.statistics.median_change_m > 0 ? '+' : ''}{forecastData.statistics.median_change_m.toFixed(3)} m
                </div>
              </div>
              <div className="bg-white p-2 rounded border border-gray-200">
                <div className="text-xs text-gray-600">Decline/Recovery Ratio</div>
                <div className="text-lg font-bold text-gray-800">
                  {(forecastData.statistics.declining_cells / Math.max(forecastData.statistics.recovering_cells, 1)).toFixed(2)}
                </div>
              </div>
            </div>
          </div>

          <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-sm text-blue-800">
              <strong>ℹ️ What:</strong> Grid-based 12-month GWL forecast using neighbor-weighted wells + GRACE anomaly.
              <br/><strong>How:</strong> KNN distance-weighted composite per cell → deseasonalize GWL + GRACE → OLS (trend + GRACE) → add back seasonality.
              <br/><strong>Significance:</strong> Red cells = declining (deeper), Green = recovering (shallower). Assumes linear trend + stationary seasonality.
              <br/><strong>⚠️ Note:</strong> This is a SPATIAL forecast (grid of points), not a TIME forecast (single location over time).
            </p>
          </div>
        </div>
      );
    }

if (selectedAdvancedModule === 'RECHARGE' && rechargeData) {
  return (
    <div className="lg:col-span-2 xl:col-span-3 bg-white rounded-xl shadow-2xl p-6 border-2 border-cyan-400">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span className="text-2xl">💧</span>
          <h3 className="text-lg font-bold text-gray-800">Managed Aquifer Recharge (MAR) Planning</h3>
        </div>
        <span className="text-sm bg-cyan-100 text-cyan-800 px-3 py-1 rounded-full font-semibold">
          {rechargeData.potential.total_recharge_potential_mcm.toFixed(2)} MCM potential
        </span>
      </div>

      {/* Analysis Parameters */}
      <div className="bg-gradient-to-r from-cyan-50 to-blue-50 p-4 rounded-lg border border-cyan-200 mb-4">
        <h4 className="font-bold text-sm mb-3 text-cyan-900">Analysis Parameters</h4>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-xs">
          <div><strong>Area:</strong> {rechargeData.analysis_parameters.area_km2} km²</div>
          <div><strong>Dominant Lithology:</strong> {rechargeData.analysis_parameters.dominant_lithology}</div>
          <div><strong>Runoff Coeff:</strong> {rechargeData.analysis_parameters.runoff_coefficient}</div>
          <div><strong>Monsoon Rainfall:</strong> {rechargeData.analysis_parameters.monsoon_rainfall_m.toFixed(3)} m</div>
          <div><strong>Capture Fraction:</strong> {(rechargeData.analysis_parameters.capture_fraction * 100).toFixed(1)}%</div>
          <div><strong>Year Analyzed:</strong> {rechargeData.analysis_parameters.year_analyzed}</div>
        </div>
      </div>

      {/* Recharge Potential */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <div className="bg-cyan-50 p-4 rounded-lg border border-cyan-200">
          <div className="text-xs text-gray-600 mb-1">Total Recharge Potential</div>
          <div className="text-3xl font-bold text-cyan-600">{rechargeData.potential.total_recharge_potential_mcm.toFixed(2)} MCM</div>
          <div className="text-xs text-gray-500 mt-1">Million Cubic Meters</div>
        </div>
        <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
          <div className="text-xs text-gray-600 mb-1">Per km² Potential</div>
          <div className="text-3xl font-bold text-blue-600">{rechargeData.potential.per_km2_mcm.toFixed(4)} MCM/km²</div>
          <div className="text-xs text-gray-500 mt-1">Normalized by area</div>
        </div>
      </div>

      {/* Structure Plan */}
      <div className="bg-white border-2 border-cyan-200 rounded-lg p-4 mb-4">
        <h4 className="font-bold text-sm mb-3 text-cyan-900">📋 Recommended Structure Plan</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-cyan-50">
              <tr>
                <th className="text-left p-2 border-b-2 border-cyan-200">Structure Type</th>
                <th className="text-center p-2 border-b-2 border-cyan-200">Units</th>
                <th className="text-center p-2 border-b-2 border-cyan-200">Total Capacity (MCM)</th>
                <th className="text-center p-2 border-b-2 border-cyan-200">Allocation (%)</th>
              </tr>
            </thead>
            <tbody>
              {rechargeData.structure_plan.map((structure, idx) => (
                <tr key={idx} className={idx % 2 === 0 ? 'bg-gray-50' : 'bg-white'}>
                  <td className="p-2 border-b border-gray-200 font-medium">{structure.structure_type}</td>
                  <td className="p-2 border-b border-gray-200 text-center">{structure.recommended_units}</td>
                  <td className="p-2 border-b border-gray-200 text-center font-semibold text-cyan-600">
                    {structure.total_capacity_mcm.toFixed(3)}
                  </td>
                  <td className="p-2 border-b border-gray-200 text-center">
                    {(structure.allocation_fraction * 100).toFixed(0)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Site-Specific Recommendations Map */}
      {rechargeData.site_recommendations.length > 0 && (
        <>
          <div className="mb-4">
            <h4 className="font-bold text-sm mb-2 text-cyan-900">🎯 Site-Specific Recommendations ({rechargeData.count} sites)</h4>
            <p className="text-xs text-gray-600">Based on current groundwater stress levels</p>
          </div>

          <div className="h-[500px] relative rounded-lg overflow-hidden">
            <MapContainer center={center} zoom={zoom} style={{ height: "100%", width: "100%" }} key={`recharge-${mapKey}`}>
              <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
              {districtGeo && <GeoJSON data={districtGeo} style={{ color: "#06B6D4", weight: 3, fillOpacity: 0.1 }} />}
              
              {rechargeData.site_recommendations.map((site, i) => {
                const getStressColor = (category: string) => {
                  switch (category) {
                    case 'Critical': return '#DC2626';
                    case 'Stressed': return '#F59E0B';
                    case 'Moderate': return '#FCD34D';
                    case 'Healthy': return '#22C55E';
                    default: return '#9CA3AF';
                  }
                };

                return (
                  <CircleMarker
                    key={`recharge_${i}`}
                    center={[site.latitude, site.longitude]}
                    radius={8}
                    fillColor={getStressColor(site.stress_category)}
                    color="white"
                    weight={2}
                    fillOpacity={0.8}
                  >
                    <Popup>
                      <div style={{ fontFamily: 'sans-serif', minWidth: '220px' }}>
                        <strong style={{ fontSize: '14px', color: getStressColor(site.stress_category) }}>
                          {site.stress_category} Site
                        </strong><br/>
                        <hr style={{ margin: '5px 0' }}/>
                        <table style={{ width: '100%', fontSize: '12px' }}>
                          <tbody>
                            <tr><td><strong>Site ID:</strong></td><td>{site.site_id}</td></tr>
                            <tr><td><strong>Current GWL:</strong></td><td>{site.current_gwl.toFixed(2)} m bgl</td></tr>
                            <tr><td><strong>Stress Level:</strong></td><td>{site.stress_category}</td></tr>
                            <tr>
                              <td><strong>Recommended:</strong></td>
                              <td className="font-semibold text-cyan-600">{site.recommended_structure}</td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    </Popup>
                  </CircleMarker>
                );
              })}
            </MapContainer>

            {/* Legend */}
            <div className="absolute bottom-4 right-4 z-[1000] bg-white/95 backdrop-blur-sm p-3 rounded-lg shadow-lg border-2 border-cyan-300">
              <div className="text-xs font-bold mb-2">Stress Category</div>
              <div className="space-y-1">
                {[
                  { label: 'Critical', category: 'Critical', structure: 'Recharge shaft' },
                  { label: 'Stressed', category: 'Stressed', structure: 'Check dam' },
                  { label: 'Moderate', category: 'Moderate', structure: 'Farm pond' },
                  { label: 'Healthy', category: 'Healthy', structure: 'Percolation tank' }
                ].map((item, idx) => {
                  const getStressColor = (category: string) => {
                    switch (category) {
                      case 'Critical': return '#DC2626';
                      case 'Stressed': return '#F59E0B';
                      case 'Moderate': return '#FCD34D';
                      case 'Healthy': return '#22C55E';
                      default: return '#9CA3AF';
                    }
                  };

                  return (
                    <div key={idx} className="flex flex-col gap-1">
                      <div className="flex items-center gap-2">
                        <div className="w-4 h-4 rounded" style={{ backgroundColor: getStressColor(item.category) }}></div>
                        <span className="text-xs font-semibold">{item.label}</span>
                      </div>
                      <div className="text-xs text-gray-600 ml-6">→ {item.structure}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </>
      )}

      {/* No site recommendations message */}
      {rechargeData.site_recommendations.length === 0 && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-4">
          <p className="text-sm text-yellow-800">
            ℹ️ No site-specific recommendations available. 
            {!rechargeData.filters.month && " Select a month to enable site-level stress analysis and recommendations."}
          </p>
        </div>
      )}

      {/* Formula and Explanation */}
      <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <h4 className="font-bold text-sm mb-2 text-blue-900">📐 Calculation Method</h4>
        <p className="text-sm text-blue-800 mb-2">
          <strong>Recharge Potential (MCM):</strong>
        </p>
        <div className="bg-white p-3 rounded border border-blue-300 font-mono text-xs mb-3">
          V = (Area_km² × 10⁶) × Rainfall_m × Runoff_Coeff × Capture_Fraction
        </div>
        <p className="text-xs text-blue-700 mb-3">
          <strong>Where:</strong><br/>
          • <strong>Area:</strong> Total aquifer area ({rechargeData.analysis_parameters.area_km2} km²)<br/>
          • <strong>Rainfall:</strong> Monsoon average ({rechargeData.analysis_parameters.monsoon_rainfall_m.toFixed(3)} m)<br/>
          • <strong>Runoff Coeff:</strong> Based on {rechargeData.analysis_parameters.dominant_lithology} ({rechargeData.analysis_parameters.runoff_coefficient})<br/>
          • <strong>Capture Fraction:</strong> Efficiency factor ({(rechargeData.analysis_parameters.capture_fraction * 100).toFixed(0)}%)
        </p>
        
        <p className="text-sm text-blue-800 mb-2">
          <strong>📊 Structure Allocation:</strong>
        </p>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-xs">
          {rechargeData.structure_plan.map((s, idx) => (
            <div key={idx} className="bg-white p-2 rounded border border-blue-200">
              <div className="font-semibold truncate">{s.structure_type}</div>
              <div className="text-gray-600">{(s.allocation_fraction * 100).toFixed(0)}% allocation</div>
            </div>
          ))}
        </div>

        <div className="mt-3 pt-3 border-t border-blue-300">
          <p className="text-sm text-blue-800">
            <strong>🎯 Site Recommendations:</strong> Based on negative z-score of current GWL<br/>
            • <strong>Critical:</strong> Very deep GWL → Recharge shaft (fast infiltration)<br/>
            • <strong>Stressed:</strong> Deep GWL → Check dam (moderate recharge)<br/>
            • <strong>Moderate:</strong> Medium GWL → Farm pond (localized storage)<br/>
            • <strong>Healthy:</strong> Shallow GWL → Percolation tank (preventive recharge)
          </p>
        </div>
      </div>
    </div>
  );
}

    // Significant Trends Visualization
    if (selectedAdvancedModule === 'SIGNIFICANT_TRENDS' && significantTrendsData) {
      return (
        <div className="lg:col-span-2 xl:col-span-3 bg-white rounded-xl shadow-2xl p-6 border-2 border-green-400">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <span className="text-2xl">📉</span>
              <h3 className="text-lg font-bold text-gray-800">Statistically Significant Trends</h3>
            </div>
            <span className="text-sm bg-green-100 text-green-800 px-3 py-1 rounded-full font-semibold">
              {significantTrendsData.count} sites (p &lt; {significantTrendsData.parameters.p_threshold})
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
            <div className="bg-red-50 p-3 rounded-lg border border-red-200">
              <div className="text-xs text-gray-600 mb-1">Declining</div>
              <div className="text-2xl font-bold text-red-
              600">{significantTrendsData.statistics.declining}</div>
            </div>
            <div className="bg-green-50 p-3 rounded-lg border border-green-200">
              <div className="text-xs text-gray-600 mb-1">Recovering</div>
              <div className="text-2xl font-bold text-green-600">{significantTrendsData.statistics.recovering}</div>
            </div>
            <div className="bg-blue-50 p-3 rounded-lg border border-blue-200">
              <div className="text-xs text-gray-600 mb-1">Mean Slope</div>
              <div className="text-xl font-bold text-blue-600">{significantTrendsData.statistics.mean_slope.toFixed(4)} m/yr</div>
            </div>
            <div className="bg-purple-50 p-3 rounded-lg border border-purple-200">
              <div className="text-xs text-gray-600 mb-1">High Significance</div>
              <div className="text-2xl font-bold text-purple-600">{significantTrendsData.statistics.high_significance}</div>
            </div>
          </div>

          <div className="h-[500px] relative rounded-lg overflow-hidden">
            <MapContainer center={center} zoom={zoom} style={{ height: "100%", width: "100%" }} key={`trends-${mapKey}`}>
              <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
              {districtGeo && <GeoJSON data={districtGeo} style={{ color: "#22C55E", weight: 3, fillOpacity: 0.1 }} />}
              
              {significantTrendsData.data.map((point, i) => (
                <CircleMarker
                  key={`trend_${i}`}
                  center={[point.latitude, point.longitude]}
                  radius={7}
                  fillColor={getTrendColor(point.slope_m_per_year)}
                  color="white"
                  weight={2}
                  fillOpacity={0.8}
                >
                  <Popup>
                    <div style={{ fontFamily: 'sans-serif', minWidth: '220px' }}>
                      <strong style={{ color: getTrendColor(point.slope_m_per_year) }}>
                        {point.trend_direction}: {Math.abs(point.slope_m_per_year).toFixed(4)} m/yr
                      </strong><br/>
                      <hr style={{ margin: '5px 0' }}/>
                      <table style={{ width: '100%', fontSize: '12px' }}>
                        <tbody>
                          <tr><td><strong>Site ID:</strong></td><td>{point.site_id}</td></tr>
                          <tr><td><strong>p-value:</strong></td><td>{point.p_value.toFixed(6)}</td></tr>
                          <tr><td><strong>Significance:</strong></td><td>{point.significance_level}</td></tr>
                          <tr><td><strong>Data Points:</strong></td><td>{point.n_months} months</td></tr>
                          <tr><td><strong>Period:</strong></td><td>{point.date_range}</td></tr>
                        </tbody>
                      </table>
                    </div>
                  </Popup>
                </CircleMarker>
              ))}
            </MapContainer>

            <div className="absolute bottom-4 right-4 z-[1000] bg-white/95 backdrop-blur-sm p-3 rounded-lg shadow-lg border-2 border-green-300">
              <div className="text-xs font-bold mb-2">Trend Direction</div>
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full" style={{ backgroundColor: '#DC2626' }}></div>
                  <span className="text-xs">Declining (p &lt; {significantTrendsData.parameters.p_threshold})</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full" style={{ backgroundColor: '#16A34A' }}></div>
                  <span className="text-xs">Recovering (p &lt; {significantTrendsData.parameters.p_threshold})</span>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-sm text-blue-800">
              <strong>ℹ️ What:</strong> Sites with statistically robust trends (Mann-Kendall or OLS with p &lt; {significantTrendsData.parameters.p_threshold}).
              <br/><strong>How:</strong> {significantTrendsData.parameters.method === 'mann_kendall' ? 'Mann-Kendall test' : 'OLS regression'} applied to annualized GWL slopes.
              <br/><strong>Significance:</strong> Focus interventions on sites with verified declining trends.
            </p>
          </div>
        </div>
      );
    }

    // Changepoints Visualization
    if (selectedAdvancedModule === 'CHANGEPOINTS' && changepointsData) {
      return (
        <div className="lg:col-span-2 xl:col-span-3 bg-white rounded-xl shadow-2xl p-6 border-2 border-yellow-400">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <span className="text-2xl">📍</span>
              <h3 className="text-lg font-bold text-gray-800">Changepoint Detection</h3>
            </div>
            <span className="text-sm bg-yellow-100 text-yellow-800 px-3 py-1 rounded-full font-semibold">
              {changepointsData.changepoints.count} sites with breakpoints
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
            <div className="bg-yellow-50 p-3 rounded-lg border border-yellow-200">
              <div className="text-xs text-gray-600 mb-1">Total Sites</div>
              <div className="text-2xl font-bold text-yellow-600">{changepointsData.statistics.total_sites_analyzed}</div>
            </div>
            <div className="bg-orange-50 p-3 rounded-lg border border-orange-200">
              <div className="text-xs text-gray-600 mb-1">With Changepoints</div>
              <div className="text-2xl font-bold text-orange-600">{changepointsData.statistics.sites_with_changepoints}</div>
            </div>
            <div className="bg-red-50 p-3 rounded-lg border border-red-200">
              <div className="text-xs text-gray-600 mb-1">Detection Rate</div>
              <div className="text-2xl font-bold text-red-600">{(changepointsData.statistics.detection_rate * 100).toFixed(1)}%</div>
            </div>
            <div className="bg-blue-50 p-3 rounded-lg border border-blue-200">
              <div className="text-xs text-gray-600 mb-1">Avg Series Length</div>
              <div className="text-xl font-bold text-blue-600">{changepointsData.statistics.avg_series_length.toFixed(0)} months</div>
            </div>
          </div>

          <div className="h-[500px] relative rounded-lg overflow-hidden">
            <MapContainer center={center} zoom={zoom} style={{ height: "100%", width: "100%" }} key={`changepoints-${mapKey}`}>
              <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
              {districtGeo && <GeoJSON data={districtGeo} style={{ color: "#EAB308", weight: 3, fillOpacity: 0.1 }} />}
              
              {changepointsData.changepoints.data.map((site, i) => {
                const size = 5 + Math.min(site.n_breakpoints * 2, 10);
                return (
                  <CircleMarker
                    key={`cp_${i}`}
                    center={[site.latitude, site.longitude]}
                    radius={size}
                    fillColor="#EAB308"
                    color="white"
                    weight={2}
                    fillOpacity={0.8}
                  >
                    <Popup>
                      <div style={{ fontFamily: 'sans-serif', minWidth: '220px' }}>
                        <strong style={{ color: '#EAB308' }}>Changepoint Detected</strong><br/>
                        <hr style={{ margin: '5px 0' }}/>
                        <table style={{ width: '100%', fontSize: '12px' }}>
                          <tbody>
                            <tr><td><strong>Site ID:</strong></td><td>{site.site_id}</td></tr>
                            <tr><td><strong>Primary Break:</strong></td><td>{new Date(site.changepoint_date).toLocaleDateString()}</td></tr>
                            <tr><td><strong>Total Breaks:</strong></td><td>{site.n_breakpoints}</td></tr>
                            <tr><td><strong>Series Length:</strong></td><td>{site.series_length} months</td></tr>
                          </tbody>
                        </table>
                        {site.all_breakpoints.length > 1 && (
                          <>
                            <hr style={{ margin: '5px 0' }}/>
                            <div style={{ fontSize: '11px', color: '#666' }}>
                              <strong>All Breakpoints:</strong><br/>
                              {site.all_breakpoints.map((bp, idx) => (
                                <div key={idx}>• {new Date(bp).toLocaleDateString()}</div>
                              ))}
                            </div>
                          </>
                        )}
                      </div>
                    </Popup>
                  </CircleMarker>
                );
              })}
            </MapContainer>

            <div className="absolute bottom-4 right-4 z-[1000] bg-white/95 backdrop-blur-sm p-3 rounded-lg shadow-lg border-2 border-yellow-300">
              <div className="text-xs font-bold mb-2">Symbol Size = # Breakpoints</div>
              <p className="text-xs text-gray-600">Larger circles indicate more structural breaks detected in GWL timeseries</p>
            </div>
          </div>

          <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-sm text-blue-800">
              <strong>ℹ️ What:</strong> Structural breaks in GWL timeseries using PELT algorithm.
              <br/><strong>How:</strong> Penalty={changepointsData.parameters.penalty}, Model={changepointsData.parameters.model}.
              <br/><strong>Significance:</strong> Identifies regime shifts from policy, irrigation expansion, or hydrology changes.
            </p>
          </div>
        </div>
      );
    }

    // Lag Correlation Visualization
    if (selectedAdvancedModule === 'LAG_CORRELATION' && lagCorrelationData) {
      return (
        <div className="lg:col-span-2 xl:col-span-3 bg-white rounded-xl shadow-2xl p-6 border-2 border-pink-400">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <span className="text-2xl">⏱️</span>
              <h3 className="text-lg font-bold text-gray-800">Rainfall-GWL Lag Correlation</h3>
            </div>
            <span className="text-sm bg-pink-100 text-pink-800 px-3 py-1 rounded-full font-semibold">
              {lagCorrelationData.count} sites analyzed
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
            <div className="bg-pink-50 p-3 rounded-lg border border-pink-200">
              <div className="text-xs text-gray-600 mb-1">Mean Lag</div>
              <div className="text-2xl font-bold text-pink-600">{lagCorrelationData.statistics.mean_lag.toFixed(1)} months</div>
            </div>
            <div className="bg-purple-50 p-3 rounded-lg border border-purple-200">
              <div className="text-xs text-gray-600 mb-1">Median Lag</div>
              <div className="text-2xl font-bold text-purple-600">{lagCorrelationData.statistics.median_lag.toFixed(1)} months</div>
            </div>
            <div className="bg-blue-50 p-3 rounded-lg border border-blue-200">
              <div className="text-xs text-gray-600 mb-1">Avg |Correlation|</div>
              <div className="text-2xl font-bold text-blue-600">{lagCorrelationData.statistics.mean_abs_correlation.toFixed(3)}</div>
            </div>
            <div className="bg-green-50 p-3 rounded-lg border border-green-200">
              <div className="text-xs text-gray-600 mb-1">Max Lag Tested</div>
              <div className="text-2xl font-bold text-green-600">{lagCorrelationData.parameters.max_lag_months} months</div>
            </div>
          </div>

          <div className="h-[500px] relative rounded-lg overflow-hidden">
            <MapContainer center={center} zoom={zoom} style={{ height: "100%", width: "100%" }} key={`lag-${mapKey}`}>
              <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
              {districtGeo && <GeoJSON data={districtGeo} style={{ color: "#EC4899", weight: 3, fillOpacity: 0.1 }} />}
              
              {lagCorrelationData.data.map((point, i) => {
                const lagColors: Record<number, string> = {
                  0: '#DC2626', 1: '#F59E0B', 2: '#FCD34D', 
                  3: '#84CC16', 4: '#22C55E', 5: '#14B8A6',
                  6: '#3B82F6', 7: '#6366F1', 8: '#8B5CF6',
                  9: '#A855F7', 10: '#D946EF', 11: '#EC4899', 12: '#F43F5E'
                };
                const color = lagColors[point.best_lag_months] || '#9CA3AF';
                
                return (
                  <CircleMarker
                    key={`lag_${i}`}
                    center={[point.latitude, point.longitude]}
                    radius={6 + (point.abs_correlation * 4)}
                    fillColor={color}
                    color="white"
                    weight={2}
                    fillOpacity={0.8}
                  >
                    <Popup>
                      <div style={{ fontFamily: 'sans-serif', minWidth: '200px' }}>
                        <strong style={{ color }}>Best Lag: {point.best_lag_months} months</strong><br/>
                        <hr style={{ margin: '5px 0' }}/>
                        <table style={{ width: '100%', fontSize: '12px' }}>
                          <tbody>
                            <tr><td><strong>Site ID:</strong></td><td>{point.site_id}</td></tr>
                            <tr><td><strong>Correlation:</strong></td><td>{point.correlation.toFixed(3)}</td></tr>
                            <tr><td><strong>|Correlation|:</strong></td><td>{point.abs_correlation.toFixed(3)}</td></tr>
                            <tr><td><strong>Relationship:</strong></td><td>{point.relationship}</td></tr>
                            <tr><td><strong>Data Points:</strong></td><td>{point.n_months_analyzed} months</td></tr>
                          </tbody>
                        </table>
                      </div>
                    </Popup>
                  </CircleMarker>
                );
              })}
            </MapContainer>

            <div className="absolute bottom-4 right-4 z-[1000] bg-white/95 backdrop-blur-sm p-3 rounded-lg shadow-lg border-2 border-pink-300 max-h-[300px] overflow-y-auto">
              <div className="text-xs font-bold mb-2">Lag (months)</div>
              <div className="space-y-1">
                {Object.entries(lagCorrelationData.statistics.lag_distribution).sort(([a], [b]) => Number(a) - Number(b)).map(([lag, count]) => {
                  const lagColors: Record<number, string> = {
                    0: '#DC2626', 1: '#F59E0B', 2: '#FCD34D', 
                    3: '#84CC16', 4: '#22C55E', 5: '#14B8A6',
                    6: '#3B82F6', 7: '#6366F1', 8: '#8B5CF6',
                    9: '#A855F7', 10: '#D946EF', 11: '#EC4899', 12: '#F43F5E'
                  };
                  return (
                    <div key={lag} className="flex items-center gap-2">
                      <div className="w-4 h-4 rounded" style={{ backgroundColor: lagColors[Number(lag)] || '#9CA3AF' }}></div>
                      <span className="text-xs">{lag} mo ({count} sites)</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-sm text-blue-800">
              <strong>ℹ️ What:</strong> Time lag between AOI rainfall and site GWL for max correlation.
              <br/><strong>How:</strong> Tests 0–{lagCorrelationData.parameters.max_lag_months} month lags via cross-correlation.
              <br/><strong>Significance:</strong> Reveals aquifer memory; short lags = responsive systems, long lags = slow infiltration.
            </p>
          </div>
        </div>
      );
    }

    // Hotspots Clustering Visualization
    if (selectedAdvancedModule === 'HOTSPOTS' && hotspotsData) {
      return (
        <div className="lg:col-span-2 xl:col-span-3 bg-white rounded-xl shadow-2xl p-6 border-2 border-rose-400">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <span className="text-2xl">🔥</span>
              <h3 className="text-lg font-bold text-gray-800">Declining GWL Hotspots (DBSCAN)</h3>
            </div>
            <span className="text-sm bg-rose-100 text-rose-800 px-3 py-1 rounded-full font-semibold">
              {hotspotsData.statistics.n_clusters} clusters found
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
            <div className="bg-rose-50 p-3 rounded-lg border border-rose-200">
              <div className="text-xs text-gray-600 mb-1">Declining Sites</div>
              <div className="text-2xl font-bold text-rose-600">{hotspotsData.statistics.total_declining_sites}</div>
            </div>
            <div className="bg-orange-50 p-3 rounded-lg border border-orange-200">
              <div className="text-xs text-gray-600 mb-1">Clustered</div>
              <div className="text-2xl font-bold text-orange-600">{hotspotsData.statistics.clustered_points}</div>
            </div>
            <div className="bg-gray-50 p-3 rounded-lg border border-gray-200">
              <div className="text-xs text-gray-600 mb-1">Noise Points</div>
              <div className="text-2xl font-bold text-gray-600">{hotspotsData.statistics.noise_points}</div>
            </div>
            <div className="bg-blue-50 p-3 rounded-lg border border-blue-200">
              <div className="text-xs text-gray-600 mb-1">Clustering Rate</div>
              <div className="text-2xl font-bold text-blue-600">{(hotspotsData.statistics.clustering_rate * 100).toFixed(1)}%</div>
            </div>
          </div>

          {hotspotsData.clusters.length > 0 && (
            <div className="bg-gradient-to-r from-rose-50 to-orange-50 p-4 rounded-lg border border-rose-200 mb-4">
              <h4 className="font-bold text-sm mb-3 text-rose-900">Cluster Details</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-h-[200px] overflow-y-auto">
                {hotspotsData.clusters.map((cluster, idx) => (
                  <div key={idx} className="bg-white p-3 rounded-lg border border-gray-200">
                    <div className="font-semibold text-sm flex items-center gap-2">
                      <div className="w-4 h-4 rounded-full" style={{ backgroundColor: getHotspotColor(cluster.cluster_id) }}></div>
                      Cluster {cluster.cluster_id}
                    </div>
                    <div className="text-xs text-gray-600 mt-1">
                      <div><strong>Sites:</strong> {cluster.n_sites}</div>
                      <div><strong>Mean Slope:</strong> {cluster.mean_slope.toFixed(4)} m/yr</div>
                      <div><strong>Max Slope:</strong> {cluster.max_slope.toFixed(4)} m/yr</div>
                      <div><strong>Centroid:</strong> {cluster.centroid_lat.toFixed(4)}, {cluster.centroid_lon.toFixed(4)}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="h-[500px] relative rounded-lg overflow-hidden">
            <MapContainer center={center} zoom={zoom} style={{ height: "100%", width: "100%" }} key={`hotspots-${mapKey}`}>
              <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
              {districtGeo && <GeoJSON data={districtGeo} style={{ color: "#F43F5E", weight: 3, fillOpacity: 0.1 }} />}
              
              {hotspotsData.data.map((point, i) => (
                <CircleMarker
                  key={`hot_${i}`}
                  center={[point.latitude, point.longitude]}
                  radius={7}
                  fillColor={getHotspotColor(point.cluster)}
                  color="white"
                  weight={2}
                  fillOpacity={point.cluster === -1 ? 0.3 : 0.8}
                >
                  <Popup>
                    <div style={{ fontFamily: 'sans-serif', minWidth: '180px' }}>
                      <strong style={{ color: getHotspotColor(point.cluster) }}>
                        {point.cluster === -1 ? 'Noise Point' : `Cluster ${point.cluster}`}
                      </strong><br/>
                      <hr style={{ margin: '5px 0' }}/>
                      <table style={{ width: '100%', fontSize: '12px' }}>
                        <tbody>
                          <tr><td><strong>Site ID:</strong></td><td>{point.site_id}</td></tr>
                          <tr><td><strong>Decline Rate:</strong></td><td>{point.slope_m_per_year.toFixed(4)} m/yr</td></tr>
                        </tbody>
                      </table>
                    </div>
                  </Popup>
                </CircleMarker>
              ))}

              {hotspotsData.clusters.map((cluster, idx) => (
                <CircleMarker
                  key={`centroid_${idx}`}
                  center={[cluster.centroid_lat, cluster.centroid_lon]}
                  radius={12}
                  fillColor={getHotspotColor(cluster.cluster_id)}
                  color="white"
                  weight={3}
                  fillOpacity={1}
                >
                  <Popup>
                    <div style={{ fontFamily: 'sans-serif', minWidth: '180px' }}>
                      <strong>Cluster {cluster.cluster_id} Centroid</strong><br/>
                      <hr style={{ margin: '5px 0' }}/>
                      <div style={{ fontSize: '12px' }}>
                        <div><strong>Sites:</strong> {cluster.n_sites}</div>
                        <div><strong>Mean Decline:</strong> {cluster.mean_slope.toFixed(4)} m/yr</div>
                      </div>
                    </div>
                  </Popup>
                </CircleMarker>
              ))}
            </MapContainer>

            <div className="absolute bottom-4 right-4 z-[1000] bg-white/95 backdrop-blur-sm p-3 rounded-lg shadow-lg border-2 border-rose-300">
              <div className="text-xs font-bold mb-2">Clusters</div>
              <div className="space-y-1 max-h-[200px] overflow-y-auto">
                {hotspotsData.clusters.slice(0, 8).map((cluster, idx) => (
                  <div key={idx} className="flex items-center gap-2">
                    <div className="w-4 h-4 rounded-full" style={{ backgroundColor: getHotspotColor(cluster.cluster_id) }}></div>
                    <span className="text-xs">Cluster {cluster.cluster_id} ({cluster.n_sites})</span>
                  </div>
                ))}
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full bg-gray-400"></div>
                  <span className="text-xs">Noise ({hotspotsData.statistics.noise_points})</span>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-sm text-blue-800">
              <strong>ℹ️ What:</strong> Spatial clusters of declining sites using DBSCAN (haversine metric).
              <br/><strong>How:</strong> eps={hotspotsData.parameters.eps_km} km, min_samples={hotspotsData.parameters.min_samples}.
              <br/><strong>Significance:</strong> Priority zones for regional intervention; large centroids = critical hotspots.
            </p>
          </div>
        </div>
      );
    }

    // GRACE Divergence Visualization
    if (selectedAdvancedModule === 'GRACE_DIVERGENCE' && divergenceData) {
      return (
        <div className="lg:col-span-2 xl:col-span-3 bg-white rounded-xl shadow-2xl p-6 border-2 border-amber-400">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <span className="text-2xl">🌐</span>
              <h3 className="text-lg font-bold text-gray-800">GRACE-Well Divergence Analysis</h3>
            </div>
            <span className="text-sm bg-amber-100 text-amber-800 px-3 py-1 rounded-full font-semibold">
              {divergenceData.count} pixels analyzed
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
            <div className="bg-amber-50 p-3 rounded-lg border border-amber-200">
              <div className="text-xs text-gray-600 mb-1">Mean Divergence</div>
              <div className="text-2xl font-bold text-amber-600">{divergenceData.statistics.mean_divergence.toFixed(3)}</div>
            </div>
            <div className="bg-blue-50 p-3 rounded-lg border border-blue-200">
              <div className="text-xs text-gray-600 mb-1">Positive Div</div>
              <div className="text-2xl font-bold text-blue-600">{divergenceData.statistics.positive_divergence_pixels}</div>
            </div>
            <div className="bg-red-50 p-3 rounded-lg border border-red-200">
              <div className="text-xs text-gray-600 mb-1">Negative Div</div>
              <div className="text-2xl font-bold text-red-600">{divergenceData.statistics.negative_divergence_pixels}</div>
            </div>
            <div className="bg-purple-50 p-3 rounded-lg border border-purple-200">
              <div className="text-xs text-gray-600 mb-1">Max |Divergence|</div>
              <div className="text-xl font-bold text-purple-600">{Math.max(Math.abs(divergenceData.statistics.max_divergence), Math.abs(divergenceData.statistics.min_divergence)).toFixed(3)}</div>
            </div>
          </div>

          <div className="h-[500px] relative rounded-lg overflow-hidden">
            <MapContainer center={center} zoom={zoom} style={{ height: "100%", width: "100%" }} key={`divergence-${mapKey}`}>
              <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
              {districtGeo && <GeoJSON data={districtGeo} style={{ color: "#F59E0B", weight: 3, fillOpacity: 0.1 }} />}
              
              {divergenceData.data.map((point, i) => (
                <CircleMarker
                  key={`div_${i}`}
                  center={[point.latitude, point.longitude]}
                  radius={5}
                  fillColor={getDivergenceColor(point.divergence)}
                  color="white"
                  weight={1}
                  fillOpacity={0.8}
                >
                  <Popup>
                    <div style={{ fontFamily: 'sans-serif', minWidth: '200px' }}>
                      <strong style={{ color: getDivergenceColor(point.divergence) }}>Divergence: {point.divergence.toFixed(3)}</strong><br/>
                      <hr style={{ margin: '5px 0' }}/>
                      <table style={{ width: '100%', fontSize: '12px' }}>
                        <tbody>
                          <tr><td><strong>GRACE z:</strong></td><td>{point.grace_z.toFixed(3)}</td></tr>
                          <tr><td><strong>Well z (interp):</strong></td><td>{point.well_z_interpolated.toFixed(3)}</td></tr>
                          <tr><td><strong>TWS:</strong></td><td>{point.tws.toFixed(2)} cm</td></tr>
                        </tbody>
                      </table>
                    </div>
                  </Popup>
                </CircleMarker>
              ))}
            </MapContainer>

            <div className="absolute bottom-4 right-4 z-[1000] bg-white/95 backdrop-blur-sm p-3 rounded-lg shadow-lg border-2 border-amber-300">
              <div className="text-xs font-bold mb-2">Divergence (z-score)</div>
              <div className="space-y-1">
                {[
                  { label: 'Strong -ve (<-2)', value: -2.5 },
                  { label: 'Moderate -ve (-2 to -1)', value: -1.5 },
                  { label: 'Weak -ve (-1 to 0)', value: -0.5 },
                  { label: 'Weak +ve (0 to 1)', value: 0.5 },
                  { label: 'Moderate +ve (1 to 2)', value: 1.5 },
                  { label: 'Strong +ve (>2)', value: 2.5 }
                ].map((item, idx) => (
                  <div key={idx} className="flex items-center gap-2">
                    <div className="w-4 h-4 rounded" style={{ backgroundColor: getDivergenceColor(item.value) }}></div>
                    <span className="text-xs">{item.label}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-sm text-blue-800">
              <strong>ℹ️ What:</strong> z_GRACE − z_GWL (interpolated) at each pixel.
              <br/><strong>How:</strong> Standardize GRACE and well data separately; compute pixel-level divergence.
              <br/><strong>Significance:</strong> Negative divergence = GRACE underestimates stress (surface/pumped zone mismatch). Positive = overestimates.
            </p>
          </div>
        </div>
      );
    }

    return null;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
      {alertMessage && (
        <div className={`fixed top-4 right-4 z-[10000] max-w-md p-4 rounded-lg shadow-2xl border-2 animate-fade-in ${
          alertType === 'error' ? 'bg-red-50 border-red-300 text-red-800' :
          alertType === 'warning' ? 'bg-yellow-50 border-yellow-300 text-yellow-800' :
          alertType === 'success' ? 'bg-green-50 border-green-300 text-green-800' :
          'bg-blue-50 border-blue-300 text-blue-800'
        }`}>
          <div className="flex items-center gap-2">
            <span className="text-2xl">
              {alertType === 'error' ? '❌' : alertType === 'warning' ? '⚠️' : alertType === 'success' ? '✅' : 'ℹ️'}
            </span>
            <span className="font-medium">{alertMessage}</span>
          </div>
        </div>
      )}

      <div className="relative bg-gradient-to-r from-blue-600 via-green-600 to-teal-600 text-white shadow-2xl">
        <div className="absolute inset-0 bg-black/10"></div>
        <div className="relative max-w-7xl mx-auto px-6 py-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold mb-2 flex items-center gap-3">
                🌍 GeoHydro Dashboard
              </h1>
              <p className="text-blue-100 text-lg">
                Advanced Groundwater Analysis & Remote Sensing Platform
              </p>
            </div>
            <button
              onClick={() => setIsChatOpen(!isChatOpen)}
              className="bg-white/20 backdrop-blur-md hover:bg-white/30 text-white px-6 py-3 rounded-xl font-semibold transition-all duration-200 flex items-center gap-2 shadow-lg border border-white/30"
            >
              <span className="text-2xl">💬</span>
              {isChatOpen ? 'Close Chat' : 'Ask AI Expert'}
            </button>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-8">
          <div className="lg:col-span-4 bg-white rounded-xl shadow-xl p-6 border-2 border-blue-200">
            <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
              <span>🎛️</span> Control Panel
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">State</label>
                <select
                  className="w-full p-3 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
                  value={selectedState}
                  onChange={(e) => {
                    setSelectedState(e.target.value);
                    setSelectedDistrict("");
                  }}
                >
                  <option value="">All India</option>
                  {states.map((s, i) => (
                    <option key={i} value={s.State}>{s.State}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">District</label>
                <select
                  className="w-full p-3 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
                  value={selectedDistrict}
                  onChange={(e) => setSelectedDistrict(e.target.value)}
                  disabled={!selectedState}
                >
                  <option value="">All Districts</option>
                  {districts.map((d, i) => (
                    <option key={i} value={d.district_name}>{d.district_name}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Year</label>
                <select
                  className="w-full p-3 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
                  value={selectedYear}
                  onChange={(e) => setSelectedYear(Number(e.target.value))}
                >
                  {availableYears.map(y => (
                    <option key={y} value={y}>{y}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Month (Optional)</label>
                <select
                  className="w-full p-3 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
                  value={selectedMonth || ""}
                  onChange={(e) => setSelectedMonth(e.target.value ? Number(e.target.value) : null)}
                >
                  <option value="">All Year</option>
                  {Array.from({ length: 12 }, (_, i) => i + 1).map(m => (
                    <option key={m} value={m}>{new Date(2000, m - 1).toLocaleString('default', { month: 'long' })}</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
              <button
                onClick={() => setShowAquifers(!showAquifers)}
                className={`p-3 rounded-lg font-semibold transition-all duration-200 border-2 ${
                  showAquifers
                    ? 'bg-purple-500 text-white border-purple-600 shadow-lg'
                    : 'bg-white text-gray-700 border-gray-300 hover:border-purple-400'
                }`}
              >
                🔷 Aquifers
              </button>

              <button
                onClick={() => setShowGrace(!showGrace)}
                className={`p-3 rounded-lg font-semibold transition-all duration-200 border-2 ${
                  showGrace
                    ? 'bg-green-500 text-white border-green-600 shadow-lg'
                    : 'bg-white text-gray-700 border-gray-300 hover:border-green-400'
                }`}
              >
                🌊 GRACE
              </button>

              <button
                onClick={() => setShowRainfall(!showRainfall)}
                className={`p-3 rounded-lg font-semibold transition-all duration-200 border-2 ${
                  showRainfall
                    ? 'bg-blue-500 text-white border-blue-600 shadow-lg'
                    : 'bg-white text-gray-700 border-gray-300 hover:border-blue-400'
                }`}
              >
                🌧️ Rainfall
              </button>

              <button
                onClick={() => setShowWells(!showWells)}
                className={`p-3 rounded-lg font-semibold transition-all duration-200 border-2 ${
                  showWells
                    ? 'bg-red-500 text-white border-red-600 shadow-lg'
                    : 'bg-white text-gray-700 border-gray-300 hover:border-red-400'
                }`}
              >
                💧 Wells
              </button>

              <button
                onClick={() => setShowTimeseries(!showTimeseries)}
                className={`p-3 rounded-lg font-semibold transition-all duration-200 border-2 ${
                  showTimeseries
                    ? 'bg-indigo-500 text-white border-indigo-600 shadow-lg'
                    : 'bg-white text-gray-700 border-gray-300 hover:border-indigo-400'
                }`}
              >
                📈 Timeseries
              </button>

              {/* NEW: Advanced Modules Button */}
              <div className="relative">
                <button
                  onClick={() => setShowAdvancedMenu(!showAdvancedMenu)}
                  className={`w-full p-3 rounded-lg font-semibold transition-all duration-200 border-2 ${
                    selectedAdvancedModule
                      ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white border-purple-600 shadow-lg'
                      : 'bg-white text-gray-700 border-gray-300 hover:border-purple-400'
                  }`}
                >
                  🔬 Advanced
                </button>

                {showAdvancedMenu && (
                  <div className="absolute top-full left-0 mt-2 w-80 bg-white rounded-lg shadow-2xl border-2 border-purple-300 z-[2000] max-h-96 overflow-y-auto">
                    <div className="p-2">
                      {[
                        { id: 'ASI', label: '1. Aquifer Suitability Index', icon: '🔷' },
                        { id: 'NETWORK_DENSITY', label: '2. Network Density', icon: '📊' },
                        { id: 'SASS', label: '3. Aquifer Stress Score', icon: '⚠️' },
                        { id: 'GRACE_DIVERGENCE', label: '4. GRACE Divergence', icon: '🌐' },
                        { id: 'FORECAST', label: '5. GWL Forecasting', icon: '📈' },
                        { id: 'RECHARGE', label: '6. Recharge Planning', icon: '💧' },
                        { id: 'SIGNIFICANT_TRENDS', label: '7. Significant Trends', icon: '📉' },
                        { id: 'CHANGEPOINTS', label: '8. Changepoint Detection', icon: '📍' },
                        { id: 'LAG_CORRELATION', label: '9. Lag Correlation', icon: '⏱️' },
                        { id: 'HOTSPOTS', label: '10. Hotspots Clustering', icon: '🔥' }
                      ].map((module) => (
                        <button
                          key={module.id}
                          onClick={() => {
                            setSelectedAdvancedModule(module.id);
                            setShowAdvancedMenu(false);
                            // Clear other visualizations
                            setShowAquifers(false);
                            setShowGrace(false);
                            setShowRainfall(false);
                            setShowWells(false);
                            setShowTimeseries(false);
                          }}
                          className={`w-full text-left p-3 rounded-lg hover:bg-purple-50 transition-all flex items-center gap-2 ${
                            selectedAdvancedModule === module.id ? 'bg-purple-100 font-semibold' : ''
                          }`}
                        >
                          <span className="text-xl">{module.icon}</span>
                          <span className="text-sm">{module.label}</span>
                        </button>
                      ))}
                      
                      {selectedAdvancedModule && (
                        <button
                          onClick={() => {
                            setSelectedAdvancedModule('');
                            setShowAdvancedMenu(false);
                          }}
                          className="w-full mt-2 p-2 bg-red-50 text-red-600 rounded-lg hover:bg-red-100 transition-all text-sm font-semibold"
                        >
                          ❌ Clear Advanced Module
                        </button>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {activeLayers.length > 0 && (
          <div className="mb-6 bg-white rounded-xl shadow-xl p-4 border-2 border-gray-200">
            <h4 className="text-sm font-bold text-gray-700 mb-2">Active Layers:</h4>
            <div className="flex flex-wrap gap-2">
              {activeLayers.map((layer, idx) => (
                <div key={idx} className="flex items-center gap-2 bg-gray-100 px-3 py-1 rounded-full border border-gray-300">
                  <span>{layer.icon}</span>
                  <span className="text-sm font-medium">{layer.name}</span>
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: layer.color === 'purple' ? '#9333EA' : layer.color === 'green' ? '#059669' : layer.color === 'blue' ? '#2563EB' : '#DC2626' }}></div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* NEW: Render Advanced Module Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6 mb-8">
          {renderAdvancedModuleContent()}
        </div>

        {/* Existing Map Grid */}
        {(showAquifers || showGrace || showRainfall || showWells) && !selectedAdvancedModule && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            {activeLayers.map((layer) => (
              <div key={layer.id} className="h-[600px]">
                {renderMap(layer.id, layer.name, layer.icon, 
                  layer.color === 'purple' ? '#9333EA' : 
                  layer.color === 'green' ? '#059669' : 
                  layer.color === 'blue' ? '#2563EB' : '#DC2626'
                )}
              </div>
            ))}
          </div>
        )}

        {showTimeseries && timeseriesResponse && !selectedAdvancedModule && (
          <div className="bg-white rounded-xl shadow-2xl p-6 mb-8 border-2 border-indigo-400">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold text-gray-800 flex items-center gap-2">
                <span>📈</span> Unified Timeseries Analysis
              </h3>
              <div className="flex gap-2">
                {(['raw', 'seasonal', 'deseasonalized'] as const).map((view) => (
                  <button
                    key={view}
                    onClick={() => setTimeseriesView(view)}
                    className={`px-4 py-2 rounded-lg font-semibold text-sm transition-all ${
                      timeseriesView === view
                        ? 'bg-indigo-500 text-white'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    {view.charAt(0).toUpperCase() + view.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            {timeseriesResponse.statistics && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                {timeseriesResponse.statistics.gwl_trend && (
                  <div className={`p-4 rounded-lg border-2 ${
                    timeseriesResponse.statistics.gwl_trend.direction === 'Declining' 
                      ? 'bg-red-50 border-red-300' 
                      : 'bg-green-50 border-green-300'
                  }`}>
                    <div className="text-sm font-semibold text-gray-700 mb-2">GWL Trend</div>
                    <div className={`text-2xl font-bold ${
                      timeseriesResponse.statistics.gwl_trend.direction === 'Declining' 
                        ? 'text-red-600' 
                        : 'text-green-600'
                    }`}>
                      {timeseriesResponse.statistics.gwl_trend.slope_per_year.toFixed(4)} m/yr
                    </div>
                    <div className="text-xs text-gray-600 mt-1">
                      R² = {timeseriesResponse.statistics.gwl_trend.r_squared.toFixed(3)} | 
                      p = {timeseriesResponse.statistics.gwl_trend.p_value.toFixed(4)}
                    </div>
                  </div>
                )}

                {timeseriesResponse.statistics.grace_trend && (
                  <div className="p-4 rounded-lg bg-green-50 border-2 border-green-300">
                    <div className="text-sm font-semibold text-gray-700 mb-2">GRACE Trend</div>
                    <div className="text-2xl font-bold text-green-600">
                      {timeseriesResponse.statistics.grace_trend.slope_per_year.toFixed(4)} cm/yr
                    </div>
                    <div className="text-xs text-gray-600 mt-1">
                      R² = {timeseriesResponse.statistics.grace_trend.r_squared.toFixed(3)}
                    </div>
                  </div>
                )}

                {timeseriesResponse.statistics.rainfall_trend && (
                  <div className="p-4 rounded-lg bg-blue-50 border-2 border-blue-300">
                    <div className="text-sm font-semibold text-gray-700 mb-2">Rainfall Trend</div>
                    <div className="text-2xl font-bold text-blue-600">
                      {timeseriesResponse.statistics.rainfall_trend.slope_per_year.toFixed(4)} mm/yr
                    </div>
                    <div className="text-xs text-gray-600 mt-1">
                      R² = {timeseriesResponse.statistics.rainfall_trend.r_squared.toFixed(3)}
                    </div>
                  </div>
                )}
              </div>
            )}

            <Plot
              data={[
                // 1. GWL Trace
                timeseriesResponse.timeseries.length > 0 && {
                  x: timeseriesResponse.timeseries.map(p => p.date),
                  y: timeseriesResponse.timeseries.map(p => 
                    timeseriesView === 'raw' ? p.avg_gwl :
                    timeseriesView === 'seasonal' ? p.gwl_seasonal :
                    p.gwl_deseasonalized
                  ),
                  type: 'scatter',
                  mode: 'lines+markers',
                  name: 'GWL (m bgl)',
                  line: { color: GWL_COLOR, width: 2 },
                  marker: { size: 6 },
                  // Connect gaps if some seasonal data is missing at edges
                  connectgaps: true, 
                  yaxis: 'y1'
                },
                
                // 2. GRACE Trace
                timeseriesResponse.timeseries.length > 0 && {
                  x: timeseriesResponse.timeseries.map(p => p.date),
                  y: timeseriesResponse.timeseries.map(p => 
                    timeseriesView === 'raw' ? p.avg_tws :
                    timeseriesView === 'seasonal' ? p.grace_seasonal :
                    p.grace_deseasonalized
                  ),
                  type: 'scatter',
                  mode: 'lines+markers',
                  name: 'GRACE TWS (cm)',
                  line: { color: GRACE_COLOR, width: 2 },
                  marker: { size: 6 },
                  connectgaps: true,
                  yaxis: 'y2'
                },

                // 3. Rainfall Trace
                timeseriesResponse.timeseries.length > 0 && {
                  x: timeseriesResponse.timeseries.map(p => p.date),
                  y: timeseriesResponse.timeseries.map(p => 
                    timeseriesView === 'raw' ? p.avg_rainfall :
                    timeseriesView === 'seasonal' ? p.rainfall_seasonal :
                    p.rainfall_deseasonalized
                  ),
                  type: 'bar', // Or 'line' for seasonal view if preferred
                  name: 'Rainfall (mm/day)',
                  marker: { color: RAIN_COLOR, opacity: 0.6 },
                  yaxis: 'y3'
                }
              ].filter(Boolean)}
              layout={{
                autosize: true,
                height: 500,
                margin: { l: 60, r: 60, t: 40, b: 60 },
                xaxis: { 
                  title: 'Date',
                  gridcolor: 'rgba(0,0,0,0.1)'
                },
                yaxis: {
                  title: 'GWL (m bgl)',
                  titlefont: { color: GWL_COLOR },
                  tickfont: { color: GWL_COLOR },
                  autorange: 'reversed',
                  gridcolor: 'rgba(0,0,0,0.1)'
                },
                yaxis2: {
                  title: 'GRACE TWS (cm)',
                  titlefont: { color: GRACE_COLOR },
                  tickfont: { color: GRACE_COLOR },
                  overlaying: 'y',
                  side: 'right',
                  showgrid: false
                },
                yaxis3: {
                  title: 'Rainfall (mm/day)',
                  titlefont: { color: RAIN_COLOR },
                  tickfont: { color: RAIN_COLOR },
                  overlaying: 'y',
                  side: 'right',
                  position: 0.85,
                  showgrid: false
                },
                plot_bgcolor: 'white',
                paper_bgcolor: 'white',
                font: { family: 'Arial, sans-serif' },
                hovermode: 'x unified',
                legend: { x: 0.01, y: 0.99, bgcolor: 'rgba(255,255,255,0.8)' }
              }}
              config={{
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['pan2d', 'select2d', 'lasso2d'],
                toImageButtonOptions: {
                  format: 'png',
                  filename: `timeseries_${timeseriesView}_${new Date().toISOString().split('T')[0]}`,
                  height: 600,
                  width: 1200,
                  scale: 2
                }
              }}
              style={{ width: '100%' }}
              useResizeHandler={true}
            />
          </div>
        )}
      </div>

      {isChatOpen && (
        <div className="fixed bottom-6 right-6 w-96 h-[600px] bg-white rounded-2xl shadow-2xl border-2 border-blue-300 flex flex-col z-[10000]">
          <div className="bg-gradient-to-r from-blue-600 to-green-600 text-white p-4 rounded-t-2xl flex items-center justify-between">
            <div>
              <h3 className="font-bold text-lg">AI Expert Assistant</h3>
              <p className="text-xs text-blue-100">Powered by Llama3.1</p>
            </div>
            <button
              onClick={() => setIsChatOpen(false)}
              className="text-white hover:bg-white/20 rounded-lg p-2 transition-all"
            >
              ✕
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {chatMessages.map((msg, idx) => (
              <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[80%] rounded-2xl p-3 ${
                  msg.role === 'user'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 text-gray-800 border border-gray-200'
                }`}>
                  <div className="text-sm whitespace-pre-wrap">{msg.content}</div>
                  {msg.sourcesUsed !== undefined && msg.sourcesUsed > 0 && (
                    <div className="text-xs mt-2 opacity-70">
                      📚 Used {msg.sourcesUsed} knowledge source{msg.sourcesUsed !== 1 ? 's' : ''}
                    </div>
                  )}
                  <div className="text-xs mt-1 opacity-60">
                    {msg.timestamp.toLocaleTimeString()}
                  </div>
                </div>
              </div>
            ))}
            {isChatLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 rounded-2xl p-3 border border-gray-200">
                  <div className="flex gap-1">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {chatMessages.length === 1 && (
            <div className="px-4 py-2 border-t border-gray-200">
              <div className="text-xs font-semibold text-gray-600 mb-2">
                {selectedAdvancedModule ? `💡 Ask about ${selectedAdvancedModule}:` : '💡 Try asking:'}
              </div>
              <div className="space-y-1 max-h-32 overflow-y-auto">
                {getSuggestedQuestions().map((q, idx) => (
                  <button
                    key={idx}
                    onClick={() => setChatInput(q)}
                    className="w-full text-left text-xs p-2 bg-blue-50 hover:bg-blue-100 rounded-lg transition-all text-blue-700"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="p-4 border-t border-gray-200">
            <div className="flex gap-2">
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyPress={handleChatKeyPress}
                placeholder="Ask about groundwater, GRACE, or advanced modules..."
                className="flex-1 p-3 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
                disabled={isChatLoading}
              />
              <button
                onClick={handleSendMessage}
                disabled={!chatInput.trim() || isChatLoading}
                className="bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 text-white px-4 py-2 rounded-lg font-semibold transition-all"
              >
                {isChatLoading ? '⏳' : '📤'}
              </button>
            </div>
          </div>
        </div>
      )}

      {isLoading && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-[10000] flex items-center justify-center">
          <div className="bg-white rounded-2xl p-8 shadow-2xl text-center">
            <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-600 mx-auto mb-4"></div>
            <p className="text-lg font-semibold text-gray-700">Loading data...</p>
          </div>
        </div>
      )}
    </div>
  );
}