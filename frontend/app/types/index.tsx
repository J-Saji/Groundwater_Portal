import type { GeoJSON } from 'geojson';

export type Geometry = GeoJSON.FeatureCollection<GeoJSON.Geometry, GeoJSON.GeoJsonProperties>;

export interface StateType {
  State: string;
}

export interface DistrictType {
  district_name: string;
  geometry: Geometry;
  center?: [number, number];
}

export interface AquiferType {
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

export interface GracePoint {
  longitude: number;
  latitude: number;
  lwe_cm: number;
  cell_area_km2?: number;
}

export interface RainfallPoint {
  longitude: number;
  latitude: number;
  rainfall_mm: number;
  days_averaged?: number;
}

export interface WellPoint {
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

export interface GWRFeature {
  type: "Feature";
  properties: {
    district: string;
    state: string;
    annual_resource_mcm: number;
    area_m2: number;
  };
  geometry: GeoJSON.Geometry;
}

export interface GWRResponse {
  status: string;
  filters: {
    year: number;
    state: string | null;
    district: string | null;
  };
  statistics: {
    total_resource_mcm: number;
    mean_resource_mcm: number;
    min_resource_mcm: number;
    max_resource_mcm: number;
    num_districts: number;
    total_area_km2: number;
    color_range?: {
      zmin: number;
      zmax: number;
    };
  };
  count: number;
  geojson: {
    type: "FeatureCollection";
    features: GWRFeature[];
  };
}

export interface GWRYearsResponse {
  status: string;
  min_year: number;
  max_year: number;
  total_years: number;
  years: number[];
}

export interface StorageVsGWRPoint {
  year: number;
  storage_change_mcm: number | null;
  gwr_resource_mcm: number | null;
  pre_monsoon_gwl: number | null;
  post_monsoon_gwl: number | null;
  fluctuation_m: number | null;
}

export interface StorageVsGWRResponse {
  filters: {
    state: string | null;
    district: string | null;
  };
  aquifer_properties: {
    total_area_km2: number;
    area_weighted_specific_yield: number;
  };
  storage_statistics: {
    years_with_data: number;
    avg_annual_storage_change_mcm: number;
    min_storage_mcm: number;
    max_storage_mcm: number;
  };
  gwr_statistics: {
    years_with_data: number;
    avg_annual_resource_mcm: number;
    min_resource_mcm: number;
    max_resource_mcm: number;
  };
  data: StorageVsGWRPoint[];
}

export interface TimeseriesPoint {
  date: string;
  avg_gwl?: number;
  avg_tws?: number;
  avg_rainfall?: number;
  monthly_rainfall_mm?: number;
  monthly_rainfall_total_mm?: number;  // For raw view: avg_rainfall * days_in_month
  days_in_month?: number;
  count?: number;
  gwl_seasonal?: number;
  grace_seasonal?: number;
  rainfall_seasonal?: number;
  gwl_deseasonalized?: number;
  grace_deseasonalized?: number;
  rainfall_deseasonalized?: number;
}

export interface TrendStatistics {
  slope_per_month?: number;
  slope_per_year: number;
  r_squared: number;
  direction: string;
  mean?: number;
  min?: number;
  max?: number;
  std?: number;
}

export interface TimeseriesStatistics {
  gwl_trend?: TrendStatistics;
  grace_trend?: TrendStatistics;
  rainfall_trend?: TrendStatistics;
  seasonal_amplitude?: number;
  seasonal_mean?: number;
  view?: string;
  note?: string;
}

export interface ActionableInsight {
  severity: 'CRITICAL' | 'HIGH' | 'MODERATE' | 'WARNING' | 'POSITIVE' | 'INFO';
  metric: string;
  finding: string;
  meaning: string;
  recommendation?: string;
  confidence?: string;
}

export interface TrendInterpretation {
  what_is_slope: string;
  slope_meaning: {
    positive_slope: string;
    negative_slope: string;
  };
  r_squared_meaning: string;
  r_squared_ranges?: {
    [key: string]: string;
  };
  current_values: TrendStatistics;
}

export interface Interpretations {
  gwl_trend?: TrendInterpretation | null;
  grace_trend?: TrendInterpretation | null;
  actionable_insights: ActionableInsight[];
}

export interface ChartConfig {
  gwl_chart_type: string;
  grace_chart_type?: string;
  rainfall_chart_type: string;
  rainfall_field: string;
  rainfall_unit: string;
  gwl_y_axis_reversed: boolean;
}

export interface TimeseriesResponse {
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
  interpretations?: Interpretations;
  error?: string;
  message?: string;
}

export interface WellsSummary {
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

export interface StorageYear {
  year: number;
  pre_monsoon_gwl: number;
  post_monsoon_gwl: number;
  fluctuation_m: number;
  storage_change_mcm: number;
}

export interface StorageResponse {
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

export interface GraceResponse {
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

export interface RainfallResponse {
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

export interface WellsResponse {
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

export interface YearRangeResponse {
  min_year: number;
  max_year: number;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  sourcesUsed?: number;
  timestamp: Date;
}

export interface MapContext {
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
    active_module?: string;
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

// Advanced Module Interfaces
export interface ASIFeature {
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

export interface ASIResponse {
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

export interface NetworkDensitySitePoint {
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

export interface NetworkDensityGridPoint {
  latitude: number;
  longitude: number;
  x: number;
  y: number;
  density_per_1000km2: number;
}

export interface NetworkDensityResponse {
  module: string;
  description: string;
  filters: { state: string | null; district: string | null };
  parameters: { radius_km: number };
  statistics: {
    total_sites: number;
    avg_strength: number;
    avg_density: number;
    avg_local_density: number;
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

export interface SASSPoint {
  site_id: string;
  latitude: number;
  longitude: number;
  sass_score: number;
  gwl_stress: number;
  grace_z: number;
  rain_z: number;
  gwl: number;
}

export interface SASSResponse {
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

export interface DivergencePoint {
  latitude: number;
  longitude: number;
  divergence: number;
  grace_z: number;
  well_z_interpolated: number;
  tws: number;
}

export interface DivergenceResponse {
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

export interface ForecastPoint {
  longitude: number;
  latitude: number;
  pred_delta_m: number;
  current_gwl: number;
  forecast_gwl: number;
  r_squared: number;
  trend_component: number;
  grace_component: number;
  n_months_training: number;
  mean_grace_contribution: number;
}

export interface ForecastResponse {
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
    mean_grace_contribution: number;
    success_rate: number;
  };
  count: number;
  data: ForecastPoint[];
}

export interface RechargeStructure {
  structure_type: string;
  recommended_units: number;
  total_capacity_mcm: number;
  allocation_fraction: number;
}

export interface RechargeSiteRecommendation {
  site_id: string;
  latitude: number;
  longitude: number;
  stress_category: string;
  recommended_structure: string;
  current_gwl: number;
}

export interface RechargeResponse {
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

export interface SignificantTrendPoint {
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

export interface SignificantTrendsResponse {
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

export interface ChangepointCoverageSite {
  site_id: string;
  latitude: number;
  longitude: number;
  n_months: number;
  date_start: string;
  date_end: string;
  span_years: number;
  analyzed: boolean;
}

export interface ChangepointSite {
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

export interface ChangepointResponse {
  module: string;
  description: string;
  filters: { state: string | null; district: string | null };
  parameters: {
    penalty: number;
    algorithm: string;
    model: string;
    min_months_required: number;
  };
  statistics: {
    total_sites: number;
    sites_analyzed: number;
    sites_with_changepoints: number;
    detection_rate: number;
    avg_series_length: number;
    avg_span_years: number;
    sites_insufficient_data: number;
    total_sites_analyzed: number;
  };
  changepoints: {
    count: number;
    data: ChangepointSite[];
    description: string;
  };
  coverage: {
    count: number;
    data: ChangepointCoverageSite[];
    description: string;
  };
}

export interface LagCorrelationPoint {
  site_id: string;
  latitude: number;
  longitude: number;
  best_lag_months: number;
  correlation: number;
  abs_correlation: number;
  relationship: string;
  n_months_analyzed: number;
}

export interface LagCorrelationResponse {
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

export interface HotspotPoint {
  site_id: string;
  latitude: number;
  longitude: number;
  slope_m_per_year: number;
  cluster: number;
}

export interface HotspotCluster {
  cluster_id: number;
  n_sites: number;
  mean_slope: number;
  max_slope: number;
  centroid_lat: number;
  centroid_lon: number;
}

export interface HotspotsResponse {
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