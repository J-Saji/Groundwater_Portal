import { useCallback } from 'react';
import { 
  MapContext, 
  AquiferType, 
  GraceResponse, 
  GracePoint,
  RainfallResponse,
  RainfallPoint,
  WellsResponse,
  WellPoint,
  WellsSummary,
  TimeseriesResponse,
  GWRResponse,
  StorageVsGWRResponse,
  ASIResponse,
  NetworkDensityResponse,
  SASSResponse,
  DivergenceResponse,
  ForecastResponse,
  RechargeResponse,
  SignificantTrendsResponse,
  ChangepointResponse,
  LagCorrelationResponse,
  HotspotsResponse
} from '../types';

interface UseMapContextProps {
  showAquifers: boolean;
  showGrace: boolean;
  showRainfall: boolean;
  showWells: boolean;
  showGWR: boolean;
  showStorageVsGWR: boolean;
  showTimeseries: boolean;
  aquifers: AquiferType[];
  graceResponse: GraceResponse | null;
  graceData: GracePoint[];
  rainfallResponse: RainfallResponse | null;
  rainfallData: RainfallPoint[];
  wellsResponse: WellsResponse | null;
  wellsData: WellPoint[];
  summaryData: WellsSummary | null;
  timeseriesResponse: TimeseriesResponse | null;
  gwrData: GWRResponse | null;
  storageVsGwrData: StorageVsGWRResponse | null;
  selectedAdvancedModule: string;
  asiData: ASIResponse | null;
  networkDensityData: NetworkDensityResponse | null;
  sassData: SASSResponse | null;
  divergenceData: DivergenceResponse | null;
  forecastData: ForecastResponse | null;
  rechargeData: RechargeResponse | null;
  significantTrendsData: SignificantTrendsResponse | null;
  changepointsData: ChangepointResponse | null;
  lagCorrelationData: LagCorrelationResponse | null;
  hotspotsData: HotspotsResponse | null;
  selectedState: string;
  selectedDistrict: string;
  selectedYear: number;
  selectedMonth: number | null;
  selectedSeason: string;
}

export function useMapContext({
  showAquifers,
  showGrace,
  showRainfall,
  showWells,
  showGWR,
  showStorageVsGWR,
  showTimeseries,
  aquifers,
  graceResponse,
  graceData,
  rainfallResponse,
  rainfallData,
  wellsResponse,
  wellsData,
  summaryData,
  timeseriesResponse,
  gwrData,
  storageVsGwrData,
  selectedAdvancedModule,
  asiData,
  networkDensityData,
  sassData,
  divergenceData,
  forecastData,
  rechargeData,
  significantTrendsData,
  changepointsData,
  lagCorrelationData,
  hotspotsData,
  selectedState,
  selectedDistrict,
  selectedYear,
  selectedMonth,
  selectedSeason
}: UseMapContextProps): MapContext {
  
  const buildMapContext = useCallback((): MapContext => {
    const activeLayers: string[] = [];
    if (showAquifers) activeLayers.push("aquifers");
    if (showGrace) activeLayers.push("grace");
    if (showRainfall) activeLayers.push("rainfall");
    if (showWells) activeLayers.push("wells");
    if (showGWR) activeLayers.push("gwr");

    const dataSummary: any = {};

    // Aquifers context
    if (showAquifers && aquifers.length > 0) {
      const aquiferTypes = aquifers.reduce((acc, a) => {
        acc[a.aquifer] = (acc[a.aquifer] || 0) + 1;
        return acc;
      }, {} as Record<string, number>);

      dataSummary.aquifers = {
        count: aquifers.length,
        types: Object.entries(aquiferTypes).map(([type, count]) => ({ type, count })),
        dominant_type: Object.entries(aquiferTypes).sort((a, b) => b[1] - a[1])[0]?.[0],
        avg_zone_m: aquifers.reduce((sum, a) => sum + (a.zone_m || 0), 0) / aquifers.length,
        avg_mbgl: aquifers.reduce((sum, a) => sum + (a.avg_mbgl || 0), 0) / aquifers.length
      };
    }

    // GRACE context
    if (showGrace && graceResponse) {
      const graceValues = graceData.map(p => p.lwe_cm);
      
      dataSummary.grace = {
        year: graceResponse.year,
        month: graceResponse.month,
        description: graceResponse.description,
        regional_average_cm: graceResponse.regional_average_cm,
        data_points: graceResponse.count,
        statistics: {
          mean_lwe: graceValues.length > 0 ? (graceValues.reduce((a, b) => a + b, 0) / graceValues.length).toFixed(3) : null,
          min_lwe: graceValues.length > 0 ? Math.min(...graceValues).toFixed(3) : null,
          max_lwe: graceValues.length > 0 ? Math.max(...graceValues).toFixed(3) : null
        },
        spatial_distribution: graceData.length > 0 ? {
          pixels_positive: graceData.filter(p => p.lwe_cm > 0).length,
          pixels_negative: graceData.filter(p => p.lwe_cm < 0).length,
          pixels_critical_low: graceData.filter(p => p.lwe_cm < -10).length
        } : null
      };
    }

    // Rainfall context
    if (showRainfall && rainfallResponse) {
      const rainfallValues = rainfallData.map(p => p.rainfall_mm);
      
      dataSummary.rainfall = {
        year: rainfallResponse.year,
        month: rainfallResponse.month,
        description: rainfallResponse.description,
        regional_average_mm_per_day: rainfallResponse.regional_average_mm_per_day,
        data_points: rainfallResponse.count,
        statistics: {
          mean_rainfall: rainfallValues.length > 0 ? (rainfallValues.reduce((a, b) => a + b, 0) / rainfallValues.length).toFixed(2) : null,
          min_rainfall: rainfallValues.length > 0 ? Math.min(...rainfallValues).toFixed(2) : null,
          max_rainfall: rainfallValues.length > 0 ? Math.max(...rainfallValues).toFixed(2) : null
        }
      };
    }

    // Wells context
    if (showWells && wellsResponse) {
      const categories = wellsData.reduce((acc, w) => {
        acc[w.gwl_category] = (acc[w.gwl_category] || 0) + 1;
        return acc;
      }, {} as Record<string, number>);

      const gwlValues = wellsData.map(w => w.gwl);
      const avgGWL = gwlValues.length > 0 ? gwlValues.reduce((sum, v) => sum + v, 0) / gwlValues.length : null;

      dataSummary.wells = {
        year: wellsResponse.filters.year,
        month: wellsResponse.filters.month,
        season: wellsResponse.filters.season,
        data_points: wellsResponse.count,
        categories: categories,
        avg_gwl: avgGWL !== null ? avgGWL.toFixed(2) : null,
        unique_sites: new Set(wellsData.map(w => w.site_id)).size
      };
    }

    // Summary context
    if (summaryData && !summaryData.error) {
      dataSummary.summary = {
        mean_gwl: summaryData.statistics.mean_gwl,
        trend_direction: summaryData.trend.trend_direction,
        slope_m_per_year: summaryData.trend.slope_m_per_year,
        temporal_coverage: summaryData.temporal_coverage
      };
    }

    // GWR context
    if (showGWR && gwrData) {
      dataSummary.gwr = {
        year: gwrData.filters.year,
        statistics: gwrData.statistics,
        polygons_count: gwrData.count
      };
    }

    // Storage vs GWR context
    if (showStorageVsGWR && storageVsGwrData) {
      dataSummary.storage_vs_gwr = {
        filters: storageVsGwrData.filters,
        aquifer_properties: storageVsGwrData.aquifer_properties,
        storage_statistics: storageVsGwrData.storage_statistics,
        gwr_statistics: storageVsGwrData.gwr_statistics
      };
    }

    // Advanced modules context
    if (selectedAdvancedModule) {
      dataSummary.active_module = selectedAdvancedModule;
    }

    if (selectedAdvancedModule === 'ASI' && asiData) {
      dataSummary.asi = {
        module: asiData.module,
        statistics: asiData.statistics,
        polygons_analyzed: asiData.count,
        methodology: asiData.methodology
      };
    }

    if (selectedAdvancedModule === 'NETWORK_DENSITY' && networkDensityData) {
      dataSummary.network_density = {
        module: networkDensityData.module,
        statistics: networkDensityData.statistics,
        site_level_count: networkDensityData.map1_site_level.count,
        grid_count: networkDensityData.map2_gridded.count
      };
    }

    if (selectedAdvancedModule === 'SASS' && sassData) {
      dataSummary.sass = {
        module: sassData.module,
        formula: sassData.formula,
        statistics: sassData.statistics,
        sites_analyzed: sassData.count
      };
    }

    if (selectedAdvancedModule === 'GRACE_DIVERGENCE' && divergenceData) {
      dataSummary.divergence = {
        module: divergenceData.module,
        statistics: divergenceData.statistics,
        pixels_analyzed: divergenceData.count
      };
    }

    if (selectedAdvancedModule === 'FORECAST' && forecastData) {
      dataSummary.forecast = {
        module: forecastData.module,
        method: forecastData.method,
        statistics: forecastData.statistics,
        forecast_months: forecastData.parameters.forecast_months,
        grid_cells: forecastData.count
      };
    }

    if (selectedAdvancedModule === 'RECHARGE' && rechargeData) {
      dataSummary.recharge = {
        module: rechargeData.module,
        potential: rechargeData.potential,
        analysis_parameters: rechargeData.analysis_parameters,
        structure_plan: rechargeData.structure_plan
      };
    }

    if (selectedAdvancedModule === 'SIGNIFICANT_TRENDS' && significantTrendsData) {
      dataSummary.significant_trends = {
        module: significantTrendsData.module,
        statistics: significantTrendsData.statistics,
        p_threshold: significantTrendsData.parameters.p_threshold
      };
    }

    if (selectedAdvancedModule === 'CHANGEPOINTS' && changepointsData) {
      dataSummary.changepoints = {
        module: changepointsData.module,
        statistics: changepointsData.statistics,
        changepoints_found: changepointsData.changepoints.count
      };
    }

    if (selectedAdvancedModule === 'LAG_CORRELATION' && lagCorrelationData) {
      dataSummary.lag_correlation = {
        module: lagCorrelationData.module,
        statistics: lagCorrelationData.statistics,
        sites_analyzed: lagCorrelationData.count
      };
    }

    if (selectedAdvancedModule === 'HOTSPOTS' && hotspotsData) {
      dataSummary.hotspots = {
        module: hotspotsData.module,
        statistics: hotspotsData.statistics,
        clusters: hotspotsData.clusters
      };
    }

    // Timeseries context
    if (showTimeseries && timeseriesResponse) {
      if (timeseriesResponse.error) {
        dataSummary.timeseries = {
          status: 'error',
          error: timeseriesResponse.error,
          view: timeseriesResponse.view
        };
      } else {
        dataSummary.timeseries = {
          status: 'success',
          view: timeseriesResponse.view,
          aggregation: timeseriesResponse.aggregation,
          data_points: timeseriesResponse.count,
          statistics: timeseriesResponse.statistics
        };
      }
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
    showAquifers, showGrace, showRainfall, showWells, showTimeseries, showGWR, showStorageVsGWR,
    aquifers, graceResponse, graceData, rainfallResponse, rainfallData, 
    wellsResponse, wellsData, summaryData, timeseriesResponse, gwrData, storageVsGwrData,
    selectedAdvancedModule, asiData, networkDensityData, sassData, 
    divergenceData, forecastData, rechargeData, significantTrendsData,
    changepointsData, lagCorrelationData, hotspotsData,
    selectedState, selectedDistrict, selectedYear, selectedMonth, selectedSeason
  ]);

  return buildMapContext();
}