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
    if (aquifers && aquifers.length > 0) {
      // Group by aquifer TYPE (Confined/Unconfined/Semi-confined)
      const aquiferAreaByType: { [key: string]: number } = {};
      const lithologyByType: { [key: string]: { [lithology: string]: number } } = {};

      aquifers.forEach(a => {
        const type = a.aquifers || a.aquifer || 'Unknown';
        const lithology = a.aquifer || 'Unknown'; // Lithology name

        // Sum area by aquifer type
        aquiferAreaByType[type] = (aquiferAreaByType[type] || 0) + (a.area_sqm || 0);

        // Track lithology breakdown within each type
        if (!lithologyByType[type]) {
          lithologyByType[type] = {};
        }
        lithologyByType[type][lithology] = (lithologyByType[type][lithology] || 0) + (a.area_sqm || 0);
      });

      // Find dominant type by area (not count)
      const dominantByArea = Object.entries(aquiferAreaByType)
        .sort((a, b) => b[1] - a[1])[0];

      const dominantType = dominantByArea?.[0] || 'Unknown';
      const dominantArea = dominantByArea?.[1] || 0;

      // Get lithology breakdown for dominant type
      const dominantLithologies = lithologyByType[dominantType] || {};
      const topLithologies = Object.entries(dominantLithologies)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3)  // Top 3 lithologies
        .map(([lith, area]) => ({
          name: lith,
          area_km2: (area / 1_000_000).toFixed(2)
        }));

      // DEBUG: Log aquifer areas
      console.log('🔍 Aquifer Area Calculation:', {
        totalAquifers: aquifers.length,
        areaByType: Object.entries(aquiferAreaByType)
          .map(([type, area]) => ({ type, area_sqm: area, area_km2: (area / 1_000_000).toFixed(2) }))
          .sort((a, b) => b.area_sqm - a.area_sqm),
        lithologyBreakdown: Object.entries(lithologyByType).map(([type, lithos]) => ({
          aquifer_type: type,
          lithologies: Object.entries(lithos).map(([lith, area]) => ({
            lithology: lith,
            area_km2: (area / 1_000_000).toFixed(2)
          })).sort((a, b) => parseFloat(b.area_km2) - parseFloat(a.area_km2))
        })),
        sampleAquifer: aquifers[0]
      });

      console.log('✅ Dominant Aquifer:', dominantType, `(${(dominantArea / 1_000_000).toFixed(2)} km²)`);
      console.log('   Top Lithologies:', topLithologies.map(l => `${l.name} (${l.area_km2} km²)`).join(', '));

      dataSummary.aquifers = {
        count: aquifers.length,
        types: Object.entries(aquiferAreaByType).map(([type, area]) => ({
          type,
          area_sqm: area,
          area_sqkm: area / 1_000_000
        })),
        dominant_type: dominantType,
        dominant_area_sqkm: dominantArea / 1_000_000,
        dominant_lithologies: topLithologies,  // NEW: Top lithologies in dominant type
        total_area_sqkm: Object.values(aquiferAreaByType).reduce((sum, a) => sum + a, 0) / 1_000_000,
        avg_zone_m: aquifers.reduce((sum, a) => sum + (a.zone_m || 0), 0) / (aquifers.length || 1),
        avg_mbgl: aquifers.reduce((sum, a) => sum + (a.avg_mbgl || 0), 0) / (aquifers.length || 1)
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
        methodology: asiData.methodology,
        interpretation: asiData.interpretation,
        key_insights: asiData.key_insights
      };
    }

    if (selectedAdvancedModule === 'NETWORK_DENSITY' && networkDensityData) {
      dataSummary.network_density = {
        module: networkDensityData.module,
        statistics: networkDensityData.statistics,
        site_level_count: networkDensityData.map1_site_level.count,
        grid_count: networkDensityData.map2_gridded.count,
        methodology: networkDensityData.methodology,
        interpretation: networkDensityData.interpretation,
        key_insights: networkDensityData.key_insights
      };
    }

    if (selectedAdvancedModule === 'SASS' && sassData) {
      dataSummary.sass = {
        module: sassData.module,
        formula: sassData.formula,
        statistics: sassData.statistics,
        sites_analyzed: sassData.count,
        methodology: sassData.methodology,
        interpretation: sassData.interpretation,
        key_insights: sassData.key_insights
      };
    }

    if (selectedAdvancedModule === 'GRACE_DIVERGENCE' && divergenceData) {
      dataSummary.divergence = {
        module: divergenceData.module,
        statistics: divergenceData.statistics,
        pixels_analyzed: divergenceData.count,
        methodology: divergenceData.methodology,
        interpretation: divergenceData.interpretation,
        key_insights: divergenceData.key_insights
      };
    }

    if (selectedAdvancedModule === 'FORECAST' && forecastData) {
      dataSummary.forecast = {
        module: forecastData.module,
        method: forecastData.method,
        statistics: forecastData.statistics,
        forecast_months: forecastData.parameters.forecast_months,
        grid_cells: forecastData.count,
        methodology: forecastData.methodology,
        interpretation: forecastData.interpretation,
        key_insights: forecastData.key_insights
      };
    }

    if (selectedAdvancedModule === 'RECHARGE' && rechargeData) {
      dataSummary.recharge = {
        module: rechargeData.module,
        potential: rechargeData.potential,
        analysis_parameters: rechargeData.analysis_parameters,
        structure_plan: rechargeData.structure_plan,
        methodology: rechargeData.methodology,
        interpretation: rechargeData.interpretation,
        key_insights: rechargeData.key_insights
      };
    }

    if (selectedAdvancedModule === 'SIGNIFICANT_TRENDS' && significantTrendsData) {
      dataSummary.significant_trends = {
        module: significantTrendsData.module,
        statistics: significantTrendsData.statistics,
        p_threshold: significantTrendsData.parameters.p_threshold,
        methodology: significantTrendsData.methodology,
        interpretation: significantTrendsData.interpretation,
        key_insights: significantTrendsData.key_insights
      };
    }

    if (selectedAdvancedModule === 'CHANGEPOINTS' && changepointsData) {
      dataSummary.changepoints = {
        module: changepointsData.module,
        statistics: changepointsData.statistics,
        changepoints_found: changepointsData.changepoints.count,
        methodology: changepointsData.methodology,
        interpretation: changepointsData.interpretation,
        key_insights: changepointsData.key_insights
      };
    }

    if (selectedAdvancedModule === 'LAG_CORRELATION' && lagCorrelationData) {
      dataSummary.lag_correlation = {
        module: lagCorrelationData.module,
        statistics: lagCorrelationData.statistics,
        sites_analyzed: lagCorrelationData.count,
        methodology: lagCorrelationData.methodology,
        interpretation: lagCorrelationData.interpretation,
        key_insights: lagCorrelationData.key_insights
      };
    }

    if (selectedAdvancedModule === 'HOTSPOTS' && hotspotsData) {
      dataSummary.hotspots = {
        module: hotspotsData.module,
        statistics: hotspotsData.statistics,
        clusters: hotspotsData.clusters,
        methodology: hotspotsData.methodology,
        interpretation: hotspotsData.interpretation,
        key_insights: hotspotsData.key_insights
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