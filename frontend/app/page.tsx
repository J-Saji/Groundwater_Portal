"use client";

import { useEffect, useState } from "react";
import axios from "axios";
import "leaflet/dist/leaflet.css";

import {
  StateType,
  DistrictType,
  AquiferType,
  GracePoint,
  GraceResponse,
  RainfallPoint,
  RainfallResponse,
  WellPoint,
  WellsResponse,
  WellsSummary,
  StorageResponse,
  TimeseriesResponse,
  GWRResponse,
  GWRYearsResponse,
  StorageVsGWRResponse,
  YearRangeResponse,
  ASIResponse,
  NetworkDensityResponse,
  SASSResponse,
  DivergenceResponse,
  ForecastResponse,
  RechargeResponse,
  SignificantTrendsResponse,
  ChangepointResponse,
  LagCorrelationResponse,
  HotspotsResponse,
  Geometry
} from './types';

import { BACKEND_URL } from './utils/constants';
import { useMapContext } from './hooks/useMapContext';

import Header from './components/Header';
import Sidebar from './components/Sidebar';
import IndiaMap from './components/IndiaMap';
import AdvancedModulesMenu from './components/AdvancedModulesMenu';
import AdvancedModulePanel from './components/AdvancedModulePanel';
import ChatBot from './components/ChatBot';
import LoadingSpinner from './components/LoadingSpinner';
import AlertToast from './components/AlertToast';
import Timeseries from './components/Timeseries';
import StorageVsGWR from './Components/StorageVsGWR';

import ASIModule from './components/AdvancedModules/ASIModule';
import NetworkDensityModule from './components/AdvancedModules/NetworkDensityModule';
import SASSModule from './components/AdvancedModules/SASSModule';
import DivergenceModule from './components/AdvancedModules/DivergenceModule';
import ForecastModule from './components/AdvancedModules/ForecastModule';
import RechargeModule from './components/AdvancedModules/RechargeModule';
import SignificantTrendsModule from './components/AdvancedModules/SignificantTrendsModule';
import ChangepointsModule from './components/AdvancedModules/ChangepointsModule';
import LagCorrelationModule from './components/AdvancedModules/LagCorrelationModule';
import HotspotsModule from './components/AdvancedModules/HotspotsModule';

export default function Home() {
  // State management
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
  const [gwrData, setGwrData] = useState<GWRResponse | null>(null);
  const [gwrYearRange, setGwrYearRange] = useState<{ min: number; max: number } | null>(null);
  const [storageVsGwrData, setStorageVsGwrData] = useState<StorageVsGWRResponse | null>(null);

  const [showGWR, setShowGWR] = useState<boolean>(false);
  const [showStorageVsGWR, setShowStorageVsGWR] = useState<boolean>(false);

  // Advanced Module States
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
  const [isSidebarExpanded, setIsSidebarExpanded] = useState(true);
  const [isTimeseriesMinimized, setIsTimeseriesMinimized] = useState(false);
  const [isStorageVsGWRMinimized, setIsStorageVsGWRMinimized] = useState(false);

  const showAlert = (message: string, type: "info" | "warning" | "error" | "success" = "info") => {
    setAlertMessage(message);
    setAlertType(type);
  };

  // Build map context using custom hook
  const mapContext = useMapContext({
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
  });

  // Advanced Module Loading
  const loadAdvancedModule = async (moduleName: string) => {
    setIsLoading(true);
    const params: any = {};
    if (selectedState) params.state = selectedState;
    if (selectedDistrict) params.district = selectedDistrict;

    try {
      switch (moduleName) {
        case 'ASI':
          const asiRes = await axios.get<ASIResponse>(`${BACKEND_URL}/api/advanced/asi`, { params });
          setAsiData(asiRes.data);
          showAlert(`ASI Analysis: ${asiRes.data.count} aquifer polygons analyzed`, "success");
          break;

        case 'NETWORK_DENSITY':
          const netRes = await axios.get<NetworkDensityResponse>(`${BACKEND_URL}/api/advanced/network-density`, { params });
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
          const sassRes = await axios.get<SASSResponse>(`${BACKEND_URL}/api/advanced/sass`, { params });
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
          const divRes = await axios.get<DivergenceResponse>(`${BACKEND_URL}/api/advanced/grace-divergence`, { params });
          setDivergenceData(divRes.data);
          showAlert(`Divergence: ${divRes.data.count} pixels analyzed`, "success");
          break;

        case 'FORECAST':
          const foreRes = await axios.get<ForecastResponse>(`${BACKEND_URL}/api/advanced/forecast`, { params });
          setForecastData(foreRes.data);
          showAlert(`Forecast: ${foreRes.data.count} grid cells predicted`, "success");
          break;

        case 'RECHARGE':
          if (!selectedMonth) {
            showAlert("Please select a month for site-specific recharge recommendations", "warning");
            params.year = selectedYear;
          } else {
            params.year = selectedYear;
            params.month = selectedMonth;
          }
          const rechRes = await axios.get<RechargeResponse>(`${BACKEND_URL}/api/advanced/recharge-planning`, { params });
          setRechargeData(rechRes.data);
          showAlert(`Recharge: ${rechRes.data.potential.total_recharge_potential_mcm.toFixed(2)} MCM potential`, "success");
          break;

        case 'SIGNIFICANT_TRENDS':
          const trendRes = await axios.get<SignificantTrendsResponse>(`${BACKEND_URL}/api/advanced/significant-trends`, { params });
          setSignificantTrendsData(trendRes.data);
          showAlert(`Significant Trends: ${trendRes.data.count} sites found`, "success");
          break;

        case 'CHANGEPOINTS':
          const cpRes = await axios.get<ChangepointResponse>(`${BACKEND_URL}/api/advanced/changepoints`, { params });
          setChangepointsData(cpRes.data);
          showAlert(`Changepoints: ${cpRes.data.changepoints.count} sites with breaks`, "success");
          break;

        case 'LAG_CORRELATION':
          const lagRes = await axios.get<LagCorrelationResponse>(`${BACKEND_URL}/api/advanced/lag-correlation`, { params });
          setLagCorrelationData(lagRes.data);
          showAlert(`Lag Correlation: ${lagRes.data.count} sites analyzed`, "success");
          break;

        case 'HOTSPOTS':
          const hotRes = await axios.get<HotspotsResponse>(`${BACKEND_URL}/api/advanced/hotspots`, { params });
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

  // Initial data fetching
  useEffect(() => {
    axios.get<YearRangeResponse>(`${BACKEND_URL}/api/wells/years`)
      .then(res => {
        setWellYearRange({ min: res.data.min_year, max: res.data.max_year });
        setSelectedYear(res.data.max_year);
      })
      .catch(() => setWellYearRange({ min: 1994, max: 2024 }));
  }, []);

  useEffect(() => {
    axios.get<StateType[]>(`${BACKEND_URL}/api/states`)
      .then(res => setStates(res.data))
      .catch(err => showAlert("Error loading states", "error"));
  }, []);

  useEffect(() => {
    axios.get<GWRYearsResponse>(`${BACKEND_URL}/api/gwr/available-years`)
      .then(res => {
        if (res.data.status === 'success') {
          setGwrYearRange({ min: res.data.min_year, max: res.data.max_year });
        }
      })
      .catch(() => console.log("GWR years not available"));
  }, []);

  // GWR data loading
  useEffect(() => {
    if (!showGWR) {
      setGwrData(null);
      return;
    }

    if (gwrYearRange && (selectedYear < gwrYearRange.min || selectedYear > gwrYearRange.max)) {
      showAlert(`GWR data only available for ${gwrYearRange.min}-${gwrYearRange.max}. Switching to ${gwrYearRange.max}.`, "warning");
      setSelectedYear(gwrYearRange.max);
      return;
    }

    setIsLoading(true);
    const params: any = { year: selectedYear, clip_to_boundary: true };
    if (selectedState) params.state = selectedState;
    if (selectedDistrict) params.district = selectedDistrict;

    axios.get<GWRResponse>(`${BACKEND_URL}/api/gwr`, { params })
      .then(res => {
        setGwrData(res.data);
        showAlert(`Loaded ${res.data.count} GWR polygons`, "success");
      })
      .catch(err => showAlert(err.response?.data?.detail || "Error loading GWR data", "error"))
      .finally(() => setIsLoading(false));
  }, [showGWR, selectedYear, selectedState, selectedDistrict, gwrYearRange]);

  // Storage vs GWR
  useEffect(() => {
    if (!showStorageVsGWR) {
      setStorageVsGwrData(null);
      return;
    }
    setIsLoading(true);
    const params: any = {};
    if (selectedState) params.state = selectedState;
    if (selectedDistrict) params.district = selectedDistrict;

    axios.get<StorageVsGWRResponse>(`${BACKEND_URL}/api/wells/storage-vs-gwr`, { params })
      .then(res => {
        setStorageVsGwrData(res.data);
        showAlert("Loaded storage vs GWR comparison", "success");
      })
      .catch(err => showAlert("Error loading storage vs GWR data", "error"))
      .finally(() => setIsLoading(false));
  }, [showStorageVsGWR, selectedState, selectedDistrict]);

  // State/District handling
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
    axios.get<DistrictType[]>(`${BACKEND_URL}/api/districts/${selectedState}`)
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
      .catch(err => showAlert("Error loading districts", "error"))
      .finally(() => setIsLoading(false));
  }, [selectedState]);

  useEffect(() => {
    if (!selectedDistrict || !selectedState) {
      if (selectedState && !selectedDistrict) setDistrictGeo(null);
      return;
    }

    setIsLoading(true);
    axios.get(`${BACKEND_URL}/api/district/${selectedState}/${selectedDistrict}`)
      .then(res => {
        setDistrictGeo(res.data.geometry);
        if (res.data.center) {
          setCenter(res.data.center);
          setZoom(9);
          setMapKey(prev => prev + 1);
        }
      })
      .catch(err => showAlert("Error loading boundary", "error"))
      .finally(() => setIsLoading(false));
  }, [selectedDistrict, selectedState]);

  // Aquifers loading
  useEffect(() => {
    if (!showAquifers) {
      setAquifers([]);
      return;
    }
    if (selectedDistrict && selectedState) {
      setIsLoading(true);
      axios.get<AquiferType[]>(`${BACKEND_URL}/api/aquifers/district/${selectedState}/${selectedDistrict}`)
        .then(res => {
          setAquifers(res.data);
          showAlert(`Loaded ${res.data.length} aquifers`, "success");
        })
        .catch(err => showAlert("Error loading aquifers", "error"))
        .finally(() => setIsLoading(false));
    } else if (selectedState) {
      setIsLoading(true);
      axios.get<AquiferType[]>(`${BACKEND_URL}/api/aquifers/state/${selectedState}`)
        .then(res => {
          setAquifers(res.data);
          showAlert(`Loaded ${res.data.length} aquifers`, "success");
        })
        .catch(err => showAlert("Error loading aquifers", "error"))
        .finally(() => setIsLoading(false));
    }
  }, [showAquifers, selectedState, selectedDistrict]);

  // GRACE loading
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

    axios.get<GraceResponse>(`${BACKEND_URL}/api/grace`, { params })
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

  // Rainfall loading
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

    axios.get<RainfallResponse>(`${BACKEND_URL}/api/rainfall`, { params })
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

  // Wells loading
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

    axios.get<WellsResponse>(`${BACKEND_URL}/api/wells`, { params })
      .then(res => {
        setWellsResponse(res.data);
        setWellsData(res.data.wells);
        showAlert(`Loaded ${res.data.count} wells`, "success");
      })
      .catch(err => showAlert("Error loading wells data", "error"))
      .finally(() => setIsLoading(false));
  }, [showWells, selectedYear, selectedMonth, selectedSeason, selectedState, selectedDistrict]);

  // Timeseries loading
  useEffect(() => {
    if (!showTimeseries) {
      setTimeseriesResponse(null);
      return;
    }
    setIsLoading(true);
    const params: any = { view: timeseriesView };
    if (selectedState) params.state = selectedState;
    if (selectedDistrict) params.district = selectedDistrict;

    axios.get<TimeseriesResponse>(`${BACKEND_URL}/api/wells/timeseries`, { params })
      .then(res => {
        setTimeseriesResponse(res.data);
        showAlert(`Loaded unified timeseries: ${res.data.count} points`, "success");
      })
      .catch(err => showAlert("Error loading timeseries", "error"))
      .finally(() => setIsLoading(false));
  }, [showTimeseries, timeseriesView, selectedState, selectedDistrict]);

  const graceYears = Array.from({ length: 24 }, (_, i) => 2002 + i);
  const rainfallYears = Array.from({ length: 31 }, (_, i) => 1994 + i);
  const wellYears = Array.from(
    { length: wellYearRange.max - wellYearRange.min + 1 },
    (_, i) => wellYearRange.min + i
  );

  const gwrYears = gwrYearRange
    ? Array.from({ length: gwrYearRange.max - gwrYearRange.min + 1 },
      (_, i) => gwrYearRange.min + i)
    : [];

  const availableYears = showGWR ? gwrYears :
    (showGrace ? graceYears :
      (showWells ? wellYears : rainfallYears));

  const handleAdvancedModuleSelect = (module: string) => {
    setSelectedAdvancedModule(module);
    setShowAdvancedMenu(false);
  };

  // Get module name for display
  const getModuleName = (moduleId: string): string => {
    const moduleNames: { [key: string]: string } = {
      'ASI': 'Aquifer Stress Index',
      'NETWORK_DENSITY': 'Network Density',
      'SASS': 'SASS Analysis',
      'GRACE_DIVERGENCE': 'GRACE Divergence',
      'FORECAST': 'GW Forecast',
      'RECHARGE': 'Recharge Planning',
      'SIGNIFICANT_TRENDS': 'Trend Analysis',
      'CHANGEPOINTS': 'Changepoints',
      'LAG_CORRELATION': 'Lag Correlation',
      'HOTSPOTS': 'Hotspots',
    };
    return moduleNames[moduleId] || moduleId;
  };

  // Render Advanced Module Content
  const renderAdvancedModuleContent = () => {
    if (!selectedAdvancedModule) return null;

    if (selectedAdvancedModule === 'ASI' && asiData) {
      return <ASIModule data={asiData} center={center} zoom={zoom} mapKey={mapKey} districtGeo={districtGeo} />;
    }

    if (selectedAdvancedModule === 'NETWORK_DENSITY' && networkDensityData) {
      return <NetworkDensityModule data={networkDensityData} center={center} zoom={zoom} mapKey={mapKey} districtGeo={districtGeo} />;
    }

    if (selectedAdvancedModule === 'SASS' && sassData) {
      return <SASSModule data={sassData} center={center} zoom={zoom} mapKey={mapKey} districtGeo={districtGeo} />;
    }

    if (selectedAdvancedModule === 'GRACE_DIVERGENCE' && divergenceData) {
      return <DivergenceModule data={divergenceData} center={center} zoom={zoom} mapKey={mapKey} districtGeo={districtGeo} />;
    }

    if (selectedAdvancedModule === 'FORECAST' && forecastData) {
      return <ForecastModule data={forecastData} center={center} zoom={zoom} mapKey={mapKey} districtGeo={districtGeo} />;
    }

    if (selectedAdvancedModule === 'RECHARGE' && rechargeData) {
      return <RechargeModule data={rechargeData} center={center} zoom={zoom} mapKey={mapKey} districtGeo={districtGeo} />;
    }

    if (selectedAdvancedModule === 'SIGNIFICANT_TRENDS' && significantTrendsData) {
      return <SignificantTrendsModule data={significantTrendsData} center={center} zoom={zoom} mapKey={mapKey} districtGeo={districtGeo} />;
    }

    if (selectedAdvancedModule === 'CHANGEPOINTS' && changepointsData) {
      return <ChangepointsModule data={changepointsData} center={center} zoom={zoom} mapKey={mapKey} districtGeo={districtGeo} />;
    }

    if (selectedAdvancedModule === 'LAG_CORRELATION' && lagCorrelationData) {
      return <LagCorrelationModule data={lagCorrelationData} center={center} zoom={zoom} mapKey={mapKey} districtGeo={districtGeo} />;
    }

    if (selectedAdvancedModule === 'HOTSPOTS' && hotspotsData) {
      return <HotspotsModule data={hotspotsData} center={center} zoom={zoom} mapKey={mapKey} districtGeo={districtGeo} />;
    }
    return null;
  };

  // Determine if split view is active (advanced module showing)
  const isSplitViewActive = !!selectedAdvancedModule;

  return (
    <div className="min-h-screen bg-slate-100">
      {alertMessage && (
        <AlertToast
          message={alertMessage}
          type={alertType}
          onClose={() => setAlertMessage("")}
        />
      )}

      {/* Header */}
      <Header onChatToggle={() => setIsChatOpen(!isChatOpen)} isChatOpen={isChatOpen} />

      {/* Left Sidebar */}
      <Sidebar
        isExpanded={isSidebarExpanded}
        onToggle={() => setIsSidebarExpanded(!isSidebarExpanded)}
        states={states}
        districts={districts}
        selectedState={selectedState}
        selectedDistrict={selectedDistrict}
        selectedYear={selectedYear}
        selectedMonth={selectedMonth}
        availableYears={availableYears}
        showAquifers={showAquifers}
        showGWR={showGWR}
        showGrace={showGrace}
        showRainfall={showRainfall}
        showWells={showWells}
        showTimeseries={showTimeseries}
        showStorageVsGWR={showStorageVsGWR}
        onStateChange={setSelectedState}
        onDistrictChange={setSelectedDistrict}
        onYearChange={setSelectedYear}
        onMonthChange={setSelectedMonth}
        onToggleAquifers={() => setShowAquifers(!showAquifers)}
        onToggleGWR={() => setShowGWR(!showGWR)}
        onToggleGrace={() => setShowGrace(!showGrace)}
        onToggleRainfall={() => setShowRainfall(!showRainfall)}
        onToggleWells={() => setShowWells(!showWells)}
        onToggleTimeseries={() => setShowTimeseries(!showTimeseries)}
        onToggleStorageVsGWR={() => setShowStorageVsGWR(!showStorageVsGWR)}
      />

      {/* Advanced Modules Menu Icon (Top Right) */}
      <button
        onClick={() => setShowAdvancedMenu(true)}
        className="fixed top-16 right-6 w-11 h-11 bg-slate-700 hover:bg-slate-600 text-white rounded-lg shadow-lg hover:shadow-xl transition-all z-40 flex items-center justify-center border border-slate-600"
        title="Advanced Analytics"
      >
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      </button>

      {/* Advanced Modules Menu */}
      <AdvancedModulesMenu
        isOpen={showAdvancedMenu}
        onClose={() => setShowAdvancedMenu(false)}
        onSelectModule={handleAdvancedModuleSelect}
        selectedModule={selectedAdvancedModule}
      />

      {/* Main Content Area */}
      <div
        className={`transition-all duration-300 ${showTimeseries || showStorageVsGWR
            ? 'p-4'
            : (showAquifers || showGWR || showGrace || showRainfall || showWells || selectedAdvancedModule ? 'p-2' : '')
          } ${isSidebarExpanded ? 'ml-80' : 'ml-14'}`}
        style={{ height: 'calc(100vh - 3.5rem)' }}
      >
        {(showAquifers || showGWR || showGrace || showRainfall || showWells || selectedAdvancedModule) ? (
          // Split View: India Map (Left) + Layer-Specific Map or Advanced Module (Right)
          <div className="h-full grid grid-cols-2 gap-4">
            {/* Left: India Map - Shows layers when advanced module is open, otherwise plain overview */}
            <div className="h-full relative" style={{ zIndex: 1 }}>
              <div className="h-full w-full rounded-lg overflow-hidden shadow-lg border border-slate-200 relative">
                <IndiaMap
                  center={selectedAdvancedModule && (showAquifers || showGWR || showGrace || showRainfall || showWells) ? center : [22.9734, 78.6569]}
                  zoom={selectedAdvancedModule && (showAquifers || showGWR || showGrace || showRainfall || showWells) ? zoom : 5}
                  mapKey={selectedAdvancedModule && (showAquifers || showGWR || showGrace || showRainfall || showWells) ? mapKey : 0}
                  districtGeo={selectedAdvancedModule && (showAquifers || showGWR || showGrace || showRainfall || showWells) ? districtGeo : null}
                  aquifers={selectedAdvancedModule && showAquifers ? aquifers : []}
                  graceData={selectedAdvancedModule && showGrace ? graceData : []}
                  rainfallData={selectedAdvancedModule && showRainfall ? rainfallData : []}
                  wellsData={selectedAdvancedModule && showWells ? wellsData : []}
                  gwrData={selectedAdvancedModule && showGWR ? gwrData : null}
                  showAquifers={selectedAdvancedModule ? showAquifers : false}
                  showGrace={selectedAdvancedModule ? showGrace : false}
                  showRainfall={selectedAdvancedModule ? showRainfall : false}
                  showWells={selectedAdvancedModule ? showWells : false}
                  showGWR={selectedAdvancedModule ? showGWR : false}
                />
                <div className="absolute top-3 left-3 bg-white px-3 py-1.5 rounded-md shadow-md border border-slate-300">
                  <p className="text-xs font-semibold text-slate-700">
                    {selectedAdvancedModule && (showAquifers || showGWR || showGrace || showRainfall || showWells) ? (
                      <>
                        {showAquifers && 'Aquifers Layer'}
                        {showGWR && 'GWR Layer'}
                        {showGrace && 'GRACE Layer'}
                        {showRainfall && 'Rainfall Layer'}
                        {showWells && 'Wells Layer'}
                        {selectedState && ` - ${selectedState}`}
                        {selectedDistrict && ` / ${selectedDistrict}`}
                      </>
                    ) : (
                      'India Overview'
                    )}
                  </p>
                </div>
              </div>
            </div>

            {/* Right: Zoomed Map with Selected Layer OR Advanced Module */}
            <div className="h-full relative" style={{ zIndex: 10 }}>
              {selectedAdvancedModule ? (
                // Advanced Module Panel
                <AdvancedModulePanel
                  moduleName={getModuleName(selectedAdvancedModule)}
                  onClose={() => setSelectedAdvancedModule('')}
                >
                  {renderAdvancedModuleContent()}
                </AdvancedModulePanel>
              ) : (
                // Regular Layer Map (Zoomed to selected state/district)
                <div className="h-full w-full rounded-lg overflow-hidden shadow-lg border border-blue-300 relative">
                  <IndiaMap
                    center={center}
                    zoom={zoom}
                    mapKey={mapKey}
                    districtGeo={districtGeo}
                    aquifers={showAquifers ? aquifers : []}
                    graceData={showGrace ? graceData : []}
                    rainfallData={showRainfall ? rainfallData : []}
                    wellsData={showWells ? wellsData : []}
                    gwrData={showGWR ? gwrData : null}
                    showAquifers={showAquifers}
                    showGrace={showGrace}
                    showRainfall={showRainfall}
                    showWells={showWells}
                    showGWR={showGWR}
                  />
                  <div className="absolute top-3 left-3 bg-white px-3 py-1.5 rounded-md shadow-md border border-blue-400">
                    <p className="text-xs font-semibold text-blue-900">
                      {showAquifers && 'Aquifers Layer'}
                      {showGWR && 'GWR Layer'}
                      {showGrace && 'GRACE Layer'}
                      {showRainfall && 'Rainfall Layer'}
                      {showWells && 'Wells Layer'}
                      {selectedState && ` - ${selectedState}`}
                      {selectedDistrict && ` / ${selectedDistrict}`}
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        ) : (
          // No Layers Active: Full View India Map
          <div className="h-full">
            <IndiaMap
              center={[22.9734, 78.6569]}
              zoom={5}
              mapKey={0}
              districtGeo={null}
              aquifers={[]}
              graceData={[]}
              rainfallData={[]}
              wellsData={[]}
              gwrData={null}
              showAquifers={false}
              showGrace={false}
              showRainfall={false}
              showWells={false}
              showGWR={false}
            />
          </div>
        )}

        {/* Timeseries Chart - Inline (when no split view) */}
        {!isSplitViewActive && showTimeseries && timeseriesResponse && (
          <div className="mt-4">
            <div className="bg-white rounded-lg shadow-lg border border-slate-200 overflow-hidden">
              <div className="bg-indigo-600 text-white px-4 py-2.5 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                  </svg>
                  <h3 className="font-semibold text-sm">Timeseries Analysis</h3>
                </div>
                <button
                  onClick={() => setShowTimeseries(false)}
                  className="hover:bg-indigo-700 p-1 rounded transition-colors"
                  title="Close"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              <div className="p-4">
                <Timeseries
                  timeseriesResponse={timeseriesResponse}
                  timeseriesView={timeseriesView}
                  onViewChange={setTimeseriesView}
                />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Floating Timeseries - Only when split view is active */}
      {isSplitViewActive && showTimeseries && timeseriesResponse && (
        <div className={`fixed ${isSidebarExpanded ? 'left-80' : 'left-14'} right-6 bottom-6 transition-all duration-300 z-30`}>
          {isTimeseriesMinimized ? (
            // Minimized Badge
            <button
              onClick={() => setIsTimeseriesMinimized(false)}
              className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-3 rounded-lg shadow-lg flex items-center justify-center gap-2 transition-all"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
              </svg>
              <span className="font-semibold">Show Timeseries Analysis</span>
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
              </svg>
            </button>
          ) : (
            // Expanded Panel
            <div className="bg-white rounded-lg shadow-2xl border border-slate-200 overflow-hidden max-h-[60vh]">
              <div className="bg-indigo-600 text-white px-4 py-2.5 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                  </svg>
                  <h3 className="font-semibold text-sm">Timeseries Analysis</h3>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setIsTimeseriesMinimized(true)}
                    className="hover:bg-indigo-700 p-1 rounded transition-colors"
                    title="Minimize"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>
                  <button
                    onClick={() => setShowTimeseries(false)}
                    className="hover:bg-indigo-700 p-1 rounded transition-colors"
                    title="Close"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              </div>
              <div className="p-4 overflow-y-auto max-h-[50vh]">
                <Timeseries
                  timeseriesResponse={timeseriesResponse}
                  timeseriesView={timeseriesView}
                  onViewChange={setTimeseriesView}
                />
              </div>
            </div>
          )}
        </div>
      )}
      {/* Storage vs GWR Chart - Floating Panel */}
      {showStorageVsGWR && storageVsGwrData && (
        <div className={`fixed ${isSidebarExpanded ? 'left-80' : 'left-14'} bottom-6 transition-all duration-300 z-30 ${isStorageVsGWRMinimized ? 'w-auto' : 'right-6'}`}>
          {isStorageVsGWRMinimized ? (
            // Minimized Badge
            <button
              onClick={() => setIsStorageVsGWRMinimized(false)}
              className="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg shadow-lg flex items-center gap-2 transition-all"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              <span className="font-medium text-sm">Storage vs GWR</span>
            </button>
          ) : (
            // Expanded Panel
            <div className="bg-white rounded-lg shadow-2xl border border-slate-300 overflow-hidden">
              <div className="bg-purple-600 text-white px-4 py-2.5 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                  <h3 className="font-semibold text-sm">Storage vs GWR Comparison</h3>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setIsStorageVsGWRMinimized(true)}
                    className="hover:bg-purple-700 p-1 rounded transition-colors"
                    title="Minimize"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
                    </svg>
                  </button>
                  <button
                    onClick={() => setShowStorageVsGWR(false)}
                    className="hover:bg-purple-700 p-1 rounded transition-colors"
                    title="Close"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              </div>
              <div className="p-4 max-h-96 overflow-y-auto">
                <StorageVsGWR storageVsGwrData={storageVsGwrData} />
              </div>
            </div>
          )}
        </div>
      )}

      {/* ChatBot - Positioned at bottom right */}
      <ChatBot
        isOpen={isChatOpen}
        onClose={() => setIsChatOpen(false)}
        mapContext={mapContext}
        backendURL={BACKEND_URL}
        selectedAdvancedModule={selectedAdvancedModule}
      />

      {/* Loading Spinner */}
      {isLoading && <LoadingSpinner />}
    </div>
  );
}