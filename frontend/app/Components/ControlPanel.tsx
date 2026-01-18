import { StateType, DistrictType } from '../types';

interface ControlPanelProps {
  states: StateType[];
  districts: DistrictType[];
  selectedState: string;
  selectedDistrict: string;
  selectedYear: number;
  selectedMonth: number | null;
  showAquifers: boolean;
  showGWR: boolean;
  showStorageVsGWR: boolean;
  showGrace: boolean;
  showRainfall: boolean;
  showWells: boolean;
  showTimeseries: boolean;
  availableYears: number[];
  gwrYearRange: { min: number; max: number } | null;
  showAdvancedMenu: boolean;
  selectedAdvancedModule: string;
  onStateChange: (state: string) => void;
  onDistrictChange: (district: string) => void;
  onYearChange: (year: number) => void;
  onMonthChange: (month: number | null) => void;
  onToggleAquifers: () => void;
  onToggleGWR: () => void;
  onToggleStorageVsGWR: () => void;
  onToggleGrace: () => void;
  onToggleRainfall: () => void;
  onToggleWells: () => void;
  onToggleTimeseries: () => void;
  onToggleAdvancedMenu: () => void;
  onSelectAdvancedModule: (module: string) => void;
}

export default function ControlPanel({
  states,
  districts,
  selectedState,
  selectedDistrict,
  selectedYear,
  selectedMonth,
  showAquifers,
  showGWR,
  showStorageVsGWR,
  showGrace,
  showRainfall,
  showWells,
  showTimeseries,
  availableYears,
  gwrYearRange,
  showAdvancedMenu,
  selectedAdvancedModule,
  onStateChange,
  onDistrictChange,
  onYearChange,
  onMonthChange,
  onToggleAquifers,
  onToggleGWR,
  onToggleStorageVsGWR,
  onToggleGrace,
  onToggleRainfall,
  onToggleWells,
  onToggleTimeseries,
  onToggleAdvancedMenu,
  onSelectAdvancedModule,
}: ControlPanelProps) {
  const advancedModules = [
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
  ];

  return (
    <div className="lg:col-span-4 bg-white rounded-xl shadow-xl p-6 border-2 border-blue-200">
      <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
        <span>🎛️</span> Control Panel
      </h3>
      
      {/* Dropdowns */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">State</label>
          <select
            className="w-full p-3 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
            value={selectedState}
            onChange={(e) => {
              onStateChange(e.target.value);
              onDistrictChange("");
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
            onChange={(e) => onDistrictChange(e.target.value)}
            disabled={!selectedState}
          >
            <option value="">All Districts</option>
            {districts.map((d, i) => (
              <option key={i} value={d.district_name}>{d.district_name}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            Year
            {showGWR && gwrYearRange && (
              <span className="ml-2 text-xs text-teal-600">
                (GWR: {gwrYearRange.min}-{gwrYearRange.max} only)
              </span>
            )}
          </label>
          <select
            className="w-full p-3 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
            value={selectedYear}
            onChange={(e) => onYearChange(Number(e.target.value))}
          >
            {availableYears.map(y => {
              const gwrAvailable = !showGWR || !gwrYearRange || 
                (y >= gwrYearRange.min && y <= gwrYearRange.max);
              
              return (
                <option key={y} value={y}>
                  {y} {!gwrAvailable && showGWR ? '⚠️ (No GWR)' : ''}
                </option>
              );
            })}
          </select>
        </div>

        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">Month (Optional)</label>
          <select
            className="w-full p-3 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
            value={selectedMonth || ""}
            onChange={(e) => onMonthChange(e.target.value ? Number(e.target.value) : null)}
          >
            <option value="">All Year</option>
            {Array.from({ length: 12 }, (_, i) => i + 1).map(m => (
              <option key={m} value={m}>{new Date(2000, m - 1).toLocaleString('default', { month: 'long' })}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Buttons */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
        <button
          onClick={onToggleAquifers}
          className={`p-3 rounded-lg font-semibold transition-all duration-200 border-2 ${
            showAquifers
              ? 'bg-purple-500 text-white border-purple-600 shadow-lg'
              : 'bg-white text-gray-700 border-gray-300 hover:border-purple-400'
          }`}
        >
          🔷 Aquifers
        </button>

        <button
          onClick={onToggleGWR}
          className={`p-3 rounded-lg font-semibold transition-all duration-200 border-2 ${
            showGWR
              ? 'bg-teal-500 text-white border-teal-600 shadow-lg'
              : 'bg-white text-gray-700 border-gray-300 hover:border-teal-400'
          }`}
        >
          💦 GWR
        </button>

        <button
          onClick={onToggleStorageVsGWR}
          className={`p-3 rounded-lg font-semibold transition-all duration-200 border-2 ${
            showStorageVsGWR
              ? 'bg-cyan-500 text-white border-cyan-600 shadow-lg'
              : 'bg-white text-gray-700 border-gray-300 hover:border-cyan-400'
          }`}
        >
          📊 Storage vs GWR
        </button>

        <button
          onClick={onToggleGrace}
          className={`p-3 rounded-lg font-semibold transition-all duration-200 border-2 ${
            showGrace
              ? 'bg-green-500 text-white border-green-600 shadow-lg'
              : 'bg-white text-gray-700 border-gray-300 hover:border-green-400'
          }`}
        >
          🌊 GRACE
        </button>

        <button
          onClick={onToggleRainfall}
          className={`p-3 rounded-lg font-semibold transition-all duration-200 border-2 ${
            showRainfall
              ? 'bg-blue-500 text-white border-blue-600 shadow-lg'
              : 'bg-white text-gray-700 border-gray-300 hover:border-blue-400'
          }`}
        >
          🌧️ Rainfall
        </button>

        <button
          onClick={onToggleWells}
          className={`p-3 rounded-lg font-semibold transition-all duration-200 border-2 ${
            showWells
              ? 'bg-red-500 text-white border-red-600 shadow-lg'
              : 'bg-white text-gray-700 border-gray-300 hover:border-red-400'
          }`}
        >
          💧 Wells
        </button>

        <button
          onClick={onToggleTimeseries}
          className={`p-3 rounded-lg font-semibold transition-all duration-200 border-2 ${
            showTimeseries
              ? 'bg-indigo-500 text-white border-indigo-600 shadow-lg'
              : 'bg-white text-gray-700 border-gray-300 hover:border-indigo-400'
          }`}
        >
          📈 Timeseries
        </button>

        {/* Advanced Modules Dropdown */}
        <div className="relative">
          <button
            onClick={onToggleAdvancedMenu}
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
                {advancedModules.map((module) => (
                  <button
                    key={module.id}
                    onClick={() => {
                      onSelectAdvancedModule(module.id);
                      onToggleAdvancedMenu();
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
                      onSelectAdvancedModule('');
                      onToggleAdvancedMenu();
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
  );
}