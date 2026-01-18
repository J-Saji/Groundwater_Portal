import dynamic from 'next/dynamic';
import { ForecastResponse, Geometry } from '../../types';

const MapContainer = dynamic(() => import('react-leaflet').then(m => m.MapContainer), { ssr: false });
const TileLayer = dynamic(() => import('react-leaflet').then(m => m.TileLayer), { ssr: false });
const GeoJSON = dynamic(() => import('react-leaflet').then(m => m.GeoJSON), { ssr: false });
const CircleMarker = dynamic(() => import('react-leaflet').then(m => m.CircleMarker), { ssr: false });
const Popup = dynamic(() => import('react-leaflet').then(m => m.Popup), { ssr: false });

interface ForecastModuleProps {
  data: ForecastResponse;
  center: [number, number];
  zoom: number;
  mapKey: number;
  districtGeo: Geometry | null;
}

export default function ForecastModule({ data, center, zoom, mapKey, districtGeo }: ForecastModuleProps) {
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
    <div className="lg:col-span-2 xl:col-span-3 bg-white rounded-lg shadow-lg overflow-hidden border border-gray-200">
      {/* Header */}
      <div className="bg-gradient-to-r from-gray-50 to-white px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 tracking-tight">GWL Forecasting (Grid-based)</h3>
            <p className="text-xs text-gray-500 mt-1">Predictive analysis using neighbor-weighted interpolation</p>
          </div>
          <div className="flex items-center gap-3 text-xs">
            <span className="px-3 py-1.5 bg-white rounded border border-gray-300 text-gray-700">
              <span className="text-gray-500">Grid Cells:</span> <span className="font-semibold text-gray-900">{data.count}</span>
            </span>
          </div>
        </div>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-px bg-gray-200 p-px">
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Mean Change (12mo)</div>
          <div className={`text-2xl font-bold tracking-tight ${data.statistics.mean_change_m > 0 ? 'text-red-600' : 'text-emerald-600'}`}>
            {data.statistics.mean_change_m > 0 ? '+' : ''}{data.statistics.mean_change_m.toFixed(3)} m
          </div>
        </div>
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Declining Cells</div>
          <div className="text-2xl font-bold text-red-600 tracking-tight">{data.statistics.declining_cells}</div>
        </div>
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Recovering Cells</div>
          <div className="text-2xl font-bold text-emerald-600 tracking-tight">{data.statistics.recovering_cells}</div>
        </div>
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Mean R²</div>
          <div className="text-2xl font-bold text-blue-600 tracking-tight">{data.statistics.mean_r_squared.toFixed(3)}</div>
        </div>
      </div>

      {/* Forecast Parameters */}
      <div className="bg-gray-50 p-4 border-b border-gray-200">
        <h4 className="text-sm font-semibold text-gray-900 mb-2">Forecast Parameters</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs text-gray-700">
          <div><strong>Horizon:</strong> {data.parameters.forecast_months} months</div>
          <div><strong>Neighbors (k):</strong> {data.parameters.k_neighbors}</div>
          <div><strong>Grid Resolution:</strong> {data.parameters.grid_resolution}×{data.parameters.grid_resolution}</div>
          <div><strong>GRACE Used:</strong> {data.parameters.grace_used ? 'Yes' : 'No'}</div>
          <div><strong>Method:</strong> {data.method}</div>
          <div><strong>Success Rate:</strong> {data.statistics.success_rate.toFixed(1)}%</div>
        </div>
      </div>

      <div className="h-[500px] relative border-b border-gray-200">
        <MapContainer center={center} zoom={zoom} style={{ height: "100%", width: "100%" }} key={`forecast-${mapKey}`}>
          <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
          {districtGeo && <GeoJSON data={districtGeo} style={{ color: "#14B8A6", weight: 3, fillOpacity: 0.1 }} />}

          {data.data.map((point, i) => (
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
                  </strong><br />
                  <hr style={{ margin: '5px 0' }} />
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
          ))}
        </MapContainer>

        <div className="absolute bottom-4 right-4 z-[1000] bg-white/95 backdrop-blur-sm p-3 rounded border border-gray-300 shadow-sm">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 font-medium">12-Month Change (m)</div>
          <div className="space-y-1">
            {[
              { label: 'Strong Decline (>2m)', change: 2.5 },
              { label: 'Moderate Decline (1-2m)', change: 1.5 },
              { label: 'Slight Decline (0-1m)', change: 0.5 },
              { label: 'Slight Recovery (0 to -1m)', change: -0.5 },
              { label: 'Moderate Recovery (-1 to -2m)', change: -1.5 },
              { label: 'Strong Recovery (<-2m)', change: -2.5 }
            ].map((item, idx) => (
              <div key={idx} className="flex items-center gap-2">
                <div className="w-4 h-4 rounded" style={{ backgroundColor: getColor(item.change) }}></div>
                <span className="text-xs">{item.label}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Grid Statistics */}
      <div className="bg-gray-50 p-4 border-b border-gray-200">
        <h4 className="text-sm font-semibold text-gray-900 mb-3">Grid Statistics</h4>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          <div className="bg-white p-2 rounded border border-gray-200">
            <div className="text-xs text-gray-600">Total Cells</div>
            <div className="text-lg font-bold text-gray-800">{data.count}</div>
          </div>
          <div className="bg-white p-2 rounded border border-gray-200">
            <div className="text-xs text-gray-600">Median Change</div>
            <div className={`text-lg font-bold ${data.statistics.median_change_m > 0 ? 'text-red-600' : 'text-emerald-600'}`}>
              {data.statistics.median_change_m > 0 ? '+' : ''}{data.statistics.median_change_m.toFixed(3)} m
            </div>
          </div>
          <div className="bg-white p-2 rounded border border-gray-200">
            <div className="text-xs text-gray-600">Decline/Recovery Ratio</div>
            <div className="text-lg font-bold text-gray-800">
              {(data.statistics.declining_cells / Math.max(data.statistics.recovering_cells, 1)).toFixed(2)}
            </div>
          </div>
        </div>
      </div>

      {/* What, How, Why Section */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-px bg-gray-200 p-px">
        {/* What */}
        <div className="p-4 bg-white border-l-4 border-l-blue-600">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 font-medium">What</div>
          <p className="text-sm text-gray-700 leading-relaxed">
            Grid-based {data.parameters.forecast_months}-month GWL forecast using neighbor-weighted wells and GRACE data
          </p>
        </div>

        {/* How */}
        <div className="p-4 bg-white border-l-4 border-l-violet-600">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 font-medium">How</div>
          <p className="text-sm text-gray-700 leading-relaxed">
            KNN-weighted composite per cell, deseasonalize, OLS regression (trend + GRACE), add back seasonality
          </p>
        </div>

        {/* Why */}
        <div className="p-4 bg-white border-l-4 border-l-emerald-600">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 font-medium">Why</div>
          <p className="text-sm text-gray-700 leading-relaxed">
            Identify spatial zones expected to decline versus recover and plan preemptive interventions
          </p>
        </div>
      </div>
    </div>
  );
}