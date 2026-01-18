import dynamic from 'next/dynamic';
import { LagCorrelationResponse, Geometry } from '../../types';
import { COLOR_PALETTE } from '../../utils/constants';

const MapContainer = dynamic(() => import('react-leaflet').then(m => m.MapContainer), { ssr: false });
const TileLayer = dynamic(() => import('react-leaflet').then(m => m.TileLayer), { ssr: false });
const GeoJSON = dynamic(() => import('react-leaflet').then(m => m.GeoJSON), { ssr: false });
const CircleMarker = dynamic(() => import('react-leaflet').then(m => m.CircleMarker), { ssr: false });
const Popup = dynamic(() => import('react-leaflet').then(m => m.Popup), { ssr: false });

interface LagCorrelationModuleProps {
  data: LagCorrelationResponse;
  center: [number, number];
  zoom: number;
  mapKey: number;
  districtGeo: Geometry | null;
}

export default function LagCorrelationModule({ data, center, zoom, mapKey, districtGeo }: LagCorrelationModuleProps) {
  const lagColors: Record<number, string> = {
    0: '#DC2626', 1: '#F59E0B', 2: '#FCD34D',
    3: '#84CC16', 4: '#22C55E', 5: '#14B8A6',
    6: '#3B82F6', 7: '#6366F1', 8: '#8B5CF6',
    9: '#A855F7', 10: '#D946EF', 11: '#EC4899', 12: '#F43F5E'
  };

  return (
    <div className="lg:col-span-2 xl:col-span-3 bg-white rounded-lg shadow-lg overflow-hidden border border-gray-200">
      {/* Header */}
      <div className="bg-gradient-to-r from-gray-50 to-white px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 tracking-tight">Rainfall-GWL Lag Correlation</h3>
            <p className="text-xs text-gray-500 mt-1">Response time analysis between precipitation and groundwater</p>
          </div>
          <div className="flex items-center gap-3 text-xs">
            <span className="px-3 py-1.5 bg-white rounded border border-gray-300 text-gray-700">
              <span className="text-gray-500">Sites:</span> <span className="font-semibold text-gray-900">{data.count}</span>
            </span>
          </div>
        </div>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-px bg-gray-200 p-px">
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Mean Lag</div>
          <div className="text-2xl font-bold text-pink-600 tracking-tight">{data.statistics.mean_lag.toFixed(1)} months</div>
        </div>
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Median Lag</div>
          <div className="text-2xl font-bold text-violet-600 tracking-tight">{data.statistics.median_lag.toFixed(1)} months</div>
        </div>
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Avg |Correlation|</div>
          <div className="text-2xl font-bold text-blue-600 tracking-tight">{data.statistics.mean_abs_correlation.toFixed(3)}</div>
        </div>
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Max Lag Tested</div>
          <div className="text-2xl font-bold text-emerald-600 tracking-tight">{data.parameters.max_lag_months} months</div>
        </div>
      </div>

      <div className="h-[500px] relative border-b border-gray-200">
        <MapContainer center={center} zoom={zoom} style={{ height: "100%", width: "100%" }} key={`lag-${mapKey}`}>
          <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
          {districtGeo && <GeoJSON data={districtGeo} style={{ color: "#EC4899", weight: 3, fillOpacity: 0.1 }} />}

          {data.data.map((point, i) => {
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
                    <strong style={{ color }}>Best Lag: {point.best_lag_months} months</strong><br />
                    <hr style={{ margin: '5px 0' }} />
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

        <div className="absolute bottom-4 right-4 z-[1000] bg-white/95 backdrop-blur-sm p-3 rounded border border-gray-300 shadow-sm max-h-[300px] overflow-y-auto">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 font-medium">Lag (months)</div>
          <div className="space-y-1">
            {Object.entries(data.statistics.lag_distribution).sort(([a], [b]) => Number(a) - Number(b)).map(([lag, count]) => (
              <div key={lag} className="flex items-center gap-2">
                <div className="w-4 h-4 rounded" style={{ backgroundColor: lagColors[Number(lag)] || '#9CA3AF' }}></div>
                <span className="text-xs">{lag} mo ({count} sites)</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* What, How, Why Section */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-px bg-gray-200 p-px">
        {/* What */}
        <div className="p-4 bg-white border-l-4 border-l-blue-600">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 font-medium">What</div>
          <p className="text-sm text-gray-700 leading-relaxed">
            Time lag (0-{data.parameters.max_lag_months} months) between rainfall and GWL for maximum correlation
          </p>
        </div>

        {/* How */}
        <div className="p-4 bg-white border-l-4 border-l-violet-600">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 font-medium">How</div>
          <p className="text-sm text-gray-700 leading-relaxed">
            Cross-correlation at each lag tested with best lag determined by highest |correlation| at {data.count} sites
          </p>
        </div>

        {/* Why */}
        <div className="p-4 bg-white border-l-4 border-l-emerald-600">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 font-medium">Why</div>
          <p className="text-sm text-gray-700 leading-relaxed">
            Short lags indicate responsive aquifers while long lags suggest slow infiltration and guides irrigation timing
          </p>
        </div>
      </div>
    </div>
  );
}