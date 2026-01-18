import dynamic from 'next/dynamic';
import { SignificantTrendsResponse, Geometry } from '../../types';
import { getTrendColor } from '../../utils/colorScales';

const MapContainer = dynamic(() => import('react-leaflet').then(m => m.MapContainer), { ssr: false });
const TileLayer = dynamic(() => import('react-leaflet').then(m => m.TileLayer), { ssr: false });
const GeoJSON = dynamic(() => import('react-leaflet').then(m => m.GeoJSON), { ssr: false });
const CircleMarker = dynamic(() => import('react-leaflet').then(m => m.CircleMarker), { ssr: false });
const Popup = dynamic(() => import('react-leaflet').then(m => m.Popup), { ssr: false });

interface SignificantTrendsModuleProps {
  data: SignificantTrendsResponse;
  center: [number, number];
  zoom: number;
  mapKey: number;
  districtGeo: Geometry | null;
}

export default function SignificantTrendsModule({ data, center, zoom, mapKey, districtGeo }: SignificantTrendsModuleProps) {
  return (
    <div className="lg:col-span-2 xl:col-span-3 bg-white rounded-lg shadow-lg overflow-hidden border border-gray-200">
      {/* Header */}
      <div className="bg-gradient-to-r from-gray-50 to-white px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 tracking-tight">Statistically Significant Trends</h3>
            <p className="text-xs text-gray-500 mt-1">Verified long-term changes beyond random variation</p>
          </div>
          <div className="flex items-center gap-3 text-xs">
            <span className="px-3 py-1.5 bg-white rounded border border-gray-300 text-gray-700">
              <span className="text-gray-500">Sites (p &lt; {data.parameters.p_threshold}):</span> <span className="font-semibold text-gray-900">{data.count}</span>
            </span>
          </div>
        </div>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-px bg-gray-200 p-px">
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Declining</div>
          <div className="text-2xl font-bold text-red-600 tracking-tight">{data.statistics.declining}</div>
        </div>
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Recovering</div>
          <div className="text-2xl font-bold text-emerald-600 tracking-tight">{data.statistics.recovering}</div>
        </div>
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Mean Slope</div>
          <div className="text-xl font-bold text-blue-600 tracking-tight">{data.statistics.mean_slope.toFixed(4)} m/yr</div>
        </div>
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">High Significance</div>
          <div className="text-2xl font-bold text-violet-600 tracking-tight">{data.statistics.high_significance}</div>
        </div>
      </div>

      <div className="h-[500px] relative border-b border-gray-200">
        <MapContainer center={center} zoom={zoom} style={{ height: "100%", width: "100%" }} key={`trends-${mapKey}`}>
          <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
          {districtGeo && <GeoJSON data={districtGeo} style={{ color: "#22C55E", weight: 3, fillOpacity: 0.1 }} />}

          {data.data.map((point, i) => (
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
                  </strong><br />
                  <hr style={{ margin: '5px 0' }} />
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

        <div className="absolute bottom-4 right-4 z-[1000] bg-white/95 backdrop-blur-sm p-3 rounded border border-gray-300 shadow-sm">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 font-medium">Trend Direction</div>
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full" style={{ backgroundColor: '#DC2626' }}></div>
              <span className="text-xs">Declining (p &lt; {data.parameters.p_threshold})</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full" style={{ backgroundColor: '#16A34A' }}></div>
              <span className="text-xs">Recovering (p &lt; {data.parameters.p_threshold})</span>
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
            Sites with statistically verified trends (p &lt; {data.parameters.p_threshold}) that separate real changes from random noise
          </p>
        </div>

        {/* How */}
        <div className="p-4 bg-white border-l-4 border-l-violet-600">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 font-medium">How</div>
          <p className="text-sm text-gray-700 leading-relaxed">
            {data.parameters.method === 'mann_kendall' ? 'Mann-Kendall test' : 'OLS regression'} on annual GWL slopes testing {data.count} sites for significance
          </p>
        </div>

        {/* Why It Matters */}
        <div className="p-4 bg-white border-l-4 border-l-emerald-600">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 font-medium">Why</div>
          <p className="text-sm text-gray-700 leading-relaxed">
            Focus resources on proven declining sites and avoid false alarms and temporary fluctuations
          </p>
        </div>
      </div>
    </div>
  );
}