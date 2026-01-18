import dynamic from 'next/dynamic';
import { SASSResponse, Geometry } from '../../types';
import { getSASSColor } from '../../utils/colorScales';

const MapContainer = dynamic(() => import('react-leaflet').then(m => m.MapContainer), { ssr: false });
const TileLayer = dynamic(() => import('react-leaflet').then(m => m.TileLayer), { ssr: false });
const GeoJSON = dynamic(() => import('react-leaflet').then(m => m.GeoJSON), { ssr: false });
const CircleMarker = dynamic(() => import('react-leaflet').then(m => m.CircleMarker), { ssr: false });
const Popup = dynamic(() => import('react-leaflet').then(m => m.Popup), { ssr: false });

interface SASSModuleProps {
  data: SASSResponse;
  center: [number, number];
  zoom: number;
  mapKey: number;
  districtGeo: Geometry | null;
}

export default function SASSModule({ data, center, zoom, mapKey, districtGeo }: SASSModuleProps) {
  return (
    <div className="lg:col-span-2 xl:col-span-3 bg-white rounded-lg shadow-lg overflow-hidden border border-gray-200">
      {/* Header */}
      <div className="bg-gradient-to-r from-gray-50 to-white px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 tracking-tight">Spatio-Temporal Aquifer Stress Score (SASS)</h3>
            <p className="text-xs text-gray-500 mt-1">Multi-source composite stress index validation</p>
          </div>
          <div className="flex items-center gap-3 text-xs">
            <span className="px-3 py-1.5 bg-white rounded border border-gray-300 text-gray-700">
              <span className="text-gray-500">Stressed Sites:</span> <span className="font-semibold text-red-600">{data.statistics.stressed_sites}</span>
            </span>
          </div>
        </div>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-px bg-gray-200 p-px">
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Mean SASS</div>
          <div className="text-2xl font-bold text-red-600 tracking-tight">{data.statistics.mean_sass.toFixed(3)}</div>
        </div>
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Max Stress</div>
          <div className="text-2xl font-bold text-orange-600 tracking-tight">{data.statistics.max_sass.toFixed(3)}</div>
        </div>
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Min Stress</div>
          <div className="text-2xl font-bold text-emerald-600 tracking-tight">{data.statistics.min_sass.toFixed(3)}</div>
        </div>
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Stressed Sites</div>
          <div className="text-2xl font-bold text-amber-600 tracking-tight">{data.statistics.stressed_sites}</div>
        </div>
      </div>

      <div className="h-[500px] relative border-b border-gray-200">
        <MapContainer center={center} zoom={zoom} style={{ height: "100%", width: "100%" }} key={`sass-${mapKey}`}>
          <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
          {districtGeo && <GeoJSON data={districtGeo} style={{ color: "#DC2626", weight: 3, fillOpacity: 0.1 }} />}

          {data.data.map((point, i) => (
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
                  <strong style={{ fontSize: '14px', color: getSASSColor(point.sass_score) }}>SASS: {point.sass_score.toFixed(3)}</strong><br />
                  <hr style={{ margin: '5px 0' }} />
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

        <div className="absolute bottom-4 right-4 z-[1000] bg-white/95 backdrop-blur-sm p-3 rounded border border-gray-300 shadow-sm">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 font-medium">SASS Score</div>
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

      {/* What, How, Why Section */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-px bg-gray-200 p-px">
        {/* What */}
        <div className="p-4 bg-white border-l-4 border-l-blue-600">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 font-medium">What</div>
          <p className="text-sm text-gray-700 leading-relaxed">
            Composite stress index combining wells + GRACE + rainfall (z-scores)
          </p>
        </div>

        {/* How */}
        <div className="p-4 bg-white border-l-4 border-l-violet-600">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 font-medium">How</div>
          <p className="text-sm text-gray-700 leading-relaxed">
            SASS = GWL_stress + GRACE_z + Rain_z for {data.count} sites ({data.filters.year}-{String(data.filters.month).padStart(2, '0')})
          </p>
        </div>

        {/* Why */}
        <div className="p-4 bg-white border-l-4 border-l-emerald-600">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 font-medium">Why</div>
          <p className="text-sm text-gray-700 leading-relaxed">
            Multi-source validation: high SASS means wells, satellites, and rainfall all show stress
          </p>
        </div>
      </div>
    </div>
  );
}