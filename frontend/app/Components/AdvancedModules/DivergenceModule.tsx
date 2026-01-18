import dynamic from 'next/dynamic';
import { DivergenceResponse, Geometry } from '../../types';
import { getDivergenceColor } from '../../utils/colorScales';

const MapContainer = dynamic(() => import('react-leaflet').then(m => m.MapContainer), { ssr: false });
const TileLayer = dynamic(() => import('react-leaflet').then(m => m.TileLayer), { ssr: false });
const GeoJSON = dynamic(() => import('react-leaflet').then(m => m.GeoJSON), { ssr: false });
const CircleMarker = dynamic(() => import('react-leaflet').then(m => m.CircleMarker), { ssr: false });
const Popup = dynamic(() => import('react-leaflet').then(m => m.Popup), { ssr: false });

interface DivergenceModuleProps {
  data: DivergenceResponse;
  center: [number, number];
  zoom: number;
  mapKey: number;
  districtGeo: Geometry | null;
}

export default function DivergenceModule({ data, center, zoom, mapKey, districtGeo }: DivergenceModuleProps) {
  return (
    <div className="lg:col-span-2 xl:col-span-3 bg-white rounded-lg shadow-lg overflow-hidden border border-gray-200">
      {/* Header */}
      <div className="bg-gradient-to-r from-gray-50 to-white px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 tracking-tight">GRACE-Well Divergence Analysis</h3>
            <p className="text-xs text-gray-500 mt-1">Satellite-ground data comparison and validation</p>
          </div>
          <div className="flex items-center gap-3 text-xs">
            <span className="px-3 py-1.5 bg-white rounded border border-gray-300 text-gray-700">
              <span className="text-gray-500">Pixels:</span> <span className="font-semibold text-gray-900">{data.count}</span>
            </span>
          </div>
        </div>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-px bg-gray-200 p-px">
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Mean Divergence</div>
          <div className="text-2xl font-bold text-amber-600 tracking-tight">{data.statistics.mean_divergence.toFixed(3)}</div>
        </div>
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Positive Div</div>
          <div className="text-2xl font-bold text-blue-600 tracking-tight">{data.statistics.positive_divergence_pixels}</div>
        </div>
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Negative Div</div>
          <div className="text-2xl font-bold text-red-600 tracking-tight">{data.statistics.negative_divergence_pixels}</div>
        </div>
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Max |Divergence|</div>
          <div className="text-xl font-bold text-violet-600 tracking-tight">
            {Math.max(Math.abs(data.statistics.max_divergence), Math.abs(data.statistics.min_divergence)).toFixed(3)}
          </div>
        </div>
      </div>

      <div className="h-[500px] relative border-b border-gray-200">
        <MapContainer center={center} zoom={zoom} style={{ height: "100%", width: "100%" }} key={`divergence-${mapKey}`}>
          <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
          {districtGeo && <GeoJSON data={districtGeo} style={{ color: "#F59E0B", weight: 3, fillOpacity: 0.1 }} />}

          {data.data.map((point, i) => (
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
                  <strong style={{ color: getDivergenceColor(point.divergence) }}>Divergence: {point.divergence.toFixed(3)}</strong><br />
                  <hr style={{ margin: '5px 0' }} />
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

        <div className="absolute bottom-4 right-4 z-[1000] bg-white/95 backdrop-blur-sm p-3 rounded border border-gray-300 shadow-sm">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 font-medium">Divergence (z-score)</div>
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

      {/* What, How, Why Section */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-px bg-gray-200 p-px">
        {/* What */}
        <div className="p-4 bg-white border-l-4 border-l-blue-600">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 font-medium">What</div>
          <p className="text-sm text-gray-700 leading-relaxed">
            Difference between GRACE satellite signal and ground-interpolated well data (z-scores)
          </p>
        </div>

        {/* How */}
        <div className="p-4 bg-white border-l-4 border-l-violet-600">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 font-medium">How</div>
          <p className="text-sm text-gray-700 leading-relaxed">
            Divergence = z_GRACE - z_GWL_interpolated at {data.count} pixels
          </p>
        </div>

        {/* Why */}
        <div className="p-4 bg-white border-l-4 border-l-emerald-600">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 font-medium">Why</div>
          <p className="text-sm text-gray-700 leading-relaxed">
            Negative = GRACE underestimates stress; Positive = overestimates. Validates satellite accuracy
          </p>
        </div>
      </div>
    </div>
  );
}