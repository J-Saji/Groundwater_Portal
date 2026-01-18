import dynamic from 'next/dynamic';
import { RechargeResponse, Geometry } from '../../types';
import { getStressColor } from '../../utils/colorScales';

const MapContainer = dynamic(() => import('react-leaflet').then(m => m.MapContainer), { ssr: false });
const TileLayer = dynamic(() => import('react-leaflet').then(m => m.TileLayer), { ssr: false });
const GeoJSON = dynamic(() => import('react-leaflet').then(m => m.GeoJSON), { ssr: false });
const CircleMarker = dynamic(() => import('react-leaflet').then(m => m.CircleMarker), { ssr: false });
const Popup = dynamic(() => import('react-leaflet').then(m => m.Popup), { ssr: false });

interface RechargeModuleProps {
  data: RechargeResponse;
  center: [number, number];
  zoom: number;
  mapKey: number;
  districtGeo: Geometry | null;
}

export default function RechargeModule({ data, center, zoom, mapKey, districtGeo }: RechargeModuleProps) {
  return (
    <div className="lg:col-span-2 xl:col-span-3 bg-white rounded-lg shadow-lg overflow-hidden border border-gray-200">
      {/* Header */}
      <div className="bg-gradient-to-r from-gray-50 to-white px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 tracking-tight">Managed Aquifer Recharge (MAR) Planning</h3>
            <p className="text-xs text-gray-500 mt-1">Site-specific structure recommendations and regional potential</p>
          </div>
          <div className="flex items-center gap-3 text-xs">
            <span className="px-3 py-1.5 bg-white rounded border border-gray-300 text-gray-700">
              <span className="text-gray-500">Potential:</span> <span className="font-semibold text-cyan-600">{data.potential.total_recharge_potential_mcm.toFixed(2)} MCM</span>
            </span>
          </div>
        </div>
      </div>

      {/* Analysis Parameters */}
      <div className="bg-gray-50 p-4 border-b border-gray-200">
        <h4 className="text-sm font-semibold text-gray-900 mb-3">Analysis Parameters</h4>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-xs text-gray-700">
          <div><strong>Area:</strong> {data.analysis_parameters.area_km2} km²</div>
          <div><strong>Dominant Lithology:</strong> {data.analysis_parameters.dominant_lithology}</div>
          <div><strong>Runoff Coeff:</strong> {data.analysis_parameters.runoff_coefficient}</div>
          <div><strong>Monsoon Rainfall:</strong> {data.analysis_parameters.monsoon_rainfall_m.toFixed(3)} m</div>
          <div><strong>Capture Fraction:</strong> {(data.analysis_parameters.capture_fraction * 100).toFixed(1)}%</div>
          <div><strong>Year Analyzed:</strong> {data.analysis_parameters.year_analyzed}</div>
        </div>
      </div>

      {/* Recharge Potential */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-px bg-gray-200 p-px">
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Total Recharge Potential</div>
          <div className="text-3xl font-bold text-cyan-600 tracking-tight">{data.potential.total_recharge_potential_mcm.toFixed(2)} MCM</div>
          <div className="text-xs text-gray-500 mt-1">Million Cubic Meters</div>
        </div>
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Per km² Potential</div>
          <div className="text-3xl font-bold text-blue-600 tracking-tight">{data.potential.per_km2_mcm.toFixed(4)} MCM/km²</div>
          <div className="text-xs text-gray-500 mt-1">Normalized by area</div>
        </div>
      </div>

      {/* Structure Plan */}
      <div className="bg-white border-t border-gray-200 p-4">
        <h4 className="text-sm font-semibold text-gray-900 mb-3">Recommended Structure Plan</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-cyan-50">
              <tr>
                <th className="text-left p-2 border-b-2 border-cyan-200">Structure Type</th>
                <th className="text-center p-2 border-b-2 border-cyan-200">Units</th>
                <th className="text-center p-2 border-b-2 border-cyan-200">Total Capacity (MCM)</th>
                <th className="text-center p-2 border-b-2 border-cyan-200">Allocation (%)</th>
              </tr>
            </thead>
            <tbody>
              {data.structure_plan.map((structure, idx) => (
                <tr key={idx} className={idx % 2 === 0 ? 'bg-gray-50' : 'bg-white'}>
                  <td className="p-2 border-b border-gray-200 font-medium">{structure.structure_type}</td>
                  <td className="p-2 border-b border-gray-200 text-center">{structure.recommended_units}</td>
                  <td className="p-2 border-b border-gray-200 text-center font-semibold text-cyan-600">
                    {structure.total_capacity_mcm.toFixed(3)}
                  </td>
                  <td className="p-2 border-b border-gray-200 text-center">
                    {(structure.allocation_fraction * 100).toFixed(0)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Site-Specific Recommendations Map */}
      {data.site_recommendations.length > 0 && (
        <>
          <div className="mb-4">
            <h4 className="text-sm font-semibold text-gray-900 mb-2">Site-Specific Recommendations ({data.count} sites)</h4>
            <p className="text-xs text-gray-600">Based on current groundwater stress levels</p>
          </div>

          <div className="h-[500px] relative border-b border-gray-200">
            <MapContainer center={center} zoom={zoom} style={{ height: "100%", width: "100%" }} key={`recharge-${mapKey}`}>
              <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
              {districtGeo && <GeoJSON data={districtGeo} style={{ color: "#06B6D4", weight: 3, fillOpacity: 0.1 }} />}

              {data.site_recommendations.map((site, i) => (
                <CircleMarker
                  key={`recharge_${i}`}
                  center={[site.latitude, site.longitude]}
                  radius={8}
                  fillColor={getStressColor(site.stress_category)}
                  color="white"
                  weight={2}
                  fillOpacity={0.8}
                >
                  <Popup>
                    <div style={{ fontFamily: 'sans-serif', minWidth: '220px' }}>
                      <strong style={{ fontSize: '14px', color: getStressColor(site.stress_category) }}>
                        {site.stress_category} Site
                      </strong><br />
                      <hr style={{ margin: '5px 0' }} />
                      <table style={{ width: '100%', fontSize: '12px' }}>
                        <tbody>
                          <tr><td><strong>Site ID:</strong></td><td>{site.site_id}</td></tr>
                          <tr><td><strong>Current GWL:</strong></td><td>{site.current_gwl.toFixed(2)} m bgl</td></tr>
                          <tr><td><strong>Stress Level:</strong></td><td>{site.stress_category}</td></tr>
                          <tr>
                            <td><strong>Recommended:</strong></td>
                            <td className="font-semibold text-cyan-600">{site.recommended_structure}</td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </Popup>
                </CircleMarker>
              ))}
            </MapContainer>

            {/* Legend */}
            <div className="absolute bottom-4 right-4 z-[1000] bg-white/95 backdrop-blur-sm p-3 rounded border border-gray-300 shadow-sm">
              <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 font-medium">Stress Category</div>
              <div className="space-y-1">
                {[
                  { label: 'Critical', category: 'Critical', structure: 'Recharge shaft' },
                  { label: 'Stressed', category: 'Stressed', structure: 'Check dam' },
                  { label: 'Moderate', category: 'Moderate', structure: 'Farm pond' },
                  { label: 'Healthy', category: 'Healthy', structure: 'Percolation tank' }
                ].map((item, idx) => (
                  <div key={idx} className="flex flex-col gap-1">
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 rounded" style={{ backgroundColor: getStressColor(item.category) }}></div>
                      <span className="text-xs font-semibold">{item.label}</span>
                    </div>
                    <div className="text-xs text-gray-600 ml-6">→ {item.structure}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </>
      )}

      {data.site_recommendations.length === 0 && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-4">
          <p className="text-sm text-yellow-800">
            ℹ️ No site-specific recommendations available. Select a month to enable site-level stress analysis.
          </p>
        </div>
      )}

      {/* What, How, Why Section */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-px bg-gray-200 p-px">
        {/* What */}
        <div className="p-4 bg-white border-l-4 border-l-blue-600">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 font-medium">What</div>
          <p className="text-sm text-gray-700 leading-relaxed">
            Regional recharge potential ({data.potential.total_recharge_potential_mcm.toFixed(2)} MCM) with site-specific structure recommendations
          </p>
        </div>

        {/* How */}
        <div className="p-4 bg-white border-l-4 border-l-violet-600">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 font-medium">How</div>
          <p className="text-sm text-gray-700 leading-relaxed">
            V = Area × Rainfall × Runoff_Coeff × Capture_Fraction with stress-based structure allocation
          </p>
        </div>

        {/* Why */}
        <div className="p-4 bg-white border-l-4 border-l-emerald-600">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 font-medium">Why</div>
          <p className="text-sm text-gray-700 leading-relaxed">
            Prioritize MAR investments with realistic recharge targets and site-appropriate structures
          </p>
        </div>
      </div>
    </div>
  );
}