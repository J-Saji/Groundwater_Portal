import dynamic from 'next/dynamic';
import { ChangepointResponse, Geometry } from '../../types';

const MapContainer = dynamic(() => import('react-leaflet').then(m => m.MapContainer), { ssr: false });
const TileLayer = dynamic(() => import('react-leaflet').then(m => m.TileLayer), { ssr: false });
const GeoJSON = dynamic(() => import('react-leaflet').then(m => m.GeoJSON), { ssr: false });
const CircleMarker = dynamic(() => import('react-leaflet').then(m => m.CircleMarker), { ssr: false });
const Popup = dynamic(() => import('react-leaflet').then(m => m.Popup), { ssr: false });

interface ChangepointsModuleProps {
  data: ChangepointResponse;
  center: [number, number];
  zoom: number;
  mapKey: number;
  districtGeo: Geometry | null;
}

export default function ChangepointsModule({ data, center, zoom, mapKey, districtGeo }: ChangepointsModuleProps) {
  return (
    <div className="lg:col-span-2 xl:col-span-3 bg-white rounded-lg shadow-lg overflow-hidden border border-gray-200">
      {/* Header */}
      <div className="bg-gradient-to-r from-gray-50 to-white px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 tracking-tight">Changepoint Detection - Dual Map View</h3>
            <p className="text-xs text-gray-500 mt-1">Structural break identification and data coverage analysis</p>
          </div>
          <div className="flex items-center gap-3 text-xs">
            <span className="px-3 py-1.5 bg-white rounded border border-gray-300 text-gray-700">
              <span className="text-gray-500">Map1:</span> <span className="font-semibold text-gray-900">{data.changepoints?.count || 0}</span> <span className="text-gray-500">| Map2:</span> <span className="font-semibold text-gray-900">{data.coverage?.count || 0}</span>
            </span>
          </div>
        </div>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-px bg-gray-200 p-px">
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Total Sites</div>
          <div className="text-2xl font-bold text-amber-600 tracking-tight">{data.statistics.total_sites}</div>
        </div>
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Analyzed (≥24mo)</div>
          <div className="text-2xl font-bold text-orange-600 tracking-tight">{data.statistics.sites_analyzed}</div>
        </div>
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">With Changepoints</div>
          <div className="text-2xl font-bold text-red-600 tracking-tight">{data.statistics.sites_with_changepoints}</div>
        </div>
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Detection Rate</div>
          <div className="text-2xl font-bold text-blue-600 tracking-tight">{data.statistics.detection_rate.toFixed(1)}%</div>
        </div>
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Insufficient Data</div>
          <div className="text-2xl font-bold text-gray-600 tracking-tight">{data.statistics.sites_insufficient_data}</div>
        </div>
      </div>

      {/* DUAL MAP LAYOUT */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-px bg-gray-200 p-px">

        {/* MAP 1: Changepoints (Structural Breaks) */}
        <div className="bg-white p-4">
          <h4 className="text-sm font-semibold text-gray-900 mb-2">
            Map 1: Changepoints (Symbol Size = # Breakpoints)
          </h4>
          <div className="h-[400px] relative border border-gray-200 rounded">
            <MapContainer
              center={center}
              zoom={zoom}
              style={{ height: "100%", width: "100%" }}
              key={`changepoints-map1-${mapKey}`}
            >
              <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

              {districtGeo && (
                <GeoJSON
                  data={districtGeo}
                  style={{ color: "#EAB308", weight: 3, fillOpacity: 0.1 }}
                />
              )}

              {data.changepoints?.data.map((site, i) => {
                const size = 5 + Math.min(site.n_breakpoints * 2, 10);
                return (
                  <CircleMarker
                    key={`cp_${i}`}
                    center={[site.latitude, site.longitude]}
                    radius={size}
                    fillColor="#EAB308"
                    color="white"
                    weight={2}
                    fillOpacity={0.8}
                  >
                    <Popup>
                      <div style={{ fontFamily: 'sans-serif', minWidth: '220px' }}>
                        <strong style={{ color: '#EAB308' }}>Changepoint Detected</strong><br />
                        <hr style={{ margin: '5px 0' }} />
                        <table style={{ width: '100%', fontSize: '12px' }}>
                          <tbody>
                            <tr><td><strong>Site ID:</strong></td><td>{site.site_id}</td></tr>
                            <tr><td><strong>Primary Break:</strong></td><td>{new Date(site.changepoint_date).toLocaleDateString()}</td></tr>
                            <tr><td><strong>Total Breaks:</strong></td><td>{site.n_breakpoints}</td></tr>
                            <tr><td><strong>Series Length:</strong></td><td>{site.series_length} months</td></tr>
                          </tbody>
                        </table>
                        {site.all_breakpoints.length > 1 && (
                          <>
                            <hr style={{ margin: '5px 0' }} />
                            <div style={{ fontSize: '11px', color: '#666' }}>
                              <strong>All Breakpoints:</strong><br />
                              {site.all_breakpoints.map((bp, idx) => (
                                <div key={idx}>• {new Date(bp).toLocaleDateString()}</div>
                              ))}
                            </div>
                          </>
                        )}
                      </div>
                    </Popup>
                  </CircleMarker>
                );
              })}
            </MapContainer>

            <div className="absolute bottom-4 right-4 z-[1000] bg-white/95 backdrop-blur-sm p-3 rounded border border-gray-300 shadow-sm">
              <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1 font-medium">Symbol Size</div>
              <p className="text-xs text-gray-600">Larger circles = more structural breaks detected</p>
            </div>
          </div>
        </div>

        {/* MAP 2: Coverage Diagnostics */}
        <div className="bg-white p-4">
          <h4 className="text-sm font-semibold text-gray-900 mb-2">
            Map 2: Coverage Diagnostics (Analyzed vs Insufficient Data)
          </h4>
          <div className="h-[400px] relative border border-gray-200 rounded">
            <MapContainer
              center={center}
              zoom={zoom}
              style={{ height: "100%", width: "100%" }}
              key={`changepoints-map2-${mapKey}`}
            >
              <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

              {districtGeo && (
                <GeoJSON
                  data={districtGeo}
                  style={{ color: "#22C55E", weight: 3, fillOpacity: 0.1 }}
                />
              )}

              {data.coverage?.data?.map((site, i) => (
                <CircleMarker
                  key={`coverage_${i}`}
                  center={[site.latitude, site.longitude]}
                  radius={site.analyzed ? 8 : 4}
                  fillColor={site.analyzed ? "#22C55E" : "#9CA3AF"}
                  color="white"
                  weight={site.analyzed ? 2 : 1}
                  fillOpacity={0.8}
                >
                  <Popup>
                    <div style={{ fontFamily: 'sans-serif', minWidth: '200px' }}>
                      <strong style={{
                        fontSize: '14px',
                        color: site.analyzed ? '#22C55E' : '#9CA3AF'
                      }}>
                        {site.analyzed ? 'Analyzed Site' : 'Insufficient Data'}
                      </strong><br />
                      <hr style={{ margin: '5px 0' }} />
                      <table style={{ width: '100%', fontSize: '12px' }}>
                        <tbody>
                          <tr><td><strong>Site ID:</strong></td><td>{site.site_id}</td></tr>
                          <tr><td><strong>Coverage:</strong></td><td>{site.span_years} years</td></tr>
                          <tr><td><strong>Months:</strong></td><td>{site.n_months} months</td></tr>
                          <tr><td><strong>Period:</strong></td><td>{new Date(site.date_start).toLocaleDateString()} to {new Date(site.date_end).toLocaleDateString()}</td></tr>
                          <tr>
                            <td><strong>Status:</strong></td>
                            <td style={{
                              fontWeight: 'bold',
                              color: site.analyzed ? '#22C55E' : '#EF4444'
                            }}>
                              {site.analyzed ? '≥24 months (Analyzed)' : '<24 months (Skipped)'}
                            </td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </Popup>
                </CircleMarker>
              ))}
            </MapContainer>

            <div className="absolute bottom-4 right-4 z-[1000] bg-white/95 backdrop-blur-sm p-3 rounded border border-gray-300 shadow-sm">
              <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 font-medium">Data Coverage</div>
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full bg-emerald-500 border border-white"></div>
                  <span className="text-xs">Analyzed (≥24 months)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-gray-400 border border-white"></div>
                  <span className="text-xs">Insufficient (&lt;24 months)</span>
                </div>
              </div>
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
            Dual-map view showing sites with structural breaks and data coverage diagnostics
          </p>
        </div>

        {/* How */}
        <div className="p-4 bg-white border-l-4 border-l-violet-600">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 font-medium">How</div>
          <p className="text-sm text-gray-700 leading-relaxed">
            PELT algorithm (penalty={data.parameters.penalty}) detects mean shifts with ≥{data.parameters.min_months_required} months required
          </p>
        </div>

        {/* Why */}
        <div className="p-4 bg-white border-l-4 border-l-emerald-600">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 font-medium">Why</div>
          <p className="text-sm text-gray-700 leading-relaxed">
            Identify regime shifts from policy changes, irrigation expansion, or climate events
          </p>
        </div>
      </div>
    </div>
  );
}