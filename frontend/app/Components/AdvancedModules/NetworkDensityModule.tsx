import dynamic from 'next/dynamic';
import { NetworkDensityResponse, Geometry } from '../../types';
import { getDensityColor } from '../../utils/colorScales';

const MapContainer = dynamic(() => import('react-leaflet').then(m => m.MapContainer), { ssr: false });
const TileLayer = dynamic(() => import('react-leaflet').then(m => m.TileLayer), { ssr: false });
const GeoJSON = dynamic(() => import('react-leaflet').then(m => m.GeoJSON), { ssr: false });
const CircleMarker = dynamic(() => import('react-leaflet').then(m => m.CircleMarker), { ssr: false });
const Popup = dynamic(() => import('react-leaflet').then(m => m.Popup), { ssr: false });

interface NetworkDensityModuleProps {
  data: NetworkDensityResponse;
  center: [number, number];
  zoom: number;
  mapKey: number;
  districtGeo: Geometry | null;
}

export default function NetworkDensityModule({ data, center, zoom, mapKey, districtGeo }: NetworkDensityModuleProps) {
  return (
    <div className="lg:col-span-2 xl:col-span-3 bg-white rounded-lg shadow-lg overflow-hidden border border-gray-200">
      {/* Header */}
      <div className="bg-gradient-to-r from-gray-50 to-white px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 tracking-tight">Well Network Density Analysis</h3>
            <p className="text-xs text-gray-500 mt-1">Spatial distribution and connectivity metrics</p>
          </div>
          <div className="flex items-center gap-3 text-xs">
            <span className="px-3 py-1.5 bg-white rounded border border-gray-300 text-gray-700">
              <span className="text-gray-500">Sites:</span> <span className="font-semibold text-gray-900">{data.map1_site_level?.count || 0}</span>
            </span>
            <span className="px-3 py-1.5 bg-white rounded border border-gray-300 text-gray-700">
              <span className="text-gray-500">Grid Cells:</span> <span className="font-semibold text-gray-900">{data.map2_gridded?.count || 0}</span>
            </span>
          </div>
        </div>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-px bg-gray-200 p-px">
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Total Sites</div>
          <div className="text-2xl font-bold text-gray-900 tracking-tight">{data.statistics?.total_sites ?? 0}</div>
        </div>
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Avg Strength</div>
          <div className="text-2xl font-bold text-blue-600 tracking-tight">{data.statistics?.avg_strength?.toFixed(3) ?? 'N/A'}</div>
        </div>
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Avg Local Density</div>
          <div className="text-xl font-bold text-emerald-600 tracking-tight">{data.statistics?.avg_local_density?.toFixed(4) ?? 'N/A'} <span className="text-sm text-gray-500">/km²</span></div>
        </div>
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Median Observations</div>
          <div className="text-2xl font-bold text-violet-600 tracking-tight">{data.statistics?.median_observations ?? 0}</div>
        </div>
      </div>

      {/* Dual Map Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-px bg-gray-200 p-px">

        {/* MAP 1: Site-Level Strength */}
        <div className="bg-gray-50 p-4">
          <div className="flex items-center justify-between mb-3">
            <h4 className="font-semibold text-sm text-gray-900">
              Site-Level Strength
            </h4>
            <span className="text-[10px] uppercase tracking-wider text-gray-500">Marker Size = Density</span>
          </div>
          <div className="h-[400px] relative rounded overflow-hidden border border-gray-300 shadow-sm">
            <MapContainer
              center={center}
              zoom={zoom}
              style={{ height: "100%", width: "100%" }}
              key={`network-map1-${mapKey}`}
            >
              <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

              {districtGeo && (
                <GeoJSON
                  data={districtGeo}
                  style={{ color: "#6366F1", weight: 3, fillOpacity: 0.1 }}
                />
              )}

              {data.map1_site_level?.data?.map((point, i) => {
                const size = 2 + (point.local_density_per_km2 * 150);  // Reduced base and multiplier
                return (
                  <CircleMarker
                    key={`net1_${i}`}
                    center={[point.latitude, point.longitude]}
                    radius={Math.min(size, 12)}  // Reduced max size from 20 to 12
                    fillColor="#6366F1"
                    color="white"
                    weight={1}
                    fillOpacity={0.7}
                  >
                    <Popup>
                      <div style={{ fontFamily: 'sans-serif', minWidth: '220px' }}>
                        <strong style={{ fontSize: '14px', color: '#6366F1' }}>
                          Site: {point.site_id}
                        </strong><br />
                        <hr style={{ margin: '5px 0' }} />
                        <table style={{ width: '100%', fontSize: '12px' }}>
                          <tbody>
                            <tr><td><strong>Strength:</strong></td><td>{point.strength.toFixed(3)}</td></tr>
                            <tr><td><strong>Local Density:</strong></td><td>{point.local_density_per_km2.toFixed(4)} /km²</td></tr>
                            <tr><td><strong>Neighbors:</strong></td><td>{point.neighbors_within_radius} within {data.parameters?.radius_km}km</td></tr>
                            <tr><td><strong>Observations:</strong></td><td>{point.n_observations}</td></tr>
                            <tr><td><strong>Trend:</strong></td><td>{point.slope_m_per_year.toFixed(4)} m/yr</td></tr>
                            <tr><td><strong>GWL Std:</strong></td><td>{point.gwl_std.toFixed(2)} m</td></tr>
                          </tbody>
                        </table>
                      </div>
                    </Popup>
                  </CircleMarker>
                );
              })}
            </MapContainer>

            <div className="absolute bottom-4 right-4 z-[1000] bg-white/95 backdrop-blur-sm px-3 py-2 rounded border border-gray-300 shadow-sm">
              <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1">Symbol Size = Density</div>
              <p className="text-xs text-gray-600">
                Larger circles = higher density within {data.parameters?.radius_km}km
              </p>
            </div>
          </div>
        </div>

        {/* MAP 2: Gridded Density Heatmap */}
        <div className="bg-gray-50 p-4">
          <div className="flex items-center justify-between mb-3">
            <h4 className="font-semibold text-sm text-gray-900">
              Gridded Density Heatmap
            </h4>
            <span className="text-[10px] uppercase tracking-wider text-gray-500">Sites per 1000 km²</span>
          </div>
          <div className="h-[400px] relative rounded overflow-hidden border border-gray-300 shadow-sm">
            <MapContainer
              center={center}
              zoom={zoom}
              style={{ height: "100%", width: "100%" }}
              key={`network-map2-${mapKey}`}
            >
              <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

              {districtGeo && (
                <GeoJSON
                  data={districtGeo}
                  style={{ color: "#F97316", weight: 3, fillOpacity: 0.1 }}
                />
              )}

              {data.map2_gridded?.data?.map((point, i) => (
                <CircleMarker
                  key={`net2_${i}`}
                  center={[point.y, point.x]}
                  radius={8}
                  fillColor={getDensityColor(point.density_per_1000km2)}
                  color="white"
                  weight={1}
                  fillOpacity={0.8}
                >
                  <Popup>
                    <div style={{ fontFamily: 'sans-serif', minWidth: '180px' }}>
                      <strong style={{ fontSize: '14px' }}>Grid Cell Density</strong><br />
                      <hr style={{ margin: '5px 0' }} />
                      <div style={{ fontSize: '12px' }}>
                        <div><strong>Density:</strong> {point.density_per_1000km2.toFixed(2)} sites/1000km²</div>
                        <div><strong>Location:</strong> {point.y.toFixed(4)}°N, {point.x.toFixed(4)}°E</div>
                      </div>
                    </div>
                  </Popup>
                </CircleMarker>
              ))}
            </MapContainer>

            <div className="absolute bottom-4 right-4 z-[1000] bg-white/95 backdrop-blur-sm p-3 rounded-lg shadow-lg border-2 border-orange-300">
              <div className="text-xs font-bold mb-2 text-orange-900">Density Scale (sites/1000km²)</div>
              <div className="space-y-1">
                {[
                  { label: 'Very Dense (>40)', density: 45 },
                  { label: 'Dense (20-40)', density: 30 },
                  { label: 'Moderate (10-20)', density: 15 },
                  { label: 'Sparse (5-10)', density: 7 },
                  { label: 'Very Sparse (<5)', density: 3 }
                ].map((item, idx) => (
                  <div key={idx} className="flex items-center gap-2">
                    <div
                      className="w-4 h-4 rounded border border-gray-400"
                      style={{ backgroundColor: getDensityColor(item.density) }}
                    ></div>
                    <span className="text-xs">{item.label}</span>
                  </div>
                ))}
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
            Dual-map view: (1) Site-level signal strength, (2) Grid-based monitoring density
          </p>
        </div>

        {/* How */}
        <div className="p-4 bg-white border-l-4 border-l-violet-600">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 font-medium">How</div>
          <p className="text-sm text-gray-700 leading-relaxed">
            Strength = |slope|/σ per site; density = sites per 1000 km² on {data.parameters?.radius_km}km radius grid
          </p>
        </div>

        {/* Why */}
        <div className="p-4 bg-white border-l-4 border-l-emerald-600">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 font-medium">Why</div>
          <p className="text-sm text-gray-700 leading-relaxed">
            Identify coverage gaps and assess which areas have reliable trend data
          </p>
        </div>
      </div>
    </div>
  );
}