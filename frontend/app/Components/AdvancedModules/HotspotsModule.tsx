import dynamic from 'next/dynamic';
import { HotspotsResponse, Geometry } from '../../types';
import { getHotspotColor } from '../../utils/colorScales';

const MapContainer = dynamic(() => import('react-leaflet').then(m => m.MapContainer), { ssr: false });
const TileLayer = dynamic(() => import('react-leaflet').then(m => m.TileLayer), { ssr: false });
const GeoJSON = dynamic(() => import('react-leaflet').then(m => m.GeoJSON), { ssr: false });
const CircleMarker = dynamic(() => import('react-leaflet').then(m => m.CircleMarker), { ssr: false });
const Popup = dynamic(() => import('react-leaflet').then(m => m.Popup), { ssr: false });

interface HotspotsModuleProps {
  data: HotspotsResponse;
  center: [number, number];
  zoom: number;
  mapKey: number;
  districtGeo: Geometry | null;
}

export default function HotspotsModule({ data, center, zoom, mapKey, districtGeo }: HotspotsModuleProps) {
  return (
    <div className="lg:col-span-2 xl:col-span-3 bg-white rounded-lg shadow-lg overflow-hidden border border-gray-200">
      {/* Header */}
      <div className="bg-gradient-to-r from-gray-50 to-white px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 tracking-tight">Declining GWL Hotspots (DBSCAN)</h3>
            <p className="text-xs text-gray-500 mt-1">Spatial clustering of declining groundwater levels</p>
          </div>
          <div className="flex items-center gap-3 text-xs">
            <span className="px-3 py-1.5 bg-white rounded border border-gray-300 text-gray-700">
              <span className="text-gray-500">Clusters:</span> <span className="font-semibold text-gray-900">{data.statistics.n_clusters}</span>
            </span>
          </div>
        </div>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-px bg-gray-200 p-px">
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Declining Sites</div>
          <div className="text-2xl font-bold text-rose-600 tracking-tight">{data.statistics.total_declining_sites}</div>
        </div>
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Clustered</div>
          <div className="text-2xl font-bold text-orange-600 tracking-tight">{data.statistics.clustered_points}</div>
        </div>
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Noise Points</div>
          <div className="text-2xl font-bold text-gray-600 tracking-tight">{data.statistics.noise_points}</div>
        </div>
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Clustering Rate</div>
          <div className="text-2xl font-bold text-blue-600 tracking-tight">{data.statistics.clustering_rate.toFixed(1)}%</div>
        </div>
      </div>


      {data.clusters.length > 0 && (
        <div className="bg-gray-50 p-4 border-b border-gray-200">
          <h4 className="text-sm font-semibold text-gray-900 mb-3">Cluster Details</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-h-[200px] overflow-y-auto">
            {data.clusters.map((cluster, idx) => (
              <div key={idx} className="bg-white p-3 rounded border border-gray-200 shadow-sm">
                <div className="font-semibold text-sm flex items-center gap-2 text-gray-900">
                  <div className="w-4 h-4 rounded-full" style={{ backgroundColor: getHotspotColor(cluster.cluster_id) }}></div>
                  Cluster {cluster.cluster_id}
                </div>
                <div className="text-xs text-gray-600 mt-1 space-y-0.5">
                  <div><strong>Sites:</strong> {cluster.n_sites}</div>
                  <div><strong>Mean Slope:</strong> {cluster.mean_slope.toFixed(4)} m/yr</div>
                  <div><strong>Max Slope:</strong> {cluster.max_slope.toFixed(4)} m/yr</div>
                  <div><strong>Centroid:</strong> {cluster.centroid_lat.toFixed(4)}, {cluster.centroid_lon.toFixed(4)}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="h-[500px] relative border-b border-gray-200">
        <MapContainer center={center} zoom={zoom} style={{ height: "100%", width: "100%" }} key={`hotspots-${mapKey}`}>
          <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
          {districtGeo && <GeoJSON data={districtGeo} style={{ color: "#F43F5E", weight: 3, fillOpacity: 0.1 }} />}

          {data.data.map((point, i) => (
            <CircleMarker
              key={`hot_${i}`}
              center={[point.latitude, point.longitude]}
              radius={point.cluster === -1 ? 5 : 7}  // Smaller for noise
              fillColor={getHotspotColor(point.cluster)}
              color={point.cluster === -1 ? "#9CA3AF" : "white"}  // Gray border for noise
              weight={point.cluster === -1 ? 1 : 2}  // Thinner border for noise
              fillOpacity={point.cluster === -1 ? 0.4 : 0.85}  // More transparent for noise
            >
              <Popup>
                <div style={{ fontFamily: 'sans-serif', minWidth: '180px' }}>
                  <strong style={{ color: point.cluster === -1 ? '#6B7280' : getHotspotColor(point.cluster) }}>
                    {point.cluster === -1 ? 'Noise Point' : `Cluster ${point.cluster}`}
                  </strong><br />
                  <hr style={{ margin: '5px 0' }} />
                  <table style={{ width: '100%', fontSize: '12px' }}>
                    <tbody>
                      <tr><td><strong>Site ID:</strong></td><td>{point.site_id}</td></tr>
                      <tr><td><strong>Decline Rate:</strong></td><td>{point.slope_m_per_year.toFixed(4)} m/yr</td></tr>
                    </tbody>
                  </table>
                </div>
              </Popup>
            </CircleMarker>
          ))}

          {data.clusters.map((cluster, idx) => (
            <CircleMarker
              key={`centroid_${idx}`}
              center={[cluster.centroid_lat, cluster.centroid_lon]}
              radius={12}
              fillColor={getHotspotColor(cluster.cluster_id)}
              color="white"
              weight={3}
              fillOpacity={1}
            >
              <Popup>
                <div style={{ fontFamily: 'sans-serif', minWidth: '180px' }}>
                  <strong>Cluster {cluster.cluster_id} Centroid</strong><br />
                  <hr style={{ margin: '5px 0' }} />
                  <div style={{ fontSize: '12px' }}>
                    <div><strong>Sites:</strong> {cluster.n_sites}</div>
                    <div><strong>Mean Decline:</strong> {cluster.mean_slope.toFixed(4)} m/yr</div>
                  </div>
                </div>
              </Popup>
            </CircleMarker>
          ))}
        </MapContainer>

        <div className="absolute bottom-4 right-4 z-[1000] bg-white/95 backdrop-blur-sm p-3 rounded-lg shadow-lg border-2 border-rose-300">
          <div className="text-xs font-bold mb-2">Clusters</div>
          <div className="space-y-1 max-h-[200px] overflow-y-auto">
            {data.clusters.slice(0, 8).map((cluster, idx) => (
              <div key={idx} className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full" style={{ backgroundColor: getHotspotColor(cluster.cluster_id) }}></div>
                <span className="text-xs">Cluster {cluster.cluster_id} ({cluster.n_sites})</span>
              </div>
            ))}
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-gray-400"></div>
              <span className="text-xs">Noise ({data.statistics.noise_points})</span>
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
            Spatial clusters of declining sites; {data.statistics.n_clusters} hotspots + {data.statistics.noise_points} isolated points
          </p>
        </div>

        {/* How */}
        <div className="p-4 bg-white border-l-4 border-l-violet-600">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 font-medium">How</div>
          <p className="text-sm text-gray-700 leading-relaxed">
            DBSCAN clustering (eps={data.parameters.eps_km}km, min={data.parameters.min_samples} sites) on declining wells
          </p>
        </div>

        {/* Why */}
        <div className="p-4 bg-white border-l-4 border-l-emerald-600">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 font-medium">Why</div>
          <p className="text-sm text-gray-700 leading-relaxed">
            Priority zones for regional intervention; clusters indicate systemic stress, not isolated issues
          </p>
        </div>
      </div>
    </div>
  );
}