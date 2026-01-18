import dynamic from 'next/dynamic';
import { ASIResponse, Geometry } from '../../types';
import { getASIColor } from '../../utils/colorScales';

const MapContainer = dynamic(() => import('react-leaflet').then(m => m.MapContainer), { ssr: false });
const TileLayer = dynamic(() => import('react-leaflet').then(m => m.TileLayer), { ssr: false });
const GeoJSON = dynamic(() => import('react-leaflet').then(m => m.GeoJSON), { ssr: false });

interface ASIModuleProps {
  data: ASIResponse;
  center: [number, number];
  zoom: number;
  mapKey: number;
  districtGeo: Geometry | null;
}

export default function ASIModule({ data, center, zoom, mapKey, districtGeo }: ASIModuleProps) {
  return (
    <div className="lg:col-span-2 xl:col-span-3 bg-white rounded-lg shadow-lg overflow-hidden border border-gray-200">
      {/* Header */}
      <div className="bg-gradient-to-r from-gray-50 to-white px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 tracking-tight">Aquifer Suitability Index (ASI)</h3>
            <p className="text-xs text-gray-500 mt-1">Groundwater storage potential ranking by specific yield</p>
          </div>
          <div className="flex items-center gap-3 text-xs">
            <span className="px-3 py-1.5 bg-white rounded border border-gray-300 text-gray-700">
              <span className="text-gray-500">Polygons:</span> <span className="font-semibold text-gray-900">{data.count}</span>
            </span>
          </div>
        </div>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-px bg-gray-200 p-px">
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Mean ASI</div>
          <div className="text-2xl font-bold text-violet-600 tracking-tight">{data.statistics.mean_asi.toFixed(2)}</div>
        </div>
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Median</div>
          <div className="text-2xl font-bold text-emerald-600 tracking-tight">{data.statistics.median_asi.toFixed(2)}</div>
        </div>
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Range</div>
          <div className="text-sm font-bold text-blue-600 tracking-tight">
            {data.statistics.min_asi.toFixed(1)}–{data.statistics.max_asi.toFixed(1)}
          </div>
        </div>
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Avg Sy</div>
          <div className="text-xl font-bold text-orange-600 tracking-tight">{data.statistics.avg_specific_yield.toFixed(4)}</div>
        </div>
        <div className="bg-white p-4">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5 font-medium">Total Area</div>
          <div className="text-lg font-bold text-teal-600 tracking-tight">{data.statistics.total_area_km2.toFixed(0)} km²</div>
        </div>
      </div>

      {/* Dominant Aquifer Info */}
      <div className="bg-gray-50 p-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-xs text-gray-600">Dominant Aquifer Type</div>
            <div className="font-bold text-lg text-gray-900">{data.statistics.dominant_aquifer}</div>
          </div>
          <div className="text-right">
            <div className="text-xs text-gray-600">Normalization Range</div>
            <div className="text-sm font-semibold text-gray-700">
              Sy: {data.methodology.quantile_stretch.low.toFixed(4)} – {data.methodology.quantile_stretch.high.toFixed(4)}
            </div>
          </div>
        </div>
      </div>

      {/* Map with Choropleth Polygons */}
      <div className="h-[500px] relative rounded-lg overflow-hidden">
        <MapContainer
          center={center}
          zoom={zoom}
          style={{ height: "100%", width: "100%" }}
          key={`asi-${mapKey}`}
        >
          <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

          {districtGeo && (
            <GeoJSON
              data={districtGeo}
              style={{ color: "#9333EA", weight: 3, fillOpacity: 0.05 }}
            />
          )}

          {data.geojson.features.map((feature, i) => {
            const asiScore = feature.properties.asi_score;
            const fillColor = getASIColor(asiScore);

            return (
              <GeoJSON
                key={`asi_polygon_${feature.id}_${i}`}
                data={feature}
                style={{
                  fillColor: fillColor,
                  fillOpacity: 0.7,
                  color: '#666',
                  weight: 1,
                  opacity: 0.8
                }}
                onEachFeature={(feature: any, layer: any) => {
                  const props = feature.properties;
                  const popupContent = `
                    <div style="font-family: sans-serif; min-width: 220px;">
                      <strong style="font-size: 15px; color: ${fillColor};">
                        ASI Score: ${props.asi_score.toFixed(2)}/5
                      </strong><br/>
                      <hr style="margin: 5px 0; border: 1px solid #ddd;"/>
                      <table style="width: 100%; font-size: 12px; margin-top: 5px;">
                        <tbody>
                          <tr><td><strong>Aquifer:</strong></td><td>${props.majoraquif || 'N/A'}</td></tr>
                          <tr><td><strong>Specific Yield:</strong></td><td>${props.specific_yield.toFixed(4)}</td></tr>
                          <tr><td><strong>Area:</strong></td><td>${(props.area_m2 / 1_000_000).toFixed(2)} km²</td></tr>
                          <tr>
                            <td><strong>Quality:</strong></td>
                            <td style="font-weight: bold; color: ${fillColor};">
                              ${asiScore >= 4 ? 'Excellent' :
                      asiScore >= 3 ? 'Good' :
                        asiScore >= 2 ? 'Moderate' :
                          asiScore >= 1 ? 'Fair' : 'Poor'}
                            </td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  `;

                  layer.bindPopup(popupContent);
                  layer.on('mouseover', function () {
                    this.setStyle({ weight: 3, color: '#333', fillOpacity: 0.9 });
                  });
                  layer.on('mouseout', function () {
                    this.setStyle({ weight: 1, color: '#666', fillOpacity: 0.7 });
                  });
                }}
              />
            );
          })}
        </MapContainer>

        {/* Legend */}
        <div className="absolute bottom-4 right-4 z-[1000] bg-white/95 backdrop-blur-sm p-3 rounded-lg shadow-lg border-2 border-purple-300">
          <div className="text-xs font-bold mb-2 text-purple-900">ASI Score (Storage Potential)</div>
          <div className="space-y-1">
            {[
              { label: 'Excellent (4-5)', score: 4.5, desc: 'Highest potential' },
              { label: 'Good (3-4)', score: 3.5, desc: 'High potential' },
              { label: 'Moderate (2-3)', score: 2.5, desc: 'Medium potential' },
              { label: 'Fair (1-2)', score: 1.5, desc: 'Low potential' },
              { label: 'Poor (0-1)', score: 0.5, desc: 'Very low potential' }
            ].map((item, idx) => (
              <div key={idx} className="flex items-center gap-2">
                <div
                  className="w-5 h-5 rounded border border-gray-400"
                  style={{ backgroundColor: getASIColor(item.score) }}
                ></div>
                <div className="flex flex-col">
                  <span className="text-xs font-semibold">{item.label}</span>
                  <span className="text-[10px] text-gray-500">{item.desc}</span>
                </div>
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
            Ranks aquifer zones by groundwater storage potential (0-5 scale) based on specific yield
          </p>
        </div>

        {/* How */}
        <div className="p-4 bg-white border-l-4 border-l-violet-600">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 font-medium">How</div>
          <p className="text-sm text-gray-700 leading-relaxed">
            Quantile normalization of specific yield (5th-95th percentile) across {data.count} aquifer polygons
          </p>
        </div>

        {/* Why */}
        <div className="p-4 bg-white border-l-4 border-l-emerald-600">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 font-medium">Why</div>
          <p className="text-sm text-gray-700 leading-relaxed">
            Prioritize high-ASI zones for artificial recharge structures and MAR projects
          </p>
        </div>
      </div>
    </div>
  );
}