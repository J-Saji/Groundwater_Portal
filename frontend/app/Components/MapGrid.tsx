import dynamic from 'next/dynamic';
import { AquiferType, GracePoint, RainfallPoint, WellPoint, GWRResponse, DistrictType, Geometry } from '../types';
import { getAquiferColor, getGraceColor, getRainfallColor, getWellColor, getGWRColor } from '../utils/colorScales';

const MapContainer = dynamic(() => import('react-leaflet').then(m => m.MapContainer), { ssr: false });
const TileLayer = dynamic(() => import('react-leaflet').then(m => m.TileLayer), { ssr: false });
const GeoJSON = dynamic(() => import('react-leaflet').then(m => m.GeoJSON), { ssr: false });
const CircleMarker = dynamic(() => import('react-leaflet').then(m => m.CircleMarker), { ssr: false });
const Popup = dynamic(() => import('react-leaflet').then(m => m.Popup), { ssr: false });

interface MapGridProps {
  activeLayers: Array<{ id: string; name: string; icon: string; show: boolean; color: string }>;
  center: [number, number];
  zoom: number;
  mapKey: number;
  districts: DistrictType[];
  districtGeo: Geometry | null;
  aquifers: AquiferType[];
  graceData: GracePoint[];
  rainfallData: RainfallPoint[];
  wellsData: WellPoint[];
  gwrData: GWRResponse | null;
  selectedDistrict: string;
}

export default function MapGrid({
  activeLayers,
  center,
  zoom,
  mapKey,
  districts,
  districtGeo,
  aquifers,
  graceData,
  rainfallData,
  wellsData,
  gwrData,
  selectedDistrict
}: MapGridProps) {

  const createAquiferPopup = (aquifer: AquiferType, colorIndex: number) => {
    return (feature: unknown, layer: any) => {
      const aquiferColor = getAquiferColor(aquifer.aquifer, colorIndex);
      const popupContent = `
        <div style="font-family: sans-serif; min-width: 200px;">
          <strong style="font-size: 15px; color: ${aquiferColor};">${aquifer.aquifer || 'N/A'}</strong><br/>
          <hr style="margin: 5px 0; border: 1px solid #ddd;"/>
          <table style="width: 100%; font-size: 12px; margin-top: 5px;">
            <tr><td><strong>Type:</strong></td><td>${aquifer.aquifers || 'N/A'}</td></tr>
            <tr><td><strong>Zone (m):</strong></td><td>${aquifer.zone_m || 'N/A'}</td></tr>
            <tr><td><strong>Avg MBGL:</strong></td><td>${aquifer.avg_mbgl || 'N/A'}</td></tr>
          </table>
        </div>
      `;
      if (layer && typeof layer.bindPopup === 'function') {
        layer.bindPopup(popupContent);
      }
    };
  };

  const formatWellPopup = (well: WellPoint): string => {
    const parts = [
      `<div style="font-family: sans-serif; min-width: 200px;">`,
      `<strong style="font-size: 14px; color: ${getWellColor(well.gwl_category)};">Groundwater Level</strong><br/>`,
      `<hr style="margin: 5px 0; border: 1px solid #ddd;"/>`,
      `<table style="width: 100%; font-size: 12px; margin-top: 5px;">`,
      `<tr><td><strong>Date:</strong></td><td>${new Date(well.date).toLocaleDateString()}</td></tr>`,
      `<tr><td><strong>GWL:</strong></td><td>${well.gwl.toFixed(2)} m bgl</td></tr>`,
      `<tr><td><strong>Category:</strong></td><td>${well.gwl_category}</td></tr>`,
      `<tr><td><strong>Site ID:</strong></td><td>${well.site_id}</td></tr>`,
    ];
    if (well.site_name) parts.push(`<tr><td><strong>Site:</strong></td><td>${well.site_name}</td></tr>`);
    if (well.site_type) parts.push(`<tr><td><strong>Type:</strong></td><td>${well.site_type}</td></tr>`);
    if (well.aquifer) parts.push(`<tr><td><strong>Aquifer:</strong></td><td>${well.aquifer}</td></tr>`);
    if (well.season) parts.push(`<tr><td><strong>Season:</strong></td><td>${well.season}</td></tr>`);
    parts.push(`</table></div>`);
    return parts.join('');
  };

  const uniqueAquiferTypesWithColors = Array.from(
    new Map(aquifers.filter(a => a.aquifer).map((a, idx) => [a.aquifer, { type: a.aquifer, color: getAquiferColor(a.aquifer, idx) }])).values()
  );

  const uniqueWellCategories = Array.from(new Set(wellsData.map(w => w.gwl_category))).map(cat => ({ category: cat, color: getWellColor(cat) }));

  const renderMap = (layerId: string, layerName: string, layerIcon: string, borderColor: string) => {
    return (
      <div className="relative h-full rounded-xl overflow-hidden shadow-2xl bg-white" style={{ borderWidth: '3px', borderColor: borderColor, borderStyle: 'solid' }}>
        <div className="absolute top-4 left-4 z-[1000] bg-white/95 backdrop-blur-sm px-4 py-2 rounded-lg shadow-lg border-2" style={{ borderColor: borderColor }}>
          <div className="flex items-center gap-2">
            <span className="text-2xl">{layerIcon}</span>
            <span className="font-bold text-gray-800">{layerName}</span>
          </div>
        </div>
        
        <MapContainer 
          center={center} 
          zoom={zoom} 
          style={{ height: "100%", width: "100%" }} 
          key={`${layerId}-${mapKey}`}
        >
          <TileLayer 
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          />
          
          {!selectedDistrict && districts.map((d, i) => (
            <GeoJSON key={`dist_${i}`} data={d.geometry} style={{ color: "#FF9800", weight: 1, fillOpacity: 0.05 }} />
          ))}
          
          {districtGeo && <GeoJSON data={districtGeo} style={{ color: borderColor, weight: 3, fillOpacity: 0.15 }} />}
          
          {layerId === 'aquifers' && aquifers.map((aquifer, i) => (
            <GeoJSON
              key={`aq_${i}`}
              data={aquifer.geometry}
              style={{ 
                color: getAquiferColor(aquifer.aquifer, i), 
                weight: 2.5, 
                fillColor: getAquiferColor(aquifer.aquifer, i),
                fillOpacity: 0.5,
                opacity: 0.8 
              }}
              onEachFeature={createAquiferPopup(aquifer, i)}
            />
          ))}

          {layerId === 'gwr' && gwrData && gwrData.geojson.features.map((feature, i) => {
            const resourceMCM = feature.properties.annual_resource_mcm;
            const zmin = gwrData.statistics.min_resource_mcm;
            const zmax = gwrData.statistics.max_resource_mcm;
            const fillColor = getGWRColor(resourceMCM, zmin, zmax);
            
            return (
              <GeoJSON
                key={`gwr_${i}`}
                data={feature}
                style={{
                  fillColor: fillColor,
                  fillOpacity: 0.7,
                  color: '#333',
                  weight: 1.5,
                  opacity: 0.8
                }}
                onEachFeature={(feature: any, layer: any) => {
                  const props = feature.properties;
                  const popupContent = `
                    <div style="font-family: sans-serif; min-width: 220px;">
                      <strong style="font-size: 15px; color: ${fillColor};">
                        GWR: ${props.annual_resource_mcm.toFixed(2)} MCM
                      </strong><br/>
                      <hr style="margin: 5px 0; border: 1px solid #ddd;"/>
                      <table style="width: 100%; font-size: 12px; margin-top: 5px;">
                        <tbody>
                          <tr><td><strong>State:</strong></td><td>${props.state}</td></tr>
                          <tr><td><strong>District:</strong></td><td>${props.district}</td></tr>
                          <tr><td><strong>Annual Resource:</strong></td><td>${props.annual_resource_mcm.toFixed(2)} MCM</td></tr>
                          <tr><td><strong>Area:</strong></td><td>${(props.area_m2 / 1_000_000).toFixed(2)} km²</td></tr>
                          <tr><td><strong>Resource/km²:</strong></td><td>${(props.annual_resource_mcm / (props.area_m2 / 1_000_000)).toFixed(3)} MCM/km²</td></tr>
                        </tbody>
                      </table>
                    </div>
                  `;
                  
                  layer.bindPopup(popupContent);
                  
                  layer.on('mouseover', function() {
                    this.setStyle({ weight: 3, color: '#000', fillOpacity: 0.9 });
                  });
                  
                  layer.on('mouseout', function() {
                    this.setStyle({ weight: 1.5, color: '#333', fillOpacity: 0.7 });
                  });
                }}
              />
            );
          })}

          {layerId === 'grace' && graceData.map((point, i) => (
            <CircleMarker
              key={`grace_${i}`}
              center={[point.latitude, point.longitude]}
              radius={5}
              fillColor={getGraceColor(point.lwe_cm)}
              color="white"
              weight={1}
              fillOpacity={0.8}
            >
              <Popup>
                <strong>GRACE LWE</strong><br/>
                Value: {point.lwe_cm.toFixed(2)} cm<br/>
                {point.cell_area_km2 && <>Cell Area: {point.cell_area_km2.toFixed(0)} km²</>}
              </Popup>
            </CircleMarker>
          ))}
          
          {layerId === 'rainfall' && rainfallData.map((point, i) => (
            <CircleMarker
              key={`rain_${i}`}
              center={[point.latitude, point.longitude]}
              radius={5}
              fillColor={getRainfallColor(point.rainfall_mm)}
              color="white"
              weight={1}
              fillOpacity={0.8}
            >
              <Popup>
                <strong>Rainfall</strong><br/>
                Value: {point.rainfall_mm.toFixed(2)} mm/day<br/>
                {point.days_averaged && point.days_averaged > 1 && <>Averaged over: {point.days_averaged} days</>}
              </Popup>
            </CircleMarker>
          ))}
          
          {layerId === 'wells' && wellsData.map((well, i) => (
            <CircleMarker
              key={`well_${well.site_id}_${i}`}
              center={[well.latitude, well.longitude]}
              radius={6}
              fillColor={getWellColor(well.gwl_category)}
              color="white"
              weight={1}
              fillOpacity={0.8}
            >
              <Popup>
                <div dangerouslySetInnerHTML={{ __html: formatWellPopup(well) }} />
              </Popup>
            </CircleMarker>
          ))}
        </MapContainer>

        <div className="absolute bottom-4 right-4 z-[1000] bg-white/95 backdrop-blur-sm p-3 rounded-lg shadow-lg max-w-[200px] max-h-[300px] overflow-y-auto border-2" style={{ borderColor: borderColor }}>
          <div className="text-xs font-bold mb-2 text-gray-700">{layerName} Legend</div>
          
          {layerId === 'aquifers' && uniqueAquiferTypesWithColors.slice(0, 5).map((item, idx) => (
            <div key={idx} className="flex items-center gap-2 mb-1">
              <div className="w-4 h-4 flex-shrink-0 rounded" style={{ backgroundColor: item.color }}></div>
              <span className="text-xs truncate">{item.type}</span>
            </div>
          ))}
          
          {layerId === 'wells' && uniqueWellCategories.map((item, idx) => (
            <div key={idx} className="flex items-center gap-2 mb-1">
              <div className="w-4 h-4 rounded-full flex-shrink-0" style={{ backgroundColor: item.color }}></div>
              <span className="text-xs">{item.category}</span>
            </div>
          ))}
          
          {layerId === 'grace' && (
            <>
              <div className="flex items-center gap-2 mb-1">
                <div className="w-4 h-4 flex-shrink-0" style={{ backgroundColor: "#8B0000" }}></div>
                <span className="text-xs">&lt; -10 cm</span>
              </div>
              <div className="flex items-center gap-2 mb-1">
                <div className="w-4 h-4 flex-shrink-0" style={{ backgroundColor: "#FF6347" }}></div>
                <span className="text-xs">-5 to 0</span>
              </div>
              <div className="flex items-center gap-2 mb-1">
                <div className="w-4 h-4 flex-shrink-0" style={{ backgroundColor: "#32CD32" }}></div>
                <span className="text-xs">0 to 5</span>
              </div>
              <div className="flex items-center gap-2 mb-1">
                <div className="w-4 h-4 flex-shrink-0" style={{ backgroundColor: "#0000CD" }}></div>
                <span className="text-xs">&gt; 10 cm</span>
              </div>
            </>
          )}
          
          {layerId === 'rainfall' && (
            <>
              <div className="flex items-center gap-2 mb-1">
                <div className="w-4 h-4 flex-shrink-0" style={{ backgroundColor: "#F0F0F0" }}></div>
                <span className="text-xs">&lt; 1 mm/day</span>
              </div>
              <div className="flex items-center gap-2 mb-1">
                <div className="w-4 h-4 flex-shrink-0" style={{ backgroundColor: "#4FC3F7" }}></div>
                <span className="text-xs">10-25</span>
              </div>
              <div className="flex items-center gap-2 mb-1">
                <div className="w-4 h-4 flex-shrink-0" style={{ backgroundColor: "#2196F3" }}></div>
                <span className="text-xs">25-50</span>
              </div>
              <div className="flex items-center gap-2 mb-1">
                <div className="w-4 h-4 flex-shrink-0" style={{ backgroundColor: "#0D47A1" }}></div>
                <span className="text-xs">&gt; 100 mm/day</span>
              </div>
            </>
          )}
          
          {layerId === 'gwr' && gwrData && (
            (() => {
              const zmin = gwrData.statistics.min_resource_mcm;
              const zmax = gwrData.statistics.max_resource_mcm;
              const range = zmax - zmin;
              
              const thresholds = [
                { label: `< ${(zmin + range * 0.1).toFixed(0)}`, value: zmin + range * 0.05 },
                { label: `${(zmin + range * 0.1).toFixed(0)}-${(zmin + range * 0.2).toFixed(0)}`, value: zmin + range * 0.15 },
                { label: `${(zmin + range * 0.2).toFixed(0)}-${(zmin + range * 0.4).toFixed(0)}`, value: zmin + range * 0.3 },
                { label: `${(zmin + range * 0.4).toFixed(0)}-${(zmin + range * 0.6).toFixed(0)}`, value: zmin + range * 0.5 },
                { label: `${(zmin + range * 0.6).toFixed(0)}-${(zmin + range * 0.8).toFixed(0)}`, value: zmin + range * 0.7 },
                { label: `> ${(zmin + range * 0.8).toFixed(0)}`, value: zmin + range * 0.9 }
              ];
              
              return (
                <>
                  <div className="text-xs font-bold mb-2 text-gray-700">GWR Resource (MCM)</div>
                  {thresholds.map((item, idx) => (
                    <div key={idx} className="flex items-center gap-2 mb-1">
                      <div 
                        className="w-4 h-4 flex-shrink-0" 
                        style={{ backgroundColor: getGWRColor(item.value, zmin, zmax) }}
                      ></div>
                      <span className="text-xs">{item.label}</span>
                    </div>
                  ))}
                  <div className="text-xs text-gray-500 mt-2 pt-2 border-t">
                    Range: {zmin.toFixed(0)} - {zmax.toFixed(0)} MCM
                  </div>
                </>
              );
            })()
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
      {activeLayers.map((layer) => (
        <div key={layer.id} className="h-[600px]">
          {renderMap(layer.id, layer.name, layer.icon, 
            layer.color === 'purple' ? '#9333EA' :
            layer.color === 'teal' ? '#14B8A6' :
            layer.color === 'green' ? '#059669' : 
            layer.color === 'blue' ? '#2563EB' : '#DC2626'
          )}
        </div>
      ))}
    </div>
  );
}