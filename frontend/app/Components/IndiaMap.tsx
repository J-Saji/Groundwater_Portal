"use client";

import { useEffect, useRef, useState } from 'react';
import { DistrictType, AquiferType, Geometry, GracePoint, RainfallPoint, WellPoint, GWRResponse, GWRFeature } from '../types';
import { getAquiferColor, getGraceColor, getRainfallColor, getWellColor, getGWRColor } from '../Utils/colorScales';
import MapLegend from './MapLegend';

interface IndiaMapProps {
    center: [number, number];
    zoom: number;
    mapKey: number;
    districtGeo: Geometry | null;

    // Layer data
    aquifers: AquiferType[];
    graceData: GracePoint[];
    rainfallData: RainfallPoint[];
    wellsData: WellPoint[];
    gwrData: GWRResponse | null;

    // Layer visibility
    showAquifers: boolean;
    showGrace: boolean;
    showRainfall: boolean;
    showWells: boolean;
    showGWR: boolean;
}

export default function IndiaMap({
    center,
    zoom,
    mapKey,
    districtGeo,
    aquifers,
    graceData,
    rainfallData,
    wellsData,
    gwrData,
    showAquifers,
    showGrace,
    showRainfall,
    showWells,
    showGWR,
}: IndiaMapProps) {
    const mapContainerRef = useRef<HTMLDivElement>(null);
    const mapRef = useRef<any>(null);
    const layersRef = useRef<any[]>([]);
    const [isClient, setIsClient] = useState(false);
    const LRef = useRef<any>(null);

    // Ensure component only renders on client
    useEffect(() => {
        setIsClient(true);
    }, []);

    // Initialize map only once
    useEffect(() => {
        if (!isClient || !mapContainerRef.current) return;

        // Dynamic import of Leaflet
        import('leaflet').then((L) => {
            LRef.current = L;

            // Fix for default marker icons
            delete (L.Icon.Default.prototype as any)._getIconUrl;
            L.Icon.Default.mergeOptions({
                iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
                iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
                shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
            });

            // Initialize map only once
            if (!mapRef.current && mapContainerRef.current) {
                mapRef.current = L.map(mapContainerRef.current, {
                    center,
                    zoom,
                    zoomControl: true,
                });

                // Add base layer
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '© OpenStreetMap contributors',
                    maxZoom: 19,
                }).addTo(mapRef.current);
            }
        });

        return () => {
            // Cleanup map on unmount
            if (mapRef.current) {
                try {
                    mapRef.current.remove();
                } catch (e) {
                    console.error('Error removing map:', e);
                }
                mapRef.current = null;
            }
        };
    }, [isClient]); // Only reinitialize when client status changes

    // Update layers and view when data changes
    useEffect(() => {
        if (!mapRef.current || !LRef.current) return;

        const map = mapRef.current;
        const L = LRef.current;

        // Clear existing custom layers
        layersRef.current.forEach(layer => {
            try {
                if (map.hasLayer(layer)) {
                    map.removeLayer(layer);
                }
            } catch (e) {
                // Silently ignore
            }
        });
        layersRef.current = [];

        // Add district boundary if available
        if (districtGeo) {
            const geoJsonLayer = L.geoJSON(districtGeo as any, {
                style: {
                    color: '#3B82F6',
                    weight: 3,
                    fillOpacity: 0.1,
                    fillColor: '#3B82F6',
                },
            });
            geoJsonLayer.addTo(map);
            layersRef.current.push(geoJsonLayer);
        }

        // Add GWR polygons
        if (showGWR && gwrData && gwrData.geojson && gwrData.geojson.features) {
            const allResources = gwrData.geojson.features.map((f: GWRFeature) => f.properties.annual_resource_mcm);
            const zmin = Math.min(...allResources);
            const zmax = Math.max(...allResources);

            gwrData.geojson.features.forEach((feature: GWRFeature) => {
                const resourceValue = feature.properties.annual_resource_mcm;
                const color = getGWRColor(resourceValue, zmin, zmax);
                const geoJsonLayer = L.geoJSON(feature, {
                    style: {
                        color: color,
                        weight: 2,
                        fillOpacity: 0.6,
                        fillColor: color,
                    },
                });
                geoJsonLayer.bindPopup(`
                    <strong>District:</strong> ${feature.properties.district}<br/>
                    <strong>State:</strong> ${feature.properties.state}<br/>
                    <strong>Resource:</strong> ${resourceValue.toFixed(2)} MCM
                `);
                geoJsonLayer.addTo(map);
                layersRef.current.push(geoJsonLayer);
            });
        }

        // Add aquifer polygons
        if (showAquifers && aquifers.length > 0) {
            aquifers.forEach((aquifer, index) => {
                if (aquifer.geometry) {
                    const color = getAquiferColor(aquifer.aquifer || aquifer.aquifers, index);
                    const geoJsonLayer = L.geoJSON(aquifer.geometry as any, {
                        style: {
                            color: color,
                            weight: 2,
                            fillOpacity: 0.5,
                            fillColor: color,
                        },
                    });
                    const zoneValue = typeof aquifer.zone_m === 'number' ? aquifer.zone_m.toFixed(2) : 'N/A';
                    geoJsonLayer.bindPopup(`
                        <strong>Aquifer:</strong> ${aquifer.aquifer || aquifer.aquifers || 'Unknown'}<br/>
                        <strong>State:</strong> ${aquifer.stname || 'N/A'}<br/>
                        <strong>Zone (m):</strong> ${zoneValue}
                    `);
                    geoJsonLayer.addTo(map);
                    layersRef.current.push(geoJsonLayer);
                }
            });
        }

        // Add GRACE points
        if (showGrace && graceData.length > 0) {
            graceData.forEach((point) => {
                const color = getGraceColor(point.lwe_cm);
                const circle = L.circleMarker([point.latitude, point.longitude], {
                    radius: 6,
                    fillColor: color,
                    color: '#fff',
                    weight: 1,
                    opacity: 1,
                    fillOpacity: 0.8,
                });
                circle.bindPopup(`
                    <strong>GRACE Point</strong><br/>
                    <strong>LWE:</strong> ${point.lwe_cm.toFixed(2)} cm
                `);
                circle.addTo(map);
                layersRef.current.push(circle);
            });
        }

        // Add rainfall points
        if (showRainfall && rainfallData.length > 0) {
            rainfallData.forEach((point) => {
                const color = getRainfallColor(point.rainfall_mm);
                const circle = L.circleMarker([point.latitude, point.longitude], {
                    radius: 5,
                    fillColor: color,
                    color: '#fff',
                    weight: 1,
                    opacity: 1,
                    fillOpacity: 0.7,
                });
                circle.bindPopup(`
                    <strong>Rainfall</strong><br/>
                    <strong>Amount:</strong> ${point.rainfall_mm.toFixed(2)} mm
                `);
                circle.addTo(map);
                layersRef.current.push(circle);
            });
        }

        // Add well points
        if (showWells && wellsData.length > 0) {
            wellsData.forEach((well) => {
                const color = getWellColor(well.gwl_category);
                const circle = L.circleMarker([well.latitude, well.longitude], {
                    radius: 5,
                    fillColor: color,
                    color: '#fff',
                    weight: 1,
                    opacity: 1,
                    fillOpacity: 0.8,
                });
                circle.bindPopup(`
                    <strong>Well</strong><br/>
                    ${well.site_name ? `<strong>Site:</strong> ${well.site_name}<br/>` : ''}
                    <strong>GWL:</strong> ${well.gwl.toFixed(2)} m<br/>
                    <strong>Category:</strong> ${well.gwl_category}
                `);
                circle.addTo(map);
                layersRef.current.push(circle);
            });
        }

        // Update map view
        try {
            map.setView(center, zoom);
        } catch (e) {
            console.error('Error setting view:', e);
        }
    }, [mapKey, center, zoom, districtGeo, aquifers, graceData, rainfallData, wellsData, gwrData, showAquifers, showGrace, showRainfall, showWells, showGWR]);

    if (!isClient) {
        return (
            <div className="h-full w-full rounded-lg overflow-hidden shadow-xl border-2 border-gray-200 flex items-center justify-center bg-gray-100">
                <div className="text-center">
                    <svg className="w-16 h-16 mx-auto text-slate-400 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                    </svg>
                    <div className="text-slate-600 font-medium">Loading map...</div>
                </div>
            </div>
        );
    }

    return (
        <div className="h-full w-full rounded-lg overflow-hidden shadow-xl border-2 border-slate-300 relative">
            <div ref={mapContainerRef} className="h-full w-full relative" style={{ zIndex: 1 }} />
            <MapLegend
                aquifers={aquifers}
                showAquifers={showAquifers}
                showGWR={showGWR}
                showGrace={showGrace}
                showRainfall={showRainfall}
                showWells={showWells}
            />
        </div>
    );
}
