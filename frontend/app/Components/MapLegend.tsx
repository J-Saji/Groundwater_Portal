"use client";

import { AquiferType } from '../types';
import { getAquiferColor } from '../Utils/colorScales';

interface MapLegendProps {
    aquifers: AquiferType[];
    showAquifers: boolean;
    showGWR: boolean;
    showGrace: boolean;
    showRainfall: boolean;
    showWells: boolean;
}

export default function MapLegend({
    aquifers,
    showAquifers,
    showGWR,
    showGrace,
    showRainfall,
    showWells,
}: MapLegendProps) {
    // Dynamically extract unique rock types from actual data
    const getUniqueAquiferTypes = () => {
        console.log('MapLegend - aquifers array:', aquifers);
        console.log('MapLegend - aquifers length:', aquifers?.length);

        if (!aquifers || aquifers.length === 0) {
            console.log('MapLegend - NO AQUIFERS DATA!');
            return [];
        }

        const typeMap = new Map<string, { name: string; color: string }>();

        aquifers.forEach((aquifer, index) => {
            const typeName = aquifer.aquifer || aquifer.aquifers || 'Unknown';
            console.log('MapLegend - Processing aquifer:', typeName, 'Color:', getAquiferColor(typeName, index));
            if (!typeMap.has(typeName)) {
                typeMap.set(typeName, {
                    name: typeName,
                    color: getAquiferColor(typeName, index)
                });
            }
        });

        const result = Array.from(typeMap.values());
        console.log('MapLegend - Final aquifer types for legend:', result);
        return result;
    };

    const aquiferTypes = getUniqueAquiferTypes();

    const graceRanges = [
        { label: '< -10 cm', color: '#8B0000' },
        { label: '-10 to -5 cm', color: '#DC143C' },
        { label: '-5 to 0 cm', color: '#FF6347' },
        { label: '0 to 5 cm', color: '#32CD32' },
        { label: '5 to 10 cm', color: '#1E90FF' },
        { label: '> 10 cm', color: '#0000CD' },
    ];

    const rainfallRanges = [
        { label: '< 1 mm', color: '#F0F0F0' },
        { label: '1-10 mm', color: '#B3E5FC' },
        { label: '10-25 mm', color: '#4FC3F7' },
        { label: '25-50 mm', color: '#2196F3' },
        { label: '50-100 mm', color: '#1976D2' },
        { label: '> 100 mm', color: '#0D47A1' },
    ];

    const wellCategories = [
        { label: 'Recharge', color: '#1E88E5' },
        { label: 'Shallow (0-30m)', color: '#43A047' },
        { label: 'Moderate (30-60m)', color: '#FB8C00' },
        { label: 'Deep (60-100m)', color: '#E53935' },
        { label: 'Very Deep (>100m)', color: '#B71C1C' },
    ];

    const hasAnyLayer = showAquifers || showGWR || showGrace || showRainfall || showWells;

    if (!hasAnyLayer) return null;

    return (
        <div className="absolute bottom-6 left-6 bg-white rounded-lg shadow-xl border-2 border-slate-300 p-4 max-w-xs z-30 max-h-[70vh] overflow-y-auto">
            <h3 className="text-sm font-bold text-slate-800 mb-3">Map Legend</h3>

            {showAquifers && aquiferTypes.length > 0 && (
                <div className="mb-4">
                    <h4 className="text-xs font-semibold text-slate-700 mb-2">Aquifer Types</h4>
                    <div className="space-y-1">
                        {aquiferTypes.map((type) => (
                            <div key={type.name} className="flex items-center gap-2">
                                <div className="w-4 h-4 rounded border border-slate-400 flex-shrink-0" style={{ backgroundColor: type.color }} />
                                <span className="text-xs text-slate-600 truncate">{type.name}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {showGWR && (
                <div className="mb-4">
                    <h4 className="text-xs font-semibold text-slate-700 mb-2">Groundwater Resources</h4>
                    <div className="flex items-center gap-1">
                        <span className="text-xs text-slate-600">Low</span>
                        <div className="flex-1 h-4 rounded" style={{
                            background: 'linear-gradient(to right, #FFFFCC, #C2E699, #78C679, #31A354, #006837, #004529)'
                        }} />
                        <span className="text-xs text-slate-600">High</span>
                    </div>
                </div>
            )}

            {showGrace && (
                <div className="mb-4">
                    <h4 className="text-xs font-semibold text-slate-700 mb-2">GRACE (LWE)</h4>
                    <div className="space-y-1">
                        {graceRanges.map((range) => (
                            <div key={range.label} className="flex items-center gap-2">
                                <div className="w-4 h-4 rounded-full border border-slate-400" style={{ backgroundColor: range.color }} />
                                <span className="text-xs text-slate-600">{range.label}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {showRainfall && (
                <div className="mb-4">
                    <h4 className="text-xs font-semibold text-slate-700 mb-2">Rainfall</h4>
                    <div className="space-y-1">
                        {rainfallRanges.map((range) => (
                            <div key={range.label} className="flex items-center gap-2">
                                <div className="w-4 h-4 rounded-full border border-slate-400" style={{ backgroundColor: range.color }} />
                                <span className="text-xs text-slate-600">{range.label}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {showWells && (
                <div className="mb-0">
                    <h4 className="text-xs font-semibold text-slate-700 mb-2">Well Depth</h4>
                    <div className="space-y-1">
                        {wellCategories.map((category) => (
                            <div key={category.label} className="flex items-center gap-2">
                                <div className="w-4 h-4 rounded-full border border-slate-400" style={{ backgroundColor: category.color }} />
                                <span className="text-xs text-slate-600">{category.label}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
