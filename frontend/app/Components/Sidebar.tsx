"use client";

import { useState } from 'react';
import { StateType, DistrictType } from '../types';

interface SidebarProps {
    isExpanded: boolean;
    onToggle: () => void;
    states: StateType[];
    districts: DistrictType[];
    selectedState: string;
    selectedDistrict: string;
    selectedYear: number;
    selectedMonth: number | null;
    availableYears: number[];

    // Layer toggles
    showAquifers: boolean;
    showGWR: boolean;
    showGrace: boolean;
    showRainfall: boolean;
    showWells: boolean;
    showTimeseries: boolean;
    showStorageVsGWR: boolean;

    // Handlers
    onStateChange: (state: string) => void;
    onDistrictChange: (district: string) => void;
    onYearChange: (year: number) => void;
    onMonthChange: (month: number | null) => void;
    onToggleAquifers: () => void;
    onToggleGWR: () => void;
    onToggleGrace: () => void;
    onToggleRainfall: () => void;
    onToggleWells: () => void;
    onToggleTimeseries: () => void;
    onToggleStorageVsGWR: () => void;
}

export default function Sidebar({
    isExpanded,
    onToggle,
    states,
    districts,
    selectedState,
    selectedDistrict,
    selectedYear,
    selectedMonth,
    availableYears,
    showAquifers,
    showGWR,
    showGrace,
    showRainfall,
    showWells,
    onStateChange,
    onDistrictChange,
    onYearChange,
    onMonthChange,
    onToggleAquifers,
    onToggleGWR,
    onToggleGrace,
    onToggleRainfall,
    onToggleWells,
    showTimeseries,
    showStorageVsGWR,
    onToggleTimeseries,
    onToggleStorageVsGWR,
}: SidebarProps) {
    // Collapsible sections state
    const [isLocationExpanded, setIsLocationExpanded] = useState(false);
    const [isTimeExpanded, setIsTimeExpanded] = useState(false);
    const layers = [
        { id: 'aquifers', name: 'Aquifers', active: showAquifers, toggle: onToggleAquifers, color: '#8B5CF6' },
        { id: 'gwr', name: 'GWR', active: showGWR, toggle: onToggleGWR, color: '#0EA5E9' },
        { id: 'grace', name: 'GRACE', active: showGrace, toggle: onToggleGrace, color: '#10B981' },
        { id: 'rainfall', name: 'Rainfall', active: showRainfall, toggle: onToggleRainfall, color: '#3B82F6' },
        { id: 'wells', name: 'Wells', active: showWells, toggle: onToggleWells, color: '#EF4444' },
    ];

    const months = [
        { value: 1, label: 'January' },
        { value: 2, label: 'February' },
        { value: 3, label: 'March' },
        { value: 4, label: 'April' },
        { value: 5, label: 'May' },
        { value: 6, label: 'June' },
        { value: 7, label: 'July' },
        { value: 8, label: 'August' },
        { value: 9, label: 'September' },
        { value: 10, label: 'October' },
        { value: 11, label: 'November' },
        { value: 12, label: 'December' },
    ];

    return (
        <div
            className={`fixed left-0 top-14 h-[calc(100vh-3.5rem)] bg-slate-50 shadow-lg transition-all duration-300 ease-in-out z-40 border-r border-slate-300 ${isExpanded ? 'w-80' : 'w-14'
                }`}
        >
            {/* Toggle Button */}
            <button
                onClick={onToggle}
                className="absolute -right-3 top-6 w-6 h-6 bg-slate-700 text-white rounded-full shadow-md hover:bg-slate-600 transition-all flex items-center justify-center z-50"
                title={isExpanded ? 'Collapse' : 'Expand'}
            >
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d={isExpanded ? "M15 19l-7-7 7-7" : "M9 5l7 7-7 7"} />
                </svg>
            </button>

            {/* Collapsed View - Icons Only */}
            {!isExpanded && (
                <div className="flex flex-col items-center gap-4 p-2 mt-8">
                    {layers.filter(l => l.active).map((layer) => (
                        <div
                            key={layer.id}
                            className="w-10 h-10 rounded-lg flex items-center justify-center transition-all cursor-pointer hover:scale-105"
                            title={layer.name}
                            style={{ backgroundColor: layer.color + '20', border: `2px solid ${layer.color}` }}
                        >
                            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: layer.color }} />
                        </div>
                    ))}
                </div>
            )}

            {/* Expanded View */}
            {isExpanded && (
                <div className="h-full overflow-y-auto p-5">
                    <h2 className="text-lg font-semibold text-slate-800 mb-5">
                        Map Configuration
                    </h2>

                    {/* Layer Selection */}
                    <div className="mb-6">
                        <h3 className="text-xs font-semibold text-slate-600 uppercase mb-2 tracking-wider">Data Layers</h3>
                        <div className="space-y-1.5">
                            {layers.map((layer) => (
                                <button
                                    key={layer.id}
                                    onClick={layer.toggle}
                                    className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-md transition-all text-sm ${layer.active
                                        ? 'bg-white border border-slate-300 shadow-sm'
                                        : 'bg-slate-100 border border-transparent hover:bg-slate-200'
                                        }`}
                                >
                                    <div
                                        className="w-3 h-3 rounded-sm border"
                                        style={{
                                            backgroundColor: layer.active ? layer.color : 'transparent',
                                            borderColor: layer.active ? layer.color : '#94a3b8'
                                        }}
                                    />
                                    <span className={`flex-1 text-left font-medium ${layer.active ? 'text-slate-900' : 'text-slate-600'}`}>
                                        {layer.name}
                                    </span>
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Analysis Tools */}
                    <div className="mb-6">
                        <h3 className="text-xs font-semibold text-slate-600 uppercase mb-2 tracking-wider">Analysis Charts</h3>
                        <div className="space-y-1.5">
                            <button
                                onClick={onToggleTimeseries}
                                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-md transition-all text-sm ${showTimeseries
                                    ? 'bg-white border border-slate-300 shadow-sm'
                                    : 'bg-slate-100 border border-transparent hover:bg-slate-200'
                                    }`}
                            >
                                <div
                                    className="w-3 h-3 rounded-sm border"
                                    style={{
                                        backgroundColor: showTimeseries ? '#6366F1' : 'transparent',
                                        borderColor: showTimeseries ? '#6366F1' : '#94a3b8'
                                    }}
                                />
                                <span className={`flex-1 text-left font-medium ${showTimeseries ? 'text-slate-900' : 'text-slate-600'}`}>
                                    Timeseries
                                </span>
                            </button>
                            <button
                                onClick={onToggleStorageVsGWR}
                                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-md transition-all text-sm ${showStorageVsGWR
                                    ? 'bg-white border border-slate-300 shadow-sm'
                                    : 'bg-slate-100 border border-transparent hover:bg-slate-200'
                                    }`}
                            >
                                <div
                                    className="w-3 h-3 rounded-sm border"
                                    style={{
                                        backgroundColor: showStorageVsGWR ? '#8B5CF6' : 'transparent',
                                        borderColor: showStorageVsGWR ? '#8B5CF6' : '#94a3b8'
                                    }}
                                />
                                <span className={`flex-1 text-left font-medium ${showStorageVsGWR ? 'text-slate-900' : 'text-slate-600'}`}>
                                    Storage vs GWR
                                </span>
                            </button>
                        </div>
                    </div>

                    {/* Location Selection - Collapsible */}
                    <div className="mb-4">
                        <button
                            onClick={() => setIsLocationExpanded(!isLocationExpanded)}
                            className="w-full flex items-center justify-between px-3 py-2.5 rounded-md bg-slate-100 hover:bg-slate-200 transition-colors"
                        >
                            <div className="flex items-center gap-2">
                                <svg className="w-4 h-4 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                                </svg>
                                <h3 className="text-sm font-semibold text-slate-700">Location</h3>
                                {(selectedState || selectedDistrict) && !isLocationExpanded && (
                                    <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full font-medium">
                                        {selectedDistrict || selectedState}
                                    </span>
                                )}
                            </div>
                            <svg
                                className={`w-4 h-4 text-slate-600 transition-transform ${isLocationExpanded ? 'rotate-180' : ''}`}
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                            >
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                            </svg>
                        </button>

                        {isLocationExpanded && (
                            <div className="mt-3 space-y-3 animate-fadeIn">
                                {/* State Select */}
                                <div>
                                    <label className="block text-xs font-medium text-slate-600 mb-1.5">State</label>
                                    <select
                                        value={selectedState}
                                        onChange={(e) => onStateChange(e.target.value)}
                                        className="w-full px-3 py-2 border border-slate-300 rounded-md focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none bg-white text-sm"
                                    >
                                        <option value="">All India</option>
                                        {states.map((state) => (
                                            <option key={state.State} value={state.State}>
                                                {state.State}
                                            </option>
                                        ))}
                                    </select>
                                </div>

                                {/* District Select */}
                                {selectedState && (
                                    <div>
                                        <label className="block text-xs font-medium text-slate-600 mb-1.5">District</label>
                                        <select
                                            value={selectedDistrict}
                                            onChange={(e) => onDistrictChange(e.target.value)}
                                            className="w-full px-3 py-2 border border-slate-300 rounded-md focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none bg-white text-sm"
                                        >
                                            <option value="">All Districts</option>
                                            {districts.map((district) => (
                                                <option key={district.district_name} value={district.district_name}>
                                                    {district.district_name}
                                                </option>
                                            ))}
                                        </select>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>

                    {/* Time Selection - Collapsible */}
                    <div className="mb-4">
                        <button
                            onClick={() => setIsTimeExpanded(!isTimeExpanded)}
                            className="w-full flex items-center justify-between px-3 py-2.5 rounded-md bg-slate-100 hover:bg-slate-200 transition-colors"
                        >
                            <div className="flex items-center gap-2">
                                <svg className="w-4 h-4 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                </svg>
                                <h3 className="text-sm font-semibold text-slate-700">Time Period</h3>
                                {!isTimeExpanded && (
                                    <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full font-medium">
                                        {selectedYear}{selectedMonth ? ` / ${months.find(m => m.value === selectedMonth)?.label.slice(0, 3)}` : ''}
                                    </span>
                                )}
                            </div>
                            <svg
                                className={`w-4 h-4 text-slate-600 transition-transform ${isTimeExpanded ? 'rotate-180' : ''}`}
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                            >
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                            </svg>
                        </button>

                        {isTimeExpanded && (
                            <div className="mt-3 space-y-3 animate-fadeIn">
                                {/* Year Select */}
                                <div>
                                    <label className="block text-xs font-medium text-slate-600 mb-1.5">Year</label>
                                    <select
                                        value={selectedYear}
                                        onChange={(e) => onYearChange(Number(e.target.value))}
                                        className="w-full px-3 py-2 border border-slate-300 rounded-md focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none bg-white text-sm"
                                    >
                                        {availableYears.map((year) => (
                                            <option key={year} value={year}>
                                                {year}
                                            </option>
                                        ))}
                                    </select>
                                </div>

                                {/* Month Select */}
                                <div>
                                    <label className="block text-xs font-medium text-slate-600 mb-1.5">Month</label>
                                    <select
                                        value={selectedMonth || ''}
                                        onChange={(e) => onMonthChange(e.target.value ? Number(e.target.value) : null)}
                                        className="w-full px-3 py-2 border border-slate-300 rounded-md focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none bg-white text-sm"
                                    >
                                        <option value="">All Year</option>
                                        {months.map((month) => (
                                            <option key={month.value} value={month.value}>
                                                {month.label}
                                            </option>
                                        ))}
                                    </select>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Active Layers Summary */}
                    {layers.some(l => l.active) && (
                        <div className="bg-white rounded-md p-3 border border-slate-200">
                            <h4 className="text-xs font-semibold text-slate-600 mb-2">Active Layers</h4>
                            <div className="flex flex-wrap gap-1.5">
                                {layers.filter(l => l.active).map((layer) => (
                                    <div
                                        key={layer.id}
                                        className="flex items-center gap-1.5 bg-slate-100 px-2 py-1 rounded text-xs font-medium"
                                    >
                                        <div className="w-2 h-2 rounded-full" style={{ backgroundColor: layer.color }} />
                                        <span className="text-slate-700">{layer.name}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
