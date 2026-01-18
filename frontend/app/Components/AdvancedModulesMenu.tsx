"use client";

interface AdvancedModulesMenuProps {
    isOpen: boolean;
    onClose: () => void;
    onSelectModule: (module: string) => void;
    selectedModule: string;
}

export default function AdvancedModulesMenu({
    isOpen,
    onClose,
    onSelectModule,
    selectedModule,
}: AdvancedModulesMenuProps) {
    const modules = [
        { id: 'ASI', name: 'Aquifer Stress Index', icon: '📊', color: '#8B5CF6', description: 'Analyze aquifer stress levels' },
        { id: 'NETWORK_DENSITY', name: 'Network Density', icon: '🕸️', color: '#EC4899', description: 'Well network distribution analysis' },
        { id: 'SASS', name: 'SASS Analysis', icon: '🎯', color: '#F59E0B', description: 'Spatiotemporal stress assessment' },
        { id: 'GRACE_DIVERGENCE', name: 'GRACE Divergence', icon: '📈', color: '#10B981', description: 'Storage anomaly detection' },
        { id: 'FORECAST', name: 'GW Forecast', icon: '🔮', color: '#3B82F6', description: 'Groundwater level predictions' },
        { id: 'RECHARGE', name: 'Recharge Planning', icon: '💧', color: '#06B6D4', description: 'Optimal recharge strategies' },
        { id: 'SIGNIFICANT_TRENDS', name: 'Trend Analysis', icon: '📉', color: '#EF4444', description: 'Significant trend detection' },
        { id: 'CHANGEPOINTS', name: 'Changepoints', icon: '⚡', color: '#F97316', description: 'Identify regime shifts' },
        { id: 'LAG_CORRELATION', name: 'Lag Correlation', icon: '🔄', color: '#84CC16', description: 'Temporal correlation analysis' },
        { id: 'HOTSPOTS', name: 'Hotspots', icon: '🔥', color: '#DC2626', description: 'Critical zones identification' },
    ];

    if (!isOpen) return null;

    return (
        <>
            {/* Menu Panel */}
            <div className="fixed right-0 top-14 h-[calc(100vh-3.5rem)] w-96 bg-white shadow-2xl z-50 overflow-y-auto border-l-2 border-slate-300 animate-slide-in-right">
                <div className="p-6">
                    {/* Header */}
                    <div className="flex items-center justify-between mb-6">
                        <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
                            <span>⚙️</span>
                            <span>Advanced Modules</span>
                        </h2>
                        <button
                            onClick={onClose}
                            className="w-8 h-8 bg-gray-100 hover:bg-gray-200 rounded-full flex items-center justify-center transition-colors"
                        >
                            ✕
                        </button>
                    </div>

                    {/* Current Selection */}
                    {selectedModule && (
                        <div className="mb-6 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-4 border-2 border-blue-400">
                            <p className="text-xs font-semibold text-gray-600 uppercase mb-1">Currently Active</p>
                            <p className="text-lg font-bold text-blue-900">
                                {modules.find(m => m.id === selectedModule)?.name || selectedModule}
                            </p>
                            <button
                                onClick={() => onSelectModule('')}
                                className="mt-2 text-xs text-blue-600 hover:text-blue-800 font-medium"
                            >
                                Clear Selection
                            </button>
                        </div>
                    )}

                    {/* Modules Grid */}
                    <div className="space-y-3">
                        {modules.map((module) => (
                            <button
                                key={module.id}
                                onClick={() => {
                                    onSelectModule(module.id);
                                    onClose();
                                }}
                                className={`w-full text-left p-4 rounded-xl transition-all border-2 ${selectedModule === module.id
                                    ? 'bg-gradient-to-r from-blue-50 to-blue-100 border-blue-400 shadow-lg'
                                    : 'bg-white border-gray-200 hover:border-gray-300 hover:shadow-md'
                                    }`}
                            >
                                <div className="flex items-start gap-3">
                                    <div
                                        className="w-12 h-12 rounded-lg flex items-center justify-center text-2xl shadow-md"
                                        style={{ backgroundColor: module.color + '20' }}
                                    >
                                        {module.icon}
                                    </div>
                                    <div className="flex-1">
                                        <h3 className="font-bold text-gray-900 mb-1">{module.name}</h3>
                                        <p className="text-xs text-gray-600">{module.description}</p>
                                    </div>
                                    {selectedModule === module.id && (
                                        <div className="w-6 h-6 bg-blue-600 rounded-full flex items-center justify-center">
                                            <span className="text-white text-xs">✓</span>
                                        </div>
                                    )}
                                </div>
                            </button>
                        ))}
                    </div>

                    {/* Info Footer */}
                    <div className="mt-6 p-4 bg-gray-50 rounded-lg border-2 border-gray-200">
                        <p className="text-xs text-gray-600">
                            💡 <strong>Tip:</strong> Select a module to view advanced analytics in split-screen mode alongside the India map.
                        </p>
                    </div>
                </div>
            </div>

            <style jsx>{`
        @keyframes slide-in-right {
          from {
            transform: translateX(100%);
          }
          to {
            transform: translateX(0);
          }
        }
        .animate-slide-in-right {
          animation: slide-in-right 0.3s ease-out;
        }
      `}</style>
        </>
    );
}
