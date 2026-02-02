"use client";

import { ReactNode } from 'react';

interface AdvancedModulePanelProps {
    children: ReactNode;
    moduleName: string;
    onClose: () => void;
}

export default function AdvancedModulePanel({
    children,
    moduleName,
    onClose,
}: AdvancedModulePanelProps) {
    return (
        <div className="h-full w-full bg-white rounded-lg overflow-hidden shadow-xl border-2 border-blue-300 flex flex-col">
            {/* Header */}
            <div className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white p-4 flex items-center justify-between flex-shrink-0">
                <h2 className="text-xl font-bold">
                    {moduleName}
                </h2>
                <button
                    onClick={onClose}
                    className="w-8 h-8 bg-white bg-opacity-20 hover:bg-opacity-30 rounded-full flex items-center justify-center transition-all"
                    title="Close module"
                >
                    ✕
                </button>
            </div>

            {/* Content - Scrollable */}
            <div className="flex-1 overflow-y-auto p-4 bg-gray-50">
                {children}
            </div>
        </div>
    );
}
