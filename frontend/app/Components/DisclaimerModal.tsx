import { useState, useEffect } from 'react';

interface DisclaimerModalProps {
    onAccept: () => void;
}

export default function DisclaimerModal({ onAccept }: DisclaimerModalProps) {
    const [isVisible, setIsVisible] = useState(false);

    useEffect(() => {
        // Check if user has already accepted the disclaimer in this session
        const hasAccepted = sessionStorage.getItem('disclaimerAccepted');
        if (!hasAccepted) {
            setIsVisible(true);
        } else {
            onAccept();
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []); // Only run once on mount

    const handleAccept = () => {
        sessionStorage.setItem('disclaimerAccepted', 'true');
        setIsVisible(false);
        onAccept();
    };

    if (!isVisible) return null;

    return (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-[100000] p-4">
            <div className="bg-white rounded-lg shadow-2xl max-w-3xl w-full max-h-[90vh] overflow-hidden animate-fadeIn border border-slate-300">
                {/* Header */}
                <div className="bg-white border-b border-slate-200 px-8 py-6">
                    <div className="flex items-start gap-4">
                        <div className="w-10 h-10 bg-slate-100 rounded flex items-center justify-center flex-shrink-0">
                            <svg className="w-6 h-6 text-slate-700" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                            </svg>
                        </div>
                        <div>
                            <h2 className="text-2xl font-semibold text-slate-900">Terms of Use</h2>
                            <p className="text-sm text-slate-500 mt-1">Please review the following information before proceeding</p>
                        </div>
                    </div>
                </div>

                {/* Content */}
                <div className="px-8 py-6 overflow-y-auto max-h-[60vh]">
                    <div className="space-y-6">
                        {/* Data Coverage Section */}
                        <div className="border-l-3 border-l-4 border-slate-300 pl-4">
                            <h3 className="font-semibold text-slate-900 mb-2 text-base">Data Coverage Limitations</h3>
                            <p className="text-sm text-slate-700 leading-relaxed">
                                This platform primarily provides data for Indian states and their districts.
                                <strong className="text-slate-900"> Data for Union Territories may not be available or may be inaccurate.</strong> Please
                                verify critical information with official government sources.
                            </p>
                        </div>

                        {/* Data Constraints Section */}
                        <div className="border-l-3 border-l-4 border-slate-300 pl-4">
                            <h3 className="font-semibold text-slate-900 mb-2 text-base">Data Constraints</h3>
                            <ul className="text-sm text-slate-700 space-y-1.5 leading-relaxed">
                                <li className="flex items-start gap-2">
                                    <span className="text-slate-400 mt-1">•</span>
                                    <span>Some regions may have incomplete or missing historical data</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-slate-400 mt-1">•</span>
                                    <span>GRACE satellite data is available from 2002 onwards</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-slate-400 mt-1">•</span>
                                    <span>Rainfall data coverage varies by year and location</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-slate-400 mt-1">•</span>
                                    <span>Groundwater well monitoring density differs across states</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-slate-400 mt-1">•</span>
                                    <span>Advanced modules require sufficient data points to generate reliable results</span>
                                </li>
                            </ul>
                        </div>

                        {/* AI Assistant Section */}
                        <div className="border-l-3 border-l-4 border-slate-300 pl-4">
                            <h3 className="font-semibold text-slate-900 mb-2 text-base">AI Assistant Limitations</h3>
                            <ul className="text-sm text-slate-700 space-y-1.5 leading-relaxed">
                                <li className="flex items-start gap-2">
                                    <span className="text-slate-400 mt-1">•</span>
                                    <span>The AI chatbot may occasionally enter response loops or provide incomplete answers</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-slate-400 mt-1">•</span>
                                    <span>Complex queries might require rephrasing for better results</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-slate-400 mt-1">•</span>
                                    <span>AI responses are generated based on available data and may not always be completely accurate</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-slate-400 mt-1">•</span>
                                    <span>For critical decisions, verify AI suggestions with domain experts</span>
                                </li>
                            </ul>
                        </div>

                        {/* Usage Recommendations Section */}
                        <div className="border-l-3 border-l-4 border-slate-300 pl-4">
                            <h3 className="font-semibold text-slate-900 mb-2 text-base">Usage Recommendations</h3>
                            <ul className="text-sm text-slate-700 space-y-1.5 leading-relaxed">
                                <li className="flex items-start gap-2">
                                    <span className="text-slate-400 mt-1">•</span>
                                    <span>Begin by selecting a state and optionally a district from the sidebar</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-slate-400 mt-1">•</span>
                                    <span>Enable relevant data layers (Aquifers, GRACE, Rainfall, Wells) to view spatial patterns</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-slate-400 mt-1">•</span>
                                    <span>Use advanced modules for in-depth analysis once a location is selected</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-slate-400 mt-1">•</span>
                                    <span>The AI assistant can help interpret results and answer methodology questions</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-slate-400 mt-1">•</span>
                                    <span>For optimal performance, avoid loading multiple heavy layers simultaneously</span>
                                </li>
                            </ul>
                        </div>

                        {/* Disclaimer Section */}
                        <div className="bg-slate-50 border border-slate-200 rounded p-4">
                            <p className="text-sm text-slate-700 leading-relaxed">
                                <strong className="text-slate-900">Disclaimer:</strong> This platform is designed for research and educational purposes.
                                While we strive for accuracy, the data and analyses should be used as supplementary tools alongside
                                official government sources and expert consultation for policy or management decisions.
                            </p>
                        </div>
                    </div>
                </div>

                {/* Footer */}
                <div className="px-8 py-5 bg-slate-50 border-t border-slate-200 flex items-center justify-between">
                    <div className="flex items-center gap-2 text-xs text-slate-500">
                        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
                        </svg>
                        <span>This notice will only appear once</span>
                    </div>
                    <button
                        onClick={handleAccept}
                        className="px-6 py-2.5 bg-slate-700 hover:bg-slate-800 text-white font-medium rounded shadow hover:shadow-md transition-all"
                    >
                        I Understand and Agree
                    </button>
                </div>
            </div>

            <style jsx>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(-10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-fadeIn {
          animation: fadeIn 0.2s ease-out;
        }
      `}</style>
        </div>
    );
}
