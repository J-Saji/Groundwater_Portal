interface HeaderProps {
  onChatToggle: () => void;
  isChatOpen: boolean;
}

export default function Header({ onChatToggle, isChatOpen }: HeaderProps) {
  return (
    <div className="fixed top-0 left-0 right-0 z-50 bg-gradient-to-r from-slate-800 via-slate-700 to-slate-800 text-white shadow-lg border-b border-slate-600">
      <div className="px-6 py-3.5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-blue-500 rounded flex items-center justify-center font-bold text-sm">
                GH
              </div>
              <h1 className="text-xl font-semibold tracking-tight">
                GeoHydro Dashboard
              </h1>
            </div>
            <span className="text-xs text-slate-300 hidden md:block font-light">
              Advanced Groundwater Analysis & Remote Sensing Platform
            </span>
          </div>
          <button
            onClick={onChatToggle}
            className="bg-slate-600 hover:bg-slate-500 text-white px-4 py-2 rounded-md font-medium transition-all duration-200 flex items-center gap-2 text-sm border border-slate-500 hover:border-slate-400"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
            </svg>
            <span className="hidden sm:inline">{isChatOpen ? 'Close AI Chat' : 'AI Assistant'}</span>
          </button>
        </div>
      </div>
    </div>
  );
}