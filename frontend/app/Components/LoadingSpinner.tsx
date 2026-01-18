export default function LoadingSpinner() {
  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-[10000] flex items-center justify-center">
      <div className="bg-white rounded-2xl p-8 shadow-2xl text-center">
        <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-600 mx-auto mb-4"></div>
        <p className="text-lg font-semibold text-gray-700">Loading data...</p>
      </div>
    </div>
  );
}