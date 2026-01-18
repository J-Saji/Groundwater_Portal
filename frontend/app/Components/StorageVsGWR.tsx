import dynamic from 'next/dynamic';
import { StorageVsGWRResponse } from '../types';

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface StorageVsGWRProps {
  storageVsGwrData: StorageVsGWRResponse;
}

export default function StorageVsGWR({ storageVsGwrData }: StorageVsGWRProps) {
  return (
    <div className="bg-white rounded-xl shadow-2xl p-6 mb-8 border-2 border-cyan-400">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-gray-800 flex items-center gap-2">
          <span>📊</span> Storage Change vs Annual GWR
        </h3>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
        <div className="bg-cyan-50 p-4 rounded-lg border border-cyan-200">
          <div className="text-xs text-gray-600 mb-1">Aquifer Area</div>
          <div className="text-2xl font-bold text-cyan-600">
            {storageVsGwrData.aquifer_properties?.total_area_km2?.toFixed(0) || 'N/A'} km²
          </div>
        </div>
        <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
          <div className="text-xs text-gray-600 mb-1">Avg Storage Change</div>
          <div className="text-2xl font-bold text-blue-600">
            {storageVsGwrData.storage_statistics?.avg_annual_storage_change_mcm?.toFixed(2) || 'N/A'} MCM
          </div>
        </div>
        <div className="bg-green-50 p-4 rounded-lg border border-green-200">
          <div className="text-xs text-gray-600 mb-1">Avg GWR</div>
          <div className="text-2xl font-bold text-green-600">
            {storageVsGwrData.gwr_statistics?.avg_annual_resource_mcm?.toFixed(2) || 'N/A'} MCM
          </div>
        </div>
        <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
          <div className="text-xs text-gray-600 mb-1">Specific Yield</div>
          <div className="text-2xl font-bold text-purple-600">
            {storageVsGwrData.aquifer_properties?.area_weighted_specific_yield?.toFixed(4) || 'N/A'}
          </div>
        </div>
      </div>

      {/* Plotly Chart */}
      <Plot
        data={[
          {
            x: storageVsGwrData.data.map(d => d.year),
            y: storageVsGwrData.data.map(d => d.storage_change_mcm),
            type: 'bar',
            name: 'Storage Change (MCM)',
            marker: { color: '#3B82F6' },
            yaxis: 'y1'
          },
          {
            x: storageVsGwrData.data.map(d => d.year),
            y: storageVsGwrData.data.map(d => d.gwr_resource_mcm),
            type: 'bar',
            name: 'Annual GWR (MCM)',
            marker: { color: '#10B981' },
            yaxis: 'y1'
          }
        ]}
        layout={{
          autosize: true,
          height: 500,
          title: 'Storage Change vs Annual Replenishable Groundwater Resource',
          xaxis: { title: 'Year' },
          yaxis: { title: 'Volume (MCM)' },
          barmode: 'group',
          plot_bgcolor: 'white',
          paper_bgcolor: 'white'
        }}
        config={{
          displayModeBar: true,
          displaylogo: false
        }}
        style={{ width: '100%' }}
        useResizeHandler={true}
      />
    </div>
  );
}