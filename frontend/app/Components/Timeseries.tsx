import dynamic from 'next/dynamic';
import { TimeseriesResponse } from '../types';
import { GWL_COLOR, GRACE_COLOR, RAIN_COLOR } from '../utils/constants';

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface TimeseriesProps {
  timeseriesResponse: TimeseriesResponse;
  timeseriesView: 'raw' | 'seasonal' | 'deseasonalized';
  onViewChange: (view: 'raw' | 'seasonal' | 'deseasonalized') => void;
}

export default function Timeseries({
  timeseriesResponse,
  timeseriesView,
  onViewChange
}: TimeseriesProps) {
  return (
    <div className="bg-white rounded-xl shadow-2xl p-6 mb-8 border-2 border-indigo-400">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-gray-800 flex items-center gap-2">
          <span>📈</span> Unified Timeseries Analysis
        </h3>
        <div className="flex gap-2">
          {(['raw', 'seasonal', 'deseasonalized'] as const).map((view) => (
            <button
              key={view}
              onClick={() => onViewChange(view)}
              className={`px-4 py-2 rounded-lg font-semibold text-sm transition-all ${timeseriesView === view
                ? 'bg-indigo-500 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
            >
              {view.charAt(0).toUpperCase() + view.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {timeseriesResponse.statistics && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          {timeseriesResponse.statistics.gwl_trend && (
            <div className={`p-4 rounded-lg border-2 ${timeseriesResponse.statistics.gwl_trend.direction === 'Declining'
              ? 'bg-red-50 border-red-300'
              : 'bg-green-50 border-green-300'
              }`}>
              <div className="text-sm font-semibold text-gray-700 mb-2">GWL Trend</div>
              <div className={`text-2xl font-bold ${timeseriesResponse.statistics.gwl_trend.direction === 'Declining'
                ? 'text-red-600'
                : 'text-green-600'
                }`}>
                {timeseriesResponse.statistics.gwl_trend.slope_per_year.toFixed(4)} m/yr
              </div>
              <div className="text-xs text-gray-600 mt-1">
                R² = {timeseriesResponse.statistics.gwl_trend.r_squared.toFixed(3)}
              </div>
            </div>
          )}

          {timeseriesResponse.statistics.grace_trend && (
            <div className="p-4 rounded-lg bg-green-50 border-2 border-green-300">
              <div className="text-sm font-semibold text-gray-700 mb-2">GRACE Trend</div>
              <div className="text-2xl font-bold text-green-600">
                {timeseriesResponse.statistics.grace_trend.slope_per_year.toFixed(4)} cm/yr
              </div>
              <div className="text-xs text-gray-600 mt-1">
                R² = {timeseriesResponse.statistics.grace_trend.r_squared.toFixed(3)}
              </div>
            </div>
          )}

          {timeseriesResponse.statistics.rainfall_trend && (
            <div className="p-4 rounded-lg bg-blue-50 border-2 border-blue-300">
              <div className="text-sm font-semibold text-gray-700 mb-2">Rainfall Trend</div>
              <div className="text-2xl font-bold text-blue-600">
                {timeseriesResponse.statistics.rainfall_trend.slope_per_year.toFixed(4)} mm/yr
              </div>
              <div className="text-xs text-gray-600 mt-1">
                R² = {timeseriesResponse.statistics.rainfall_trend.r_squared.toFixed(3)}
              </div>
            </div>
          )}
        </div>
      )}


      {/* Professional Trend Analysis Section */}
      {timeseriesResponse.interpretations?.actionable_insights &&
        timeseriesResponse.interpretations.actionable_insights.length > 0 && (
          <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden mb-6">
            {/* Header */}
            <div className="bg-gradient-to-r from-slate-700 to-slate-900 px-6 py-4">
              <div className="flex items-center justify-between">
                <h4 className="text-lg font-semibold text-white flex items-center gap-3">
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                  Trend Analysis & Key Insights
                </h4>
                <span className="text-xs text-slate-300 font-medium tracking-wide">
                  STATISTICAL INTERPRETATION
                </span>
              </div>
            </div>

            {/* Insights Content */}
            <div className="p-6 bg-gradient-to-br from-gray-50 to-slate-50">
              <div className="space-y-4">
                {timeseriesResponse.interpretations.actionable_insights.map((insight: any, idx: number) => {
                  const severityConfig = {
                    'CRITICAL': {
                      bg: 'bg-gradient-to-r from-red-50 to-red-100',
                      border: 'border-l-4 border-red-600',
                      badge: 'bg-red-600 text-white',
                      icon: '⚠',
                      textColor: 'text-red-900'
                    },
                    'HIGH': {
                      bg: 'bg-gradient-to-r from-orange-50 to-orange-100',
                      border: 'border-l-4 border-orange-600',
                      badge: 'bg-orange-600 text-white',
                      icon: '▲',
                      textColor: 'text-orange-900'
                    },
                    'MODERATE': {
                      bg: 'bg-gradient-to-r from-yellow-50 to-yellow-100',
                      border: 'border-l-4 border-yellow-600',
                      badge: 'bg-yellow-600 text-white',
                      icon: '◆',
                      textColor: 'text-yellow-900'
                    },
                    'WARNING': {
                      bg: 'bg-gradient-to-r from-amber-50 to-amber-100',
                      border: 'border-l-4 border-amber-600',
                      badge: 'bg-amber-600 text-white',
                      icon: '◈',
                      textColor: 'text-amber-900'
                    },
                    'POSITIVE': {
                      bg: 'bg-gradient-to-r from-emerald-50 to-green-100',
                      border: 'border-l-4 border-emerald-600',
                      badge: 'bg-emerald-600 text-white',
                      icon: '✓',
                      textColor: 'text-emerald-900'
                    },
                    'INFO': {
                      bg: 'bg-gradient-to-r from-blue-50 to-indigo-100',
                      border: 'border-l-4 border-blue-600',
                      badge: 'bg-blue-600 text-white',
                      icon: 'i',
                      textColor: 'text-blue-900'
                    }
                  };

                  const config = severityConfig[insight.severity as keyof typeof severityConfig] || severityConfig['INFO'];

                  return (
                    <div
                      key={idx}
                      className={`${config.bg} ${config.border} rounded-lg shadow-md hover:shadow-lg transition-all duration-200`}
                    >
                      <div className="p-5">
                        {/* Header Row */}
                        <div className="flex items-start justify-between mb-3">
                          <div className="flex items-center gap-3">
                            <div className={`${config.badge} w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm shadow-sm`}>
                              {config.icon}
                            </div>
                            <div>
                              <div className="flex items-center gap-2">
                                <span className="text-xs font-bold uppercase tracking-wider text-gray-600 letterspacing-wide">
                                  {insight.metric}
                                </span>
                                {insight.confidence && (
                                  <span className="text-xs px-2.5 py-0.5 rounded-full bg-white shadow-sm border border-gray-200 font-semibold text-gray-700">
                                    {insight.confidence} Confidence
                                  </span>
                                )}
                              </div>
                            </div>
                          </div>
                          <span className={`${config.badge} px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wide shadow-sm`}>
                            {insight.severity}
                          </span>
                        </div>

                        {/* Finding */}
                        <div className={`${config.textColor} mb-3`}>
                          <div className="font-bold text-base mb-1.5 leading-tight">
                            {insight.finding}
                          </div>
                          <div className="text-sm leading-relaxed opacity-90">
                            {insight.meaning}
                          </div>
                        </div>

                        {/* Recommendation */}
                        {insight.recommendation && (
                          <div className="mt-4 pt-4 border-t border-gray-300 border-opacity-40">
                            <div className="flex items-start gap-2">
                              <svg className="w-5 h-5 text-gray-600 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                              </svg>
                              <div className="flex-1">
                                <span className="text-xs font-semibold uppercase tracking-wide text-gray-600 block mb-1">
                                  Recommended Action
                                </span>
                                <p className="text-sm text-gray-800 leading-relaxed">
                                  {insight.recommendation}
                                </p>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Professional Metrics Explanation Section */}
              <div className="mt-6 pt-6 border-t-2 border-gray-200">
                <details className="group">
                  <summary className="cursor-pointer list-none">
                    <div className="flex items-center justify-between p-4 bg-white rounded-lg border border-gray-200 hover:border-slate-400 hover:shadow-md transition-all duration-200">
                      <div className="flex items-center gap-3">
                        <svg className="w-5 h-5 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <span className="font-semibold text-slate-800">
                          Statistical Metrics Reference Guide
                        </span>
                      </div>
                      <svg className="w-5 h-5 text-slate-500 group-open:rotate-180 transition-transform duration-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                      </svg>
                    </div>
                  </summary>

                  <div className="mt-4 bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden">
                    <div className="divide-y divide-gray-200">
                      {/* Slope Section */}
                      <div className="p-5">
                        <div className="flex items-start gap-3">
                          <div className="flex-shrink-0 w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                            <span className="text-blue-700 font-bold text-lg">m</span>
                          </div>
                          <div className="flex-1">
                            <h5 className="font-bold text-gray-900 mb-2 text-sm">
                              Slope Coefficient (m/yr or cm/yr)
                            </h5>
                            <p className="text-sm text-gray-600 mb-3 leading-relaxed">
                              Represents the annual rate of change in the measured parameter over time.
                              A linear regression model is used to calculate the long-term trend.
                            </p>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                              <div className="bg-slate-50 rounded-md p-3 border-l-3 border-blue-500">
                                <div className="text-xs font-semibold text-gray-600 uppercase mb-1">
                                  Groundwater Level (GWL)
                                </div>
                                <div className="text-xs text-gray-700 space-y-1">
                                  <div><strong>Positive (+):</strong> Water table deepening (depletion)</div>
                                  <div><strong>Negative (−):</strong> Water table rising (recovery)</div>
                                </div>
                              </div>
                              <div className="bg-slate-50 rounded-md p-3 border-l-3 border-green-500">
                                <div className="text-xs font-semibold text-gray-600 uppercase mb-1">
                                  GRACE Total Water Storage
                                </div>
                                <div className="text-xs text-gray-700 space-y-1">
                                  <div><strong>Positive (+):</strong> Storage increasing</div>
                                  <div><strong>Negative (−):</strong> Storage decreasing</div>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* R² Section */}
                      <div className="p-5 bg-gradient-to-r from-gray-50 to-slate-50">
                        <div className="flex items-start gap-3">
                          <div className="flex-shrink-0 w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
                            <span className="text-purple-700 font-bold text-sm">R²</span>
                          </div>
                          <div className="flex-1">
                            <h5 className="font-bold text-gray-900 mb-2 text-sm">
                              Coefficient of Determination (R²)
                            </h5>
                            <p className="text-sm text-gray-600 mb-3 leading-relaxed">
                              Statistical measure indicating the proportion of variance in the dependent variable
                              that is predictable from the independent variable (time). Ranges from 0 to 1.
                            </p>
                            <div className="space-y-2">
                              <div className="flex items-center gap-3">
                                <div className="w-20 h-2 bg-gradient-to-r from-green-500 to-green-600 rounded-full"></div>
                                <div className="flex-1">
                                  <div className="text-xs font-semibold text-gray-900">0.70 - 1.00: Strong Trend</div>
                                  <div className="text-xs text-gray-600">High predictability, data closely follows trendline</div>
                                </div>
                              </div>
                              <div className="flex items-center gap-3">
                                <div className="w-20 h-2 bg-gradient-to-r from-yellow-500 to-yellow-600 rounded-full"></div>
                                <div className="flex-1">
                                  <div className="text-xs font-semibold text-gray-900">0.30 - 0.70: Moderate Trend</div>
                                  <div className="text-xs text-gray-600">Moderate predictability with some variation</div>
                                </div>
                              </div>
                              <div className="flex items-center gap-3">
                                <div className="w-20 h-2 bg-gradient-to-r from-red-500 to-red-600 rounded-full"></div>
                                <div className="flex-1">
                                  <div className="text-xs font-semibold text-gray-900">0.00 - 0.30: Weak Trend</div>
                                  <div className="text-xs text-gray-600">Low predictability, high variance, trend unreliable</div>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </details>
              </div>
            </div>
          </div>
        )}

      <Plot
        data={[
          // GWL Trace
          timeseriesResponse.timeseries.length > 0 && {
            x: timeseriesResponse.timeseries.map(p => p.date),
            y: timeseriesResponse.timeseries.map(p =>
              timeseriesView === 'raw' ? p.avg_gwl :
                timeseriesView === 'seasonal' ? p.gwl_seasonal :
                  p.gwl_deseasonalized
            ),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'GWL (m bgl)',
            line: { color: GWL_COLOR, width: 2 },
            marker: { size: 6 },
            connectgaps: true,
            yaxis: 'y1'
          },

          // GRACE Trace
          timeseriesResponse.timeseries.length > 0 && {
            x: timeseriesResponse.timeseries.map(p => p.date),
            y: timeseriesResponse.timeseries.map(p =>
              timeseriesView === 'raw' ? p.avg_tws :
                timeseriesView === 'seasonal' ? p.grace_seasonal :
                  p.grace_deseasonalized
            ),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'GRACE TWS (cm)',
            line: { color: GRACE_COLOR, width: 2 },
            marker: { size: 6 },
            connectgaps: true,
            yaxis: 'y2'
          },

          // Rainfall Trace
          timeseriesResponse.timeseries.length > 0 && {
            x: timeseriesResponse.timeseries.map(p => p.date),
            y: timeseriesResponse.timeseries.map(p =>
              timeseriesView === 'raw' ? p.monthly_rainfall_total_mm :  // Monthly total (mm) for raw
                timeseriesView === 'seasonal' ? p.rainfall_seasonal :
                  p.rainfall_deseasonalized
            ),
            type: timeseriesView === 'raw' ? 'bar' : 'scatter',  // Bar for raw, line for others
            mode: timeseriesView === 'raw' ? undefined : 'lines+markers',
            name: timeseriesView === 'raw' ? 'Rainfall (mm/month)' : 'Rainfall',
            marker: timeseriesView === 'raw' ? { color: RAIN_COLOR, opacity: 0.6 } : { size: 6 },
            line: timeseriesView === 'raw' ? undefined : { color: RAIN_COLOR, width: 2 },
            connectgaps: timeseriesView === 'raw' ? undefined : true,
            yaxis: 'y3'
          }
        ].filter(Boolean)}
        layout={{
          autosize: true,
          height: 500,
          margin: { l: 60, r: 60, t: 40, b: 60 },
          xaxis: {
            title: 'Date',
            gridcolor: 'rgba(0,0,0,0.1)'
          },
          yaxis: {
            title: 'GWL (m bgl)',
            titlefont: { color: GWL_COLOR },
            tickfont: { color: GWL_COLOR },
            autorange: 'reversed',
            gridcolor: 'rgba(0,0,0,0.1)'
          },
          yaxis2: {
            title: 'GRACE TWS (cm)',
            titlefont: { color: GRACE_COLOR },
            tickfont: { color: GRACE_COLOR },
            overlaying: 'y',
            side: 'right',
            showgrid: false
          },
          yaxis3: {
            title: 'Rainfall (mm/day)',
            titlefont: { color: RAIN_COLOR },
            tickfont: { color: RAIN_COLOR },
            overlaying: 'y',
            side: 'right',
            position: 0.85,
            showgrid: false
          },
          plot_bgcolor: 'white',
          paper_bgcolor: 'white',
          font: { family: 'Arial, sans-serif' },
          hovermode: 'x unified',
          legend: { x: 0.01, y: 0.99, bgcolor: 'rgba(255,255,255,0.8)' }
        }}
        config={{
          displayModeBar: true,
          displaylogo: false,
          modeBarButtonsToRemove: ['pan2d', 'select2d', 'lasso2d'],
          toImageButtonOptions: {
            format: 'png',
            filename: `timeseries_${timeseriesView}_${new Date().toISOString().split('T')[0]}`,
            height: 600,
            width: 1200,
            scale: 2
          }
        }}
        style={{ width: '100%' }}
        useResizeHandler={true}
      />
    </div>
  );
}