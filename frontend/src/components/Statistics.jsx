import React from 'react';

const Statistics = ({ statistics, detections }) => {
  if (!statistics || !detections || detections.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-md p-8 text-center border border-gray-200">
        <div className="mb-4">
          <svg
            className="w-16 h-16 mx-auto text-gray-300"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
            />
          </svg>
        </div>
        <h2 className="text-xl font-bold mb-2 text-gray-800">📊 Statistics</h2>
        <p className="text-gray-500">No statistics data available</p>
        <p className="text-sm text-gray-400 mt-1">Detection results will appear here</p>
      </div>
    );
  }

  const classCounts = {};
  detections.forEach(det => {
    classCounts[det.class] = (classCounts[det.class] || 0) + 1;
  });

  const getConfidenceColor = (conf) => {
    if (conf >= 0.7) return 'text-green-600';
    if (conf >= 0.5) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
      <h2 className="text-xl font-bold mb-4 flex items-center text-gray-800">
        <span className="mr-2">📊</span>
        Statistics
      </h2>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-blue-50 border border-blue-100 rounded-lg p-5 hover:shadow-md transition-shadow cursor-default">
          <div className="text-sm text-gray-600 mb-2 font-medium">Total Detections</div>
          <div className="text-3xl font-bold text-blue-600">{statistics.total}</div>
        </div>
        <div className="bg-emerald-50 border border-emerald-100 rounded-lg p-5 hover:shadow-md transition-shadow cursor-default">
          <div className="text-sm text-gray-600 mb-2 font-medium">Avg Confidence</div>
          <div className={`text-3xl font-bold ${getConfidenceColor(statistics.avg_confidence)}`}>
            {(statistics.avg_confidence * 100).toFixed(1)}%
          </div>
        </div>
        <div className="bg-purple-50 border border-purple-100 rounded-lg p-5 hover:shadow-md transition-shadow cursor-default">
          <div className="text-sm text-gray-600 mb-2 font-medium">Unique Classes</div>
          <div className="text-3xl font-bold text-purple-600">{statistics.classes?.length || 0}</div>
        </div>
      </div>

      {/* Class Distribution */}
      {statistics.classes && statistics.classes.length > 0 && (
        <div className="mb-6 bg-gray-50 rounded-lg p-5 border border-gray-200">
          <h3 className="font-semibold text-base mb-4 flex items-center text-gray-800">
            <span className="mr-2">🦁</span>
            Class Distribution
          </h3>
          <div className="space-y-3">
            {Object.entries(classCounts)
              .sort((a, b) => b[1] - a[1])
              .map(([className, count]) => {
                const percentage = (count / statistics.total) * 100;
                return (
                  <div key={className} className="flex items-center gap-3">
                    <div className="w-28 text-sm font-medium text-gray-700">{className}</div>
                    <div className="flex-1 bg-gray-200 rounded-full h-6 relative overflow-hidden">
                      <div
                        className="bg-blue-500 h-full rounded-full transition-all duration-300"
                        style={{ width: `${percentage}%` }}
                      />
                      <div className="absolute inset-0 flex items-center justify-center text-xs font-medium text-gray-700">
                        {count} ({percentage.toFixed(1)}%)
                      </div>
                    </div>
                  </div>
                );
              })}
          </div>
        </div>
      )}

      {/* Confidence Range */}
      {statistics.min_confidence !== undefined && (
        <div className="bg-gray-50 rounded-lg p-5 border border-gray-200">
          <h3 className="font-semibold text-base mb-4 flex items-center text-gray-800">
            <span className="mr-2">📈</span>
            Confidence Range
          </h3>
          <div className="grid grid-cols-3 gap-3">
            <div className="text-center p-3 bg-white rounded-lg border border-gray-200">
              <div className="text-xs text-gray-600 mb-1">Minimum</div>
              <div className={`text-lg font-bold ${getConfidenceColor(statistics.min_confidence)}`}>
                {(statistics.min_confidence * 100).toFixed(1)}%
              </div>
            </div>
            <div className="text-center p-3 bg-white rounded-lg border border-gray-200">
              <div className="text-xs text-gray-600 mb-1">Average</div>
              <div className={`text-lg font-bold ${getConfidenceColor(statistics.avg_confidence)}`}>
                {(statistics.avg_confidence * 100).toFixed(1)}%
              </div>
            </div>
            <div className="text-center p-3 bg-white rounded-lg border border-gray-200">
              <div className="text-xs text-gray-600 mb-1">Maximum</div>
              <div className={`text-lg font-bold ${getConfidenceColor(statistics.max_confidence)}`}>
                {(statistics.max_confidence * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Statistics;

