import React, { useState } from 'react';

const ResultsTable = ({ detections, onRowClick }) => {
  const [sortBy, setSortBy] = useState(null);
  const [sortOrder, setSortOrder] = useState('desc');

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.7) return 'text-green-600 font-semibold';
    if (confidence >= 0.5) return 'text-yellow-600 font-semibold';
    return 'text-red-600 font-semibold';
  };

  const handleSort = (field) => {
    if (sortBy === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(field);
      setSortOrder('desc');
    }
  };

  const sortedDetections = [...detections].sort((a, b) => {
    if (!sortBy) return 0;
    const aVal = a[sortBy];
    const bVal = b[sortBy];
    const multiplier = sortOrder === 'asc' ? 1 : -1;
    return (aVal > bVal ? 1 : -1) * multiplier;
  });

  if (!detections || detections.length === 0) {
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
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
        </div>
        <p className="text-gray-600 font-medium">No detection results</p>
        <p className="text-sm text-gray-400 mt-1">Upload an image and click Detect to see results</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
      <h2 className="text-xl font-bold mb-4 flex items-center text-gray-800">
        <span className="mr-2">📋</span>
        Detection Results
        <span className="ml-3 px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-sm font-medium">
          {detections.length} objects
        </span>
      </h2>
      <div className="overflow-x-auto -mx-2">
        <div className="inline-block min-w-full align-middle">
          <table className="w-full border-collapse">
            <thead>
              <tr className="bg-gray-50">
                <th
                  className="border border-gray-200 px-3 md:px-4 py-2.5 md:py-3 text-left cursor-pointer hover:bg-gray-100 font-semibold text-gray-700 transition-colors text-xs md:text-sm"
                  onClick={() => handleSort('id')}
                >
                  <span className="flex items-center gap-1">
                    # {sortBy === 'id' && (sortOrder === 'asc' ? '↑' : '↓')}
                  </span>
                </th>
                <th
                  className="border border-gray-200 px-3 md:px-4 py-2.5 md:py-3 text-left cursor-pointer hover:bg-gray-100 font-semibold text-gray-700 transition-colors text-xs md:text-sm"
                  onClick={() => handleSort('class')}
                >
                  <span className="flex items-center gap-1">
                    Animal {sortBy === 'class' && (sortOrder === 'asc' ? '↑' : '↓')}
                  </span>
                </th>
                <th
                  className="border border-gray-200 px-3 md:px-4 py-2.5 md:py-3 text-left cursor-pointer hover:bg-gray-100 font-semibold text-gray-700 transition-colors text-xs md:text-sm"
                  onClick={() => handleSort('confidence')}
                >
                  <span className="flex items-center gap-1">
                    Confidence {sortBy === 'confidence' && (sortOrder === 'asc' ? '↑' : '↓')}
                  </span>
                </th>
                <th className="border border-gray-200 px-3 md:px-4 py-2.5 md:py-3 text-left font-semibold text-gray-700 text-xs md:text-sm hidden md:table-cell">Location</th>
                <th className="border border-gray-200 px-3 md:px-4 py-2.5 md:py-3 text-left font-semibold text-gray-700 text-xs md:text-sm hidden lg:table-cell">Size</th>
              </tr>
            </thead>
            <tbody>
              {sortedDetections.map((detection, index) => (
                <tr
                  key={detection.id}
                  onClick={() => onRowClick && onRowClick(detection)}
                  className="hover:bg-blue-50 cursor-pointer transition-colors border-b border-gray-100"
                >
                  <td className="border border-gray-200 px-3 md:px-4 py-2.5 md:py-3 font-medium text-gray-800 text-sm">
                    {detection.id}
                  </td>
                  <td className="border border-gray-200 px-3 md:px-4 py-2.5 md:py-3 font-medium text-gray-800 text-sm">
                    {detection.class}
                  </td>
                  <td className={`border border-gray-200 px-3 md:px-4 py-2.5 md:py-3 font-semibold text-sm ${getConfidenceColor(detection.confidence)}`}>
                    {(detection.confidence * 100).toFixed(2)}%
                  </td>
                  <td className="border border-gray-200 px-3 md:px-4 py-2.5 md:py-3 text-xs md:text-sm text-gray-600 hidden md:table-cell">
                    ({detection.bbox[0].toFixed(0)}, {detection.bbox[1].toFixed(0)}) → ({detection.bbox[2].toFixed(0)}, {detection.bbox[3].toFixed(0)})
                  </td>
                  <td className="border border-gray-200 px-3 md:px-4 py-2.5 md:py-3 text-xs md:text-sm text-gray-600 hidden lg:table-cell">
                    {detection.width.toFixed(0)} × {detection.height.toFixed(0)} px
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default ResultsTable;

