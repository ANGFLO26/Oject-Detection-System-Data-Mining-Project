import React, { useState, useMemo } from 'react';
import { t } from '../utils/translations';
import { translateClass, capitalizeFirst } from '../utils/classTranslations';

const ResultsTable = ({ detections }) => {
  const [sortBy, setSortBy] = useState('confidence'); // 'confidence' | 'name'
  const [sortOrder, setSortOrder] = useState('desc'); // 'asc' | 'desc'

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.7) return 'bg-gradient-to-r from-green-100 to-emerald-100 text-green-900 border-green-400 shadow-green-200';
    if (confidence >= 0.5) return 'bg-gradient-to-r from-yellow-100 to-amber-100 text-yellow-900 border-yellow-400 shadow-yellow-200';
    return 'bg-gradient-to-r from-red-100 to-pink-100 text-red-900 border-red-400 shadow-red-200';
  };

  const getConfidenceBadge = (confidence) => {
    if (confidence >= 0.7) return 'bg-gradient-to-r from-green-500 to-emerald-600';
    if (confidence >= 0.5) return 'bg-gradient-to-r from-yellow-500 to-amber-600';
    return 'bg-gradient-to-r from-red-500 to-pink-600';
  };

  // Sorted detections
  const sortedDetections = useMemo(() => {
    if (!detections || detections.length === 0) return [];
    
    const sorted = [...detections].sort((a, b) => {
      if (sortBy === 'confidence') {
        return sortOrder === 'desc' 
          ? b.confidence - a.confidence 
          : a.confidence - b.confidence;
      } else {
        // Sort by name
        const nameA = capitalizeFirst(translateClass(a.class));
        const nameB = capitalizeFirst(translateClass(b.class));
        return sortOrder === 'desc'
          ? nameB.localeCompare(nameA)
          : nameA.localeCompare(nameB);
      }
    });
    
    return sorted;
  }, [detections, sortBy, sortOrder]);

  const handleSort = (newSortBy) => {
    if (sortBy === newSortBy) {
      setSortOrder(prev => prev === 'desc' ? 'asc' : 'desc');
    } else {
      setSortBy(newSortBy);
      setSortOrder('desc');
    }
  };

  if (!detections || detections.length === 0) {
    return null;
  }

  return (
    <div className="p-6 sm:p-8">
      {/* Header với gradient */}
      <div className="mb-6 pb-4 border-b-2 border-gray-200">
        {/* Title Row */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg flex-shrink-0">
              <span className="text-2xl">📋</span>
            </div>
            <h2 className="text-xl sm:text-2xl font-extrabold text-gray-800">
              {t('Detected Objects')}
            </h2>
          </div>
          
          {/* Count Badge */}
          <div className="px-4 py-2 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-xl text-base sm:text-lg font-extrabold shadow-lg flex items-center justify-center min-w-[50px]">
            {detections.length}
          </div>
        </div>
        
        {/* Sort Controls Row */}
        <div className="flex items-center justify-end gap-2">
          <span className="text-xs sm:text-sm text-gray-600 font-medium mr-2 hidden sm:inline">
            {t('Sort by')}:
          </span>
          <button
            onClick={() => handleSort('confidence')}
            className={`px-4 py-2 rounded-lg text-xs sm:text-sm font-bold transition-all flex items-center gap-1.5 ${
              sortBy === 'confidence'
                ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300 hover:shadow-md'
            }`}
            title={t('Sort by confidence')}
          >
            <span>{t('Confidence')}</span>
            {sortBy === 'confidence' && (
              <span className="text-base">{sortOrder === 'desc' ? '↓' : '↑'}</span>
            )}
          </button>
          <button
            onClick={() => handleSort('name')}
            className={`px-4 py-2 rounded-lg text-xs sm:text-sm font-bold transition-all flex items-center gap-1.5 ${
              sortBy === 'name'
                ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300 hover:shadow-md'
            }`}
            title={t('Sort by name')}
          >
            <span>{t('Name')}</span>
            {sortBy === 'name' && (
              <span className="text-base">{sortOrder === 'desc' ? '↓' : '↑'}</span>
            )}
          </button>
        </div>
      </div>
      
      <div className="space-y-4 max-h-[500px] overflow-y-auto scrollbar-thin pr-2">
        {sortedDetections.map((detection, index) => (
          <div
            key={detection.id || index}
            className={`border-2 rounded-2xl p-5 transition-all duration-300 hover:shadow-xl hover:scale-[1.02] ${getConfidenceColor(detection.confidence)}`}
            role="listitem"
            aria-label={`${capitalizeFirst(translateClass(detection.class))} với độ tin cậy ${(detection.confidence * 100).toFixed(1)} phần trăm`}
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-3 flex-1">
                <div className={`w-10 h-10 ${getConfidenceBadge(detection.confidence)} rounded-xl flex items-center justify-center text-white font-extrabold shadow-md`}>
                  {index + 1}
                </div>
                <h3 className="font-extrabold text-lg sm:text-xl text-gray-900">
                  {capitalizeFirst(translateClass(detection.class))}
                </h3>
              </div>
            </div>
            <div className="flex items-center justify-between bg-white bg-opacity-50 rounded-xl px-4 py-2">
              <span className="text-sm sm:text-base font-bold text-gray-700">{t('Confidence:')}</span>
              <div className="flex items-center gap-2">
                <div className="w-24 h-3 bg-gray-200 rounded-full overflow-hidden">
                  <div 
                    className={`h-full ${getConfidenceBadge(detection.confidence)} transition-all duration-500`}
                    style={{ width: `${detection.confidence * 100}%` }}
                  ></div>
                </div>
                <span className="text-base sm:text-lg font-extrabold text-gray-900 min-w-[50px] text-right">
                  {(detection.confidence * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ResultsTable;
