import React from 'react';

const SettingsPanel = ({ confThreshold, iouThreshold, onConfChange, onIouChange, onDetect, onCompare, isProcessing }) => {
  return (
    <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
      <h2 className="text-xl font-bold mb-6 text-gray-800 flex items-center">
        <span className="text-2xl mr-2">⚙️</span>
        Settings
      </h2>

      {/* Confidence Threshold */}
      <div className="mb-5 bg-gray-50 rounded-lg p-4 border border-gray-200">
        <label className="block text-sm font-semibold text-gray-700 mb-3">
          <span className="flex items-center justify-between">
            <span>Confidence Threshold</span>
            <span className="text-lg font-bold text-blue-600 bg-blue-50 px-2 py-0.5 rounded">
              {confThreshold.toFixed(2)}
            </span>
          </span>
        </label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={confThreshold}
          onChange={(e) => onConfChange(parseFloat(e.target.value))}
          disabled={isProcessing}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600 disabled:opacity-50"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1.5">
          <span>Low (0.0)</span>
          <span>High (1.0)</span>
        </div>
        <p className="text-xs text-gray-600 mt-2 font-medium">
          {confThreshold < 0.3 ? 'More detections' : confThreshold < 0.6 ? 'Balanced' : 'Only confident detections'}
        </p>
      </div>

      {/* IoU Threshold */}
      <div className="mb-5 bg-gray-50 rounded-lg p-4 border border-gray-200">
        <label className="block text-sm font-semibold text-gray-700 mb-3">
          <span className="flex items-center justify-between">
            <span>IoU Threshold</span>
            <span className="text-lg font-bold text-gray-700 bg-gray-100 px-2 py-0.5 rounded">
              {iouThreshold.toFixed(2)}
            </span>
          </span>
        </label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={iouThreshold}
          onChange={(e) => onIouChange(parseFloat(e.target.value))}
          disabled={isProcessing}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-gray-600 disabled:opacity-50"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1.5">
          <span>Low (0.0)</span>
          <span>High (1.0)</span>
        </div>
        <p className="text-xs text-gray-600 mt-2 font-medium">
          IoU threshold for Non-Maximum Suppression
        </p>
      </div>

      {/* Quick Presets */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-3">Quick Presets:</label>
        <div className="grid grid-cols-3 gap-2">
          <button
            onClick={() => {
              onConfChange(0.1);
              onIouChange(0.45);
            }}
            disabled={isProcessing}
            className={`px-3 py-2 text-xs rounded font-medium disabled:opacity-50 transition-all ${
              Math.abs(confThreshold - 0.1) < 0.01
                ? 'bg-blue-100 text-blue-700 border-2 border-blue-300'
                : 'bg-gray-100 text-gray-700 border border-gray-200 hover:bg-gray-200'
            }`}
          >
            Low (0.1)
          </button>
          <button
            onClick={() => {
              onConfChange(0.25);
              onIouChange(0.45);
            }}
            disabled={isProcessing}
            className={`px-3 py-2 text-xs rounded font-medium disabled:opacity-50 transition-all ${
              Math.abs(confThreshold - 0.25) < 0.01
                ? 'bg-blue-100 text-blue-700 border-2 border-blue-300'
                : 'bg-gray-100 text-gray-700 border border-gray-200 hover:bg-gray-200'
            }`}
          >
            Medium (0.25)
          </button>
          <button
            onClick={() => {
              onConfChange(0.5);
              onIouChange(0.45);
            }}
            disabled={isProcessing}
            className={`px-3 py-2 text-xs rounded font-medium disabled:opacity-50 transition-all ${
              Math.abs(confThreshold - 0.5) < 0.01
                ? 'bg-blue-100 text-blue-700 border-2 border-blue-300'
                : 'bg-gray-100 text-gray-700 border border-gray-200 hover:bg-gray-200'
            }`}
          >
            High (0.5)
          </button>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="space-y-2.5 mt-5">
        <button
          onClick={onDetect}
          disabled={isProcessing}
          className="w-full px-6 py-3 bg-blue-600 text-white rounded-lg font-semibold shadow-sm hover:bg-blue-700 hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center gap-2"
        >
          {isProcessing ? (
            <>
              <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              <span>Processing...</span>
            </>
          ) : (
            <>
              <span>🔍</span>
              <span>Detect</span>
            </>
          )}
        </button>
        <button
          onClick={onCompare}
          disabled={isProcessing}
          className="w-full px-6 py-2.5 bg-gray-600 text-white rounded-lg font-medium shadow-sm hover:bg-gray-700 hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center gap-2"
        >
          <span>📊</span>
          <span>Compare Thresholds</span>
        </button>
      </div>
    </div>
  );
};

export default SettingsPanel;

