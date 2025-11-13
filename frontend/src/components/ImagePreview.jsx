import React, { useState } from 'react';

const ImagePreview = ({ originalImage, resultImage, detections, onBoxClick }) => {
  const [activeTab, setActiveTab] = useState('original');
  const [zoom, setZoom] = useState(1);

  const handleZoomIn = () => setZoom(prev => Math.min(prev + 0.1, 3));
  const handleZoomOut = () => setZoom(prev => Math.max(prev - 0.1, 0.5));
  const handleResetZoom = () => setZoom(1);

  const displayImage = activeTab === 'original' ? originalImage : resultImage;

  return (
    <div className="bg-white rounded-lg shadow-md p-4 md:p-6 border border-gray-200">
      {/* Tabs */}
      <div className="flex justify-center border-b border-gray-200 mb-4 md:mb-6">
        <button
          onClick={() => setActiveTab('original')}
          className={`px-4 md:px-6 py-2.5 md:py-3 font-medium transition-all duration-200 relative text-sm md:text-base ${
            activeTab === 'original'
              ? 'text-blue-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
        >
          <span className="mr-1.5 md:mr-2">📷</span>
          <span className="hidden sm:inline">Original Image</span>
          <span className="sm:hidden">Original</span>
          {activeTab === 'original' && (
            <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-600"></span>
          )}
        </button>
        <button
          onClick={() => setActiveTab('result')}
          disabled={!resultImage}
          className={`px-4 md:px-6 py-2.5 md:py-3 font-medium transition-all duration-200 relative text-sm md:text-base ${
            activeTab === 'result'
              ? 'text-emerald-600'
              : 'text-gray-500 hover:text-gray-700'
          } ${!resultImage ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          <span className="mr-1.5 md:mr-2">✅</span>
          <span className="hidden sm:inline">Result ({detections?.length || 0})</span>
          <span className="sm:hidden">Result</span>
          {activeTab === 'result' && (
            <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-emerald-600"></span>
          )}
        </button>
      </div>

      {/* Image Display */}
      <div className="relative bg-gray-50 rounded-lg overflow-hidden border border-gray-200" style={{ minHeight: '400px' }}>
        {displayImage ? (
          <div className="flex items-center justify-center p-4 md:p-6">
            <div className="relative max-w-full">
              <img
                src={displayImage}
                alt={activeTab === 'original' ? 'Original' : 'Result'}
                style={{
                  transform: `scale(${zoom})`,
                  transition: 'transform 0.2s ease',
                  maxWidth: '100%',
                  height: 'auto',
                }}
                className="rounded-lg shadow-lg border border-gray-200"
              />
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center h-96 text-gray-400">
            <div className="text-center px-4">
              <div className="mb-4">
                <svg
                  className="w-16 h-16 md:w-20 md:h-20 mx-auto opacity-50"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                  />
                </svg>
              </div>
              <p className="font-medium text-sm md:text-base">No image to display</p>
              <p className="text-xs text-gray-400 mt-1">Upload an image to see results</p>
            </div>
          </div>
        )}
      </div>

      {/* Zoom Controls */}
      {displayImage && (
        <div className="mt-4 flex justify-center items-center gap-2">
          <button
            onClick={handleZoomOut}
            className="px-3 py-1.5 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors text-sm font-medium disabled:opacity-50"
            disabled={zoom <= 0.5}
            title="Zoom out"
          >
            ➖
          </button>
          <button
            onClick={handleResetZoom}
            className="px-4 py-1.5 bg-gray-700 text-white rounded-lg hover:bg-gray-800 transition-colors text-sm font-medium"
            title="Reset zoom"
          >
            🔍 {Math.round(zoom * 100)}%
          </button>
          <button
            onClick={handleZoomIn}
            className="px-3 py-1.5 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors text-sm font-medium disabled:opacity-50"
            disabled={zoom >= 3}
            title="Zoom in"
          >
            ➕
          </button>
        </div>
      )}
    </div>
  );
};

export default ImagePreview;

