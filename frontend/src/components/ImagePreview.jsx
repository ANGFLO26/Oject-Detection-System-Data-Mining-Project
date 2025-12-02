import React, { useState, useRef, useEffect } from 'react';
import { t } from '../utils/translations';

const ImagePreview = ({ originalImage, resultImage, detections, onBoxClick }) => {
  const [activeTab, setActiveTab] = useState('original');
  const [zoom, setZoom] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const imageContainerRef = useRef(null);

  const handleZoomIn = () => {
    setZoom(prev => {
      const newZoom = Math.min(prev + 0.1, 3);
      if (newZoom > 1) {
        // Reset position khi zoom in
        setPosition({ x: 0, y: 0 });
      }
      return newZoom;
    });
  };
  
  const handleZoomOut = () => {
    setZoom(prev => {
      const newZoom = Math.max(prev - 0.1, 0.5);
      if (newZoom <= 1) {
        // Reset position khi zoom về 1
        setPosition({ x: 0, y: 0 });
      }
      return newZoom;
    });
  };
  
  const handleResetZoom = () => {
    setZoom(1);
    setPosition({ x: 0, y: 0 });
  };

  // Handle mouse/touch drag for panning
  const handleStart = (clientX, clientY) => {
    if (zoom > 1) {
      setIsDragging(true);
      setDragStart({
        x: clientX - position.x,
        y: clientY - position.y
      });
    }
  };

  const handleMouseDown = (e) => {
    if (zoom > 1) {
      e.preventDefault();
      handleStart(e.clientX, e.clientY);
    }
  };

  const handleTouchStart = (e) => {
    if (zoom > 1 && e.touches.length === 1) {
      e.preventDefault();
      const touch = e.touches[0];
      handleStart(touch.clientX, touch.clientY);
    }
  };

  // Reset position khi đổi tab hoặc zoom về 1
  useEffect(() => {
    if (zoom <= 1) {
      setPosition({ x: 0, y: 0 });
    }
  }, [activeTab, zoom]);

  // Mouse/touch move và up handlers
  useEffect(() => {
    if (!isDragging || zoom <= 1) return;

    const handleMove = (clientX, clientY) => {
      const newX = clientX - dragStart.x;
      const newY = clientY - dragStart.y;
      
      // Giới hạn pan trong bounds của container
      if (imageContainerRef.current) {
        const container = imageContainerRef.current;
        const containerRect = container.getBoundingClientRect();
        const maxX = (containerRect.width * (zoom - 1)) / 2;
        const maxY = (containerRect.height * (zoom - 1)) / 2;
        
        setPosition({
          x: Math.max(-maxX, Math.min(maxX, newX)),
          y: Math.max(-maxY, Math.min(maxY, newY))
        });
      }
    };

    const handleMouseMove = (e) => {
      e.preventDefault();
      handleMove(e.clientX, e.clientY);
    };

    const handleTouchMove = (e) => {
      if (e.touches.length === 1) {
        e.preventDefault();
        const touch = e.touches[0];
        handleMove(touch.clientX, touch.clientY);
      }
    };

    const handleEnd = () => {
      setIsDragging(false);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleEnd);
    document.addEventListener('touchmove', handleTouchMove, { passive: false });
    document.addEventListener('touchend', handleEnd);
    
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleEnd);
      document.removeEventListener('touchmove', handleTouchMove);
      document.removeEventListener('touchend', handleEnd);
    };
  }, [isDragging, dragStart, zoom]);

  const displayImage = activeTab === 'original' ? originalImage : resultImage;

  return (
    <div className="bg-white rounded-2xl shadow-xl p-5 md:p-7 border-2 border-gray-200">
      {/* Tabs - Cải thiện với gradient */}
      <div className="flex justify-center border-b-2 border-gray-200 mb-5 md:mb-7">
        <button
          onClick={() => setActiveTab('original')}
          className={`px-6 md:px-8 py-3 md:py-4 font-bold transition-all duration-300 relative text-sm md:text-base rounded-t-xl ${
            activeTab === 'original'
              ? 'text-blue-700 bg-gradient-to-b from-blue-50 to-white'
              : 'text-gray-600 hover:text-gray-800 hover:bg-gray-50'
          }`}
          aria-label={t('Original Image')}
        >
          <span className="mr-2 text-lg">📷</span>
          <span className="hidden sm:inline">{t('Original Image')}</span>
          <span className="sm:hidden">{t('Original')}</span>
          {activeTab === 'original' && (
            <span className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-t-full"></span>
          )}
        </button>
        <button
          onClick={() => setActiveTab('result')}
          disabled={!resultImage}
          className={`px-6 md:px-8 py-3 md:py-4 font-bold transition-all duration-300 relative text-sm md:text-base rounded-t-xl ${
            activeTab === 'result'
              ? 'text-emerald-700 bg-gradient-to-b from-emerald-50 to-white'
              : 'text-gray-600 hover:text-gray-800 hover:bg-gray-50'
          } ${!resultImage ? 'opacity-50 cursor-not-allowed' : ''}`}
          aria-label={t('Result')}
        >
          <span className="mr-2 text-lg">✅</span>
          <span className="hidden sm:inline">{t('Result')} ({detections?.length || 0})</span>
          <span className="sm:hidden">{t('Result')}</span>
          {activeTab === 'result' && (
            <span className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-emerald-500 to-green-600 rounded-t-full"></span>
          )}
        </button>
      </div>

      {/* Image Display */}
      <div 
        ref={imageContainerRef}
        className="relative bg-gradient-to-br from-gray-50 to-gray-100 rounded-2xl overflow-hidden border-2 border-gray-300 shadow-inner" 
        style={{ minHeight: '400px' }}
        onMouseDown={handleMouseDown}
        onTouchStart={handleTouchStart}
      >
        {displayImage ? (
          <div 
            className="flex items-center justify-center p-4 md:p-6"
            style={{
              cursor: zoom > 1 ? (isDragging ? 'grabbing' : 'grab') : 'default',
              userSelect: 'none'
            }}
          >
            <div 
              className="relative max-w-full"
              style={{
                transform: `translate(${position.x}px, ${position.y}px) scale(${zoom})`,
                transition: isDragging ? 'none' : 'transform 0.2s ease',
              }}
            >
              <img
                src={displayImage}
                alt={activeTab === 'original' ? t('Original') : t('Result')}
                style={{
                  maxWidth: '100%',
                  height: 'auto',
                  pointerEvents: 'none',
                }}
                className="rounded-xl shadow-2xl border-2 border-white"
                draggable={false}
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
              <p className="font-medium text-sm md:text-base">{t('No image to display')}</p>
              <p className="text-xs text-gray-400 mt-1">{t('Upload an image to see results')}</p>
            </div>
          </div>
        )}
      </div>

      {/* Zoom Controls - Cải thiện với gradient */}
      {displayImage && (
        <div className="mt-6 flex justify-center items-center gap-3">
          <button
            onClick={handleZoomOut}
            className="px-5 py-3 bg-gradient-to-r from-gray-100 to-gray-200 text-gray-800 rounded-xl hover:from-gray-200 hover:to-gray-300 transition-all duration-300 text-base font-bold disabled:opacity-50 shadow-md hover:shadow-lg transform hover:scale-105 active:scale-95"
            disabled={zoom <= 0.5}
            title="Zoom out"
            aria-label="Zoom out"
          >
            ➖
          </button>
          <button
            onClick={handleResetZoom}
            className="px-6 py-3 bg-gradient-to-r from-gray-700 to-gray-800 text-white rounded-xl hover:from-gray-800 hover:to-gray-900 transition-all duration-300 text-base font-bold shadow-lg hover:shadow-xl transform hover:scale-105 active:scale-95"
            title={zoom > 1 ? "Reset zoom và vị trí" : "Reset zoom"}
            aria-label="Reset zoom"
          >
            🔍 {Math.round(zoom * 100)}%
            {zoom > 1 && <span className="ml-2 text-xs hidden sm:inline">(Kéo để di chuyển)</span>}
          </button>
          <button
            onClick={handleZoomIn}
            className="px-5 py-3 bg-gradient-to-r from-gray-100 to-gray-200 text-gray-800 rounded-xl hover:from-gray-200 hover:to-gray-300 transition-all duration-300 text-base font-bold disabled:opacity-50 shadow-md hover:shadow-lg transform hover:scale-105 active:scale-95"
            disabled={zoom >= 3}
            title="Zoom in"
            aria-label="Zoom in"
          >
            ➕
          </button>
        </div>
      )}
    </div>
  );
};

export default ImagePreview;

