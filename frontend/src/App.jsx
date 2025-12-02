import React, { useState, useEffect, useRef } from 'react';
import HomeView from './components/HomeView';
import ImageUpload from './components/ImageUpload';
import ImagePreview from './components/ImagePreview';
import ResultsTable from './components/ResultsTable';
import CameraView from './components/CameraView';
import { detectObjects, getModelInfo } from './services/api';
import { t } from './utils/translations';
import { audioService } from './services/audioService';

function App() {
  // Mode management: 'home' | 'camera' | 'image'
  const [mode, setMode] = useState('home');
  
  // Image mode states
  const [selectedFile, setSelectedFile] = useState(null);
  const [originalImage, setOriginalImage] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [detections, setDetections] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  
  // System states
  const [modelInfo, setModelInfo] = useState(null);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);
  const [isAudioEnabled, setIsAudioEnabled] = useState(true);
  const lastImageDetectionsRef = useRef(null); // Lưu detections cho nút "Đọc lại"

  // Load model info và phát welcome message (chỉ 1 lần)
  useEffect(() => {
    getModelInfo()
      .then(info => {
        setModelInfo(info);
        // Cấu hình audio và phát welcome message (chỉ 1 lần)
        if (audioService.isSupported()) {
          audioService.setEnabled(true);
          audioService.speakWelcome();
        }
      })
      .catch(err => {
        console.error('Failed to load model info:', err);
        let errorMsg = t('Cannot connect to backend. Please check the server.');
        
        if (err.code === 'ECONNABORTED') {
          errorMsg = t('Connection timeout. Please check if backend server is running.');
        } else if (err.code === 'ERR_NETWORK' || !err.response) {
          errorMsg = t('Cannot connect to server. Please check if the backend is running.');
        } else if (err.response?.status === 500) {
          errorMsg = t('Backend server error. Please check the server logs.');
        }
        
        setError(errorMsg);
      });

    // Cleanup audio service khi unmount
    return () => {
      audioService.stop();
    };
  }, []);

  // Đồng bộ trạng thái bật/tắt audio với audioService
  useEffect(() => {
    audioService.setEnabled(isAudioEnabled);
    if (!isAudioEnabled) {
      audioService.stop();
    }
  }, [isAudioEnabled]);

  // Handle mode selection
  const handleSelectMode = (selectedMode) => {
    setMode(selectedMode);
    // Reset states khi chuyển mode
    if (selectedMode === 'image') {
      handleClearImage();
    }
  };

  // Handle back to home
  const handleBackToHome = () => {
    setMode('home');
    handleClearImage();
    audioService.stop(); // Dừng audio khi quay về home
  };

  // Clear image states
  const handleClearImage = () => {
    setSelectedFile(null);
    setOriginalImage(null);
    setResultImage(null);
    setDetections([]);
    setError(null);
    setSuccessMessage(null);
  };

  // Handle image select (single image only)
  const handleImageSelect = (file) => {
    setSelectedFile(file);
    setError(null);
    setResultImage(null);
    setDetections([]);
    setSuccessMessage(null);

    // Create preview URL với error handling
    const reader = new FileReader();
    reader.onload = (e) => {
      setOriginalImage(e.target.result);
    };
    reader.onerror = () => {
      setError(t('Failed to read image file. Please try another image.'));
      setSelectedFile(null);
    };
    reader.onabort = () => {
      setError(t('Image reading was cancelled.'));
      setSelectedFile(null);
    };
    try {
      reader.readAsDataURL(file);
    } catch (err) {
      setError(t('Failed to process image. Please check if the file is valid.'));
      setSelectedFile(null);
    }
  };

  // Handle detect (single image)
  const handleDetect = async () => {
    if (!selectedFile) {
      setError(t('Please select an image first!'));
      return;
    }

    setIsProcessing(true);
    setError(null);
    setResultImage(null);
    setDetections([]);
    setSuccessMessage(null);
    audioService.stop(); // Dừng audio cũ

    try {
      const confThreshold = modelInfo?.default_conf_threshold || 0.25;
      const iouThreshold = modelInfo?.default_iou_threshold || 0.45;
      
      const result = await detectObjects(selectedFile, confThreshold, iouThreshold);
      setResultImage(result.image_base64);
      setDetections(result.detections || []);
      lastImageDetectionsRef.current = result.detections || [];
      
      // Phát âm kết quả với logic mới (gom theo lớp) nếu có đối tượng
      if (result.detections && result.detections.length > 0) {
        setSuccessMessage(`${t('Detected')} ${result.detections.length} ${t('object(s)!')}`);
        setTimeout(() => setSuccessMessage(null), 3000);
        if (isAudioEnabled) {
          audioService.speakDetections(result.detections, 2000);
        }
      } else {
        setSuccessMessage(t('No objects detected in this image.'));
        setTimeout(() => setSuccessMessage(null), 3000);
        lastImageDetectionsRef.current = [];
        if (isAudioEnabled && audioService.isSupported()) {
          audioService.speakSystemMessage('Không phát hiện đối tượng nào', 1);
        }
      }
    } catch (err) {
      // Cải thiện error handling
      let errorMessage = t('An error occurred during detection. Please try again.');
      
      if (err.response?.data?.detail) {
        errorMessage = err.response.data.detail;
      } else if (err.message) {
        errorMessage = err.message;
      } else if (err.code === 'ECONNABORTED') {
        errorMessage = t('Request timeout. The image may be too large. Please try with a smaller image.');
      } else if (err.code === 'ERR_NETWORK') {
        errorMessage = t('Cannot connect to server. Please check if the backend is running.');
      }
      
      setError(errorMessage);
      console.error('Detection error:', err);
      
      // Phát âm thông báo lỗi
      if (isAudioEnabled && audioService.isSupported()) {
        audioService.speakSystemMessage('Có lỗi xảy ra khi nhận diện', 5);
      }
    } finally {
      setIsProcessing(false);
    }
  };

  // Điều khiển audio cho chế độ ảnh
  const handleToggleAudio = () => {
    setIsAudioEnabled(prev => !prev);
  };

  const handleRepeatAudioImage = () => {
    const detectionsToRepeat = lastImageDetectionsRef.current;
    if (!isAudioEnabled || !detectionsToRepeat || detectionsToRepeat.length === 0) {
      return;
    }
    audioService.stop();
    audioService.speakDetections(detectionsToRepeat, 2000);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-indigo-50">
      {/* Header */}
      <header className="bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 shadow-lg sticky top-0 z-50 border-b-4 border-blue-700">
        <div className="container mx-auto px-4 sm:px-6 py-4 sm:py-5 max-w-7xl">
          {/* Top row: Title and Status */}
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-3">
            {/* Left: Logo & Title */}
            <div className="flex-1 min-w-0 flex items-center gap-3 sm:gap-4">
              <div className="flex-shrink-0 w-12 h-12 sm:w-14 sm:h-14 bg-white rounded-xl flex items-center justify-center shadow-lg">
                <span className="text-2xl sm:text-3xl">🎯</span>
              </div>
              <div className="min-w-0">
                <h1 className="text-xl sm:text-2xl md:text-3xl font-extrabold text-white leading-tight drop-shadow-md">
                  {t('Object Detection System')}
                </h1>
                <p className="text-blue-100 mt-1 text-sm sm:text-base font-medium">
                  {mode === 'home' ? t('Chọn chế độ để bắt đầu') : 
                   mode === 'camera' ? t('Nhận diện đối tượng theo thời gian thực') :
                   t('Tải ảnh lên để nhận diện đối tượng')}
                </p>
              </div>
            </div>

            {/* Right: Model Status */}
            {modelInfo && (
              <div className="flex-shrink-0 text-right hidden sm:block">
                <div className="text-xs text-blue-200 mb-1 font-medium">{t('Model Status')}</div>
                <div className="inline-flex items-center px-4 py-2 bg-white bg-opacity-20 backdrop-blur-sm text-white border-2 border-white border-opacity-30 rounded-xl shadow-lg">
                  <span className="w-3 h-3 bg-green-300 rounded-full mr-2 animate-pulse shadow-lg"></span>
                  <span className="font-bold text-sm">{t('Ready')}</span>
                </div>
              </div>
            )}
          </div>

          {/* Bottom row: Back to Home button (centered, khi không ở home) */}
          {mode !== 'home' && (
            <div className="flex justify-center">
              <button
                onClick={handleBackToHome}
                className="px-6 py-3 bg-white text-gray-800 border-2 border-white rounded-xl shadow-2xl hover:bg-gray-50 hover:shadow-3xl hover:scale-110 active:scale-95 transition-all duration-300 font-bold text-base sm:text-lg flex items-center justify-center gap-2 whitespace-nowrap"
                aria-label={t('Back to Home')}
              >
                <span className="text-xl">🏠</span>
                <span>{t('Back to Home')}</span>
              </button>
            </div>
          )}
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 sm:px-6 py-8 sm:py-10 max-w-7xl">
        {/* Error Message */}
        {error && (
          <div className="mb-6 sm:mb-8 bg-gradient-to-r from-red-50 via-pink-50 to-red-50 border-l-4 border-red-600 text-red-900 px-5 sm:px-7 py-4 sm:py-5 rounded-2xl shadow-xl">
            <div className="flex items-center justify-between">
              <div className="flex items-center flex-1 min-w-0 gap-3">
                <div className="w-10 h-10 bg-red-600 rounded-full flex items-center justify-center flex-shrink-0 shadow-lg">
                  <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                </div>
                <span className="font-bold text-base sm:text-lg break-words">{error}</span>
              </div>
              <button
                onClick={() => setError(null)}
                className="ml-4 w-8 h-8 bg-red-600 text-white rounded-full hover:bg-red-700 flex-shrink-0 flex items-center justify-center transition-all duration-300 hover:scale-110 shadow-md"
                aria-label="Close error message"
              >
                ✕
              </button>
            </div>
          </div>
        )}

        {/* Success Message */}
        {successMessage && (
          <div className="mb-6 sm:mb-8 bg-gradient-to-r from-green-50 via-emerald-50 to-green-50 border-l-4 border-green-600 text-green-900 px-5 sm:px-7 py-4 sm:py-5 rounded-2xl shadow-xl">
            <div className="flex items-center justify-between">
              <div className="flex items-center flex-1 min-w-0 gap-3">
                <div className="w-10 h-10 bg-green-600 rounded-full flex items-center justify-center flex-shrink-0 shadow-lg">
                  <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                </div>
                <span className="font-bold text-base sm:text-lg break-words">{successMessage}</span>
              </div>
              <button
                onClick={() => setSuccessMessage(null)}
                className="ml-4 w-8 h-8 bg-green-600 text-white rounded-full hover:bg-green-700 flex-shrink-0 flex items-center justify-center transition-all duration-300 hover:scale-110 shadow-md"
                aria-label="Close success message"
              >
                ✕
              </button>
            </div>
          </div>
        )}

        {/* Render based on mode */}
        {mode === 'home' && (
          <HomeView onSelectMode={handleSelectMode} />
        )}

        {mode === 'camera' && (
          <CameraView
            isActive={true}
            onClose={handleBackToHome}
          />
        )}

        {mode === 'image' && (
          <>
            {/* Upload Zone */}
            <div className="mb-6 sm:mb-8">
              {!originalImage ? (
                <ImageUpload
                  onImageSelect={handleImageSelect}
                  isProcessing={isProcessing}
                />
              ) : (
                <div className="bg-white rounded-2xl shadow-lg border-2 border-blue-200 p-5 sm:p-6 hover:shadow-xl transition-all duration-300">
                  <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                    <div className="flex items-center gap-4 flex-1 min-w-0">
                      <div className="w-12 h-12 bg-gradient-to-br from-green-400 to-emerald-500 rounded-xl flex items-center justify-center shadow-md flex-shrink-0">
                        <span className="text-2xl">✅</span>
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="text-base font-bold text-gray-800 mb-1">
                          {t('Image loaded')}
                        </div>
                        <div className="text-sm text-gray-600 truncate font-medium">
                          {selectedFile?.name || t('No file selected')}
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={handleClearImage}
                      disabled={isProcessing}
                      className="px-6 py-3 bg-gradient-to-r from-gray-100 to-gray-200 text-gray-800 rounded-xl font-bold text-sm shadow-md hover:from-gray-200 hover:to-gray-300 hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 flex items-center gap-2 whitespace-nowrap transform hover:scale-105 active:scale-95"
                      aria-label={t('Upload More')}
                    >
                      <span className="text-lg">📤</span>
                      <span>{t('Upload More')}</span>
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Main Content Area - Two Column Layout */}
            {originalImage && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 sm:gap-6 mb-6 sm:mb-8">
                {/* Left Column: Image Preview */}
                <div className="lg:col-span-2">
                  <div className="bg-white rounded-2xl shadow-xl border-2 border-gray-200 p-5 sm:p-7 relative hover:shadow-2xl transition-all duration-300">
                    {/* Status Badge */}
                    {detections.length > 0 && (
                      <div className="absolute top-5 right-5 z-10">
                        <div className="bg-gradient-to-r from-green-500 to-emerald-600 text-white px-4 py-2 rounded-xl text-sm font-bold flex items-center gap-2 shadow-lg border-2 border-white">
                          <span className="text-lg">✓</span>
                          <span>{t('Detected')}</span>
                        </div>
                      </div>
                    )}

                    {/* Loading Overlay - Skeleton */}
                    {isProcessing && (
                      <div className="absolute inset-0 bg-white bg-opacity-95 rounded-2xl flex items-center justify-center z-20 backdrop-blur-sm">
                        <div className="w-full h-full flex flex-col items-center justify-center p-8">
                          {/* Skeleton cho image */}
                          <div className="w-full max-w-2xl bg-gradient-to-r from-gray-200 via-gray-300 to-gray-200 rounded-xl h-64 mb-6 animate-pulse shadow-lg"></div>
                          {/* Loading spinner với double ring */}
                          <div className="text-center">
                            <div className="relative w-16 h-16 mx-auto mb-4">
                              <div className="absolute inset-0 border-4 border-blue-400 border-t-transparent rounded-full animate-spin"></div>
                              <div className="absolute inset-2 border-4 border-purple-400 border-t-transparent rounded-full animate-spin" style={{ animationDirection: 'reverse', animationDuration: '0.8s' }}></div>
                            </div>
                            <p className="text-gray-700 font-bold text-lg">{t('Processing image...')}</p>
                            <p className="text-gray-500 text-sm mt-1">{t('Đang phân tích ảnh...')}</p>
                          </div>
                        </div>
                      </div>
                    )}

                    <ImagePreview
                      originalImage={originalImage}
                      resultImage={resultImage}
                      detections={detections}
                    />
                  </div>
                </div>

                {/* Right Column: Detect Button, Audio Controls & Results */}
                <div className="lg:col-span-1 space-y-5 sm:space-y-6">
                  {/* Detect Button */}
                  <button
                    onClick={handleDetect}
                    disabled={isProcessing}
                    className="w-full px-6 py-5 bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white rounded-2xl font-extrabold text-xl shadow-2xl hover:from-blue-700 hover:via-indigo-700 hover:to-purple-700 hover:shadow-3xl disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 flex items-center justify-center gap-3 transform hover:scale-105 active:scale-95 border-2 border-white"
                    aria-label={t('Detect Objects')}
                  >
                    {isProcessing ? (
                      <>
                        <svg className="animate-spin h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        <span>{t('Processing...')}</span>
                      </>
                    ) : (
                      <>
                        <span className="text-3xl drop-shadow-lg">🔍</span>
                        <span className="drop-shadow-md">{t('Detect Objects')}</span>
                      </>
                    )}
                  </button>

                  {/* Audio Controls for Image Mode */}
                  <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
                    <button
                      type="button"
                      onClick={handleToggleAudio}
                      className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-xl font-bold shadow-md transition-all ${
                        isAudioEnabled
                          ? 'bg-green-500 hover:bg-green-600 text-white'
                          : 'bg-gray-600 hover:bg-gray-700 text-gray-200'
                      }`}
                    >
                      <span className="text-lg">{isAudioEnabled ? '🔊' : '🔇'}</span>
                      <span className="text-sm sm:text-base">
                        {isAudioEnabled ? t('Audio On') : t('Audio Off')}
                      </span>
                    </button>

                    <button
                      type="button"
                      onClick={handleRepeatAudioImage}
                      disabled={!lastImageDetectionsRef.current || !isAudioEnabled}
                      className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-xl font-bold shadow-md transition-all ${
                        !lastImageDetectionsRef.current || !isAudioEnabled
                          ? 'bg-gray-400 text-gray-200 cursor-not-allowed'
                          : 'bg-blue-500 hover:bg-blue-600 text-white'
                      }`}
                    >
                      <span className="text-lg">🔁</span>
                      <span className="text-sm sm:text-base">{t('Repeat')}</span>
                    </button>
                  </div>

                  {/* Results Table */}
                  {detections.length > 0 ? (
                    <div className="bg-white rounded-2xl shadow-xl border-2 border-green-200 overflow-hidden">
                      <ResultsTable detections={detections} />
                    </div>
                  ) : resultImage ? (
                    <div className="bg-white rounded-2xl shadow-lg border-2 border-gray-200 p-8 sm:p-10">
                      <div className="text-center">
                        <div className="w-20 h-20 bg-gradient-to-br from-gray-200 to-gray-300 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-md">
                          <span className="text-5xl">🔍</span>
                        </div>
                        <p className="text-base font-bold text-gray-700 mb-2">{t('No objects detected')}</p>
                        <p className="text-sm text-gray-500">{t('Try with a different image')}</p>
                      </div>
                    </div>
                  ) : (
                    <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-2xl shadow-lg border-2 border-blue-200 p-8 sm:p-10">
                      <div className="text-center">
                        <div className="w-20 h-20 bg-gradient-to-br from-blue-400 to-indigo-500 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-lg">
                          <span className="text-5xl">⏳</span>
                        </div>
                        <p className="text-base font-bold text-gray-800 mb-2">{t('Click Detect to start')}</p>
                        <p className="text-sm text-gray-600">{t('Results will appear here')}</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

          </>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-gradient-to-r from-gray-800 via-gray-900 to-black border-t-4 border-blue-600 mt-16 py-8">
        <div className="container mx-auto px-4 sm:px-6 max-w-7xl">
          <div className="flex flex-col sm:flex-row items-center justify-center gap-3 text-gray-300 text-sm">
            <span className="font-bold text-white">{t('Object Detection System')}</span>
            <span className="hidden sm:inline text-gray-500">•</span>
            <span className="text-gray-400">{t('Powered by')}</span>
            <span className="font-extrabold text-blue-400">YOLO</span>
            <span className="text-gray-500">&</span>
            <span className="font-extrabold text-cyan-400">React</span>
          </div>
          <div className="mt-3 text-xs text-gray-500 text-center">
            © 2024 - {t('AI Powered Object Detection')}
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
