import React, { useState, useEffect } from 'react';
import ImageUpload from './components/ImageUpload';
import ImagePreview from './components/ImagePreview';
import SettingsPanel from './components/SettingsPanel';
import ResultsTable from './components/ResultsTable';
import Statistics from './components/Statistics';
import { detectAnimal, detectBatch, compareThresholds, getModelInfo } from './services/api';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [originalImage, setOriginalImage] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [detections, setDetections] = useState([]);
  const [statistics, setStatistics] = useState(null);
  const [confThreshold, setConfThreshold] = useState(0.25);
  const [iouThreshold, setIouThreshold] = useState(0.45);
  const [isProcessing, setIsProcessing] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [error, setError] = useState(null);
  const [comparisonResults, setComparisonResults] = useState(null);
  const [batchFiles, setBatchFiles] = useState([]);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [batchResults, setBatchResults] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);

  useEffect(() => {
    // Load model info on mount
    getModelInfo()
      .then(info => {
        setModelInfo(info);
        if (info.default_conf_threshold) {
          setConfThreshold(info.default_conf_threshold);
        }
        if (info.default_iou_threshold) {
          setIouThreshold(info.default_iou_threshold);
        }
      })
      .catch(err => {
        console.error('Failed to load model info:', err);
        setError('Cannot connect to backend. Please check the server.');
      });
  }, []);

  // Keyboard shortcuts for navigation
  useEffect(() => {
    if (batchFiles.length <= 1) return;

    const handleKeyPress = (e) => {
      if (!isProcessing && !e.target.matches('input, textarea, select')) {
        if (e.key === 'ArrowLeft' && currentImageIndex > 0) {
          e.preventDefault();
          const prevIndex = currentImageIndex - 1;
          loadImageAndResults(batchFiles[prevIndex], prevIndex);
        } else if (e.key === 'ArrowRight' && currentImageIndex < batchFiles.length - 1) {
          e.preventDefault();
          const nextIndex = currentImageIndex + 1;
          loadImageAndResults(batchFiles[nextIndex], nextIndex);
        }
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [batchFiles, currentImageIndex, isProcessing, batchResults]);

  const handleClearAll = () => {
    setSelectedFile(null);
    setBatchFiles([]);
    setCurrentImageIndex(0);
    setBatchResults(null);
    setOriginalImage(null);
    setResultImage(null);
    setDetections([]);
    setStatistics(null);
    setComparisonResults(null);
    setError(null);
    setSuccessMessage(null);
  };

  const handleImageSelect = (file) => {
    setSelectedFile(file);
    setBatchFiles([]);
    setCurrentImageIndex(0);
    setBatchResults(null);
    setError(null);
    setResultImage(null);
    setDetections([]);
    setStatistics(null);
    setComparisonResults(null);

    // Create preview URL
    const reader = new FileReader();
    reader.onload = (e) => {
      setOriginalImage(e.target.result);
    };
    reader.readAsDataURL(file);
  };

  const handleBatchSelect = (files) => {
    if (files.length > 0) {
      setBatchFiles(files);
      setCurrentImageIndex(0);
      setBatchResults(null);
      setError(null);
      setResultImage(null);
      setDetections([]);
      setStatistics(null);
      setComparisonResults(null);
      
      // Load first image preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setOriginalImage(e.target.result);
      };
      reader.readAsDataURL(files[0]);
      setSelectedFile(files[0]);
    }
  };

  // Helper function to load image and results
  const loadImageAndResults = (file, index) => {
    setCurrentImageIndex(index);
    setSelectedFile(file);
    setComparisonResults(null);
    
    // Load image preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setOriginalImage(e.target.result);
    };
    reader.readAsDataURL(file);
    
    // Load results from batchResults if available
    if (batchResults && batchResults.results) {
      const fileName = file.name;
      const result = batchResults.results.find(r => r.filename === fileName);
      if (result) {
        setResultImage(result.image_base64);
        setDetections(result.detections);
        // Calculate statistics for this image
        if (result.detections && result.detections.length > 0) {
          const confidences = result.detections.map(d => d.confidence);
          const classes = [...new Set(result.detections.map(d => d.class))];
          setStatistics({
            total: result.detections.length,
            avg_confidence: confidences.reduce((a, b) => a + b, 0) / confidences.length,
            min_confidence: Math.min(...confidences),
            max_confidence: Math.max(...confidences),
            classes: classes
          });
        } else {
          setStatistics(null);
        }
      } else {
        setResultImage(null);
        setDetections([]);
        setStatistics(null);
      }
    } else {
      setResultImage(null);
      setDetections([]);
      setStatistics(null);
    }
  };

  const handleNextImage = () => {
    if (batchFiles.length > 0 && currentImageIndex < batchFiles.length - 1) {
      const nextIndex = currentImageIndex + 1;
      loadImageAndResults(batchFiles[nextIndex], nextIndex);
    }
  };

  const handlePrevImage = () => {
    if (batchFiles.length > 0 && currentImageIndex > 0) {
      const prevIndex = currentImageIndex - 1;
      loadImageAndResults(batchFiles[prevIndex], prevIndex);
    }
  };

  // Keyboard shortcuts for navigation
  useEffect(() => {
    if (batchFiles.length <= 1) return;

    const handleKeyPress = (e) => {
      if (!isProcessing && !e.target.matches('input, textarea, select')) {
        if (e.key === 'ArrowLeft' && currentImageIndex > 0) {
          e.preventDefault();
          handlePrevImage();
        } else if (e.key === 'ArrowRight' && currentImageIndex < batchFiles.length - 1) {
          e.preventDefault();
          handleNextImage();
        }
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [batchFiles, currentImageIndex, isProcessing]);

  const handleDetect = async () => {
    if (!selectedFile) {
      setError('Please select an image first!');
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      // If multiple images selected, detect all at once
      if (batchFiles.length > 1) {
        const batchResult = await detectBatch(batchFiles, confThreshold, iouThreshold);
        setBatchResults(batchResult);
        
        // Display results for current image
        if (batchResult.results) {
          const currentFileName = batchFiles[currentImageIndex].name;
          const result = batchResult.results.find(r => r.filename === currentFileName);
          if (result) {
            setResultImage(result.image_base64);
            setDetections(result.detections);
            // Calculate statistics for current image
            if (result.detections && result.detections.length > 0) {
              const confidences = result.detections.map(d => d.confidence);
              const classes = [...new Set(result.detections.map(d => d.class))];
              setStatistics({
                total: result.detections.length,
                avg_confidence: confidences.reduce((a, b) => a + b, 0) / confidences.length,
                min_confidence: Math.min(...confidences),
                max_confidence: Math.max(...confidences),
                classes: classes
              });
            } else {
              setStatistics(null);
            }
          }
        }
      } else {
        // Single image detection
        const result = await detectAnimal(selectedFile, confThreshold, iouThreshold);
        setResultImage(result.image_base64);
        setDetections(result.detections);
        setStatistics(result.statistics);
      }
      
      setComparisonResults(null);
      
      // Show success message
      if (batchFiles.length > 1) {
        setSuccessMessage(`Successfully processed ${batchFiles.length} image${batchFiles.length > 1 ? 's' : ''}!`);
      } else {
        setSuccessMessage('Detection completed successfully!');
      }
      setTimeout(() => setSuccessMessage(null), 5000);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred during detection. Please try again.');
      console.error('Detection error:', err);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleCompare = async () => {
    if (!selectedFile) {
      setError('Please select an image first!');
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      const result = await compareThresholds(selectedFile, [0.1, 0.25, 0.5, 0.75]);
      setComparisonResults(result.comparisons);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred during comparison. Please try again.');
      console.error('Comparison error:', err);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200 sticky top-0 z-50">
        <div className="container mx-auto px-6 py-4 max-w-6xl">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-3">
                <span className="text-3xl sm:text-4xl flex-shrink-0">🐾</span>
                <div className="min-w-0">
                  <h1 className="text-2xl sm:text-3xl font-bold text-gray-800 leading-tight">
                    Animal Detection System
                  </h1>
                  <p className="text-gray-600 mt-0.5 text-xs sm:text-sm font-medium">Animal Detection using YOLO - AI Powered</p>
                </div>
              </div>
            </div>
            {modelInfo && (
              <div className="flex items-center flex-shrink-0">
                <div className="text-right">
                  <div className="text-xs text-gray-500 mb-1">Model Status</div>
                  <div className="inline-flex items-center px-3 sm:px-4 py-1.5 sm:py-2 bg-emerald-100 text-emerald-700 border border-emerald-200 rounded-full shadow-sm">
                    <span className="w-2 h-2 bg-emerald-500 rounded-full mr-2 animate-pulse"></span>
                    <span className="font-semibold text-xs sm:text-sm">Ready</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-8 max-w-6xl">
        {/* Error Message */}
        {error && (
          <div className="mb-8 bg-red-50 border-l-4 border-red-500 text-red-700 px-6 py-4 rounded-lg shadow-md max-w-4xl mx-auto">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <svg className="w-5 h-5 mr-2 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
                <span className="font-semibold">{error}</span>
              </div>
              <button
                onClick={() => setError(null)}
                className="ml-4 text-red-700 hover:text-red-900"
                aria-label="Close error message"
              >
                ✕
              </button>
            </div>
          </div>
        )}

        {/* Success Message */}
        {successMessage && (
          <div className="mb-8 bg-green-50 border-l-4 border-green-500 text-green-700 px-6 py-4 rounded-lg shadow-md max-w-4xl mx-auto">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <svg className="w-5 h-5 mr-2 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                <span className="font-semibold">{successMessage}</span>
              </div>
              <button
                onClick={() => setSuccessMessage(null)}
                className="ml-4 text-green-700 hover:text-green-900"
                aria-label="Close success message"
              >
                ✕
              </button>
            </div>
          </div>
        )}

        {/* Upload Section */}
        <div className="mb-12">
          <ImageUpload
            onImageSelect={handleImageSelect}
            onBatchSelect={handleBatchSelect}
            isProcessing={isProcessing}
          />
        </div>

        {/* Image Navigation - Show when multiple images selected */}
        {batchFiles.length > 1 && originalImage && (
          <div className="mb-6 max-w-3xl mx-auto">
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                <div className="flex items-center gap-3 flex-1 min-w-0">
                  <span className="text-xl flex-shrink-0">📁</span>
                  <div className="min-w-0">
                    <div className="font-semibold text-gray-800">Multiple Images Selected</div>
                    <div className="text-sm text-gray-600 truncate" title={batchFiles[currentImageIndex]?.name}>
                      {batchFiles[currentImageIndex]?.name || `Image ${currentImageIndex + 1}`}
                    </div>
                    <div className="text-xs text-gray-500 mt-0.5">
                      Image {currentImageIndex + 1} of {batchFiles.length}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={handlePrevImage}
                    disabled={currentImageIndex === 0 || isProcessing}
                    className="px-4 py-2 bg-white border border-gray-300 text-gray-700 rounded-lg font-medium shadow-sm hover:bg-gray-50 hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center gap-2"
                    title="Previous image (←)"
                    aria-label="Previous image"
                  >
                    <span>←</span>
                    <span className="hidden sm:inline">Previous</span>
                  </button>
                  <button
                    onClick={handleNextImage}
                    disabled={currentImageIndex === batchFiles.length - 1 || isProcessing}
                    className="px-4 py-2 bg-white border border-gray-300 text-gray-700 rounded-lg font-medium shadow-sm hover:bg-gray-50 hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center gap-2"
                    title="Next image (→)"
                    aria-label="Next image"
                  >
                    <span className="hidden sm:inline">Next</span>
                    <span>→</span>
                  </button>
                  <button
                    onClick={handleClearAll}
                    disabled={isProcessing}
                    className="px-4 py-2 bg-red-50 border border-red-200 text-red-700 rounded-lg font-medium shadow-sm hover:bg-red-100 hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
                    title="Clear all and start fresh"
                    aria-label="Clear all"
                  >
                    Clear
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Info Section - Only show when no image selected */}
        {!originalImage && (
          <div className="max-w-3xl mx-auto mt-12 mb-8">
            <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
              <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                <span className="mr-2">ℹ️</span>
                How to Use
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="flex flex-col items-center text-center p-4 bg-gray-50 rounded-lg">
                  <div className="text-3xl mb-2">📤</div>
                  <h4 className="font-semibold text-gray-800 mb-1 text-sm">1. Upload</h4>
                  <p className="text-xs text-gray-600">Select or drag & drop an image</p>
                </div>
                <div className="flex flex-col items-center text-center p-4 bg-gray-50 rounded-lg">
                  <div className="text-3xl mb-2">⚙️</div>
                  <h4 className="font-semibold text-gray-800 mb-1 text-sm">2. Configure</h4>
                  <p className="text-xs text-gray-600">Adjust detection thresholds</p>
                </div>
                <div className="flex flex-col items-center text-center p-4 bg-gray-50 rounded-lg">
                  <div className="text-3xl mb-2">🔍</div>
                  <h4 className="font-semibold text-gray-800 mb-1 text-sm">3. Detect</h4>
                  <p className="text-xs text-gray-600">View detection results</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Comparison Results */}
        {comparisonResults && (
          <div className="mb-10 bg-white rounded-lg shadow-md p-6 border border-gray-200 max-w-5xl mx-auto">
            <h2 className="text-xl font-bold mb-5 flex items-center text-gray-800">
              <span className="mr-2">📊</span>
              Compare Thresholds
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {Object.entries(comparisonResults).map(([threshold, data], index) => (
                <div 
                  key={threshold} 
                  className="bg-gray-50 border border-gray-200 rounded-lg p-4 hover:bg-gray-100 transition-colors"
                >
                  <div className="text-xs font-medium text-gray-600 mb-1">Threshold</div>
                  <div className="text-xl font-bold text-gray-800 mb-2">{threshold}</div>
                  <div className="text-2xl font-bold text-blue-600 mb-1">{data.count}</div>
                  <div className="text-xs text-gray-500">detections</div>
                  <div className="mt-2 pt-2 border-t border-gray-200">
                    <div className="text-xs text-gray-600">
                      {data.classes.length > 0 ? (
                        <span className="font-medium">{data.classes.join(', ')}</span>
                      ) : (
                        <span className="italic">None</span>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Main Layout */}
        {originalImage && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
            {/* Left Column: Settings */}
            <div className="lg:col-span-1 order-2 lg:order-1">
              <SettingsPanel
                confThreshold={confThreshold}
                iouThreshold={iouThreshold}
                onConfChange={setConfThreshold}
                onIouChange={setIouThreshold}
                onDetect={handleDetect}
                onCompare={handleCompare}
                isProcessing={isProcessing}
              />
            </div>

            {/* Right Column: Image Preview */}
            <div className="lg:col-span-2 order-1 lg:order-2">
              <ImagePreview
                originalImage={originalImage}
                resultImage={resultImage}
                detections={detections}
              />
            </div>
          </div>
        )}

        {/* Results Section */}
        {detections.length > 0 && (
          <div className="space-y-6 max-w-6xl mx-auto">
            <ResultsTable detections={detections} />
            <Statistics statistics={statistics} detections={detections} />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-12 py-6">
        <div className="container mx-auto px-6 max-w-6xl">
          <div className="flex flex-col sm:flex-row items-center justify-center gap-2 text-gray-600 text-sm">
            <span className="font-medium">Animal Detection System</span>
            <span className="hidden sm:inline">•</span>
            <span>Powered by</span>
            <span className="font-semibold text-gray-800">YOLO</span>
            <span>&</span>
            <span className="font-semibold text-gray-800">React</span>
          </div>
          <div className="mt-2 text-xs text-gray-500 text-center">
            © 2024 - AI Powered Animal Detection
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;

