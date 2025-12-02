import React, { useRef, useState } from 'react';
import { t } from '../utils/translations';

const ImageUpload = ({ onImageSelect, isProcessing }) => {
  const fileInputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);
  const [validationError, setValidationError] = useState(null);

  const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
  const ALLOWED_TYPES = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/webp', 'image/tiff'];

  const validateFile = (file) => {
    if (!file) {
      return t('No file selected. Please select an image file.');
    }
    
    if (!file.type || !ALLOWED_TYPES.includes(file.type)) {
      return t('File format not supported. Supported formats: JPG, PNG, BMP, WEBP, TIFF');
    }
    
    if (file.size === 0) {
      return t('File is empty. Please select a valid image file.');
    }
    
    if (file.size > MAX_FILE_SIZE) {
      return t('File size exceeds maximum limit of 10MB. Please select a smaller image.');
    }
    
    return null;
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    setValidationError(null);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const files = Array.from(e.dataTransfer.files);
      const validFiles = [];
      const errors = [];

      files.forEach(file => {
        const error = validateFile(file);
        if (error) {
          errors.push(error);
        } else {
          validFiles.push(file);
        }
      });

      if (errors.length > 0) {
        setValidationError(errors.join('; '));
      }

      if (validFiles.length > 0) {
        // Chỉ lấy file đầu tiên (single image only)
        onImageSelect(validFiles[0]);
      }
    }
  };

  const handleFileSelect = (e) => {
    setValidationError(null);
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const error = validateFile(file);
      if (error) {
        setValidationError(error);
      } else {
        onImageSelect(file);
      }
    }
    e.target.value = ''; // Reset input
  };


  return (
    <div className="w-full max-w-4xl mx-auto">
      {/* Upload Zone - Cải thiện với gradient và animation */}
      <div
        className={`border-2 border-dashed rounded-3xl p-12 md:p-16 text-center transition-all duration-300 ${
          dragActive
            ? 'border-blue-500 bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 scale-[1.02] shadow-2xl ring-4 ring-blue-200'
            : 'border-gray-300 bg-white hover:border-blue-400 hover:shadow-xl hover:bg-gradient-to-br hover:from-gray-50 hover:to-blue-50'
        } ${isProcessing ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => !isProcessing && fileInputRef.current?.click()}
        role="button"
        tabIndex={0}
        aria-label={t('Drag and drop image here or click to select')}
        onKeyDown={(e) => {
          if ((e.key === 'Enter' || e.key === ' ') && !isProcessing) {
            e.preventDefault();
            fileInputRef.current?.click();
          }
        }}
      >
        <div className="flex flex-col items-center justify-center space-y-6">
          <div className={`relative transition-all duration-300 ${dragActive ? 'scale-110 rotate-5' : 'hover:scale-105'}`}>
            <div className="w-24 h-24 md:w-28 md:h-28 bg-gradient-to-br from-blue-500 via-indigo-500 to-purple-600 rounded-3xl flex items-center justify-center shadow-2xl">
              <svg
                className="w-12 h-12 md:w-16 md:h-16 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                />
              </svg>
            </div>
          </div>
          <div className="space-y-2">
            <p className="text-xl md:text-2xl font-extrabold text-gray-800">
              {dragActive ? (
                <span className="text-blue-600 animate-pulse">{t('Drop image here!')}</span>
              ) : (
                t('Drag and drop image here or click to select')
              )}
            </p>
            <p className="text-sm md:text-base text-gray-600 font-semibold">
              {t('Supported formats: JPG, PNG, BMP, WEBP, TIFF')}
            </p>
          </div>
        </div>
      </div>

      {/* Validation Error */}
      {validationError && (
        <div className="mt-6 bg-gradient-to-r from-red-50 to-pink-50 border-2 border-red-400 text-red-800 px-5 py-4 rounded-2xl text-sm shadow-lg">
          <div className="flex items-start gap-3">
            <div className="w-8 h-8 bg-red-500 rounded-full flex items-center justify-center flex-shrink-0">
              <span className="text-lg">⚠️</span>
            </div>
            <span className="font-semibold pt-1">{validationError}</span>
          </div>
        </div>
      )}

      {/* Action Button - Chỉ single image */}
      <div className="mt-8 flex justify-center">
        <button
          onClick={() => !isProcessing && fileInputRef.current?.click()}
          disabled={isProcessing}
          className="px-10 py-4 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-2xl font-bold text-lg shadow-xl hover:from-blue-700 hover:to-indigo-700 hover:shadow-2xl disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 flex items-center justify-center gap-3 transform hover:scale-105 active:scale-95"
          aria-label={t('Select Single Image')}
        >
          <span className="text-2xl">📸</span>
          <span>{t('Select Single Image')}</span>
        </button>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileSelect}
        className="hidden"
        disabled={isProcessing}
      />
    </div>
  );
};

export default ImageUpload;

