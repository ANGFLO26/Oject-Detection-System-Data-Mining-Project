import React, { useRef, useState } from 'react';

const ImageUpload = ({ onImageSelect, onBatchSelect, isProcessing }) => {
  const fileInputRef = useRef(null);
  const batchInputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);
  const [validationError, setValidationError] = useState(null);

  const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
  const ALLOWED_TYPES = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/webp', 'image/tiff'];

  const validateFile = (file) => {
    if (!ALLOWED_TYPES.includes(file.type)) {
      return `File "${file.name}" is not a supported image format. Supported: JPG, PNG, BMP, WEBP, TIFF`;
    }
    if (file.size > MAX_FILE_SIZE) {
      return `File "${file.name}" is too large. Maximum size: ${(MAX_FILE_SIZE / 1024 / 1024).toFixed(0)}MB`;
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
        if (validFiles.length === 1) {
          onImageSelect(validFiles[0]);
        } else {
          onBatchSelect(validFiles);
        }
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

  const handleBatchSelect = (e) => {
    setValidationError(null);
    if (e.target.files && e.target.files.length > 0) {
      const files = Array.from(e.target.files);
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
        onBatchSelect(validFiles);
      }
    }
    e.target.value = ''; // Reset input
  };

  return (
    <div className="w-full max-w-3xl mx-auto">
      {/* Upload Zone */}
      <div
        className={`border-2 border-dashed rounded-xl p-10 md:p-14 text-center transition-all duration-300 ${
          dragActive
            ? 'border-blue-400 bg-blue-50 scale-[1.01] shadow-lg'
            : 'border-gray-300 bg-white hover:border-gray-400 hover:shadow-md'
        } ${isProcessing ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => !isProcessing && fileInputRef.current?.click()}
      >
        <div className="flex flex-col items-center justify-center space-y-4">
          <div className={`relative ${dragActive ? 'scale-110' : ''} transition-transform duration-300`}>
            <svg
              className="w-16 h-16 md:w-20 md:h-20 text-gray-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
              />
            </svg>
          </div>
          <div className="space-y-1.5">
            <p className="text-lg md:text-xl font-bold text-gray-800">
              {dragActive ? 'Drop image here!' : 'Drag and drop image here or click to select'}
            </p>
            <p className="text-xs md:text-sm text-gray-500 font-medium">
              Supported formats: JPG, PNG, BMP, WEBP, TIFF
            </p>
          </div>
        </div>
      </div>

      {/* Validation Error */}
      {validationError && (
        <div className="mt-4 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm">
          <div className="flex items-start">
            <span className="mr-2">⚠️</span>
            <span>{validationError}</span>
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="mt-5 flex flex-col sm:flex-row gap-3 justify-center items-stretch sm:items-center">
        <button
          onClick={() => !isProcessing && fileInputRef.current?.click()}
          disabled={isProcessing}
          className="flex-1 sm:flex-none px-6 py-3 bg-blue-600 text-white rounded-lg font-medium shadow-sm hover:bg-blue-700 hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center gap-2"
        >
          <span>📸</span>
          <span>Select Single Image</span>
        </button>
        <button
          onClick={() => !isProcessing && batchInputRef.current?.click()}
          disabled={isProcessing}
          className="flex-1 sm:flex-none px-6 py-3 bg-gray-700 text-white rounded-lg font-medium shadow-sm hover:bg-gray-800 hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center gap-2"
        >
          <span>📁</span>
          <span>Select Multiple Images</span>
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
      <input
        ref={batchInputRef}
        type="file"
        accept="image/*"
        multiple
        onChange={handleBatchSelect}
        className="hidden"
        disabled={isProcessing}
      />
    </div>
  );
};

export default ImageUpload;

