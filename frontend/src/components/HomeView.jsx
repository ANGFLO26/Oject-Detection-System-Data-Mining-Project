import React from 'react';
import { t } from '../utils/translations';

const HomeView = ({ onSelectMode }) => {
  return (
    <div className="min-h-[70vh] flex items-center justify-center py-12 sm:py-16 animate-fade-in">
      <div className="w-full max-w-5xl px-4 sm:px-6">
        {/* Title */}
        <div className="text-center mb-12 sm:mb-16 animate-slide-down">
          <h2 className="text-3xl sm:text-4xl md:text-5xl font-extrabold text-gray-800 mb-4">
            {t('Chọn Chế Độ')}
          </h2>
          <p className="text-lg sm:text-xl text-gray-600 font-medium">
            {t('Chọn cách bạn muốn nhận diện đối tượng')}
          </p>
        </div>

        {/* Two Options */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 sm:gap-8">
          {/* Camera Option */}
          <button
            onClick={() => onSelectMode('camera')}
            className="group relative bg-white rounded-3xl shadow-2xl border-2 border-purple-200 p-8 sm:p-10 hover:border-purple-400 hover:shadow-3xl transition-all duration-300 transform hover:scale-105 active:scale-95 animate-slide-up"
            style={{ animationDelay: '0.1s' }}
            aria-label={t('Chọn chế độ Camera')}
          >
            <div className="flex flex-col items-center text-center">
              {/* Icon */}
              <div className="w-24 h-24 sm:w-28 sm:h-28 bg-gradient-to-br from-purple-500 via-indigo-600 to-purple-700 rounded-3xl flex items-center justify-center mb-6 shadow-xl group-hover:scale-110 transition-transform duration-300">
                <span className="text-5xl sm:text-6xl">📹</span>
              </div>
              
              {/* Title */}
              <h3 className="text-2xl sm:text-3xl font-extrabold text-gray-800 mb-3">
                {t('Camera')}
              </h3>
              
              {/* Description */}
              <p className="text-base sm:text-lg text-gray-600 mb-6 font-medium">
                {t('Nhận diện đối tượng theo thời gian thực qua camera')}
              </p>
              
              {/* Features */}
              <div className="w-full space-y-2 text-left">
                <div className="flex items-center gap-2 text-sm text-gray-700">
                  <span className="text-green-500 font-bold">✓</span>
                  <span>{t('Real-time detection')}</span>
                </div>
                <div className="flex items-center gap-2 text-sm text-gray-700">
                  <span className="text-green-500 font-bold">✓</span>
                  <span>{t('Audio feedback tự động')}</span>
                </div>
                <div className="flex items-center gap-2 text-sm text-gray-700">
                  <span className="text-green-500 font-bold">✓</span>
                  <span>{t('Bounding boxes hiển thị')}</span>
                </div>
              </div>
              
              {/* Arrow indicator */}
              <div className="mt-6 text-purple-600 text-2xl font-bold group-hover:translate-x-2 transition-transform duration-300">
                →
              </div>
            </div>
          </button>

          {/* Image Option */}
          <button
            onClick={() => onSelectMode('image')}
            className="group relative bg-white rounded-3xl shadow-2xl border-2 border-blue-200 p-8 sm:p-10 hover:border-blue-400 hover:shadow-3xl transition-all duration-300 transform hover:scale-105 active:scale-95 animate-slide-up"
            style={{ animationDelay: '0.2s' }}
            aria-label={t('Chọn chế độ Hình ảnh')}
          >
            <div className="flex flex-col items-center text-center">
              {/* Icon */}
              <div className="w-24 h-24 sm:w-28 sm:h-28 bg-gradient-to-br from-blue-500 via-indigo-600 to-cyan-600 rounded-3xl flex items-center justify-center mb-6 shadow-xl group-hover:scale-110 transition-transform duration-300">
                <span className="text-5xl sm:text-6xl">📷</span>
              </div>
              
              {/* Title */}
              <h3 className="text-2xl sm:text-3xl font-extrabold text-gray-800 mb-3">
                {t('Hình Ảnh')}
              </h3>
              
              {/* Description */}
              <p className="text-base sm:text-lg text-gray-600 mb-6 font-medium">
                {t('Tải ảnh lên để nhận diện đối tượng')}
              </p>
              
              {/* Features */}
              <div className="w-full space-y-2 text-left">
                <div className="flex items-center gap-2 text-sm text-gray-700">
                  <span className="text-green-500 font-bold">✓</span>
                  <span>{t('Upload 1 ảnh')}</span>
                </div>
                <div className="flex items-center gap-2 text-sm text-gray-700">
                  <span className="text-green-500 font-bold">✓</span>
                  <span>{t('Kết quả chi tiết')}</span>
                </div>
                <div className="flex items-center gap-2 text-sm text-gray-700">
                  <span className="text-green-500 font-bold">✓</span>
                  <span>{t('Audio feedback')}</span>
                </div>
              </div>
              
              {/* Arrow indicator */}
              <div className="mt-6 text-blue-600 text-2xl font-bold group-hover:translate-x-2 transition-transform duration-300">
                →
              </div>
            </div>
          </button>
        </div>
      </div>
    </div>
  );
};

export default HomeView;
