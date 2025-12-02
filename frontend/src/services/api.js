import axios from 'axios';

// Tự động phát hiện API URL dựa trên environment
// Nếu frontend chạy trên IP khác localhost, backend cũng nên chạy trên IP đó
const getApiBaseUrl = () => {
  // Kiểm tra nếu đang chạy trên IP khác localhost
  const hostname = window.location.hostname;
  
  // Nếu là localhost hoặc 127.0.0.1, dùng localhost:8000
  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    return 'http://localhost:8000';
  }
  
  // Nếu là IP khác (ví dụ: 192.168.1.21), dùng cùng IP với port 8000
  return `http://${hostname}:8000`;
};

const API_BASE_URL = getApiBaseUrl();

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'multipart/form-data',
  },
  timeout: 60000, // 60 seconds timeout cho detection requests
});

export const getModelInfo = async () => {
  try {
    const response = await api.get('/api/model-info', { timeout: 10000 }); // 10s cho model info
    return response.data;
  } catch (error) {
    console.error('Error fetching model info:', error);
    // Cải thiện error message
    if (error.code === 'ECONNABORTED') {
      throw new Error('Connection timeout. Please check if backend server is running.');
    } else if (error.code === 'ERR_NETWORK' || !error.response) {
      throw new Error('Cannot connect to backend server. Please check if the server is running.');
    }
    throw error;
  }
};

export const detectObjects = async (file, confThreshold = 0.25, iouThreshold = 0.45, signal = null) => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('conf_threshold', confThreshold);
    formData.append('iou_threshold', iouThreshold);

    const config = signal ? { signal } : {};
    const response = await api.post('/api/detect', formData, config);
    return response.data;
  } catch (error) {
    // Nếu bị cancel bởi AbortController, không throw error
    if (error.name === 'CanceledError' || error.code === 'ERR_CANCELED' || (signal && signal.aborted)) {
      throw new Error('Request canceled');
    }
    console.error('Error detecting objects:', error);
    // Cải thiện error messages
    if (error.code === 'ECONNABORTED') {
      const customError = new Error('Request timeout. The image may be too large or the server is busy. Please try again.');
      customError.response = { data: { detail: 'Request timeout. Please try with a smaller image or try again later.' } };
      throw customError;
    } else if (error.code === 'ERR_NETWORK' || !error.response) {
      const customError = new Error('Network error. Cannot connect to server.');
      customError.response = { data: { detail: 'Cannot connect to server. Please check your internet connection and ensure the backend is running.' } };
      throw customError;
    } else if (error.response?.status === 400) {
      // Bad request - có thể là file size hoặc format
      const detail = error.response.data?.detail || 'Invalid request. Please check your image file.';
      const customError = new Error(detail);
      customError.response = { data: { detail } };
      throw customError;
    } else if (error.response?.status === 500) {
      // Server error
      const detail = error.response.data?.detail || 'Server error occurred. Please try again later.';
      const customError = new Error(detail);
      customError.response = { data: { detail } };
      throw customError;
    }
    throw error;
  }
};

// detectBatch đã được xóa vì không còn sử dụng (chỉ hỗ trợ single image mode)

export const detectVideoFrame = async (
  file, 
  confThreshold = 0.25, 
  iouThreshold = 0.45,
  sessionId = null,
  signal = null
) => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('conf_threshold', confThreshold);
    formData.append('iou_threshold', iouThreshold);
    if (sessionId) {
      formData.append('session_id', sessionId);
    }

    const config = signal ? { signal } : {};
    const response = await api.post('/api/detect-video', formData, config);
    return response.data;
  } catch (error) {
    // Nếu bị cancel bởi AbortController, không throw error
    if (error.name === 'CanceledError' || error.code === 'ERR_CANCELED' || (signal && signal.aborted)) {
      throw new Error('Request canceled');
    }
    console.error('Error detecting video frame:', error);
    // Cải thiện error messages
    if (error.code === 'ECONNABORTED') {
      const customError = new Error('Request timeout. The frame may be too large or the server is busy. Please try again.');
      customError.response = { data: { detail: 'Request timeout. Please try with a smaller frame or try again later.' } };
      throw customError;
    } else if (error.code === 'ERR_NETWORK' || !error.response) {
      const customError = new Error('Network error. Cannot connect to server.');
      customError.response = { data: { detail: 'Cannot connect to server. Please check your internet connection and ensure the backend is running.' } };
      throw customError;
    } else if (error.response?.status === 400) {
      const detail = error.response.data?.detail || 'Invalid request. Please check your frame file.';
      const customError = new Error(detail);
      customError.response = { data: { detail } };
      throw customError;
    } else if (error.response?.status === 500) {
      const detail = error.response.data?.detail || 'Server error occurred. Please try again later.';
      const customError = new Error(detail);
      customError.response = { data: { detail } };
      throw customError;
    }
    throw error;
  }
};

export const resetTrackingSession = async (sessionId) => {
  try {
    const formData = new FormData();
    formData.append('session_id', sessionId);
    const response = await api.post('/api/reset-tracking-session', formData);
    return response.data;
  } catch (error) {
    console.error('Error resetting tracking session:', error);
    throw error;
  }
};

export const compareThresholds = async (file, thresholds = [0.1, 0.25, 0.5, 0.75]) => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('thresholds', JSON.stringify(thresholds));

    const response = await api.post('/api/compare-thresholds', formData);
    return response.data;
  } catch (error) {
    console.error('Error comparing thresholds:', error);
    throw error;
  }
};

export default api;

