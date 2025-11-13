import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'multipart/form-data',
  },
});

export const getModelInfo = async () => {
  try {
    const response = await api.get('/api/model-info');
    return response.data;
  } catch (error) {
    console.error('Error fetching model info:', error);
    throw error;
  }
};

export const detectAnimal = async (file, confThreshold = 0.25, iouThreshold = 0.45) => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('conf_threshold', confThreshold);
    formData.append('iou_threshold', iouThreshold);

    const response = await api.post('/api/detect', formData);
    return response.data;
  } catch (error) {
    console.error('Error detecting animal:', error);
    throw error;
  }
};

export const detectBatch = async (files, confThreshold = 0.25, iouThreshold = 0.45) => {
  try {
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });
    formData.append('conf_threshold', confThreshold);
    formData.append('iou_threshold', iouThreshold);

    const response = await api.post('/api/detect-batch', formData);
    return response.data;
  } catch (error) {
    console.error('Error batch detecting:', error);
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

