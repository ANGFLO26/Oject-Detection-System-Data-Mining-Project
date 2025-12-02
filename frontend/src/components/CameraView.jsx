import React, { useRef, useEffect, useState, useCallback } from 'react';
import { detectObjects, detectVideoFrame, resetTrackingSession } from '../services/api';
import { audioService } from '../services/audioService';
import { t } from '../utils/translations';
import { translateClass, capitalizeFirst } from '../utils/classTranslations';
import ResultsTable from './ResultsTable';

const CameraView = ({ isActive, onClose }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);
  const fallbackTimeoutRef = useRef(null);
  const loadedMetadataHandlerRef = useRef(null);
  
  // Request Queue với AbortController
  const abortControllerRef = useRef(null);
  const isDetectingRef = useRef(false); // Dùng ref để tránh stale closure
  const skippedFramesRef = useRef(0); // Track skipped frames
  
  // Audio feedback timer (hiện không dùng cooldown phức tạp, nhưng vẫn giữ ref để dễ mở rộng sau)
  const audioCooldownTimerRef = useRef(null);
  
  // Lưu danh sách track_id đã được đọc để không đọc lại
  const announcedTrackIdsRef = useRef(new Set());
  // Lưu danh sách detections đã được đọc gần nhất để có thể "Đọc lại"
  const lastAnnouncedDetectionsRef = useRef(null);
  
  const [isDetecting, setIsDetecting] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);
  const [detectionInterval, setDetectionInterval] = useState(500); // ms giữa các lần detect
  const [frameCount, setFrameCount] = useState(0);
  const [lastDetections, setLastDetections] = useState([]); // Giữ lại để backward compatibility
  const [activeTracks, setActiveTracks] = useState(new Map()); // track_id -> track data
  const [sessionId] = useState(() => `session_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`); // Unique session ID
  const [fps, setFps] = useState(0); // FPS counter
  const [detectionRate, setDetectionRate] = useState(0); // Detection rate
  const [isAudioEnabled, setIsAudioEnabled] = useState(true); // Trạng thái bật/tắt audio

  // Đồng bộ trạng thái audio với audioService
  useEffect(() => {
    audioService.setEnabled(isAudioEnabled);
    if (!isAudioEnabled) {
      audioService.stop();
    }
  }, [isAudioEnabled]);

  // Khởi động camera
  const startCamera = useCallback(async () => {
    try {
      setError(null);
      
      // Dừng camera cũ nếu có
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
      
      // Đợi một chút để đảm bảo cleanup hoàn tất
      await new Promise(resolve => setTimeout(resolve, 100));
      
      // Yêu cầu quyền truy cập camera
      // Ưu tiên camera trước cho laptop/desktop, camera sau cho mobile
      const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: isMobile ? 'environment' : 'user' // Camera sau cho mobile, camera trước cho laptop
        }
      });

      streamRef.current = stream;
      
      if (videoRef.current) {
        const video = videoRef.current;
        
        // Set srcObject trước
        video.srcObject = stream;
        
        // Đợi video sẵn sàng
        const handleLoadedMetadata = async () => {
          try {
            // Kiểm tra readyState trước khi play
            if (video && video.readyState >= 2) {
              await video.play();
              setIsStreaming(true);
            }
          } catch (playError) {
            // Xử lý lỗi play một cách graceful
            if (playError.name !== 'AbortError' && playError.name !== 'NotAllowedError') {
              console.warn('Video play warning:', playError);
            }
            // Vẫn set streaming nếu video đã load được metadata
            if (video && video.readyState >= 2) {
              setIsStreaming(true);
            }
          }
        };
        
        loadedMetadataHandlerRef.current = handleLoadedMetadata;
        video.addEventListener('loadedmetadata', handleLoadedMetadata, { once: true });
        
        // Fallback: nếu onloadedmetadata không fire, thử play sau 500ms
        fallbackTimeoutRef.current = setTimeout(async () => {
          if (video && video.readyState >= 2) {
            try {
              await video.play();
              setIsStreaming(true);
            } catch (playError) {
              if (playError.name !== 'AbortError' && playError.name !== 'NotAllowedError') {
                console.warn('Video play fallback warning:', playError);
              }
              // Vẫn cho phép streaming nếu video đã sẵn sàng
              if (video.readyState >= 2) {
                setIsStreaming(true);
              }
            }
          }
          fallbackTimeoutRef.current = null;
        }, 500);
      }
    } catch (err) {
      console.error('Error accessing camera:', err);
      
      // Cải thiện error messages dựa trên loại lỗi
      let errorMessage = 'Không thể truy cập camera.';
      
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        errorMessage = 'Quyền truy cập camera bị từ chối. Vui lòng cho phép truy cập camera trong cài đặt trình duyệt.';
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        errorMessage = 'Không tìm thấy camera. Vui lòng kiểm tra xem camera đã được kết nối chưa.';
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        errorMessage = 'Camera đang được sử dụng bởi ứng dụng khác. Vui lòng đóng ứng dụng khác và thử lại.';
      } else if (err.name === 'OverconstrainedError' || err.name === 'ConstraintNotSatisfiedError') {
        errorMessage = 'Camera không hỗ trợ yêu cầu. Vui lòng thử lại.';
      } else if (err.name === 'SecurityError') {
        errorMessage = 'Lỗi bảo mật. Vui lòng đảm bảo bạn đang sử dụng HTTPS hoặc localhost.';
      } else {
        errorMessage = 'Không thể truy cập camera. Vui lòng kiểm tra quyền truy cập và thử lại.';
      }
      
      setError(errorMessage);
      if (audioService.isSupported()) {
        audioService.speakSystemMessage('Không thể truy cập camera', 5);
      }
    }
  }, []);

  // Xử lý audio feedback cho camera mode
  // Yêu cầu: Chỉ đọc đối tượng MỚI (track_id mới), không đọc lại nếu còn ở đó
  const handleCameraAudioFeedback = useCallback((detections) => {
    if (!detections || detections.length === 0) {
      return;
    }

    // Lưu lại danh sách detections đã đọc gần nhất để có thể "Đọc lại"
    lastAnnouncedDetectionsRef.current = detections;

    // Gọi speakDetections với delay cố định 2s giữa các đối tượng
    if (isAudioEnabled) {
      audioService.speakDetections(detections, 2000);
    }
  }, [isAudioEnabled]);

  const stopCamera = useCallback(() => {
    // Cancel pending request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    if (fallbackTimeoutRef.current) {
      clearTimeout(fallbackTimeoutRef.current);
      fallbackTimeoutRef.current = null;
    }

    // Clear audio cooldown timer
    if (audioCooldownTimerRef.current) {
      clearTimeout(audioCooldownTimerRef.current);
      audioCooldownTimerRef.current = null;
    }
    // Reset danh sách track_id đã được đọc và detections đã đọc gần nhất
    announcedTrackIdsRef.current = new Set();
    lastAnnouncedDetectionsRef.current = null;

    if (videoRef.current && loadedMetadataHandlerRef.current) {
      videoRef.current.removeEventListener('loadedmetadata', loadedMetadataHandlerRef.current);
      loadedMetadataHandlerRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    // Reset tracking session ở backend để cleanup
    if (sessionId) {
      resetTrackingSession(sessionId).catch(err => {
        // Ignore errors khi cleanup session (có thể session đã bị cleanup tự động)
        console.warn('Failed to reset tracking session:', err);
      });
    }

    setIsStreaming(false);
    setIsDetecting(false);
    isDetectingRef.current = false;
    setFrameCount(0);
    setLastDetections([]);
    setActiveTracks(new Map()); // Reset tracks
    skippedFramesRef.current = 0;
    setFps(0);
    setDetectionRate(0);
  }, [sessionId]);

  // Nút bật/tắt audio
  const handleToggleAudio = useCallback(() => {
    setIsAudioEnabled(prev => !prev);
  }, []);

  // Nút "Đọc lại" - đọc lại nhóm đối tượng đã đọc gần nhất
  const handleRepeatAudio = useCallback(() => {
    const detections = lastAnnouncedDetectionsRef.current;
    if (!detections || detections.length === 0) {
      return;
    }
    audioService.stop();
    if (isAudioEnabled) {
      audioService.speakDetections(detections, 2000);
    }
  }, [isAudioEnabled]);

  // Capture frame và gửi đi detect với Request Queue + Frame Skipping + Image Optimization
  const captureAndDetect = useCallback(async () => {
    // Frame Skipping: Skip nếu đang detect
    if (!videoRef.current || !isStreaming || isDetectingRef.current) {
      skippedFramesRef.current += 1;
      return;
    }

    // Cancel request cũ nếu có
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    // Tạo AbortController mới
    const abortController = new AbortController();
    abortControllerRef.current = abortController;
    isDetectingRef.current = true;
    setIsDetecting(true);
    setFrameCount(prev => prev + 1);

    const startTime = performance.now();

    try {
      // Capture frame từ video
      const video = videoRef.current;
      const canvas = canvasRef.current;
      
      if (!canvas) {
        isDetectingRef.current = false;
        setIsDetecting(false);
        abortControllerRef.current = null;
        return;
      }

      // Image Optimization: Resize để giảm file size
      const maxWidth = 320; // Giảm từ 640 xuống 320
      const maxHeight = 240; // Giảm từ 480 xuống 240
      const videoWidth = video.videoWidth || 640;
      const videoHeight = video.videoHeight || 480;
      
      // Tính toán scale để giữ aspect ratio
      const scale = Math.min(maxWidth / videoWidth, maxHeight / videoHeight, 1);
      const canvasWidth = Math.floor(videoWidth * scale);
      const canvasHeight = Math.floor(videoHeight * scale);
      
      canvas.width = canvasWidth;
      canvas.height = canvasHeight;
      
      // Lưu canvas size thực tế vào canvas element để dùng cho bounding box scaling
      canvas._actualWidth = canvasWidth;
      canvas._actualHeight = canvasHeight;
      
      const ctx = canvas.getContext('2d');
      // Draw với scale
      ctx.drawImage(video, 0, 0, canvasWidth, canvasHeight);

      // Convert canvas to blob với quality thấp hơn (0.6 thay vì 0.8)
      try {
        canvas.toBlob(async (blob) => {
          // Check nếu request đã bị cancel
          if (abortController.signal.aborted) {
            isDetectingRef.current = false;
            setIsDetecting(false);
            return;
          }

          if (!blob) {
            isDetectingRef.current = false;
            setIsDetecting(false);
            abortControllerRef.current = null;
            return;
          }

          try {
            // Tạo File object từ blob
            const file = new File([blob], `frame_${Date.now()}.jpg`, { type: 'image/jpeg' });
            
            // Gọi API detect-video với tracking (thay vì detect thông thường)
            const result = await detectVideoFrame(file, 0.25, 0.45, sessionId, abortController.signal);
            
            // Check lại nếu request đã bị cancel
            if (abortController.signal.aborted) {
              return;
            }
            
            const endTime = performance.now();
            const detectionTime = endTime - startTime;
            
            // Update detection rate
            setDetectionRate(prev => {
              const newRate = Math.round(1000 / detectionTime);
              return Math.floor((prev * 0.7) + (newRate * 0.3)); // Moving average
            });
            
            if (result && result.tracks && result.tracks.length > 0) {
              // Update active tracks map
              setActiveTracks(prev => {
                const newTracksMap = new Map(prev);
                
                // Update với tracks mới
                const currentTrackIds = new Set();
                result.tracks.forEach(track => {
                  const trackId = track.track_id;
                  currentTrackIds.add(trackId);
                  newTracksMap.set(trackId, {
                    ...track,
                    last_seen: Date.now()
                  });
                });
                
                // Remove old tracks (không xuất hiện trong frame này)
                // Giữ lại tracks không xuất hiện < 2 giây (có thể bị tạm thời che khuất)
                const now = Date.now();
                for (const [id, track] of newTracksMap.entries()) {
                  if (!currentTrackIds.has(id)) {
                    if (now - track.last_seen > 2000) {
                      newTracksMap.delete(id);
                    }
                  }
                }
                
                return newTracksMap;
              });
              
              // Convert tracks to detections format cho backward compatibility
              const detectionsForDisplay = result.tracks.map(t => ({
                id: t.track_id,
                class: t.class,
                class_id: t.class_id,
                confidence: t.confidence,
                bbox: t.bbox,
                width: t.bbox[2] - t.bbox[0],
                height: t.bbox[3] - t.bbox[1],
                track_id: t.track_id,
                is_new: t.is_new
              }));
              setLastDetections(detectionsForDisplay);
              
              // AUDIO LOGIC MỚI DỰA TRÊN TRACKING:
              // 1. Lọc ra các track MỚI (chưa từng được đọc)
              const newTracks = result.tracks.filter(t => !announcedTrackIdsRef.current.has(t.track_id));
              
              if (newTracks.length > 0) {
                // Nếu audio đang bận đọc nhóm trước → KHÔNG đánh dấu đã đọc, đợi lần sau khi audio rảnh
                if (!audioService.isAnnouncingObjects && audioService.scheduledTimeouts.length === 0) {
                  // 2. Convert tracks mới sang format detections cho audio
                  const newDetections = newTracks.map(t => ({
                    class: t.class,
                    confidence: t.confidence,
                    bbox: t.bbox
                  }));

                  // 3. Phát audio CHỈ cho đối tượng mới
                  handleCameraAudioFeedback(newDetections);

                  // 4. Đánh dấu các track_id này là đã được đọc
                  newTracks.forEach(t => {
                    announcedTrackIdsRef.current.add(t.track_id);
                  });
                }
              }

              // 5. Cleanup: nếu track_id không còn trong frame hiện tại → xóa khỏi danh sách đã đọc
              const currentTrackIds = new Set(result.tracks.map(t => t.track_id));
              announcedTrackIdsRef.current.forEach(id => {
                if (!currentTrackIds.has(id)) {
                  announcedTrackIdsRef.current.delete(id);
                }
              });
            } else {
              // Không có tracks
              setLastDetections([]);
              // Clear old tracks sau 2 giây
              setActiveTracks(prev => {
                const newMap = new Map();
                const now = Date.now();
                for (const [id, track] of prev.entries()) {
                  if (now - track.last_seen < 2000) {
                    newMap.set(id, track);
                  }
                }
                return newMap;
              });
              
              // Clear audio cooldown (nếu có)
              if (audioCooldownTimerRef.current) {
                clearTimeout(audioCooldownTimerRef.current);
                audioCooldownTimerRef.current = null;
              }
            }
          } catch (err) {
            // Ignore canceled errors
            if (err.message === 'Request canceled' || abortController.signal.aborted) {
              return;
            }
            console.error('Detection error:', err);
          } finally {
            if (!abortController.signal.aborted) {
              isDetectingRef.current = false;
              setIsDetecting(false);
              abortControllerRef.current = null;
            }
          }
        }, 'image/jpeg', 0.6); // Giảm quality từ 0.8 xuống 0.6
      } catch (blobError) {
        console.error('Error converting canvas to blob:', blobError);
        isDetectingRef.current = false;
        setIsDetecting(false);
        abortControllerRef.current = null;
      }
    } catch (err) {
      console.error('Capture error:', err);
      isDetectingRef.current = false;
      setIsDetecting(false);
      abortControllerRef.current = null;
    }
  }, [isStreaming]);

  // Effect để quản lý camera lifecycle
  useEffect(() => {
    if (isActive) {
      startCamera();
    } else {
      stopCamera();
    }

    return () => {
      stopCamera();
    };
  }, [isActive, startCamera, stopCamera]);

  // Effect để bắt đầu detection loop với frame skipping
  useEffect(() => {
    if (isActive && isStreaming) {
      // Clear interval cũ nếu có
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      
      // Bắt đầu detection loop
      // Interval sẽ tự skip frame nếu isDetectingRef.current = true
      intervalRef.current = setInterval(() => {
        captureAndDetect();
      }, detectionInterval);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [isActive, isStreaming, detectionInterval, captureAndDetect]);

  // FPS counter - Track actual processed frames
  useEffect(() => {
    if (!isStreaming) {
      setFps(0);
      return;
    }

    let lastFrameCount = frameCount;
    const fpsInterval = setInterval(() => {
      const currentFrameCount = frameCount;
      const framesProcessed = currentFrameCount - lastFrameCount;
      setFps(framesProcessed);
      lastFrameCount = currentFrameCount;
    }, 1000);

    return () => clearInterval(fpsInterval);
  }, [isStreaming, frameCount]);

  // Cleanup khi unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  // Function để vẽ bounding boxes
  const drawBoundingBoxes = useCallback(() => {
    if (!overlayCanvasRef.current || !videoRef.current) {
      return;
    }

    const overlay = overlayCanvasRef.current;
    const video = videoRef.current;
    
    // Lấy kích thước thực tế của video element (sau khi render)
    const videoRect = video.getBoundingClientRect();
    const videoWidth = video.videoWidth || 640;
    const videoHeight = video.videoHeight || 480;
    
    if (videoWidth === 0 || videoHeight === 0) return;
    
    // Canvas đã được resize, lấy size thực tế từ canvas element
    // Nếu không có, dùng default 320x240
    const actualCanvasWidth = canvasRef.current?._actualWidth || 320;
    const actualCanvasHeight = canvasRef.current?._actualHeight || 240;
    
    // Scale factor từ canvas size thực tế về video display size
    const scaleX = videoRect.width / actualCanvasWidth;
    const scaleY = videoRect.height / actualCanvasHeight;
    
    // Set canvas size to match video element display size
    overlay.width = videoRect.width;
    overlay.height = videoRect.height;
    
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // Vẽ bounding boxes từ activeTracks (hoặc lastDetections nếu chưa có tracks)
    const tracksToDraw = Array.from(activeTracks.values()).length > 0 
      ? Array.from(activeTracks.values())
      : lastDetections;
    
    if (tracksToDraw.length > 0) {
      tracksToDraw.forEach(track => {
        // Đọc bbox từ API format: bbox là mảng [x1, y1, x2, y2]
        const bbox = track.bbox || [];
        if (bbox.length < 4) return;
        
        const [x1, y1, x2, y2] = bbox;
        const className = track.class || '';
        const confidence = track.confidence || 0;
        const trackId = track.track_id || track.id || '?';
        const isNew = track.is_new || false;
        
        // Scale coordinates từ video stream size sang display size
        let scaledX1 = x1 * scaleX;
        let scaledY1 = y1 * scaleY;
        let scaledX2 = x2 * scaleX;
        let scaledY2 = y2 * scaleY;
        
        // Vì video có transform: scaleX(-1) (mirror), cần đảo ngược tọa độ X
        // để bounding boxes hiển thị đúng vị trí
        const canvasWidth = overlay.width;
        const tempX1 = scaledX1;
        scaledX1 = canvasWidth - scaledX2;
        scaledX2 = canvasWidth - tempX1;
        
        const scaledWidth = scaledX2 - scaledX1;
        const scaledHeight = scaledY2 - scaledY1;
        
        // Màu sắc dựa trên is_new và confidence
        let color;
        if (isNew) {
          color = 'rgba(34, 197, 94, 0.9)'; // Bright green for new tracks
        } else if (confidence > 0.7) {
          color = 'rgba(59, 130, 246, 0.9)'; // Blue for high confidence
        } else if (confidence > 0.5) {
          color = 'rgba(251, 191, 36, 0.9)'; // Yellow for medium
        } else {
          color = 'rgba(239, 68, 68, 0.9)'; // Red for low
        }
        
        // Vẽ bounding box
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(scaledX1, scaledY1, scaledWidth, scaledHeight);
        
        // Vẽ background cho label với track ID
        let label = `ID:${trackId} ${capitalizeFirst(translateClass(className))} ${Math.round(confidence * 100)}%`;
        if (isNew) {
          label = `NEW ${label}`;
        }
        ctx.font = 'bold 14px Arial';
        const textMetrics = ctx.measureText(label);
        const textWidth = textMetrics.width;
        const textHeight = 18;
        
        // Đảm bảo label không bị vượt quá canvas
        const labelX = Math.max(0, Math.min(scaledX1, overlay.width - textWidth - 8));
        const labelY = Math.max(textHeight + 4, scaledY1);
        
        ctx.fillStyle = color;
        ctx.fillRect(labelX, labelY - textHeight - 4, textWidth + 8, textHeight);
        
        // Vẽ text
        ctx.fillStyle = 'white';
        ctx.fillText(label, labelX + 4, labelY - 6);
      });
    }
  }, [lastDetections, activeTracks]);

  // Effect để vẽ bounding boxes khi detections thay đổi
  useEffect(() => {
    drawBoundingBoxes();
  }, [drawBoundingBoxes]);

  // Effect để resize canvas khi video resize
  useEffect(() => {
    if (!videoRef.current || !isStreaming) return;

    const video = videoRef.current;
    const handleResize = () => {
      drawBoundingBoxes();
    };

    // Listen for video loadedmetadata
    video.addEventListener('loadedmetadata', handleResize);
    // Listen for window resize
    window.addEventListener('resize', handleResize);

    return () => {
      video.removeEventListener('loadedmetadata', handleResize);
      window.removeEventListener('resize', handleResize);
    };
  }, [isStreaming, drawBoundingBoxes]);

  if (!isActive) {
    return null;
  }

  return (
    <div className="fixed inset-0 bg-gradient-to-br from-gray-900 via-black to-gray-900 z-50 flex flex-col">
      {/* Header - Cải thiện với gradient */}
      <div className="bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 shadow-2xl border-b-4 border-blue-500">
        <div className="container mx-auto px-4 sm:px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-white bg-opacity-20 backdrop-blur-sm rounded-xl flex items-center justify-center shadow-lg border-2 border-white border-opacity-30">
                <span className="text-2xl">📹</span>
              </div>
              <div>
                <h2 className="text-xl sm:text-2xl font-extrabold text-white drop-shadow-lg">
                  {t('Camera Mode')}
                </h2>
                <p className="text-sm text-blue-100 font-medium mt-1">
                  {isStreaming ? (
                    <span className="flex items-center gap-2 flex-wrap">
                      <span className="w-2 h-2 bg-green-300 rounded-full animate-pulse"></span>
                      {t('Streaming')} • {t('Frame')}: <span className="font-bold">{frameCount}</span>
                      {fps > 0 && <span>• FPS: <span className="font-bold">{fps}</span></span>}
                      {detectionRate > 0 && <span>• Rate: <span className="font-bold">{detectionRate}/s</span></span>}
                    </span>
                  ) : (
                    <span className="flex items-center gap-2">
                      <span className="w-2 h-2 bg-yellow-300 rounded-full animate-pulse"></span>
                      {t('Initializing...')}
                    </span>
                  )}
                </p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="px-6 py-3 bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 text-white rounded-xl font-bold shadow-xl hover:shadow-2xl transition-all duration-300 flex items-center gap-2 transform hover:scale-105 active:scale-95 border-2 border-white border-opacity-30"
            >
              <span>✕</span>
              <span className="hidden sm:inline">{t('Close')}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Main Content - Two Column Layout giống Image mode */}
      <div className="flex-1 overflow-y-auto bg-gradient-to-br from-gray-50 via-blue-50 to-indigo-50">
        <div className="container mx-auto px-4 sm:px-6 py-4 sm:py-6 h-full">
          {error && (
            <div className="mb-4 bg-gradient-to-r from-red-600 to-red-700 text-white p-4 rounded-2xl shadow-2xl border-2 border-red-400">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-white bg-opacity-20 rounded-full flex items-center justify-center flex-shrink-0">
                  <span className="text-xl">⚠️</span>
                </div>
                <p className="font-bold text-base">{error}</p>
              </div>
            </div>
          )}

          {/* Grid Layout: 2 cột trên desktop, 1 cột trên mobile */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 sm:gap-6 h-full">
            {/* Left Column: Video Feed (2/3 width trên desktop) */}
            <div className="lg:col-span-2 flex flex-col">
              <div className="relative w-full h-full min-h-[400px] bg-black rounded-2xl overflow-hidden shadow-2xl border-4 border-white border-opacity-30 flex items-center justify-center">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-full object-contain"
                  style={{ transform: 'scaleX(-1)' }} // Mirror effect
                />
          
                {/* Overlay canvas for bounding boxes */}
                <canvas
                  ref={overlayCanvasRef}
                  className="absolute top-0 left-0 pointer-events-none"
                  style={{ 
                    width: '100%',
                    height: '100%'
                  }}
                />
                
                {/* Detection indicator - Nhỏ gọn, không che video */}
                {isDetecting && (
                  <div className="absolute top-4 left-4 bg-gradient-to-r from-blue-600 to-indigo-600 text-white px-4 py-2 rounded-full font-bold shadow-2xl border-2 border-white backdrop-blur-sm flex items-center gap-2 z-20">
                    <div className="relative w-4 h-4">
                      <div className="absolute inset-0 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    </div>
                    <span className="text-sm">{t('Detecting...')}</span>
                  </div>
                )}

                {/* Detection count badge - Cải thiện */}
                {lastDetections.length > 0 && (
                  <div className="absolute top-4 right-4 bg-gradient-to-r from-green-500 to-emerald-600 text-white px-4 py-2 rounded-full font-extrabold shadow-2xl border-2 border-white backdrop-blur-sm">
                    <div className="flex items-center gap-2">
                      <span className="text-lg">🎯</span>
                      <span>{lastDetections.length} {t('objects')}</span>
                    </div>
                  </div>
                )}
              </div>

              {/* Hidden canvas for frame capture */}
              <canvas ref={canvasRef} className="hidden" />
            </div>

            {/* Right Column: Results Table (1/3 width trên desktop) */}
            <div className="lg:col-span-1">
              <div className="bg-white rounded-2xl shadow-xl border-2 border-gray-200 p-5 sm:p-6 h-full overflow-y-auto">
                {activeTracks.size > 0 || lastDetections.length > 0 ? (
                  <ResultsTable detections={Array.from(activeTracks.values()).length > 0 
                    ? Array.from(activeTracks.values()).map(t => ({
                        id: t.track_id || t.id,
                        class: t.class,
                        class_id: t.class_id,
                        confidence: t.confidence,
                        bbox: t.bbox,
                        width: t.bbox ? t.bbox[2] - t.bbox[0] : 0,
                        height: t.bbox ? t.bbox[3] - t.bbox[1] : 0
                      }))
                    : lastDetections} />
                ) : (
                  <div className="text-center py-8">
                    <div className="w-20 h-20 bg-gradient-to-br from-gray-200 to-gray-300 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-md">
                      <span className="text-5xl">🔍</span>
                    </div>
                    <p className="text-base font-bold text-gray-700 mb-2">{t('No objects detected')}</p>
                    <p className="text-sm text-gray-500">{t('Waiting for detection...')}</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Controls - Trạng thái + điều khiển audio */}
      <div className="bg-gradient-to-r from-gray-800 via-gray-900 to-gray-800 border-t-4 border-blue-500 shadow-2xl">
        <div className="container mx-auto px-4 sm:px-6 py-5">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            {/* Status Indicator */}
            <div className="flex items-center gap-3 bg-white bg-opacity-10 backdrop-blur-sm rounded-xl px-5 py-3 border-2 border-white border-opacity-20">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center shadow-lg">
                <span className="text-xl">📊</span>
              </div>
              <div>
                <div className="text-xs text-gray-300 mb-1 font-medium">{t('Status')}</div>
                {isStreaming ? (
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 bg-green-400 rounded-full animate-pulse shadow-lg"></span>
                    <span className="text-white font-bold">{t('Active')}</span>
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 bg-gray-400 rounded-full"></span>
                    <span className="text-gray-400 font-bold">{t('Inactive')}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Audio Controls */}
            <div className="flex items-center gap-3 bg-white bg-opacity-10 backdrop-blur-sm rounded-xl px-5 py-3 border-2 border-white border-opacity-20">
              <button
                type="button"
                onClick={handleToggleAudio}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-bold shadow-md transition-all ${
                  isAudioEnabled
                    ? 'bg-green-500 hover:bg-green-600 text-white'
                    : 'bg-gray-600 hover:bg-gray-700 text-gray-200'
                }`}
              >
                <span className="text-lg">{isAudioEnabled ? '🔊' : '🔇'}</span>
                <span>{isAudioEnabled ? t('Audio On') : t('Audio Off')}</span>
              </button>

              <button
                type="button"
                onClick={handleRepeatAudio}
                disabled={!lastAnnouncedDetectionsRef.current || !isAudioEnabled}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-bold shadow-md transition-all ${
                  !lastAnnouncedDetectionsRef.current || !isAudioEnabled
                    ? 'bg-gray-500 text-gray-200 cursor-not-allowed'
                    : 'bg-blue-500 hover:bg-blue-600 text-white'
                }`}
              >
                <span className="text-lg">🔁</span>
                <span>{t('Repeat')}</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CameraView;

