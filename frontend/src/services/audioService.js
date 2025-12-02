// Audio Service - Text-to-Speech cho người khiếm thị
import { formatDetectionMessage, formatMultipleDetections, getClassPriority } from '../utils/classTranslations';

class AudioService {
  constructor() {
    this.synthesis = window.speechSynthesis;
    this.isEnabled = true;
    this.volume = 1.0;
    this.rate = 1.0;
    this.pitch = 1.0;
    this.language = 'vi-VN';
    this.isSpeaking = false;
    this.queue = [];
    this.lastSpeakTime = 0;
    this.debounceTime = 2000; // 2 giây giữa các lần phát âm
    this.maxQueueSize = 5; // Tối đa 5 messages trong queue
    this.hasWelcomed = false; // Flag để chỉ phát welcome message 1 lần
    this.lastDetectionsHash = null; // Hash để so sánh detections
    this.scheduledTimeouts = []; // Lưu các timeout để có thể clear
    this.debounceTimer = null; // Timer cho debounce trong camera mode
    this.isAnnouncingObjects = false; // Flag để track đang phát từng đối tượng
  }

  // Kiểm tra browser support
  isSupported() {
    return 'speechSynthesis' in window;
  }

  // Bật/tắt audio
  setEnabled(enabled) {
    this.isEnabled = enabled;
    if (!enabled) {
      this.stop();
      this.queue = [];
    }
  }

  // Cài đặt volume (0.0 - 1.0)
  setVolume(volume) {
    this.volume = Math.max(0, Math.min(1, volume));
  }

  // Cài đặt tốc độ (0.1 - 10.0)
  setRate(rate) {
    this.rate = Math.max(0.1, Math.min(10, rate));
  }

  // Cài đặt pitch (0 - 2)
  setPitch(pitch) {
    this.pitch = Math.max(0, Math.min(2, pitch));
  }

  // Dừng phát âm hiện tại
  stop() {
    if (this.synthesis.speaking) {
      this.synthesis.cancel();
    }
    this.isSpeaking = false;
    this.queue = [];
    // Clear tất cả scheduled timeouts
    this.scheduledTimeouts.forEach(timeout => clearTimeout(timeout));
    this.scheduledTimeouts = [];
    // Clear debounce timer
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = null;
    }
    // Reset flag đang phát từng đối tượng
    this.isAnnouncingObjects = false;
  }

  // Reset welcome flag (khi cần)
  resetWelcome() {
    this.hasWelcomed = false;
  }

  // Phát âm text
  speak(text, priority = 1) {
    if (!this.isEnabled || !this.isSupported()) {
      return;
    }

    // Kiểm tra debounce
    const now = Date.now();
    if (now - this.lastSpeakTime < this.debounceTime && priority < 5) {
      // Thêm vào queue nếu priority thấp
      if (this.queue.length < this.maxQueueSize) {
        this.queue.push({ text, priority, timestamp: now });
      }
      return;
    }

    // Nếu đang phát âm, thêm vào queue
    if (this.isSpeaking) {
      if (this.queue.length < this.maxQueueSize) {
        this.queue.push({ text, priority, timestamp: now });
      }
      return;
    }

    // Phát âm ngay
    this._speakNow(text);
  }

  // Phát âm ngay lập tức (internal)
  _speakNow(text) {
    // Kiểm tra lại support trước khi phát
    if (!this.isSupported() || !this.isEnabled) {
      return;
    }

    // Cancel bất kỳ speech nào đang chạy để tránh conflict
    if (this.synthesis.speaking) {
      this.synthesis.cancel();
    }

    this.isSpeaking = true;
    this.lastSpeakTime = Date.now();

    try {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = this.language;
      utterance.volume = this.volume;
      utterance.rate = this.rate;
      utterance.pitch = this.pitch;

      utterance.onend = () => {
        this.isSpeaking = false;
        // Xử lý queue
        this._processQueue();
      };

      utterance.onerror = (error) => {
        // Chỉ log error nếu không phải là user cancel hoặc interruption
        if (error.error !== 'interrupted' && error.error !== 'canceled') {
          console.warn('Speech synthesis error:', error.error || 'Unknown error');
        }
        this.isSpeaking = false;
        // Chỉ process queue nếu không phải lỗi nghiêm trọng
        if (error.error !== 'synthesis-failed' && error.error !== 'synthesis-unavailable') {
          this._processQueue();
        }
      };

      this.synthesis.speak(utterance);
    } catch (err) {
      console.warn('Failed to create speech utterance:', err);
      this.isSpeaking = false;
    }
  }

  // Xử lý queue
  _processQueue() {
    if (this.queue.length === 0) {
      return;
    }

    // Sắp xếp queue theo priority
    this.queue.sort((a, b) => b.priority - a.priority);

    // Lấy message có priority cao nhất
    const next = this.queue.shift();

    // Kiểm tra debounce lại
    const now = Date.now();
    if (now - next.timestamp < this.debounceTime && next.priority < 5) {
      // Nếu vẫn còn trong debounce time, thêm lại vào queue
      this.queue.push(next);
      return;
    }

    // Phát âm
    this._speakNow(next.text);
  }

  // Phát âm detection đơn lẻ
  speakDetection(detection, position = null) {
    if (!detection) return;

    const message = formatDetectionMessage(detection, position);
    const priority = getClassPriority(detection.class);
    this.speak(message, priority);
  }

  // Phát âm nhiều detections:
  // - Nếu chỉ có 1 đối tượng: đọc chi tiết 1 lần
  // - Nếu có nhiều đối tượng: GOM THEO LỚP (class) và đọc tổng hợp:
  //   "Phát hiện 2 xe tải. Phát hiện 1 người." → không lặp lại từng xe tải
  // Bảo vệ: Không phát lại nếu đang phát từng đối tượng
  // onComplete: Callback được gọi khi phát xong câu tổng hợp
  speakDetections(detections, delayBetweenObjects = 2000, onComplete = null) {
    if (!detections || detections.length === 0) {
      this.speak('Không phát hiện đối tượng nào', 1);
      if (onComplete) {
        // Gọi callback ngay nếu không có detections
        setTimeout(() => onComplete(), 1000);
      }
      return;
    }

    // Bảo vệ: Nếu đang phát từng đối tượng → Không phát lại
    // Đảm bảo phát hết tất cả đối tượng trước khi phát thông tin mới
    if (this.isAnnouncingObjects || this.scheduledTimeouts.length > 0) {
      console.log('[AudioService] Đang phát từng đối tượng, bỏ qua speakDetections() mới');
      return;
    }

    // Clear các timeout cũ (nếu có, nhưng thường không có vì đã check ở trên)
    this.scheduledTimeouts.forEach(timeout => clearTimeout(timeout));
    this.scheduledTimeouts = [];

    // Sắp xếp theo priority để ưu tiên đọc các lớp quan trọng trước
    const sorted = [...detections].sort((a, b) => {
      const priorityA = getClassPriority(a.class);
      const priorityB = getClassPriority(b.class);
      return priorityB - priorityA;
    });

    const count = sorted.length;

    if (count === 1) {
      // Nếu chỉ có 1 đối tượng, phát chi tiết một lần
      this.speakDetection(sorted[0]);
      if (onComplete) {
        setTimeout(() => {
          onComplete();
        }, 3000);
      }
      return;
    }

    // Có nhiều đối tượng: dùng formatMultipleDetections để GOM THEO LỚP
    const summaryMessage = formatMultipleDetections(sorted);
    if (summaryMessage) {
      // Đọc một câu tổng hợp, ví dụ: "Phát hiện 2 xe tải. Phát hiện 1 người."
      this.speak(summaryMessage, 5);
      if (onComplete) {
        // Ước lượng thời gian đọc xong (dựa trên độ dài câu)
        const estimatedDuration = Math.max(3000, summaryMessage.length * 80); // ~80ms/ký tự
        setTimeout(() => {
          onComplete();
        }, estimatedDuration);
      }
    }
  }

  // Phát âm detections với kiểm tra thay đổi (cho camera mode)
  // Giải pháp 1.1: Hash chỉ so sánh class names (không so sánh confidence)
  // Giải pháp 3.1: Thêm debounce 2-3 giây để chỉ phát khi detections ổn định
  // Giải pháp 4: Bảo vệ timeouts đang chạy - không interrupt khi đang phát
  speakDetectionsIfChanged(detections, delayBetweenObjects = 5000) {
    if (!detections || detections.length === 0) {
      // Nếu không có detections, reset hash và clear debounce timer
      this.lastDetectionsHash = null;
      if (this.debounceTimer) {
        clearTimeout(this.debounceTimer);
        this.debounceTimer = null;
      }
      return;
    }

    // Bảo vệ: Nếu đang phát từng đối tượng → Không làm gì
    // Đảm bảo phát hết tất cả đối tượng trước khi phát thông tin mới
    if (this.isAnnouncingObjects || this.scheduledTimeouts.length > 0) {
      console.log('[AudioService] Đang phát từng đối tượng, bỏ qua speakDetectionsIfChanged()');
      return;
    }

    // Giải pháp 1.1: Tạo hash chỉ từ class names (không so sánh confidence)
    // Lý do: Người khiếm thị chỉ cần biết có đối tượng gì, không cần biết confidence
    // Giảm false positive khi confidence dao động nhẹ
    const detectionsHash = JSON.stringify(
      detections
        .map(d => d.class) // Chỉ lấy class name
        .sort() // Sắp xếp để so sánh
    );

    // Chỉ phát nếu có thay đổi về class names
    if (detectionsHash !== this.lastDetectionsHash) {
      this.lastDetectionsHash = detectionsHash;
      
      // Giải pháp 3.1: Debounce 2 giây để chỉ phát khi detections ổn định
      // Chỉ reset debounce timer nếu không đang phát (đã check ở trên)
      if (this.debounceTimer) {
        clearTimeout(this.debounceTimer);
        this.debounceTimer = null;
      }
      
      // Debounce: Chỉ phát sau 2 giây không có thay đổi
      // Lý do: Real-time detection thường có detections ổn định sau vài giây
      // Giảm spam audio khi detections thay đổi liên tục
      this.debounceTimer = setTimeout(() => {
        // Kiểm tra lại trước khi phát (phòng trường hợp đang phát trong lúc đợi debounce)
        if (!this.isAnnouncingObjects && this.scheduledTimeouts.length === 0) {
          this.speakDetections(detections, delayBetweenObjects);
        }
        this.debounceTimer = null;
      }, 2000); // 2 giây debounce
    }
  }

  // Phát âm thông báo hệ thống
  speakSystemMessage(message, priority = 3) {
    this.speak(message, priority);
  }

  // Phát welcome message (chỉ 1 lần)
  speakWelcome() {
    if (!this.hasWelcomed) {
      this.speak('Hệ thống sẵn sàng', 5);
      this.hasWelcomed = true;
    }
  }
}

// Export singleton instance
export const audioService = new AudioService();
export default audioService;

