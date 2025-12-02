// Class name mapping từ tiếng Anh sang tiếng Việt
// Dựa trên COCO dataset 80 classes
export const classTranslations = {
  // People & Objects
  "person": "người",
  "bicycle": "xe đạp",
  "car": "xe ô tô",
  "motorcycle": "xe máy",
  "airplane": "máy bay",
  "bus": "xe buýt",
  "train": "tàu hỏa",
  "truck": "xe tải",
  "boat": "thuyền",
  
  // Traffic objects
  "traffic light": "đèn giao thông",
  "fire hydrant": "vòi cứu hỏa",
  "stop sign": "biển báo dừng",
  "parking meter": "đồng hồ đỗ xe",
  "bench": "ghế dài",
  
  // Objects
  "bird": "chim",
  "cat": "mèo",
  "dog": "chó",
  "horse": "ngựa",
  "sheep": "cừu",
  "cow": "bò",
  "elephant": "voi",
  "bear": "gấu",
  "zebra": "ngựa vằn",
  "giraffe": "hươu cao cổ",
  
  // Other objects
  "backpack": "ba lô",
  "umbrella": "ô",
  "handbag": "túi xách",
  "tie": "cà vạt",
  "suitcase": "vali",
  "frisbee": "đĩa bay",
  "skis": "ván trượt tuyết",
  "snowboard": "ván trượt tuyết",
  "sports ball": "bóng thể thao",
  "kite": "diều",
  "baseball bat": "gậy bóng chày",
  "baseball glove": "găng tay bóng chày",
  "skateboard": "ván trượt",
  "surfboard": "ván lướt sóng",
  "tennis racket": "vợt tennis",
  "bottle": "chai",
  "wine glass": "ly rượu",
  "cup": "cốc",
  "fork": "nĩa",
  "knife": "dao",
  "spoon": "thìa",
  "bowl": "bát",
  "banana": "chuối",
  "apple": "táo",
  "sandwich": "bánh mì kẹp",
  "orange": "cam",
  "broccoli": "bông cải xanh",
  "carrot": "cà rốt",
  "hot dog": "xúc xích",
  "pizza": "pizza",
  "donut": "bánh rán",
  "cake": "bánh ngọt",
  "chair": "ghế",
  "couch": "ghế sofa",
  "potted plant": "cây cảnh",
  "bed": "giường",
  "dining table": "bàn ăn",
  "toilet": "bồn cầu",
  "tv": "tivi",
  "laptop": "máy tính xách tay",
  "mouse": "chuột máy tính",
  "remote": "điều khiển từ xa",
  "keyboard": "bàn phím",
  "cell phone": "điện thoại",
  "microwave": "lò vi sóng",
  "oven": "lò nướng",
  "toaster": "máy nướng bánh",
  "sink": "bồn rửa",
  "refrigerator": "tủ lạnh",
  "book": "sách",
  "clock": "đồng hồ",
  "vase": "lọ hoa",
  "scissors": "kéo",
  "teddy bear": "gấu bông",
  "hair drier": "máy sấy tóc",
  "toothbrush": "bàn chải đánh răng",
};

// Priority system cho audio (số càng cao = ưu tiên càng cao)
export const classPriorities = {
  // High priority - Safety critical
  "person": 10,
  "car": 9,
  "motorcycle": 9,
  "bus": 9,
  "truck": 9,
  "bicycle": 8,
  "traffic light": 8,
  "stop sign": 8,
  
  // Medium priority
  "train": 7,
  "boat": 7,
  "airplane": 6,
  "bird": 5,
  "cat": 5,
  "dog": 5,
  
  // Low priority - Default
  default: 1,
};

// Helper function để translate class name
export const translateClass = (className) => {
  return classTranslations[className.toLowerCase()] || className;
};

// Hàm viết hoa chữ đầu tiên
export const capitalizeFirst = (str) => {
  if (!str) return str;
  return str.charAt(0).toUpperCase() + str.slice(1);
};

// Helper function để get priority
export const getClassPriority = (className) => {
  return classPriorities[className.toLowerCase()] || classPriorities.default;
};

// Helper function để format detection message
export const formatDetectionMessage = (detection, position = null) => {
  const className = translateClass(detection.class);
  
  let message = `Phát hiện ${className}`;
  
  if (position) {
    message += ` ở ${position}`;
  }

  return message;
};

// Helper function để format multiple detections
export const formatMultipleDetections = (detections) => {
  if (detections.length === 0) return null;
  
  // Group by class
  const grouped = {};
  detections.forEach(det => {
    const className = translateClass(det.class);
    if (!grouped[className]) {
      grouped[className] = 0;
    }
    grouped[className]++;
  });
  
  // Sort by priority
  const sorted = Object.entries(grouped)
    .map(([className, count]) => ({
      className,
      count,
      priority: getClassPriority(detections.find(d => translateClass(d.class) === className)?.class || '')
    }))
    .sort((a, b) => b.priority - a.priority);
  
  // Format message
  const messages = sorted.map(({ className, count }) => {
    if (count === 1) {
      return `Phát hiện ${className}`;
    } else {
      return `Phát hiện ${count} ${className}`;
    }
  });
  
  return messages.join('. ') + '.';
};

