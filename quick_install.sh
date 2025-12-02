#!/bin/bash

# Script cài đặt nhanh với CPU-only version (tiết kiệm thời gian)
# Dùng cho hệ thống KHÔNG có GPU hoặc muốn cài đặt nhanh

echo "⚡ Quick Install - CPU-Only Version"
echo "===================================="
echo ""

cd backend

# Kiểm tra virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet

# Uninstall torch nếu đã cài (để tránh conflict)
echo "🧹 Cleaning up existing torch installation (if any)..."
pip uninstall -y torch torchvision 2>/dev/null || true

# Cài CPU-only version của torch (NHANH HƠN - chỉ ~200MB)
echo "📥 Installing PyTorch CPU-only version (~200MB instead of 3GB)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet

# Cài các dependencies khác (không có torch trong requirements.txt nữa)
echo "📥 Installing other dependencies..."
pip install -r requirements.txt --quiet

# Fix numpy version compatibility (opencv-python cần numpy < 2.0)
echo "🔧 Fixing numpy version compatibility..."
pip install "numpy>=1.24.4,<2.0.0" --force-reinstall --quiet

# Kiểm tra cài đặt
echo ""
echo "✅ Installation complete!"
echo ""
echo "🔍 Verifying installation..."

# Test imports
python3 -c "import numpy; print('✅ numpy:', numpy.__version__)" 2>/dev/null || echo "❌ numpy: FAILED"
python3 -c "import pydantic; print('✅ pydantic:', pydantic.__version__)" 2>/dev/null || echo "❌ pydantic: FAILED"
python3 -c "import scipy; print('✅ scipy:', scipy.__version__)" 2>/dev/null || echo "❌ scipy: FAILED"
python3 -c "import filterpy; print('✅ filterpy: OK')" 2>/dev/null || echo "❌ filterpy: FAILED"
python3 -c "import torch; print('✅ torch:', torch.__version__, '(CPU-only)')" 2>/dev/null || echo "❌ torch: FAILED"
python3 -c "from ultralytics import YOLO; print('✅ ultralytics: OK')" 2>/dev/null || echo "❌ ultralytics: FAILED"
python3 -c "from deepsort import DeepSortTracker; print('✅ DeepSORT: OK')" 2>/dev/null || echo "❌ DeepSORT: FAILED"
python3 -c "from tracker import VideoTracker; print('✅ VideoTracker: OK')" 2>/dev/null || echo "❌ VideoTracker: FAILED"

echo ""
echo "🎉 Quick installation finished!"
echo ""
echo "📝 Note: Using CPU-only version. System will work but slower than GPU version."
echo "💡 To use GPU version later, run: ./install_gpu.sh"
echo ""

