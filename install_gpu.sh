#!/bin/bash

# Script cài đặt với GPU support (download lớn ~3GB, nhưng nhanh hơn khi chạy)
# Dùng cho hệ thống CÓ GPU NVIDIA

echo "🚀 GPU Installation - Full CUDA Support"
echo "========================================"
echo ""

# Kiểm tra GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  WARNING: nvidia-smi not found!"
    echo "   This system may not have NVIDIA GPU."
    echo "   Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "❌ Installation cancelled."
        exit 1
    fi
else
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1
    echo ""
fi

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
pip install --upgrade pip

# Uninstall CPU-only torch nếu có
echo "🧹 Cleaning up existing torch installation (if any)..."
pip uninstall -y torch torchvision 2>/dev/null || true

# Cài full CUDA version (LỚN - ~3GB, nhưng nhanh khi chạy)
echo "📥 Installing PyTorch with CUDA support (~3GB download)..."
echo "   This may take 15-30 minutes depending on internet speed..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Cài các dependencies khác
echo "📥 Installing other dependencies..."
pip install -r requirements.txt

# Kiểm tra cài đặt
echo ""
echo "✅ Installation complete!"
echo ""
echo "🔍 Verifying installation..."

# Test imports
python3 -c "import torch; print('✅ torch:', torch.__version__); print('   CUDA available:', torch.cuda.is_available())" 2>/dev/null || echo "❌ torch: FAILED"
python3 -c "from ultralytics import YOLO; print('✅ ultralytics: OK')" 2>/dev/null || echo "❌ ultralytics: FAILED"
python3 -c "from deepsort import DeepSortTracker; print('✅ DeepSORT: OK')" 2>/dev/null || echo "❌ DeepSORT: FAILED"
python3 -c "from tracker import VideoTracker; print('✅ VideoTracker: OK')" 2>/dev/null || echo "❌ VideoTracker: FAILED"

echo ""
echo "🎉 GPU installation finished!"
echo ""

