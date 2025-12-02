#!/bin/bash

# Script để chạy backend
echo "🚀 Starting Object Detection Backend..."
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

# Kiểm tra dependencies
if [ ! -f "venv/bin/uvicorn" ] || ! python3 -c "from deepsort import DeepSortTracker" 2>/dev/null; then
    echo "⚠️  Dependencies not fully installed!"
    echo ""
    echo "📥 Please install dependencies first:"
    echo "   Option 1 (Quick - CPU only): ./quick_install.sh"
    echo "   Option 2 (Full - GPU support): ./install_gpu.sh"
    echo ""
    echo "   Or manually: pip install -r requirements.txt"
    echo ""
    exit 1
fi

# Chạy backend
echo "✅ Starting FastAPI server..."
echo "📍 Backend will run at: http://localhost:8000"
echo "📚 API docs at: http://localhost:8000/docs"
echo ""
python app.py

