#!/bin/bash

# Script để chạy backend
echo "🚀 Starting Animal Detection Backend..."
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

# Cài đặt dependencies nếu chưa có
if [ ! -f "venv/bin/uvicorn" ]; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
fi

# Chạy backend
echo "✅ Starting FastAPI server..."
echo "📍 Backend will run at: http://localhost:8000"
echo "📚 API docs at: http://localhost:8000/docs"
echo ""
python app.py

