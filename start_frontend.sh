#!/bin/bash

# Script để chạy frontend
echo "🚀 Starting Object Detection Frontend..."
echo ""

cd frontend

# Kiểm tra node_modules
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Chạy frontend
echo "✅ Starting React development server..."
echo "📍 Frontend will run at: http://localhost:3000"
echo ""
npm start

