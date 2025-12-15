#!/bin/bash

# Quick start script for WhisperLiveKit + Open WebUI integration
# This script helps you quickly set up and test the integration

set -e

echo "=========================================="
echo "WhisperLiveKit + Open WebUI Quick Start"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is available
if ! docker compose version &> /dev/null && ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "   Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"
echo ""

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    USE_GPU=true
else
    echo "⚠️  No NVIDIA GPU detected. Will use CPU-only mode."
    echo "   Note: CPU mode is slower. Consider using a GPU for better performance."
    USE_GPU=false
fi

echo ""
echo "Starting services..."
echo ""

# Use the appropriate compose file
if [ "$USE_GPU" = true ]; then
    docker compose -f docker-compose.openwebui.yml up -d
else
    echo "⚠️  GPU support requires modifying docker-compose.openwebui.yml to use Dockerfile.cpu"
    echo "   Starting with GPU configuration anyway. This may fail if GPU is not available."
    docker compose -f docker-compose.openwebui.yml up -d
fi

echo ""
echo "Waiting for services to start..."
sleep 10

# Check if services are running
WHISPER_RUNNING=$(docker ps --filter "name=whisperlivekit" --filter "status=running" -q)
OPENWEBUI_RUNNING=$(docker ps --filter "name=open-webui" --filter "status=running" -q)

if [ -z "$WHISPER_RUNNING" ]; then
    echo "❌ WhisperLiveKit failed to start. Check logs with:"
    echo "   docker logs whisperlivekit"
    exit 1
fi

if [ -z "$OPENWEBUI_RUNNING" ]; then
    echo "❌ Open WebUI failed to start. Check logs with:"
    echo "   docker logs open-webui"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Services are running!"
echo "=========================================="
echo ""
echo "🎤 WhisperLiveKit: http://localhost:8000"
echo "💬 Open WebUI:     http://localhost:3000"
echo ""
echo "Next steps:"
echo "1. Open http://localhost:3000 in your browser"
echo "2. Create an account or sign in"
echo "3. Start a new chat"
echo "4. Click the microphone icon 🎤"
echo "5. Speak your message - it will be transcribed in real-time!"
echo ""
echo "Useful commands:"
echo "  View logs:       docker logs -f whisperlivekit"
echo "  Stop services:   docker compose -f docker-compose.openwebui.yml down"
echo "  Restart:         docker compose -f docker-compose.openwebui.yml restart"
echo ""
echo "For more information, see OPEN_WEBUI_INTEGRATION.md"
echo ""
