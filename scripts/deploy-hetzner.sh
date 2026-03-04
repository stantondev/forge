#!/bin/bash
#
# Forge — Hetzner GPU Server Deployment Script
#
# Prerequisites:
#   1. A Hetzner GPU dedicated server (GEX44 or higher)
#   2. SSH access to the server
#   3. Ubuntu 22.04+ installed on the server
#
# Usage:
#   ssh root@YOUR_SERVER_IP 'bash -s' < scripts/deploy-hetzner.sh
#
# This script sets up everything from scratch on a fresh Hetzner server.
#
set -e

echo "=== Forge — Hetzner GPU Server Setup ==="
echo ""

# --- System updates ---
echo "[1/7] Updating system..."
apt-get update -qq && apt-get upgrade -y -qq

# --- Install Docker ---
echo "[2/7] Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
fi

# --- Install Docker Compose ---
echo "[3/7] Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    apt-get install -y -qq docker-compose-plugin
fi

# --- Install NVIDIA Container Toolkit (for GPU passthrough to Docker) ---
echo "[4/7] Installing NVIDIA Container Toolkit..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: NVIDIA drivers not found. Install GPU drivers first."
    echo "For Hetzner: apt install nvidia-driver-535"
else
    distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L "https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list" | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -qq
    apt-get install -y -qq nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
fi

# --- Clone Forge ---
echo "[5/7] Setting up Forge..."
mkdir -p /opt/forge
cd /opt/forge

# If repo exists, pull latest; otherwise clone
if [ -d ".git" ]; then
    git pull
else
    echo "Place your Forge repo here or git clone it."
    echo "Example: git clone https://github.com/YOUR_USERNAME/forge.git /opt/forge"
fi

# --- Enable GPU in docker-compose ---
echo "[6/7] Enabling GPU support..."
# Uncomment GPU lines in docker-compose.yml
if [ -f "docker-compose.yml" ]; then
    sed -i 's/# deploy:/deploy:/g' docker-compose.yml
    sed -i 's/#   resources:/  resources:/g' docker-compose.yml
    sed -i 's/#     reservations:/    reservations:/g' docker-compose.yml
    sed -i 's/#       devices:/      devices:/g' docker-compose.yml
    sed -i 's/#         - driver: nvidia/        - driver: nvidia/g' docker-compose.yml
    sed -i 's/#           count: all/          count: all/g' docker-compose.yml
    sed -i 's/#           capabilities: \[gpu\]/          capabilities: \[gpu\]/g' docker-compose.yml
fi

# --- Start services ---
echo "[7/7] Starting Forge..."
docker compose up -d

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
sleep 10

# Pull the models
echo "Pulling DeepSeek-R1 8B (this may take a few minutes)..."
docker exec forge-ollama ollama pull deepseek-r1:8b
docker exec forge-ollama ollama pull nomic-embed-text

echo ""
echo "=== Forge is running! ==="
echo ""
echo "  Chat UI:  http://YOUR_SERVER_IP:3000"
echo "  API:      http://YOUR_SERVER_IP:8000/v1/chat/completions"
echo "  Ollama:   http://YOUR_SERVER_IP:11434"
echo ""
echo "IMPORTANT: Set up a firewall to restrict access!"
echo "  ufw allow ssh"
echo "  ufw allow from YOUR_IP to any port 3000"
echo "  ufw allow from YOUR_IP to any port 8000"
echo "  ufw enable"
echo ""
echo "For HTTPS, set up a reverse proxy with Caddy or nginx."
