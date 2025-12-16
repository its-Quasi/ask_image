#!/bin/bash

# Installation script for Grounding DINO + SAM2 Pipeline
# This script installs all required dependencies

set -e  # Exit on error

echo "=========================================="
echo "Installing Grounding DINO + SAM2 Pipeline"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment (recommended)
read -p "Do you want to create a virtual environment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Virtual environment activated"
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CPU version - modify if you have GPU)
echo "Installing PyTorch (CPU version)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For GPU support, uncomment the line below and comment the one above:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install base requirements
echo "Installing base requirements..."
pip install -r requirements.txt

# Install Grounding DINO
# echo "Installing Grounding DINO..."
# pip install git+https://github.com/IDEA-Research/GroundingDINO.git

# # Install SAM2
# echo "Installing SAM2..."
# pip install git+https://github.com/facebookresearch/segment-anything-2.git

echo "=========================================="
echo "Installation completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Download model checkpoints (see README.md)"
echo "2. Place models in the 'models/' directory"
echo "3. Run: python app.py"
echo ""
echo "Note: You need to download the following models:"
echo "  - Grounding DINO: groundingdino_swint_ogc.pth"
echo "  - Grounding DINO Config: GroundingDINO_SwinT_OGC.py"
echo "  - SAM2: sam2_hiera_large.pt"
echo ""
