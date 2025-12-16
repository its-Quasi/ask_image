#!/bin/bash

# Script to download all required model checkpoints
# Run this script after installing dependencies

set -e  # Exit on error

echo "=========================================="
echo "Downloading Model Checkpoints"
echo "=========================================="
echo ""

# Create models directory
mkdir -p models
cd models

echo "This script will download approximately 1.6 GB of model files."
echo "Make sure you have enough disk space and a stable internet connection."
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Download cancelled."
    exit 1
fi

echo ""
echo "=========================================="
echo "1/3 Downloading Grounding DINO checkpoint"
echo "=========================================="

if [ -f "groundingdino_swint_ogc.pth" ]; then
    echo "✓ groundingdino_swint_ogc.pth already exists, skipping..."
else
    wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
    echo "✓ Grounding DINO checkpoint downloaded"
fi

echo ""
echo "=========================================="
echo "2/3 Downloading Grounding DINO config"
echo "=========================================="

if [ -f "GroundingDINO_SwinT_OGC.py" ]; then
    echo "✓ GroundingDINO_SwinT_OGC.py already exists, skipping..."
else
    wget https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py
    echo "✓ Grounding DINO config downloaded"
fi

echo ""
echo "=========================================="
echo "3/3 Downloading SAM2 checkpoint"
echo "=========================================="

echo "Choose SAM2 model variant:"
echo "  1) Large (recommended, ~900 MB, best accuracy)"
echo "  2) Base+ (smaller, ~270 MB, good accuracy)"
echo "  3) Small (smallest, ~180 MB, decent accuracy)"
echo ""
read -p "Enter choice [1-3]: " sam_choice

case $sam_choice in
    1)
        SAM_FILE="sam2_hiera_large.pt"
        SAM_URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
        echo "Downloading SAM2 Large..."
        ;;
    2)
        SAM_FILE="sam2_hiera_base_plus.pt"
        SAM_URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"
        echo "Downloading SAM2 Base+..."
        echo "NOTE: You need to update app/core/config.py:"
        echo "  SAM2_CHECKPOINT = MODELS_DIR / 'sam2_hiera_base_plus.pt'"
        echo "  SAM2_CONFIG = 'sam2_hiera_b+.yaml'"
        ;;
    3)
        SAM_FILE="sam2_hiera_small.pt"
        SAM_URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"
        echo "Downloading SAM2 Small..."
        echo "NOTE: You need to update app/core/config.py:"
        echo "  SAM2_CHECKPOINT = MODELS_DIR / 'sam2_hiera_small.pt'"
        echo "  SAM2_CONFIG = 'sam2_hiera_s.yaml'"
        ;;
    *)
        echo "Invalid choice. Downloading Large by default..."
        SAM_FILE="sam2_hiera_large.pt"
        SAM_URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
        ;;
esac

if [ -f "$SAM_FILE" ]; then
    echo "✓ $SAM_FILE already exists, skipping..."
else
    wget $SAM_URL
    echo "✓ SAM2 checkpoint downloaded"
fi

cd ..

echo ""
echo "=========================================="
echo "Download completed successfully!"
echo "=========================================="
echo ""
echo "Downloaded files:"
ls -lh models/
echo ""
echo "Next steps:"
echo "1. Verify models with: python verify_models.py"
echo "2. Run the application: python app.py"
echo ""
