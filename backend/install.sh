#!/bin/bash
pip install --upgrade pip
echo "Installing base requirements..."
pip install -r requirements-base.txt

echo "Choose PyTorch installation:"
echo "1) CPU only "
echo "2) GPU with CUDA 11.8"
echo "3) GPU with CUDA 12.1"
echo "4) Manual installation (enter specific version)"

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "Installing PyTorch CPU version..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        ;;
    2)
        echo "Installing PyTorch with CUDA 11.8..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ;;
    3)
        echo "Installing PyTorch with CUDA 12.1..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ;;
    4)
        read -p "Enter the PyTorch version you want to install (e.g., 2.0.1+cu118): " version
        if [[ $version =~ ^[0-9]+\.[0-9]+\.[0-9]+(\+cu[0-9]+)?$ ]]; then
            echo "Installing PyTorch version $version..."
            pip install torch==$version torchvision==$version torchaudio==$version --index-url https://download.pytorch.org/whl
        else
            echo "Invalid version format. Please use a valid version (e.g., 2.0.1+cu118)."
            exit 1
        fi
        ;;
    *)
        echo "Invalid choice. Installing CPU version by default..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        ;;
esac

echo "Please now deactivate and reactivate your virtual environment before continuing."
echo "Then run: pip install --no-build-isolation basicsr==1.3.4.9"

exit 0
