#!/bin/bash


# Detect GPU type and set the appropriate CUDA architecture
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader)

case $gpu_name in
    *P100*)
        export TORCH_CUDA_ARCH_LIST="6.0"
        ;;
    *V100*)
        export TORCH_CUDA_ARCH_LIST="7.0"
        ;;
    *A100*)
        export TORCH_CUDA_ARCH_LIST="8.0"
        ;;
    *A40*)
        export TORCH_CUDA_ARCH_LIST="8.6"
        ;;
    *L40*)
        export TORCH_CUDA_ARCH_LIST="8.9"
        ;;
    *)
        echo "Unknown GPU: $gpu_name"
        exit 1
        ;;
esac

echo "Set TORCH_CUDA_ARCH_LIST to $TORCH_CUDA_ARCH_LIST"

# Set CUDA 12.5 environment variables
export CUDA_HOME=/usr/local/cuda-12.5
export PATH=/usr/local/cuda-12.5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64:$LD_LIBRARY_PATH

echo "CUDA environment variables set for CUDA 12.5"

# Remove Torch extensions cache
rm -rf ~/.cache/torch_extensions
echo "Torch extensions cache cleared"