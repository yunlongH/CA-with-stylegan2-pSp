#!/bin/bash
# This script sets the TORCH_CUDA_ARCH_LIST environment variable based on the detected GPU

# Detect the GPU name
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n 1)

echo "Detected GPU: $GPU_NAME"

# Map GPU names to CUDA architectures
case "$GPU_NAME" in
  "NVIDIA A100"* )
    export TORCH_CUDA_ARCH_LIST="8.0"
    ;;
  "NVIDIA H100"* )
    export TORCH_CUDA_ARCH_LIST="9.0"
    ;;
  "NVIDIA V100"* )
    export TORCH_CUDA_ARCH_LIST="7.0"
    ;;
  "NVIDIA T4"* )
    export TORCH_CUDA_ARCH_LIST="7.5"
    ;;
  "NVIDIA RTX 3090"* )
    export TORCH_CUDA_ARCH_LIST="8.6"
    ;;
  "NVIDIA RTX 4090"* )
    export TORCH_CUDA_ARCH_LIST="8.9"
    ;;
  *)
    echo "Unknown GPU: $GPU_NAME. TORCH_CUDA_ARCH_LIST not set."
    ;;
esac

# Print the set value or a message if not set
echo "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST:-Not set}"



python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
