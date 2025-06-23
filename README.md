# CUDA IPC Tensor Sharing Example

This repository demonstrates efficient inter-process sharing of GPU tensors using:
- **CUDA IPC** for zero-copy GPU memory sharing between processes
- **POSIX pipes** for inter-process communication and synchronization
- **LibTorch** (PyTorch C++ API) for tensor operations

## Key Features
- Producer/consumer model for tensor sharing
- IPC handle serialization and transfer
- Multi-process coordination using `posix_spawn`

## Use Cases
- Distributed deep learning inference
- Multi-process GPU pipelines
- Zero-copy tensor sharing between applications

## Build Requirements
- CUDA Toolkit
- LibTorch (C++ distribution)
- C++17 compatible compiler
- POSIX-compliant OS (Linux WSL2 tested)
