# CUDA IPC Tensor Sharing Example

This repository demonstrates inter-process sharing of GPU tensors using:
- **CUDA IPC** for zero-copy GPU memory sharing between processes
- **POSIX pipes** for inter-process communication and synchronization
- **LibTorch** (PyTorch C++ API) for tensor operations

## Key Features
- Producer/consumer model for tensor sharing
- IPC handle serialization and transfer
- Multi-process coordination using `posix_spawn`

## Building and Running

### Prerequisites
- CUDA Toolkit (>= 11.0)
- LibTorch C++ distribution (compatible with your CUDA version)
- CMake (>= 3.18)
- C++17 compatible compiler
- Linux OS (tested on Ubuntu WSL2 22.04)

## Building and Running the Sample

### Prerequisites
- CUDA Toolkit (≥ 11.0)
- LibTorch C++ distribution (compatible with your CUDA version)
- CMake (≥ 3.18)
- C++17 compatible compiler (GCC ≥ 9.0 or Clang ≥ 10.0 recommended)
- Linux OS (tested on Ubuntu 20.04/22.04)

### Build Instructions
1. **Set environment variable**:
   ```bash
   export Torch_DIR=/path/to/libtorch/share/cmake/Torch
   ```
2. **Configure with CMake**:
   ```bash
    mkdir build
    cd build
    cmake ..
   ```
3. **Build the project**:
   ```bash
    cmake --build .
   ```
4. **Run the executable**:
   ```
   ./cuda_ipc_get_mem_handle_producer_consumer_sample
   ```

### Sample output

Tensors are zeroed after `cudaIpcGetMemHandle` is called after `cudaMemcpy`:
```bash
[2104842] Parent process starting
[2104842] Pipes created
[2104842] Spawning producer
[2104842] Producer spawned with PID: 2104843
[2104842] Spawning consumer
[2104842] Consumer spawned with PID: 2104844
[2104842] Parent closed all pipe ends
[2104842] Parent waiting for children
[2104844] Child process started with role: consumer
[2104844] Consumer starting
[2104843] Child process started with role: producer
[2104843] Producer starting
[2104844] Consumer waiting for producer done signal
[2104843] Producer creating tensor #1
#1: Tensor to send before cudaIpcGetMemHandle:  1
 2
[ CUDAIntType{2} ]
#1: Tensor to send after cudaIpcGetMemHandle:  0
 0
[ CUDAIntType{2} ]
[2104843] Producer sent index for #1
[2104843] Producer sent IPC handle 0xE0FD0AE12F5600000B1E200000000000080000000000000000020000000000000001000000000000210000000000000015000000000000400000000000000000 for # 1
[2104843] Producer creating tensor #2
#2: Tensor to send before cudaIpcGetMemHandle:  2
 4
[ CUDAIntType{2} ]
#2: Tensor to send after cudaIpcGetMemHandle:  0
 0
[ CUDAIntType{2} ]
[2104843] Producer sent index for #2
[2104843] Producer sent IPC handle 0xE0FD0AE12F5600000B1E200000000000080000000000000000020000000000000001000000000000220000000000000016000000000000400000000000000000 for # 2
[2104843] Producer creating tensor #3
#3: Tensor to send before cudaIpcGetMemHandle:  3
 6
[ CUDAIntType{2} ]
#3: Tensor to send after cudaIpcGetMemHandle:  0
 0
[ CUDAIntType{2} ]
[2104843] Producer sent index for #3
[2104843] Producer sent IPC handle 0xE0FD0AE12F5600000B1E200000000000080000000000000000020000000000000001000000000000230000000000000017000000000000400000000000000000 for # 3
[2104843] Producer creating tensor #4
#4: Tensor to send before cudaIpcGetMemHandle:  4
 8
[ CUDAIntType{2} ]
#4: Tensor to send after cudaIpcGetMemHandle:  0
 0
[ CUDAIntType{2} ]
[2104843] Producer sent index for #4
[2104843] Producer sent IPC handle 0xE0FD0AE12F5600000B1E200000000000080000000000000000020000000000000001000000000000240000000000000018000000000000400000000000000000 for # 4
[2104843] Producer creating tensor #5
#5: Tensor to send before cudaIpcGetMemHandle:   5
 10
[ CUDAIntType{2} ]
#5: Tensor to send after cudaIpcGetMemHandle:  0
 0
[ CUDAIntType{2} ]
[2104843] Producer sent index for #5
[2104843] Producer sent IPC handle 0xE0FD0AE12F5600000B1E200000000000080000000000000000020000000000000001000000000000250000000000000019000000000000400000000000000000 for # 5
[2104843] Producer creating tensor #6
#6: Tensor to send before cudaIpcGetMemHandle:   6
 12
[ CUDAIntType{2} ]
#6: Tensor to send after cudaIpcGetMemHandle:  0
 0
[ CUDAIntType{2} ]
[2104843] Producer sent index for #6
[2104843] Producer sent IPC handle 0xE0FD0AE12F5600000B1E20000000000008000000000000000002000000000000000100000000000026000000000000001A000000000000400000000000000000 for # 6
[2104843] Producer creating tensor #7
#7: Tensor to send before cudaIpcGetMemHandle:   7
 14
[ CUDAIntType{2} ]
#7: Tensor to send after cudaIpcGetMemHandle:  0
 0
[ CUDAIntType{2} ]
[2104843] Producer sent index for #7
[2104843] Producer sent IPC handle 0xE0FD0AE12F5600000B1E20000000000008000000000000000002000000000000000100000000000027000000000000001B000000000000400000000000000000 for # 7
[2104843] Producer creating tensor #8
#8: Tensor to send before cudaIpcGetMemHandle:   8
 16
[ CUDAIntType{2} ]
#8: Tensor to send after cudaIpcGetMemHandle:  0
 0
[ CUDAIntType{2} ]
[2104843] Producer sent index for #8
[2104843] Producer sent IPC handle 0xE0FD0AE12F5600000B1E20000000000008000000000000000002000000000000000100000000000028000000000000001C000000000000400000000000000000 for # 8
[2104843] Producer creating tensor #9
#9: Tensor to send before cudaIpcGetMemHandle:   9
 18
[ CUDAIntType{2} ]
#9: Tensor to send after cudaIpcGetMemHandle:  0
 0
[ CUDAIntType{2} ]
[2104843] Producer sent index for #9
[2104843] Producer sent IPC handle 0xE0FD0AE12F5600000B1E20000000000008000000000000000002000000000000000100000000000029000000000000001D000000000000400000000000000000 for # 9
[2104843] Producer finished sending tensors
[2104843] Producer sent done signal
[2104843] Producer waiting for consumer done
[2104844] Consumer received producer done signal
[2104844] Consumer processing tensor #1
[2104844] Consumer received index for #1
[2104844] Received handle: 0xE0FD0AE12F5600000B1E200000000000080000000000000000020000000000000001000000000000210000000000000015000000000000400000000000000000
[2104844] Consumer opened IPC handle at 0x204c00000
[2104844] Consumer created tensor from blob
#1: Tensor received:  0
 0
[ CUDAIntType{2} ]
[2104844] Consumer processing tensor #2
[2104844] Consumer received index for #2
[2104844] Received handle: 0xE0FD0AE12F5600000B1E200000000000080000000000000000020000000000000001000000000000220000000000000016000000000000400000000000000000
[2104844] Consumer opened IPC handle at 0x204c00000
[2104844] Consumer created tensor from blob
#2: Tensor received:  0
 0
[ CUDAIntType{2} ]
[2104844] Consumer processing tensor #3
[2104844] Consumer received index for #3
[2104844] Received handle: 0xE0FD0AE12F5600000B1E200000000000080000000000000000020000000000000001000000000000230000000000000017000000000000400000000000000000
[2104844] Consumer opened IPC handle at 0x204c00000
[2104844] Consumer created tensor from blob
#3: Tensor received:  0
 0
[ CUDAIntType{2} ]
[2104844] Consumer processing tensor #4
[2104844] Consumer received index for #4
[2104844] Received handle: 0xE0FD0AE12F5600000B1E200000000000080000000000000000020000000000000001000000000000240000000000000018000000000000400000000000000000
[2104844] Consumer opened IPC handle at 0x204c00000
[2104844] Consumer created tensor from blob
#4: Tensor received:  0
 0
[ CUDAIntType{2} ]
[2104844] Consumer processing tensor #5
[2104844] Consumer received index for #5
[2104844] Received handle: 0xE0FD0AE12F5600000B1E200000000000080000000000000000020000000000000001000000000000250000000000000019000000000000400000000000000000
[2104844] Consumer opened IPC handle at 0x204c00000
[2104844] Consumer created tensor from blob
#5: Tensor received:  0
 0
[ CUDAIntType{2} ]
[2104844] Consumer processing tensor #6
[2104844] Consumer received index for #6
[2104844] Received handle: 0xE0FD0AE12F5600000B1E20000000000008000000000000000002000000000000000100000000000026000000000000001A000000000000400000000000000000
[2104844] Consumer opened IPC handle at 0x204c00000
[2104844] Consumer created tensor from blob
#6: Tensor received:  0
 0
[ CUDAIntType{2} ]
[2104844] Consumer processing tensor #7
[2104844] Consumer received index for #7
[2104844] Received handle: 0xE0FD0AE12F5600000B1E20000000000008000000000000000002000000000000000100000000000027000000000000001B000000000000400000000000000000
[2104844] Consumer opened IPC handle at 0x204c00000
[2104844] Consumer created tensor from blob
#7: Tensor received:  0
 0
[ CUDAIntType{2} ]
[2104844] Consumer processing tensor #8
[2104844] Consumer received index for #8
[2104844] Received handle: 0xE0FD0AE12F5600000B1E20000000000008000000000000000002000000000000000100000000000028000000000000001C000000000000400000000000000000
[2104844] Consumer opened IPC handle at 0x204c00000
[2104844] Consumer created tensor from blob
#8: Tensor received:  0
 0
[ CUDAIntType{2} ]
[2104844] Consumer processing tensor #9
[2104844] Consumer received index for #9
[2104844] Received handle: 0xE0FD0AE12F5600000B1E20000000000008000000000000000002000000000000000100000000000029000000000000001D000000000000400000000000000000
[2104844] Consumer opened IPC handle at 0x204c00000
[2104844] Consumer created tensor from blob
#9: Tensor received:  0
 0
[ CUDAIntType{2} ]
[2104844] Consumer sent done signal
Consumer exits
[2104843] Producer received consumer done
Producer exits
[2104842] Producer exited
[2104842] Consumer exited
```



