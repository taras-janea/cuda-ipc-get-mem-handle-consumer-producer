// MIT License
//
// Copyright (c) 2025 [Janea Systems]
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// =============================================================================
// CUDA IPC Tensor Sharing Example
//
// This repository demonstrates inter-process communication of CUDA tensors
// using:
// - CUDA IPC handles for GPU memory sharing
// - POSIX pipes for inter-process communication
// - LibTorch for tensor operations
//
// Features:
// - Producer process creates GPU tensors and shares them via IPC
// - Consumer process accesses shared GPU memory without copy
// - Synchronization using pipe-based signaling
// - Error handling and robust data transfer
// =============================================================================

#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <iostream>
#include <spawn.h>
#include <sstream>
#include <sys/wait.h>
#include <torch/torch.h>
#include <unistd.h>
#include <vector>

namespace {
#define DEBUG_LOG(msg)                                                         \
  do {                                                                         \
    std::ostringstream oss;                                                    \
    oss << "[" << getpid() << "] " << msg;                                     \
    std::cout << oss.str() << std::endl;                                       \
    std::cout.flush();                                                         \
  } while (0)

std::string cudaIpcHandleToString(const cudaIpcMemHandle_t &handle) {
  const uint8_t *data = reinterpret_cast<const uint8_t *>(&handle);
  std::string result = "0x";
  for (int i = 0; i < sizeof(cudaIpcMemHandle_t); ++i) {
    char res[3] = {0};
    snprintf(res, sizeof(res), "%02X", static_cast<uint8_t>(data[i]));
    result += std::string(res);
  }
  return result;
}
} // namespace

void producer(int tensor_pipe_write, int producer_done_write,
              int consumer_done_read) {
  try {
    DEBUG_LOG("Producer starting");
    cudaSetDevice(0);
    cudaFree(0);

    using uptr = std::unique_ptr<void, std::function<void(void *)>>;

    // Store tensors to keep memory allocated
    std::vector<uptr> allocations;
    for (int i = 1; i <= 9; i++) {
      DEBUG_LOG("Producer creating tensor #" + std::to_string(i));

      // Allocate CUDA memory
      void *d_ptr;
      int data[2] = {i, i * 2};
      cudaError_t err = cudaMalloc(&d_ptr, sizeof(data));
      if (err != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed: " +
                                 std::string(cudaGetErrorString(err)));
      }
      allocations.push_back(uptr(d_ptr, [](void *ptr) { cudaFree(ptr); }));

      // Create tensor from raw memory
      std::vector<int64_t> sizes = {2};
      torch::Tensor gpu_tensor = torch::from_blob(
          d_ptr, sizes,
          torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0));

      // Fill tensor with data
      cudaMemcpy(d_ptr, data, sizeof(data), cudaMemcpyHostToDevice);

      {
        std::cout << "#" << i << ": Tensor to send before cudaIpcGetMemHandle: "
                  << gpu_tensor << std::endl;
      }

      // Get IPC handle
      cudaIpcMemHandle_t handle;
      err = cudaIpcGetMemHandle(&handle, gpu_tensor.data_ptr());
      if (err != cudaSuccess) {
        std::stringstream ss;
        ss << "cudaIpcGetMemHandle failed: " << cudaGetErrorString(err)
           << " for tensor at " << gpu_tensor.data_ptr();
        throw std::runtime_error(ss.str());
      }

      {
        std::cout << "#" << i << ": Tensor to send after cudaIpcGetMemHandle: "
                  << gpu_tensor << std::endl;
      }

      // Send tensor index
      if (write(tensor_pipe_write, &i, sizeof(i)) != sizeof(i)) {
        throw std::runtime_error("Failed to write tensor index");
      }
      DEBUG_LOG("Producer sent index for #" + std::to_string(i));

      // Send IPC handle
      if (write(tensor_pipe_write, &handle, sizeof(handle)) != sizeof(handle)) {
        throw std::runtime_error("Failed to write IPC handle");
      }
      DEBUG_LOG("Producer sent IPC handle " + cudaIpcHandleToString(handle) +
                " for # " + std::to_string(i));
    }

    DEBUG_LOG("Producer finished sending tensors");
    char done_byte = 'D';
    if (write(producer_done_write, &done_byte, 1) != 1) {
      throw std::runtime_error("Failed to signal producer done");
    }
    DEBUG_LOG("Producer sent done signal");
    cudaDeviceSynchronize();

    DEBUG_LOG("Producer waiting for consumer done");
    char ack_byte;
    if (read(consumer_done_read, &ack_byte, 1) != 1) {
      throw std::runtime_error("Failed to receive consumer done signal");
    }
    DEBUG_LOG("Producer received consumer done");

    std::cout << "Producer exits" << std::endl;
  } catch (const std::exception &e) {
    DEBUG_LOG(std::string("Producer error: ") + e.what());
    exit(1);
  }
}

void consumer(int tensor_pipe_read, int producer_done_read,
              int consumer_done_write) {
  try {
    DEBUG_LOG("Consumer starting");
    cudaSetDevice(0);
    cudaFree(0);

    DEBUG_LOG("Consumer waiting for producer done signal");
    char done_byte;
    if (read(producer_done_read, &done_byte, 1) != 1) {
      throw std::runtime_error("Failed to receive producer done signal");
    }
    DEBUG_LOG("Consumer received producer done signal");

    for (int i = 1; i <= 9; i++) {
      DEBUG_LOG("Consumer processing tensor #" + std::to_string(i));

      // Read tensor index
      int idx;
      ssize_t n_read = read(tensor_pipe_read, &idx, sizeof(idx));
      if (n_read != sizeof(idx)) {
        throw std::runtime_error("Failed to read tensor index");
      }
      DEBUG_LOG("Consumer received index for #" + std::to_string(idx));

      // Read IPC handle
      cudaIpcMemHandle_t handle;
      n_read = read(tensor_pipe_read, &handle, sizeof(handle));
      if (n_read != sizeof(handle)) {
        throw std::runtime_error("Failed to read IPC handle");
      }

      DEBUG_LOG("Received handle: " + cudaIpcHandleToString(handle));

      // Open shared memory handle
      void *d_ptr;
      cudaError_t err =
          cudaIpcOpenMemHandle(&d_ptr, handle, cudaIpcMemLazyEnablePeerAccess);
      if (err != cudaSuccess) {
        std::stringstream ss;
        ss << "cudaIpcOpenMemHandle failed: " << cudaGetErrorString(err);
        throw std::runtime_error(ss.str());
      }
      DEBUG_LOG("Consumer opened IPC handle at " << d_ptr);

      // Create tensor with custom deleter
      auto deleter = [](void *ptr) { cudaIpcCloseMemHandle(ptr); };

      // Create tensor from shared memory
      torch::Tensor tensor = torch::from_blob(
          d_ptr, {2}, deleter,
          torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
      DEBUG_LOG("Consumer created tensor from blob");
      std::cout << "#" << idx << ": Tensor received: " << tensor << std::endl;
    }

    // Signal consumer is done
    char ack_byte = 'A';
    if (write(consumer_done_write, &ack_byte, 1) != 1) {
      throw std::runtime_error("Failed to signal consumer done");
    }
    DEBUG_LOG("Consumer sent done signal");

    std::cout << "Consumer exits" << std::endl;
  } catch (const std::exception &e) {
    DEBUG_LOG(std::string("Consumer error: ") + e.what());
    exit(1);
  }
}

int main(int argc, char *argv[]) {
  // If called with arguments, run as worker process
  if (argc == 5) {
    DEBUG_LOG("Child process started with role: " + std::string(argv[1]));

    int tensor_pipe = atoi(argv[2]);
    int done_pipe1 = atoi(argv[3]);
    int done_pipe2 = atoi(argv[4]);

    if (strcmp(argv[1], "producer") == 0) {
      producer(tensor_pipe, done_pipe1, done_pipe2);
    } else if (strcmp(argv[1], "consumer") == 0) {
      consumer(tensor_pipe, done_pipe1, done_pipe2);
    }
    return 0;
  }

  DEBUG_LOG("Parent process starting");

  // Create communication pipes
  int tensor_pipe[2];
  int producer_done_pipe[2]; // Producer -> Consumer
  int consumer_done_pipe[2]; // Consumer -> Producer

  if (pipe(tensor_pipe)) {
    perror("tensor_pipe creation failed");
    return 1;
  }
  if (pipe(producer_done_pipe)) {
    perror("producer_done_pipe creation failed");
    return 1;
  }
  if (pipe(consumer_done_pipe)) {
    perror("consumer_done_pipe creation failed");
    return 1;
  }
  DEBUG_LOG("Pipes created");

  // Prepare arguments for producer
  char tensor_pipe_str[16], producer_done_str[16], consumer_done_str[16];
  snprintf(tensor_pipe_str, sizeof(tensor_pipe_str), "%d", tensor_pipe[1]);
  snprintf(producer_done_str, sizeof(producer_done_str), "%d",
           producer_done_pipe[1]);
  snprintf(consumer_done_str, sizeof(consumer_done_str), "%d",
           consumer_done_pipe[0]);

  // Spawn producer
  DEBUG_LOG("Spawning producer");
  pid_t producer_pid;
  char *producer_args[] = {argv[0],
                           (char *)"producer",
                           tensor_pipe_str,
                           producer_done_str, // Write end of producer_done_pipe
                           consumer_done_str, // Read end of consumer_done_pipe
                           NULL};

  if (posix_spawn(&producer_pid, argv[0], NULL, NULL, producer_args, environ)) {
    perror("posix_spawn producer failed");
    return 1;
  }
  DEBUG_LOG("Producer spawned with PID: " << producer_pid);

  // Prepare arguments for consumer
  snprintf(tensor_pipe_str, sizeof(tensor_pipe_str), "%d", tensor_pipe[0]);
  snprintf(producer_done_str, sizeof(producer_done_str), "%d",
           producer_done_pipe[0]);
  snprintf(consumer_done_str, sizeof(consumer_done_str), "%d",
           consumer_done_pipe[1]);

  // Spawn consumer
  DEBUG_LOG("Spawning consumer");
  pid_t consumer_pid;
  char *consumer_args[] = {argv[0],
                           (char *)"consumer",
                           tensor_pipe_str,
                           producer_done_str, // Read end of producer_done_pipe
                           consumer_done_str, // Write end of consumer_done_pipe
                           NULL};

  if (posix_spawn(&consumer_pid, argv[0], NULL, NULL, consumer_args, environ)) {
    perror("posix_spawn consumer failed");
    return 1;
  }
  DEBUG_LOG("Consumer spawned with PID: " << consumer_pid);

  // Close pipe ends in parent
  close(tensor_pipe[0]);
  close(tensor_pipe[1]);
  close(producer_done_pipe[0]);
  close(producer_done_pipe[1]);
  close(consumer_done_pipe[0]);
  close(consumer_done_pipe[1]);
  DEBUG_LOG("Parent closed all pipe ends");

  // Wait for children
  DEBUG_LOG("Parent waiting for children");
  waitpid(producer_pid, NULL, 0);
  DEBUG_LOG("Producer exited");
  waitpid(consumer_pid, NULL, 0);
  DEBUG_LOG("Consumer exited");

  return 0;
}