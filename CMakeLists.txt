cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuda_ipc_get_mem_handle_producer_consumer_sample LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find LibTorch package
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

# Add executable
add_executable(cuda_ipc_get_mem_handle_producer_consumer_sample
               main.cpp
)

# Get CUDA library directory
get_filename_component(CUDA_LIB_DIR ${CUDA_cudart_LIBRARY} DIRECTORY)

# Link libraries
target_link_libraries(cuda_ipc_get_mem_handle_producer_consumer_sample
    PRIVATE
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_cudart_LIBRARY}
)

# Set include directories
target_include_directories(cuda_ipc_get_mem_handle_producer_consumer_sample
    PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

# Set link directories
target_link_directories(cuda_ipc_get_mem_handle_producer_consumer_sample
    PRIVATE
    ${CUDA_LIB_DIR}
)

# Set compiler flags
target_compile_options(cuda_ipc_get_mem_handle_producer_consumer_sample
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# Set CUDA architecture
set_target_properties(cuda_ipc_get_mem_handle_producer_consumer_sample PROPERTIES
    CUDA_ARCHITECTURES "native"
)

