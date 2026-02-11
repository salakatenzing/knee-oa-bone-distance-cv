cmake_minimum_required(VERSION 3.18)
project(CartilageThickness CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    message(STATUS "CUDA toolkit root dir: ${CUDA_TOOLKIT_ROOT_DIR}")
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# CUDA architecture settings
# Adjust these based on your GPU (e.g., 61 for GTX 1080, 75 for RTX 2080, 86 for RTX 3090)
set(CMAKE_CUDA_ARCHITECTURES 61 75 86)

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

# Source files
set(CUDA_SOURCES
    main.cu
    cuda_kernels.cu
)

# Create executable
add_executable(cartilage_thickness ${CUDA_SOURCES})

# Set CUDA properties
set_target_properties(cartilage_thickness PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Link libraries
target_link_libraries(cartilage_thickness
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
)

# Installation rules
install(TARGETS cartilage_thickness DESTINATION bin)

# Print build configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
