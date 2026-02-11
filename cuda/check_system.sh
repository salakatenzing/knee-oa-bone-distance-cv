#!/bin/bash
# System check script for CUDA Cartilage Thickness Calculator

echo "=========================================="
echo "CUDA Cartilage Thickness - System Check"
echo "=========================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check CUDA
echo "Checking CUDA Toolkit..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo -e "${GREEN}✓${NC} CUDA Toolkit found: $CUDA_VERSION"
else
    echo -e "${RED}✗${NC} CUDA Toolkit not found"
    echo "  Install from: https://developer.nvidia.com/cuda-downloads"
    EXIT_CODE=1
fi

# Check NVIDIA Driver
echo ""
echo "Checking NVIDIA Driver..."
if command -v nvidia-smi &> /dev/null; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n1)
    echo -e "${GREEN}✓${NC} NVIDIA Driver found: $DRIVER_VERSION"
    echo "  GPU: $GPU_NAME"
    echo "  Memory: $GPU_MEMORY"

    # Check compute capability
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1)
    echo "  Compute Capability: $COMPUTE_CAP"

    # Parse compute capability
    MAJOR=$(echo $COMPUTE_CAP | cut -d'.' -f1)
    if [ "$MAJOR" -ge 6 ]; then
        echo -e "  ${GREEN}✓${NC} GPU is compatible (compute capability >= 6.0)"
    else
        echo -e "  ${YELLOW}⚠${NC} GPU may not be fully compatible (compute capability < 6.0)"
    fi
else
    echo -e "${RED}✗${NC} NVIDIA Driver not found"
    echo "  Install NVIDIA drivers for your GPU"
    EXIT_CODE=1
fi

# Check OpenCV
echo ""
echo "Checking OpenCV..."
if pkg-config --exists opencv4; then
    OPENCV_VERSION=$(pkg-config --modversion opencv4)
    echo -e "${GREEN}✓${NC} OpenCV found: $OPENCV_VERSION"
elif pkg-config --exists opencv; then
    OPENCV_VERSION=$(pkg-config --modversion opencv)
    echo -e "${GREEN}✓${NC} OpenCV found: $OPENCV_VERSION"
else
    echo -e "${RED}✗${NC} OpenCV not found"
    echo "  Install with: sudo apt-get install libopencv-dev"
    EXIT_CODE=1
fi

# Check CMake
echo ""
echo "Checking CMake..."
if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -n1 | awk '{print $3}')
    echo -e "${GREEN}✓${NC} CMake found: $CMAKE_VERSION"

    # Check if version is >= 3.18
    CMAKE_MAJOR=$(echo $CMAKE_VERSION | cut -d'.' -f1)
    CMAKE_MINOR=$(echo $CMAKE_VERSION | cut -d'.' -f2)
    if [ "$CMAKE_MAJOR" -gt 3 ] || ([ "$CMAKE_MAJOR" -eq 3 ] && [ "$CMAKE_MINOR" -ge 18 ]); then
        echo -e "  ${GREEN}✓${NC} CMake version is sufficient (>= 3.18)"
    else
        echo -e "  ${YELLOW}⚠${NC} CMake version may be too old (< 3.18)"
    fi
else
    echo -e "${YELLOW}⚠${NC} CMake not found (optional, can use Makefile)"
    echo "  Install with: sudo apt-get install cmake"
fi

# Check compiler
echo ""
echo "Checking C++ Compiler..."
if command -v g++ &> /dev/null; then
    GCC_VERSION=$(g++ --version | head -n1 | awk '{print $NF}')
    echo -e "${GREEN}✓${NC} g++ found: $GCC_VERSION"
else
    echo -e "${RED}✗${NC} g++ not found"
    echo "  Install with: sudo apt-get install build-essential"
    EXIT_CODE=1
fi

# Check for required packages
echo ""
echo "Checking system packages..."
if pkg-config --exists libpng; then
    echo -e "${GREEN}✓${NC} libpng found"
else
    echo -e "${YELLOW}⚠${NC} libpng not found (may cause issues)"
fi

# Summary
echo ""
echo "=========================================="
if [ -z "$EXIT_CODE" ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    echo "You can now build the project:"
    echo "  Using CMake: mkdir build && cd build && cmake .. && make"
    echo "  Using Make:  make"
else
    echo -e "${RED}✗ Some checks failed${NC}"
    echo ""
    echo "Please install missing dependencies before building."
fi
echo "=========================================="

exit ${EXIT_CODE:-0}
