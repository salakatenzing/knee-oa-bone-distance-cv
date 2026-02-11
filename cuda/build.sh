#!/bin/bash
# Automated build script for CUDA Cartilage Thickness Calculator

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}CUDA Cartilage Thickness - Build Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Parse command line arguments
BUILD_TYPE="Release"
USE_CMAKE=true
CLEAN_BUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --make)
            USE_CMAKE=false
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --help)
            echo "Usage: ./build.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --make      Use Makefile instead of CMake"
            echo "  --debug     Build in debug mode"
            echo "  --clean     Clean before building"
            echo "  --help      Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check system
echo -e "${YELLOW}Checking system requirements...${NC}"
if [ -f "./check_system.sh" ]; then
    if ! ./check_system.sh; then
        echo -e "${RED}System check failed. Please install missing dependencies.${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Warning: check_system.sh not found, skipping checks${NC}"
fi

echo ""
echo -e "${YELLOW}Building with $([[ $USE_CMAKE == true ]] && echo "CMake" || echo "Make")...${NC}"
echo ""

if $USE_CMAKE; then
    # CMake build
    BUILD_DIR="build"

    if $CLEAN_BUILD && [ -d "$BUILD_DIR" ]; then
        echo -e "${YELLOW}Cleaning previous build...${NC}"
        rm -rf "$BUILD_DIR"
    fi

    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    echo -e "${BLUE}Running CMake configuration...${NC}"
    cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..

    echo ""
    echo -e "${BLUE}Building...${NC}"
    make -j$(nproc)

    echo ""
    if [ -f "cartilage_thickness" ]; then
        echo -e "${GREEN}✓ Build successful!${NC}"
        echo ""
        echo -e "${GREEN}Executable location: $(pwd)/cartilage_thickness${NC}"
        echo ""
        echo "To run:"
        echo "  cd $BUILD_DIR"
        echo "  ./cartilage_thickness"
    else
        echo -e "${RED}✗ Build failed - executable not found${NC}"
        exit 1
    fi

else
    # Make build
    if $CLEAN_BUILD; then
        echo -e "${YELLOW}Cleaning previous build...${NC}"
        make clean
    fi

    echo -e "${BLUE}Building with Make...${NC}"
    make -j$(nproc)

    echo ""
    if [ -f "cartilage_thickness" ]; then
        echo -e "${GREEN}✓ Build successful!${NC}"
        echo ""
        echo -e "${GREEN}Executable location: $(pwd)/cartilage_thickness${NC}"
        echo ""
        echo "To run:"
        echo "  ./cartilage_thickness"
    else
        echo -e "${RED}✗ Build failed - executable not found${NC}"
        exit 1
    fi
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
