# Simple Makefile for CUDA Cartilage Thickness Calculator
# Alternative to CMake for quick builds

# Compiler settings
NVCC = nvcc
CXX = g++

# OpenCV configuration (adjust paths if needed)
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv)
OPENCV_LIBS = $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv)

# CUDA architecture (adjust for your GPU)
# 61 = GTX 1080/1070, 75 = RTX 2080, 86 = RTX 3090
CUDA_ARCH = -arch=sm_61 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_86,code=sm_86

# Compiler flags
NVCC_FLAGS = -O3 --use_fast_math -std=c++14 $(CUDA_ARCH)
NVCC_FLAGS += -Xcompiler -fopenmp
NVCC_FLAGS += $(OPENCV_CFLAGS)

# Linker flags
LDFLAGS = $(OPENCV_LIBS) -lcudart

# Source files
SOURCES = main.cu cuda_kernels.cu
TARGET = cartilage_thickness

# Build target
all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

# Clean build artifacts
clean:
	rm -f $(TARGET) *.o

# Run the program
run: $(TARGET)
	./$(TARGET)

# Check CUDA installation
check-cuda:
	@echo "Checking CUDA installation..."
	@nvcc --version || echo "ERROR: nvcc not found. Install CUDA Toolkit."
	@nvidia-smi || echo "ERROR: nvidia-smi not found. Install NVIDIA drivers."

# Check OpenCV installation
check-opencv:
	@echo "Checking OpenCV installation..."
	@pkg-config --modversion opencv4 || pkg-config --modversion opencv || echo "ERROR: OpenCV not found."

# Check all dependencies
check: check-cuda check-opencv
	@echo "All dependencies OK!"

# Install to system path (requires sudo)
install: $(TARGET)
	sudo cp $(TARGET) /usr/local/bin/

# Uninstall from system path
uninstall:
	sudo rm -f /usr/local/bin/$(TARGET)

# Help message
help:
	@echo "CUDA Cartilage Thickness Calculator - Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make          - Build the program"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make run      - Build and run the program"
	@echo "  make check    - Check if dependencies are installed"
	@echo "  make install  - Install to /usr/local/bin (requires sudo)"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "GPU Architecture:"
	@echo "  Edit CUDA_ARCH in Makefile to match your GPU"
	@echo "  Current: $(CUDA_ARCH)"

.PHONY: all clean run check check-cuda check-opencv install uninstall help
