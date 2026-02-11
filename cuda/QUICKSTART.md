# Quick Start Guide

Get up and running with the CUDA Cartilage Thickness Calculator in 5 minutes!

## Prerequisites Check

Run the system check script to verify your setup:

```bash
./check_system.sh
```

This will verify:
- ✓ CUDA Toolkit installation
- ✓ NVIDIA GPU and drivers
- ✓ OpenCV libraries
- ✓ CMake and compiler

## Quick Build (Option 1: Using Make)

The fastest way to build:

```bash
# Check dependencies
make check

# Build
make

# Run
./cartilage_thickness
```

## Quick Build (Option 2: Using CMake)

For more advanced configuration:

```bash
mkdir build
cd build
cmake ..
make -j4
./cartilage_thickness
```

## Verify Installation

If built successfully, you should see:

```bash
$ ./cartilage_thickness
Loaded X cases
Processing case 1/X (ID: ...)
...
```

## Troubleshooting Quick Fixes

### Problem: "nvcc: command not found"
```bash
# Add CUDA to your PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Problem: "OpenCV not found"
```bash
sudo apt-get update
sudo apt-get install libopencv-dev
```

### Problem: "CUDA error: out of memory"
Edit `main.cu` and reduce memory allocation:
```cpp
const int MAX_WIDTH = 256;   // Reduce from 512
const int MAX_HEIGHT = 256;  // Reduce from 512
```

### Problem: "Architecture mismatch"
Find your GPU's compute capability at https://developer.nvidia.com/cuda-gpus

Edit `Makefile`:
```makefile
CUDA_ARCH = -arch=sm_XX  # Replace XX with your compute capability
```

Or edit `CMakeLists.txt`:
```cmake
set(CMAKE_CUDA_ARCHITECTURES XX)
```

## Expected Performance

| Dataset Size | Octave Time | CUDA Time | Speedup |
|--------------|-------------|-----------|---------|
| 197 cases (31,520 images) | ~15 minutes | ~30-120 seconds | 10-30x |
| 1 case (160 images) | ~4 seconds | ~0.2-0.5 seconds | 8-20x |

*Times vary based on GPU model and image sizes*

## Usage Examples

### Basic Usage
```bash
# Process all cases in All_Images/
./cartilage_thickness
```

### Custom Image Folder
```bash
./cartilage_thickness TestImages/case1/ casenames.csv
```

### Batch Processing
```bash
# Process multiple test cases
for case in case1 case2 case3; do
    ./cartilage_thickness TestImages/$case/ casenames_$case.csv
done
```

## Input File Structure

```
project/
├── All_Images/
│   ├── 1_001_pred.png
│   ├── 1_002_pred.png
│   ├── ...
│   └── 197_160_pred.png
├── casenames.csv
└── cuda/
    └── cartilage_thickness (executable)
```

## Output

The program creates `test_distance_vector.csv`:

```csv
12.34,15.67,13.89
11.23,14.56,12.78
...
```

Each row has 3 values:
1. Anterior region thickness
2. Middle region thickness
3. Posterior region thickness

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Optimize for your specific GPU architecture
- Integrate into your processing pipeline
- Compare results with Octave version for validation

## Performance Monitoring

Monitor GPU usage while running:

```bash
# In another terminal
watch -n 1 nvidia-smi
```

You should see:
- GPU utilization: 70-100%
- Memory usage: < 500 MB per case
- Temperature: Should stay within safe limits

## Need Help?

1. Run `./check_system.sh` to diagnose issues
2. Check the [README.md](README.md) troubleshooting section
3. Verify image file format (grayscale PNG)
4. Check CUDA and driver compatibility

## Validation

To validate against Octave:

```bash
# Run Octave version
octave main.m
mv test_distance_vector.csv octave_results.csv

# Run CUDA version
./cartilage_thickness
mv test_distance_vector.csv cuda_results.csv

# Compare (should be very close)
python3 -c "
import numpy as np
octave = np.loadtxt('octave_results.csv', delimiter=',')
cuda = np.loadtxt('cuda_results.csv', delimiter=',')
diff = np.abs(octave - cuda)
print(f'Max difference: {diff.max():.6f}')
print(f'Mean difference: {diff.mean():.6f}')
"
```

Small differences (< 0.1) are expected due to floating-point precision.
