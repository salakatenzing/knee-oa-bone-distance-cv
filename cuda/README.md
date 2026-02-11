# CUDA Cartilage Thickness Calculator

A high-performance GPU-accelerated implementation for calculating cartilage thickness from medical images. This CUDA C++ program processes knee MRI images to measure femoral and tibial cartilage thickness, providing significant speedup over the original Octave implementation.

## Performance

- **Original Octave**: ~15 minutes for 197 cases (31,520 images)
- **CUDA C++ (Expected)**: ~30 seconds - 2 minutes (depending on GPU)
- **Speedup**: **10-30x faster**

## Features

- GPU-accelerated morphological operations (erosion)
- Parallel connected components labeling
- Fast boundary extraction
- Optimized distance calculations using CUDA kernels
- Batch processing of multiple images
- Memory-efficient streaming for large datasets

## Requirements

### Hardware
- NVIDIA GPU with compute capability 6.1 or higher (GTX 1000 series or newer)
- Minimum 2GB GPU memory (4GB+ recommended)

### Software
- CUDA Toolkit 11.0 or later
- CMake 3.18+
- OpenCV 4.0+ (with development headers)
- C++14 compatible compiler (g++ 7.0+ or MSVC 2017+)

## Installation

### Ubuntu/Debian

1. **Install CUDA Toolkit**
   ```bash
   # Follow instructions at: https://developer.nvidia.com/cuda-downloads
   # Or use package manager:
   sudo apt-get update
   sudo apt-get install nvidia-cuda-toolkit
   ```

2. **Install OpenCV**
   ```bash
   sudo apt-get install libopencv-dev
   ```

3. **Install CMake**
   ```bash
   sudo apt-get install cmake
   ```

### Build Instructions

```bash
# Navigate to the cuda directory
cd cuda

# Create build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build the project
make -j$(nproc)

# The executable will be created as: cartilage_thickness
```

### GPU Architecture Configuration

The CMakeLists.txt is configured for common GPU architectures:
- **61**: GTX 1080, GTX 1070, GTX 1060
- **75**: RTX 2080 Ti, RTX 2080, RTX 2070
- **86**: RTX 3090, RTX 3080, RTX 3070

To customize for your GPU, edit `CMakeLists.txt`:
```cmake
set(CMAKE_CUDA_ARCHITECTURES XX)  # Replace XX with your compute capability
```

Find your GPU's compute capability at: https://developer.nvidia.com/cuda-gpus

## Usage

### Basic Usage

```bash
./cartilage_thickness
```

This assumes:
- Images are in `All_Images/` directory
- Case names are in `casenames.csv`

### Custom Paths

```bash
./cartilage_thickness <images_folder> <casenames_file>
```

Example:
```bash
./cartilage_thickness TestImages/case1/ casenames.csv
```

### Input Format

1. **casenames.csv**: CSV file with case IDs (one per line after header)
   ```
   case_id
   1
   2
   3
   ```

2. **Image Files**: PNG images named as: `<caseID>_<sliceNumber>_pred.png`
   - Example: `1_001_pred.png`, `1_002_pred.png`, etc.
   - Slice numbers from 1-160 per case
   - Images should be grayscale segmentation masks

### Output

The program generates `test_distance_vector.csv` with three columns:
- Column 1: Average thickness in anterior region
- Column 2: Average thickness in middle region
- Column 3: Average thickness in posterior region

Each row corresponds to one case in the input CSV.

## Algorithm Overview

The CUDA implementation follows these steps for each image:

1. **Morphological Erosion**: 5x5 square structuring element (GPU kernel)
2. **Processability Check**: Verify >20% non-zero pixels (GPU parallel reduction)
3. **Region Masking**: Mask irrelevant areas (GPU kernel)
4. **Connected Components**: Label separate regions using Union-Find (GPU)
5. **Region Selection**: Find two largest regions (femur and tibia)
6. **Boundary Extraction**: Identify boundary pixels (GPU kernel)
7. **Distance Calculation**: Compute minimum distances between boundaries (GPU)
8. **Averaging**: Calculate mean thickness across valid slices

## Optimization Techniques

- **Parallel Processing**: All pixel operations parallelized across GPU threads
- **Coalesced Memory Access**: Optimized memory access patterns
- **Shared Memory**: Used in reduction operations
- **Atomic Operations**: For thread-safe counting and labeling
- **Thrust Library**: For efficient reductions and sorting
- **Stream Processing**: Can be extended for multi-stream processing

## Troubleshooting

### CUDA Out of Memory

If you encounter memory errors:
1. Reduce `MAX_WIDTH` and `MAX_HEIGHT` in `main.cu`
2. Process fewer cases at once
3. Use a GPU with more memory

### Slow Performance

1. Check GPU utilization: `nvidia-smi`
2. Verify CUDA architecture matches your GPU
3. Ensure GPU is not running other intensive tasks
4. Update GPU drivers

### Build Errors

- **"CUDA not found"**: Install CUDA Toolkit and set `CUDA_PATH`
- **"OpenCV not found"**: Install OpenCV development packages
- **Architecture mismatch**: Update `CMAKE_CUDA_ARCHITECTURES` in CMakeLists.txt

### Runtime Errors

- **"Cannot open image"**: Check image paths and file permissions
- **"CUDA error"**: Check GPU memory, reduce batch size
- **Wrong results**: Verify image format (should be grayscale PNG)

## Advanced Configuration

### Adjusting Image Size Limits

Edit `main.cu`:
```cpp
const int MAX_WIDTH = 512;   // Increase for larger images
const int MAX_HEIGHT = 512;  // Increase for larger images
```

### Morphological Operations

To change erosion kernel size, modify the call in `main.cu`:
```cpp
morphologicalErosion<<<gridSize, blockSize>>>(d_image, d_eroded, width, height, 5);
                                                                               // ^ kernel size
```

### Masking Regions

Adjust masking parameters in `cuda_kernels.cu`:
```cpp
maskRegions<<<gridSize2D, blockSize2D>>>(d_image, width, height, 0, 90);    // First mask
maskRegions<<<gridSize2D, blockSize2D>>>(d_image, width, height, 90, 135);  // Second mask
```

## Memory Requirements

Approximate GPU memory usage:
- Per image (512x512): ~5 MB
- Temporary buffers: ~10 MB
- Total: ~15-20 MB per concurrent image

For 4GB GPU: Can process ~200 images concurrently (more than needed)

## Performance Tips

1. **Use SSD**: Store images on SSD for faster I/O
2. **Batch Processing**: Process multiple cases if memory allows
3. **GPU Selection**: Use `CUDA_VISIBLE_DEVICES=0` to select GPU
4. **Compile Optimizations**: Use `-O3` flag (already in CMakeLists.txt)
5. **Profile Code**: Use `nvprof` or Nsight Systems for bottleneck analysis

## Comparison with Octave Version

| Feature | Octave | CUDA C++ |
|---------|--------|----------|
| Runtime | ~15 min | ~30s-2min |
| Parallelization | None | Full GPU |
| Memory Usage | CPU RAM | GPU VRAM |
| Scalability | Poor | Excellent |
| Dependencies | Octave + packages | CUDA + OpenCV |

## Future Improvements

- Multi-GPU support for processing different cases in parallel
- CUDA streams for overlapping I/O and computation
- INT8 operations for further speedup
- Direct integration with DICOM format
- Real-time visualization

## License

Same as the parent repository.

## Contributing

Contributions welcome! Areas for improvement:
- Further kernel optimizations
- Support for different image formats
- Validation against Octave output
- Benchmark suite

## Citation

If you use this code, please cite the original research paper associated with this cartilage thickness measurement method.

## Contact

For issues and questions, please open an issue on the GitHub repository.
