#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include <cuda_runtime.h>

// Morphological erosion kernel
__global__ void morphologicalErosion(const unsigned char* input, unsigned char* output,
                                     int width, int height, int seSize);

// Count non-zero pixels
__global__ void countNonZero(const unsigned char* image, int* counts, int numPixels);

// Mask regions kernel
__global__ void maskRegions(unsigned char* image, int width, int height,
                           int maskStartY, int maskEndY);

// Connected components labeling
void connectedComponentsLabeling(unsigned char* d_image, int* d_labels,
                                int width, int height, int& numRegions);

// Find largest regions
void findLargestRegions(int* d_labels, int width, int height, int numRegions,
                       int& region1Label, int& region2Label);

// Extract boundary points
void extractBoundary(int* d_labels, int regionLabel, int width, int height,
                    int* d_boundaryX, int* d_boundaryY, int& numBoundaryPoints);

// Calculate minimum distances between two boundaries
float calculateBoundaryDistance(int* d_boundary1X, int* d_boundary1Y, int numPoints1,
                               int* d_boundary2X, int* d_boundary2Y, int numPoints2,
                               float* d_distances);

// High-level function to check if image is processable
bool checkProcessable(unsigned char* d_image, int width, int height, int& nonZeroCount);

// High-level function to calculate cartilage thickness
float calculateCartilageThickness(unsigned char* d_image, int* d_labels,
                                 float* d_distances, int* d_temp,
                                 int width, int height);

#endif // CUDA_KERNELS_CUH
