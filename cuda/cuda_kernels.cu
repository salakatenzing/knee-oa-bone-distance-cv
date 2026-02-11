#include "cuda_kernels.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <cmath>
#include <algorithm>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error in %s line %d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
        } \
    } while(0)

// Morphological erosion with square structuring element
__global__ void morphologicalErosion(const unsigned char* input, unsigned char* output,
                                     int width, int height, int seSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int halfSize = seSize / 2;
    unsigned char minVal = 255;

    // Check all pixels in structuring element
    for (int dy = -halfSize; dy <= halfSize; dy++) {
        for (int dx = -halfSize; dx <= halfSize; dx++) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                unsigned char val = input[ny * width + nx];
                if (val < minVal) {
                    minVal = val;
                }
            } else {
                minVal = 0; // Border handling
            }
        }
    }

    output[y * width + x] = minVal;
}

// Count non-zero pixels using parallel reduction
__global__ void countNonZeroKernel(const unsigned char* image, int* output, int numPixels) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (idx < numPixels && image[idx] != 0) ? 1 : 0;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// Mask specific regions of image
__global__ void maskRegions(unsigned char* image, int width, int height,
                           int maskStartY, int maskEndY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    if (y >= maskStartY && y < maskEndY) {
        image[y * width + x] = 255;
    }
}

// Simple connected components using Union-Find on GPU
__global__ void initLabels(int* labels, int numPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPixels) {
        labels[idx] = idx;
    }
}

__device__ int find(int* labels, int x) {
    int y = x;
    while (labels[y] != y) {
        y = labels[y];
    }

    // Path compression
    while (labels[x] != x) {
        int z = labels[x];
        labels[x] = y;
        x = z;
    }

    return y;
}

__device__ void unionLabels(int* labels, int a, int b) {
    bool done;
    do {
        done = true;
        a = find(labels, a);
        b = find(labels, b);

        if (a < b) {
            int old = atomicMin(&labels[b], a);
            done = (old == b);
            b = old;
        } else if (b < a) {
            int old = atomicMin(&labels[a], b);
            done = (old == a);
            a = old;
        }
    } while (!done);
}

__global__ void labelConnectedComponents(unsigned char* image, int* labels,
                                        int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    if (image[idx] == 0) {
        labels[idx] = -1; // Background
        return;
    }

    // Check 4-connected neighbors
    if (x > 0 && image[idx - 1] != 0) {
        unionLabels(labels, idx, idx - 1);
    }
    if (y > 0 && image[idx - width] != 0) {
        unionLabels(labels, idx, idx - width);
    }
}

__global__ void flattenLabels(int* labels, int numPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPixels && labels[idx] >= 0) {
        labels[idx] = find(labels, idx);
    }
}

// Count pixels per label
__global__ void countRegionSizes(int* labels, int* sizes, int numPixels, int numRegions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPixels && labels[idx] >= 0) {
        int label = labels[idx];
        if (label < numRegions) {
            atomicAdd(&sizes[label], 1);
        }
    }
}

void connectedComponentsLabeling(unsigned char* d_image, int* d_labels,
                                int width, int height, int& numRegions) {
    int numPixels = width * height;

    dim3 blockSize(256);
    dim3 gridSize((numPixels + blockSize.x - 1) / blockSize.x);

    // Initialize labels
    initLabels<<<gridSize, blockSize>>>(d_labels, numPixels);
    CUDA_CHECK(cudaGetLastError());

    // Label connected components (multiple iterations for convergence)
    dim3 blockSize2D(16, 16);
    dim3 gridSize2D((width + blockSize2D.x - 1) / blockSize2D.x,
                    (height + blockSize2D.y - 1) / blockSize2D.y);

    for (int iter = 0; iter < 10; iter++) {
        labelConnectedComponents<<<gridSize2D, blockSize2D>>>(d_image, d_labels, width, height);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Flatten labels
    flattenLabels<<<gridSize, blockSize>>>(d_labels, numPixels);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Count unique labels (approximate)
    numRegions = numPixels; // Conservative upper bound
}

void findLargestRegions(int* d_labels, int width, int height, int numRegions,
                       int& region1Label, int& region2Label) {
    int numPixels = width * height;

    // Allocate memory for region sizes
    int* d_sizes;
    CUDA_CHECK(cudaMalloc(&d_sizes, numRegions * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_sizes, 0, numRegions * sizeof(int)));

    // Count sizes
    dim3 blockSize(256);
    dim3 gridSize((numPixels + blockSize.x - 1) / blockSize.x);
    countRegionSizes<<<gridSize, blockSize>>>(d_labels, d_sizes, numPixels, numRegions);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy sizes to host and find two largest
    std::vector<int> h_sizes(numRegions);
    CUDA_CHECK(cudaMemcpy(h_sizes.data(), d_sizes, numRegions * sizeof(int),
                         cudaMemcpyDeviceToHost));

    // Find two largest regions
    int maxSize1 = 0, maxSize2 = 0;
    region1Label = -1;
    region2Label = -1;

    for (int i = 0; i < numRegions; i++) {
        if (h_sizes[i] > maxSize1) {
            maxSize2 = maxSize1;
            region2Label = region1Label;
            maxSize1 = h_sizes[i];
            region1Label = i;
        } else if (h_sizes[i] > maxSize2) {
            maxSize2 = h_sizes[i];
            region2Label = i;
        }
    }

    CUDA_CHECK(cudaFree(d_sizes));
}

__global__ void extractBoundaryKernel(int* labels, int regionLabel, int width, int height,
                                     int* boundaryX, int* boundaryY, int* count) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    if (labels[idx] == regionLabel) {
        // Check if this is a boundary pixel (has a neighbor with different label)
        bool isBoundary = false;

        if (x > 0 && labels[idx - 1] != regionLabel) isBoundary = true;
        if (x < width - 1 && labels[idx + 1] != regionLabel) isBoundary = true;
        if (y > 0 && labels[idx - width] != regionLabel) isBoundary = true;
        if (y < height - 1 && labels[idx + width] != regionLabel) isBoundary = true;

        if (isBoundary) {
            int pos = atomicAdd(count, 1);
            if (pos < width * height) { // Safety check
                boundaryX[pos] = x;
                boundaryY[pos] = y;
            }
        }
    }
}

void extractBoundary(int* d_labels, int regionLabel, int width, int height,
                    int* d_boundaryX, int* d_boundaryY, int& numBoundaryPoints) {
    int* d_count;
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    extractBoundaryKernel<<<gridSize, blockSize>>>(d_labels, regionLabel, width, height,
                                                   d_boundaryX, d_boundaryY, d_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&numBoundaryPoints, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_count));
}

__global__ void calculateMinDistancesKernel(int* boundary1X, int* boundary1Y, int numPoints1,
                                           int* boundary2X, int* boundary2Y, int numPoints2,
                                           float* minDistances) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numPoints1) return;

    int x1 = boundary1X[idx];
    int y1 = boundary1Y[idx];

    float minDist = 1e9f;

    // Find minimum distance to any point in boundary2
    for (int j = 0; j < numPoints2; j++) {
        int x2 = boundary2X[j];
        int y2 = boundary2Y[j];

        float dx = (float)(x1 - x2);
        float dy = (float)(y1 - y2);
        float dist = sqrtf(dx * dx + dy * dy);

        if (dist < minDist) {
            minDist = dist;
        }
    }

    minDistances[idx] = minDist;
}

float calculateBoundaryDistance(int* d_boundary1X, int* d_boundary1Y, int numPoints1,
                               int* d_boundary2X, int* d_boundary2Y, int numPoints2,
                               float* d_distances) {
    if (numPoints1 == 0 || numPoints2 == 0) {
        return 0.0f;
    }

    dim3 blockSize(256);
    dim3 gridSize((numPoints1 + blockSize.x - 1) / blockSize.x);

    calculateMinDistancesKernel<<<gridSize, blockSize>>>(
        d_boundary1X, d_boundary1Y, numPoints1,
        d_boundary2X, d_boundary2Y, numPoints2,
        d_distances);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Calculate mean using thrust
    thrust::device_ptr<float> ptr(d_distances);
    float sum = thrust::reduce(ptr, ptr + numPoints1, 0.0f, thrust::plus<float>());

    return sum / numPoints1;
}

bool checkProcessable(unsigned char* d_image, int width, int height, int& nonZeroCount) {
    int numPixels = width * height;

    int* d_count;
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

    dim3 blockSize(256);
    dim3 gridSize((numPixels + blockSize.x - 1) / blockSize.x);

    countNonZeroKernel<<<gridSize, blockSize, blockSize.x * sizeof(int)>>>(
        d_image, d_count, numPixels);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&nonZeroCount, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_count));

    float nonZeroPercentage = (nonZeroCount / (float)numPixels) * 100.0f;

    return nonZeroPercentage > 20.0f;
}

float calculateCartilageThickness(unsigned char* d_image, int* d_labels,
                                 float* d_distances, int* d_temp,
                                 int width, int height) {
    // Apply masking
    dim3 blockSize2D(16, 16);
    dim3 gridSize2D((width + blockSize2D.x - 1) / blockSize2D.x,
                    (height + blockSize2D.y - 1) / blockSize2D.y);

    maskRegions<<<gridSize2D, blockSize2D>>>(d_image, width, height, 0, 90);
    CUDA_CHECK(cudaGetLastError());
    maskRegions<<<gridSize2D, blockSize2D>>>(d_image, width, height, 90, 135);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Connected components labeling
    int numRegions;
    connectedComponentsLabeling(d_image, d_labels, width, height, numRegions);

    // Find two largest regions
    int femurLabel, tibiaLabel;
    findLargestRegions(d_labels, width, height, numRegions, femurLabel, tibiaLabel);

    if (femurLabel < 0 || tibiaLabel < 0) {
        return 0.0f; // Not enough regions
    }

    // Allocate boundary arrays
    int maxBoundarySize = width * height;
    int* d_femurX;
    int* d_femurY;
    int* d_tibiaX;
    int* d_tibiaY;

    CUDA_CHECK(cudaMalloc(&d_femurX, maxBoundarySize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_femurY, maxBoundarySize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tibiaX, maxBoundarySize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tibiaY, maxBoundarySize * sizeof(int)));

    // Extract boundaries
    int numFemurPoints, numTibiaPoints;
    extractBoundary(d_labels, femurLabel, width, height, d_femurX, d_femurY, numFemurPoints);
    extractBoundary(d_labels, tibiaLabel, width, height, d_tibiaX, d_tibiaY, numTibiaPoints);

    // Calculate average distance
    float avgThickness = calculateBoundaryDistance(d_femurX, d_femurY, numFemurPoints,
                                                   d_tibiaX, d_tibiaY, numTibiaPoints,
                                                   d_distances);

    // Cleanup
    CUDA_CHECK(cudaFree(d_femurX));
    CUDA_CHECK(cudaFree(d_femurY));
    CUDA_CHECK(cudaFree(d_tibiaX));
    CUDA_CHECK(cudaFree(d_tibiaY));

    return avgThickness;
}
