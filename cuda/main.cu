#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "cuda_kernels.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Read case names from CSV
std::vector<int> readCaseNames(const std::string& filename) {
    std::vector<int> cases;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return cases;
    }

    std::string line;
    std::getline(file, line); // Skip header

    while (std::getline(file, line)) {
        if (!line.empty()) {
            cases.push_back(std::stoi(line));
        }
    }

    return cases;
}

// Generate filename for a given case and slice
std::string generateFilename(const std::string& folder, int caseNum, int sliceNum) {
    std::ostringstream oss;
    oss << folder << caseNum << "_";

    if (sliceNum < 10) {
        oss << "00" << sliceNum;
    } else if (sliceNum < 100) {
        oss << "0" << sliceNum;
    } else {
        oss << sliceNum;
    }

    oss << "_pred.png";
    return oss.str();
}

// Split thickness array into 3 parts and compute mean for each
void splitThickness(const std::vector<float>& thickness, float result[3]) {
    if (thickness.empty()) {
        result[0] = result[1] = result[2] = 0.0f;
        return;
    }

    int size = thickness.size();
    int chunk = size / 3;

    // First chunk
    float sum1 = 0.0f;
    for (int i = 0; i < chunk; i++) {
        sum1 += thickness[i];
    }
    result[0] = sum1 / chunk;

    // Second chunk
    float sum2 = 0.0f;
    for (int i = chunk; i < 2 * chunk; i++) {
        sum2 += thickness[i];
    }
    result[1] = sum2 / chunk;

    // Third chunk (may be larger)
    float sum3 = 0.0f;
    int count3 = 0;
    for (int i = 2 * chunk; i < size; i++) {
        sum3 += thickness[i];
        count3++;
    }
    result[2] = sum3 / count3;
}

// Process a single image
bool processImage(const std::string& filepath, float& averageThickness,
                  unsigned char* d_image, unsigned char* d_eroded,
                  int* d_labels, float* d_distances, int* d_temp,
                  int width, int height) {

    // Read image
    cv::Mat img = cv::imread(filepath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        return false;
    }

    width = img.cols;
    height = img.rows;
    int numPixels = width * height;

    // Copy image to device
    CUDA_CHECK(cudaMemcpy(d_image, img.data, numPixels, cudaMemcpyHostToDevice));

    // Morphological erosion with 5x5 square structuring element
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    morphologicalErosion<<<gridSize, blockSize>>>(d_image, d_eroded, width, height, 5);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check if processable
    int nonZeroCount;
    bool isProcessable = checkProcessable(d_eroded, width, height, nonZeroCount);

    if (!isProcessable) {
        return false;
    }

    // Calculate average cartilage thickness
    float thickness = calculateCartilageThickness(d_eroded, d_labels, d_distances,
                                                   d_temp, width, height);

    if (thickness > 0 && thickness < 100) {
        averageThickness = thickness;
        return true;
    }

    return false;
}

int main(int argc, char** argv) {
    std::string imagesFolder = "All_Images/";
    std::string caseFile = "casenames.csv";

    if (argc > 1) {
        imagesFolder = argv[1];
    }
    if (argc > 2) {
        caseFile = argv[2];
    }

    // Read case names
    std::vector<int> cases = readCaseNames(caseFile);
    std::cout << "Loaded " << cases.size() << " cases" << std::endl;

    // Allocate GPU memory (assuming max image size 512x512)
    const int MAX_WIDTH = 512;
    const int MAX_HEIGHT = 512;
    const int MAX_PIXELS = MAX_WIDTH * MAX_HEIGHT;

    unsigned char* d_image;
    unsigned char* d_eroded;
    int* d_labels;
    float* d_distances;
    int* d_temp;

    CUDA_CHECK(cudaMalloc(&d_image, MAX_PIXELS));
    CUDA_CHECK(cudaMalloc(&d_eroded, MAX_PIXELS));
    CUDA_CHECK(cudaMalloc(&d_labels, MAX_PIXELS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_distances, MAX_PIXELS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temp, MAX_PIXELS * sizeof(int)));

    // Store results for all cases
    std::vector<std::vector<float>> allThicknesses(cases.size(), std::vector<float>(3, 0.0f));

    // Process each case
    for (size_t c = 0; c < cases.size(); c++) {
        std::vector<float> localThickness;

        std::cout << "Processing case " << (c + 1) << "/" << cases.size()
                  << " (ID: " << cases[c] << ")" << std::endl;

        // Process 160 slices per case
        for (int i = 1; i <= 160; i++) {
            std::string filepath = generateFilename(imagesFolder, cases[c], i);

            float avgThickness = 0.0f;
            if (processImage(filepath, avgThickness, d_image, d_eroded,
                           d_labels, d_distances, d_temp, MAX_WIDTH, MAX_HEIGHT)) {
                localThickness.push_back(avgThickness);
            }
        }

        // Split into 3 regions and compute means
        float result[3];
        splitThickness(localThickness, result);

        allThicknesses[c][0] = result[0];
        allThicknesses[c][1] = result[1];
        allThicknesses[c][2] = result[2];

        std::cout << "  Valid slices: " << localThickness.size()
                  << ", Thicknesses: [" << result[0] << ", "
                  << result[1] << ", " << result[2] << "]" << std::endl;
    }

    // Write results to CSV
    std::ofstream outFile("test_distance_vector.csv");
    for (size_t c = 0; c < allThicknesses.size(); c++) {
        outFile << allThicknesses[c][0] << ","
                << allThicknesses[c][1] << ","
                << allThicknesses[c][2] << std::endl;
    }
    outFile.close();

    // Cleanup
    CUDA_CHECK(cudaFree(d_image));
    CUDA_CHECK(cudaFree(d_eroded));
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_temp));

    std::cout << "Processing Completed. Output stored in 'test_distance_vector.csv'" << std::endl;

    return 0;
}
