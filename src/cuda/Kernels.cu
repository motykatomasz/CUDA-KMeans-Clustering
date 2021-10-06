#include <iostream>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "Kernels.hpp"


__global__ void updateLabelsKernel(const float* const dDataset, unsigned int* dLabels, float* dCentroids, bool* dUpdated,
    const unsigned int k, const unsigned int vectorSize, const unsigned int numSamples) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int sampleIdx = x * vectorSize;

    if (x < numSamples) {

        if (x==0) {
            *dUpdated = false;
        }
        
        // First k threads in the block read centroids to shared memory
        extern __shared__ float sharedCentroids[];
        if (threadIdx.x < k) {
            for (int i = 0; i < vectorSize; ++i) {
                sharedCentroids[threadIdx.x * vectorSize + i] = dCentroids[threadIdx.x * vectorSize + i];
            }
        }

        __syncthreads();

        int label = 0;
        float minDistance = INFINITY;
        float distance = 0.0f;

        // Find the closest cluster center
        for (int c = 0; c < k; ++c) {
            distance = 0.0f;
            for (int i = 0; i < vectorSize; ++i) {
                int centroidIdx = c*vectorSize + i;
                distance +=  powf(dDataset[sampleIdx + i] - sharedCentroids[centroidIdx], 2);
            }
            
            distance = sqrtf(distance);

            if (distance < minDistance) {
                minDistance = distance;
                label = c;
            }
        }

        __syncthreads();

        // Set the updated flag if there was any change
        if (dLabels[x] != label) {
            *dUpdated = true;
            dLabels[x] = label;
        }
    }
}


__global__ void updateCentroidsKernel(const float* const dDataset, unsigned int* const dLabels, float* const dCentroids,
        const unsigned int k, const unsigned int vectorSize, const unsigned int numSamples) {

    // For now let's assume i have no more than 1024 data points.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int sampleIdx = x * vectorSize;

    // Define shared memory for aggregating data coordinated and counting in order to compute new centroid
    extern __shared__ float newCentroidsAndCounter[];

    // Split shared memory for aggregated coordinates and counter
    float* newCentroids = newCentroidsAndCounter;
    int* counter = (int*)& newCentroidsAndCounter[k*vectorSize];

    // Go through all the samples
    if (x < numSamples) {
        unsigned int label = dLabels[x];

        // Initialize k counters
        if (x < k) {
            counter[x] = 0;
        }

        // Aggragate coordinates of a sample assigned to the same centroid
        for (int i=0; i<vectorSize; i++) {
            atomicAdd(&(newCentroids[label*vectorSize + i]), dDataset[sampleIdx + i]);
        }

        // Increase the counter for appropriate centroid
        atomicAdd(&(counter[label]), 1);

        __syncthreads();

        // Calculate new centroid
        if (x < k) {
            if (counter[x] != 0) {
                for (int i=0; i<vectorSize; i++) {
                    dCentroids[sampleIdx+i] = newCentroids[sampleIdx+i] / (float)counter[x];
                }
            }
        }
    }
}


bool updateLabels(const float* const dDataSet, unsigned int* const dLabels, float* const dCentroids, bool* dUpdated,
    unsigned int k, unsigned int numFeatures, unsigned int numSamples) {

    const dim3 blockSize(1024, 1, 1);
    
    int numGrid = ceil((float)(numSamples)/1024);
    const dim3 gridSize(numGrid, 1, 1);

    updateLabelsKernel<<<gridSize, blockSize, k*numFeatures*sizeof(float)>>>(dDataSet, dLabels, dCentroids, dUpdated, k, numFeatures, numSamples);

    cudaDeviceSynchronize();
    
    // TODO The updated flag is not working
    bool updated = false;
    bool* ptr = &updated;
    checkCudaErrors(cudaMemcpy(ptr, dUpdated, sizeof(bool), cudaMemcpyDeviceToHost));

    std::cout<< updated << std::endl;

    return updated;
}

void printIntermediateLabels(unsigned int* dLabels, unsigned int numSamples) {

    unsigned int* interLabels = new unsigned int[numSamples];

    checkCudaErrors(cudaMemcpy(interLabels, dLabels, numSamples * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    for (int i=0; i< numSamples; ++i){
        std::cout << interLabels[i] << " ";
    }

    std::cout << std::endl;
}

void updateCentroids(const float* dDataSet, unsigned int* dLabels, float* dCentroids,
    unsigned int k, unsigned int numFeatures, unsigned int numSamples) {

    const dim3 blockSize(1024, 1, 1);

    updateCentroidsKernel<<<1, blockSize, k*numFeatures*sizeof(float) + k*sizeof(int)>>>(dDataSet, dLabels, dCentroids, k, numFeatures, numSamples);

    cudaDeviceSynchronize();
}