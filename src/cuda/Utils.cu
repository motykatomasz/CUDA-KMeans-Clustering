#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "Utils.hpp"
#include "CudaErrors.hpp"

void prepare(const std::shared_ptr<DataSet> &dataSet, const float* const hCentroids, float** dDataSet, unsigned int** dLabels, 
        float** dCentroids, bool** dUpdated, unsigned int k) {

    size_t numSamples = dataSet->samples.size();
    size_t sizeFeatureVector = dataSet->numFeatures * sizeof(float);

    float* tempDataSet, *tempCentroids;
    unsigned int* tempLabels;
    bool* tempUpdated;

    checkCudaErrors(cudaMalloc(&tempDataSet, numSamples * sizeFeatureVector));
    checkCudaErrors(cudaMalloc(&tempLabels, numSamples * sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc(&tempCentroids, k * sizeFeatureVector));
    checkCudaErrors(cudaMalloc(&tempUpdated, sizeof(bool)));


    checkCudaErrors(cudaMemcpy(tempDataSet, dataSet->samplesArray, numSamples * sizeFeatureVector, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(tempCentroids, hCentroids, k * sizeFeatureVector, cudaMemcpyHostToDevice));

    *dDataSet = tempDataSet;
    *dLabels = tempLabels;
    *dCentroids = tempCentroids;
    *dUpdated = tempUpdated;
}


void clearPointers(float* d_dataset, unsigned int* d_labels, float* d_centroids, bool* d_updated, unsigned int* h_labels, float* h_centroids, unsigned int numSamples,
        unsigned int k, unsigned int numFeatures) {

    std::cout << h_labels << std::endl;

    checkCudaErrors(cudaMemcpy(h_labels, d_labels, numSamples * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_centroids, d_centroids, k * numFeatures * sizeof(float), cudaMemcpyDeviceToHost));

    // std::cout << "Centroids after copy (1):" << std::endl;
    // for (int i=0; i < k; ++i){
    //     std::cout << "Centroid " << i << ":";
    //     for (int j=0; j < numFeatures; ++j){
    //         std::cout << h_centroids[i*numFeatures + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }


    // for (int i=0; i< numSamples; ++i){
    //     std::cout << h_labels[i] << " ";
    // }

    // std::cout << std::endl;

    checkCudaErrors(cudaFree(d_dataset));
    checkCudaErrors(cudaFree(d_labels));
    checkCudaErrors(cudaFree(d_centroids));
    checkCudaErrors(cudaFree(d_updated));
}