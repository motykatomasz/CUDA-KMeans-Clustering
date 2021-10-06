#include "CudaErrors.hpp"

bool updateLabels(const float* dDataSet, unsigned int* dLabels, float* dCentroids, bool* dUpdated, 
    unsigned int k, unsigned int vectorSize, unsigned int numSamples);

void updateCentroids(const float* dDataSet, unsigned int* dLabels, float* dCentroids,
    unsigned int k, unsigned int vectorSize, unsigned int numSamples);

void printIntermediateLabels(unsigned int* dLabels, unsigned int numSamples);