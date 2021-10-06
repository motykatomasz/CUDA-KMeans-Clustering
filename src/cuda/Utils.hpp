#include "../utils/DataSet.hpp"

void prepare(const std::shared_ptr<DataSet> &data_set, const float* const h_centroids, float** d_dataset, unsigned int** d_labels, 
        float** d_centroids,  bool** d_updated, unsigned int k);

void clearPointers(float* d_dataset, unsigned int* d_labels, float* d_centroids,  bool* d_updated, unsigned int* h_labels, float* h_centroids,
        unsigned int numSamples, unsigned int k, unsigned int vectorSize);