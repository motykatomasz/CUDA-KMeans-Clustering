#include <fstream>

#include "KMeansCuda.hpp"

KMeansCuda::KMeansCuda(const std::shared_ptr<DataSet> &dataSet,
                       unsigned int numClusters,
                       unsigned int maxIterations)
    : dataSet(dataSet),
      numClusters(numClusters),
      maxIterations(maxIterations) {

  hCentroids = new float[numClusters * dataSet->numFeatures];
  hLabels = new unsigned int[dataSet->samples.size()];
}

void KMeansCuda::selectRandomCentroids() {
  UniformRandomGenerator<long> rg;
  size_t numFeatures = dataSet->numFeatures;
  // For each cluster centroid, randomly select a feature vector as initialization.
  for (int i=0; i < numClusters; ++i) {
    long sample = rg.next() % dataSet->size();
    for (int j=0; j < numFeatures; ++j){
      hCentroids[i*numFeatures + j] = dataSet->samplesArray[sample*numFeatures + j];
    }
  }
}

void KMeansCuda::initialize() {
  selectRandomCentroids();
  prepare(dataSet, hCentroids, &dDataSet, &dLabels, &dCentroids, &dUpdated, numClusters);
  std::cout << &dDataSet << std::endl;
}

void KMeansCuda::clear() {
  clearPointers(dDataSet, dLabels, dCentroids, dUpdated, hLabels, hCentroids, 
    dataSet->samples.size(), numClusters, dataSet->numFeatures);
}

void KMeansCuda::iterate() {
  converged = !updateLabels(dDataSet, dLabels, dCentroids, dUpdated,
    numClusters, dataSet->numFeatures, dataSet->samples.size());

  updateCentroids(dDataSet, dLabels, dCentroids, numClusters, dataSet->numFeatures, dataSet->samples.size());

  iteration++;
}

void KMeansCuda::run() {
  while (!converged && (iteration < maxIterations)) {
    Timer t;
    t.start();
    iterate();
    t.stop();
    std::cout << iteration << " - " << t.seconds() << " s." << std::endl;
  }
  clear();
}

void KMeansCuda::printState(std::ostream &labelsOut, std::ostream &centroidsOut) {
    // Print labels for all vectors
  for (size_t i = 0; i < dataSet->size(); ++i) {
    labelsOut << i << ", " << hLabels[i] << ", " << dataSet->vector(i).toString() << std::endl;
  }

  // Print centroids
  for (size_t c = 0; c < numClusters; c++) {
    for (size_t i = 0; i < dataSet->numFeatures; ++i) {
      centroidsOut << c << ", " << hCentroids[c*(dataSet->numFeatures) + i];
    }
    centroidsOut << std::endl;
  }

}

void KMeansCuda::dumpLabels(std::string fileName) {
  std::ofstream file(fileName, std::ios::binary);

  // Write labels as uint16_t
  for (size_t i = 0; i < dataSet->size(); ++i) {
    file.write((char *) &hLabels[i], sizeof(size_t));
  }
}
