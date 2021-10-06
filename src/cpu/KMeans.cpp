#include <fstream>
#include "KMeans.hpp"
#include "../utils/Timer.hpp"

KMeans::KMeans(const std::shared_ptr<DataSet> &data_set,
                       unsigned int num_clusters,
                       unsigned int max_iterations)
    : data_set(data_set),
      num_clusters(num_clusters),
      max_iterations(max_iterations) {
  // Initialize all labels to 0
  labels.reserve(data_set->samples.size());
  for (size_t i = 0; i < data_set->samples.size(); i++) {
    labels.push_back(0);
  }

  // Initialize the centroids to be all zero.
  centroids = std::vector<FeatureVec>(num_clusters, FeatureVec(data_set->numFeatures, 0.0f));
}

float KMeans::calculateEuclideanDistance(FeatureVec a, FeatureVec b) {
  assert(a.size() == b.size());
  float dist = 0.0f;
  // Loop over all features
  for (size_t f = 0; f < a.size(); f++) {
    // Accumulate the squared difference
    dist += std::pow(a[f] - b[f], 2);
  }
  // Return the square root
  return std::sqrt(dist);
}

size_t KMeans::findClosestCentroidIndex(size_t vec_index) {
  float closest = INFINITY;
  size_t index = 0;

  for (size_t c = 0; c < centroids.size(); c++) {
    float dist = calculateEuclideanDistance(data_set->vector(vec_index), centroids[c]);

    if (dist < closest) {
      closest = dist;
      index = c;
    }
  }
  return index;
}

void KMeans::selectRandomCentroids() {
  UniformRandomGenerator<long> rg;
  // For each cluster centroid, randomly select a feature vector as initialization.
  std::cout << "asd" << std::endl;
  for (auto &vector : centroids) {
    vector = data_set->samples[rg.next() % data_set->size()];
  }
}

bool KMeans::updateLabels() {
  bool updated = false;
  // For each feature vector, find the current closest centroid
  for (size_t i = 0; i < data_set->size(); i++) {
    size_t closest = findClosestCentroidIndex(i);
    if (labels[i] != closest) {
      labels[i] = closest;
      updated = true;
    }
  }
  return updated;
}

void KMeans::clearCentroids() {
  for (auto &centroid : centroids) {
    centroid.clear();
  }
}

void KMeans::updateCentroids() {
  clearCentroids();

  for (size_t c = 0; c < centroids.size(); c++) {
    size_t num_assigned = 0;

    // Find all vectors that belong to this centroid label
    for (size_t v = 0; v < data_set->size(); v++) {
      if (labels[v] == c) {
        num_assigned++;
        // Accumulate the centroid feature values
        for (size_t f = 0; f < centroids[c].size(); f++) {
          centroids[c][f] += data_set->samples[v][f];
        }
      }
    }
    // Average out each feature if new assignments were made
    if (num_assigned != 0) {
      for (size_t f = 0; f < data_set->numFeatures; f++) {
        centroids[c][f] /= (float) num_assigned;
      }
    }
  }
}

void KMeans::initialize() {
  selectRandomCentroids();
  updateLabels();
}

void KMeans::iterate() {
  updateCentroids();
  converged = !updateLabels();
  iteration++;
  std::cout << iteration << std::endl;
}

void KMeans::run() {
  while (!converged && (iteration < max_iterations)) {
    Timer t;
    t.start();
    iterate();
    t.stop();
    std::cout << iteration << " - " << t.seconds() << " s." << std::endl;
  }
}

void KMeans::printState(std::ostream &labels_out, std::ostream &centroids_out) {
  // Print labels for all vectors
  for (size_t v = 0; v < data_set->size(); v++) {
    labels_out << v << ", " << labels[v] << ", " << data_set->vector(v).toString() << std::endl;
  }

  // Print centroids
  for (size_t c = 0; c < num_clusters; c++) {
    centroids_out << c << ", " << centroids[c].toString() << std::endl;
  }
}

void KMeans::dumpLabels(std::string file_name) {
  std::ofstream file(file_name, std::ios::binary);

  // Write labels as uint16_t
  for (size_t v = 0; v < data_set->size(); v++) {
    file.write((char *) &labels[v], sizeof(size_t));
  }
}
