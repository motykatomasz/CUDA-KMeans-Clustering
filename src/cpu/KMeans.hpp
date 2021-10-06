#pragma once

#include <memory>
#include <iostream>

#include "../IKMeans.hpp"

#include "../utils/DataSet.hpp"
#include "../utils/RandomGenerator.hpp"

struct KMeans : public IKMeans {

  /// The dataset.
  const std::shared_ptr<DataSet> data_set;

  /// The number of clusters.
  unsigned int num_clusters = 0;

  /// The iteration threshold.
  unsigned int max_iterations = 0;

  /// The labels of the data samples.
  std::vector<size_t> labels;

  /// The current centroids.
  std::vector<FeatureVec> centroids;

  /// Whether the algorithm has converged.
  bool converged = false;

  /// The current iteration.
  unsigned int iteration = 0;

  /**
   *  Construct a new KMeans context.
   *
   * @param data_set                The data set to work on.
   * @param num_clusters            The number of clusters to calculate centroids for.
   * @param max_iterations   Distance scaling when iterations threshold is reached.
   */
  KMeans(const std::shared_ptr<DataSet> &data_set,
             unsigned int num_clusters,
             unsigned int max_iterations);

  /**
   *  Find the centroid closest to the feature vector /p vec.
   * @param vec A feature vector.
   * @return The index of the closest centroid.
   */
  size_t findClosestCentroidIndex(size_t vec_index);

  /**
   * Calculate the Euclidean distance between feature vector A and B
   * @param a A vector
   * @param b Another vector
   * @return The Euclidean Distance
   */
  float calculateEuclideanDistance(FeatureVec a, FeatureVec b);

  /// Select the centroids to be random points in the data set.
  void selectRandomCentroids();

  /**
   *  Update the labels of the feature vectors in the data set.
   * @return Whether labels were changed. Useful to check for convergence.
   */
  bool updateLabels();

  /// Reset the centroid feature values to zero.
  void clearCentroids();

  /// Calculate the new position of the centroids according to the labels.
  void updateCentroids();

  /// Initialize the KMeans clustering algorithm
  void initialize();

  /// Run a single iteration.
  void iterate();

  ///  Run all iterations until convergence.
  void run();

  /// Print the state of the clustering algorithm. Not recommended for large data sets.
  void printState(std::ostream &labels_out = std::cout, std::ostream &centroids_out = std::cout);

  /// Dump the labels to a file.
  void dumpLabels(std::string file_name);
};