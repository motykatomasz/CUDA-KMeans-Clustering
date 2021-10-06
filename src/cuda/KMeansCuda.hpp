#pragma once

#include <memory>
#include <iostream>

#include "../IKMeans.hpp"

#include "../utils/DataSet.hpp"
#include "../utils/RandomGenerator.hpp"
#include "../utils/Timer.hpp"

#include "Utils.hpp"
#include "Kernels.hpp"

struct KMeansCuda : public IKMeans {

  /// The data set to work on.
  const std::shared_ptr<DataSet>& dataSet;

  /// The number of clusters to generate a clustering for.
  unsigned int numClusters = 0;

  /// Pointers to the variables kept on GPU.
  float *dDataSet;
  float *dCentroids, *hCentroids;
  unsigned int* dLabels, *hLabels;
  bool* dUpdated;

  /// Whether the algorithm has converged.
  bool converged = false;

  /// The iteration at which the algorithm is operating currently.
  unsigned int iteration = 0;

  /// Maximum amount of iterations.
  unsigned int maxIterations = 0;

  /**
   * Constructor
   *
   * @param dataSet                The data set to work on.
   * @param numClusters            The number of clusters to calculate centroids for.
   * @param maxIterations   Distance scaling when iterations threshold is reached.
   */
  KMeansCuda(const std::shared_ptr<DataSet> &dataSet,
             unsigned int numClusters,
             unsigned int maxIterations);


  /// Select the centroids to be random points in the data set.
  void selectRandomCentroids();

  /// Reset the centroid feature values to zero.
  void clearCentroids();

  /// Initialize the KMeans clustering algorithm
  void initialize();

  void clear();

  /// Run a single iteration.
  void iterate();

  ///  Run all iterations until convergence.
  void run();

  /// Print the state of the clustering algorithm. Not recommended for large data sets.
  void printState(std::ostream &labelsOut = std::cout, std::ostream &centroidsOut = std::cout);

  /// Dump the labels to a file.
  void dumpLabels(std::string fileName);
};