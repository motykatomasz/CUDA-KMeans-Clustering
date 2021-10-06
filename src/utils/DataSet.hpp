#pragma once

#include <vector>
#include <memory>
#include <stdexcept>

#include "FeatureVec.hpp"

// A data set with vectors
struct DataSet {
  size_t numFeatures = 0;
  std::vector<FeatureVec> samples;
  float* samplesArray;

  /// Construct a new data set with \p num_features features in the feature vectors.
  explicit DataSet(size_t numFeatures = 1) : numFeatures(numFeatures) {};

  /**
   *      Add a feature vector to the data set.
   * @param f   The feature vector (a copy is made).
   */
  void addVector(std::vector<float> &f);

  /**
   *  Converts dataset to simple array represation (for CUDA).
   */
  void toArray();

  /// Access the vector at index \p idx
  inline FeatureVec &vector(size_t idx) { return samples[idx]; }

  /// Access the vector at index \p idx
  inline FeatureVec &operator[](size_t idx) { return vector(idx); }

  /// Return the number of feature vectors in the data set.
  inline size_t size() const { return samples.size(); }

  /// Write the DataSet to file
  void toFile(std::string fileName);

  /// Load a DataSet from a file
  static std::shared_ptr<DataSet> fromFile(const std::string &fileName);

  /// Create a random DataSet
  static std::shared_ptr<DataSet> random(size_t numFeatures, size_t numSamples);
};