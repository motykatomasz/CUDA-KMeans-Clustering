#pragma once

#include <vector>
#include <string>
#include <cassert>
#include <cmath>

/**
*  A feature vector
*
* This structure just wraps around a std::vector<float>, but provides some additional convenience functions.
*/
struct FeatureVec {
  /// The values of this feature vector
  std::vector<float> values;
  FeatureVec() = default;

  /// Constructors
  explicit FeatureVec(std::vector<float> vec) : values(std::move(vec)) {}
  FeatureVec(size_t num_features, float initial_value);
  FeatureVec(std::initializer_list<float> l);

  /// Access the element at index \p idx
  inline float &operator[](size_t idx) { return values[idx]; }

  /// Return the size of this feature vector
  inline size_t size() const { return values.size(); }

  /// Clear this feature vector.
  void clear();

  /// Print this vector to stdout
  std::string toString();
};