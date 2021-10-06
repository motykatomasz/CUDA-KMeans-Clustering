#include <utility>

#include "FeatureVec.hpp"

FeatureVec::FeatureVec(size_t num_features, float initial_value) {
  values = std::vector<float>(num_features, initial_value);
}

FeatureVec::FeatureVec(std::initializer_list<float> l) {
  values = std::vector<float>(l);
}

void FeatureVec::clear() {
  for (auto &value : values) {
    value = 0.0f;
  }
}

std::string FeatureVec::toString() {
  std::string str;
  for (size_t f = 0; f < values.size(); f++) {
    str += std::to_string(values[f]);
    str += f < values.size() - 1 ? ", " : "";
  }
  return str;
}