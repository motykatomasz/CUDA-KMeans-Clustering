#include <fstream>
#include <iostream>

#include "DataSet.hpp"
#include "RandomGenerator.hpp"

void DataSet::addVector(std::vector<float> &f) {
  if (numFeatures == f.size()) {
    samples.emplace_back(f);
  } else {
    throw std::runtime_error("Feature vector is of different length than number of required features in DataSet.");
  }
}

void DataSet::toArray() {
  samplesArray = new float[samples.size() * numFeatures];

  for (int i=0; i < samples.size(); ++i) {
    for (int j=0; j < numFeatures; ++j) {
      samplesArray[i*numFeatures + j] = samples[i][j];
    }
  }
}

void DataSet::toFile(std::string file_name) {
  // Open the file
  std::ofstream file(file_name, std::ios::binary);

  if (file.good()) {
    // Write number of features
    file.write((char *) &numFeatures, sizeof(size_t));
    // Write size
    auto s = size();
    file.write((char *) &s, sizeof(size_t));

    // Write vectors
    for (size_t v = 0; v < size(); v++) {
      for (size_t f = 0; f < numFeatures; f++) {
        file.write((char *) &vector(v)[f], sizeof(float));
      }
    }

    std::cout << "Dataset created" << std::endl;
  } else {
    throw std::runtime_error("Could not write to file.");
  }
}

std::shared_ptr<DataSet> DataSet::fromFile(const std::string &file_name) {
  auto ds = std::make_shared<DataSet>();

  std::ifstream file(file_name, std::ios::binary);

  if (file.good()) {
    // Read num features
    file.read((char *) &ds->numFeatures, sizeof(size_t));
    size_t size;
    // Read number of vectors
    file.read((char *) &size, sizeof(size_t));
    ds->samples.reserve(size);
    for (size_t v = 0; v < size; v++) {
      std::vector<float> vec(ds->numFeatures);
      for (size_t f = 0; f < ds->numFeatures; f++) {
        file.read((char *) &vec[f], sizeof(float));
      };
      ds->addVector(vec);
    }
    return ds;
  } else {
    throw std::runtime_error("Could not load from file");
  }
}
std::shared_ptr<DataSet> DataSet::random(size_t numFeatures, size_t numSamples) {

  // Randomize feature vectors
  UniformRandomGenerator<float> rg(1);

  auto ds = std::make_shared<DataSet>(numFeatures);

  for (int i = 0; i < numSamples; i++) {
    std::vector<float> vector(ds->numFeatures);
    for (int f = 0; f < ds->numFeatures; f++) {
      vector[f] = rg.next();
    }
    ds->addVector(vector);
  }
  return ds;
}
