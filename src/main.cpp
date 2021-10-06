#include <iostream>
#include <fstream>
#include <memory>
#include <getopt.h>

#include "utils/Timer.hpp"
#include "utils/DataSet.hpp"
#include "cpu/KMeans.hpp"
#include "cuda/KMeansCuda.hpp"

#include "IKMeans.hpp"

// Configuration
struct Config {
  std::string inputFile;
  std::string outputFile;

  bool generateDataset = false;
  bool plotOutputs = false;

  unsigned int numClusters = 4;
  unsigned int maxIterations = 1000;

  unsigned long numFeatures = 2;
  unsigned long numSamples = 1024;

  bool runCUDA = false;

  static void usage(char *argv[]) {
    std::cerr << "Options:\n"
                 "  -h            Show help and exit.\n"
                 "  -i <input>    Read data set from input file <input>.\n"
                 "\n"
                 "KMeans algorithm:\n"
                 "  -k K          Number of centroids.\n"
                 "  -t T          Threshold iterations.\n"
                 "\n"
                 "Device:\n"
                 "  -g            Run on GPU (true/false).\n"
                 "Benchmarking and testing:\n"
                 "  -p            Save result in CSV files for Python plotting.\n"
                 "  -d            Generate an artificial data set for debugging purposes (dataset.kmd).\n"
                 "  -f F          Number of features for artificial data set.\n"
                 "  -v V          Number of vectors for artificial data set.\n";

    std::cerr.flush();
    exit(0);
  }

  void generateDatasetFile() {
    auto ds = DataSet::random(numFeatures, numSamples);
    ds->toFile("dataset.kmd");
    std::cout<< numFeatures << std::endl;
    std::cout<< numSamples << std::endl;
  }
};

Config parseConfiguration(int argc, char *argv[]) {
    Config conf;

  if (argc == 1) {
    Config::usage(argv);
  }

  int opt;
  while ((opt = getopt(argc, argv, "hi:o:k:t:gpdf:v:")) != -1) {
    switch (opt) {

      case 'h': {
        Config::usage(argv);
        break;
      }

      case 'i': {
        conf.inputFile = std::string(optarg);
        break;
      }

      case 'k': {
        char *end;
        conf.numClusters = (unsigned int) std::strtol(optarg, &end, 10);
        break;
      }

      case 't': {
        char *end;
        conf.maxIterations = (unsigned int) std::strtol(optarg, &end, 10);
        break;
      }

      case 'f': {
        char *end;
        conf.numFeatures = (unsigned long) std::strtol(optarg, &end, 10);
        break;
      }

      case 'v': {
        char *end;
        conf.numSamples = (unsigned long) std::strtol(optarg, &end, 10);
        break;
      }

      case 'p': {
        conf.plotOutputs = true;
        break;
      }

      case 'd': {
        conf.generateDataset = true;
        break;
      }

      case 'g': {
        conf.runCUDA = true;
        break;
      }

      case '?':
        if ((optopt == 'i') || (optopt == 'o')) {
          std::cerr << "Options -i and -o require an argument." << std::endl;
          Config::usage(argv);
        }
        break;

      default: {
        std::cerr << "Some invalid option was supplied." << std::endl;
        Config::usage(argv);
        break;
      }

    }
  }
  return conf;
}

void run(Config& conf){
  if (conf.generateDataset) {
    conf.generateDatasetFile();
  }

  if (!conf.inputFile.empty()) {
    Timer t;

    // Load data
    t.start();
    auto ds = DataSet::fromFile(conf.inputFile);
    ds->toArray();
    t.stop();
    std::cout << "Loading dataset           : " << t.seconds() << " s." << std::endl;

    IKMeans* km;
    if (conf.runCUDA) {
      km = new KMeansCuda(ds, conf.numClusters, conf.maxIterations);
    } else {
      km = new KMeans(ds, conf.numClusters, conf.maxIterations);
    }

    // Initialize algorithm
    t.start();
    km->initialize();
    t.stop();
    std::cout << "Initialization  : " << t.seconds() << " s." << std::endl;

    // Run algorithm
    t.start();
    km->run();
    t.stop();
    std::cout << "Reached convergence after : " << t.seconds() << " s." << std::endl;
    std::cout << "Iterations                : " << km->iteration << std::endl;


    if (conf.plotOutputs) {
      std::ofstream pointsOut("points.csv");
      std::ofstream resultOut("centroids.csv");

      if (pointsOut.good() && resultOut.good()) {
        km->printState(pointsOut, resultOut);
      } else {
        std::cerr << "Could not create CSV output files." << std::endl;
      }
    }
  } else {
    std::cerr << "No input file specified." << std::endl;
  }
}

int main(int argc, char *argv[]) {
  Config conf = parseConfiguration(argc, argv);
  run(conf);

  return 0;
}
