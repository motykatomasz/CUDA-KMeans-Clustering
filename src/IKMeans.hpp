#pragma once

class IKMeans {
   public:
      unsigned int iteration = 0;

      virtual void initialize() = 0;
      virtual void run() = 0;
      virtual void dumpLabels(std::string fileName) = 0;
      virtual void printState(std::ostream &labels_out, std::ostream &centroids_out) = 0;
};