#pragma once

#include <random>

/**
 *  A pseudo-random number generator.
 * @tparam T The type of number to generate.
 */
template<typename T>
struct UniformRandomGenerator {
  ///  Construct a new RandomGenerator<T> with a seed.
  explicit UniformRandomGenerator(int seed = 0);

  ///  Return a new number.
  inline T next();
};

template<>
struct UniformRandomGenerator<float> {
  std::mt19937 gen;
  std::uniform_real_distribution<float> dis;

  explicit UniformRandomGenerator(int seed = 0) : gen(std::mt19937(seed)) {}

  inline float next() { return dis(gen); }
};

template<>
struct UniformRandomGenerator<long> {
  std::mt19937 gen;
  std::uniform_int_distribution<long> dis;

  explicit UniformRandomGenerator(int seed = 0) : gen(std::mt19937(seed)) {}

  inline long next() { return dis(gen); }
};