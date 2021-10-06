#pragma once

#include <chrono>
#include <sstream>
#include <iomanip>
#include <iostream>

struct Timer {
  using system_clock = std::chrono::system_clock;
  using nanoseconds = std::chrono::nanoseconds;
  using time_point = std::chrono::time_point<system_clock, nanoseconds>;
  using duration = std::chrono::duration<double>;

  Timer() = default;

  ///  Timer start point.
  time_point start_{};
  ///  Timer stop point.
  time_point stop_{};

  ///  Start the timer.
  inline void start() { start_ = std::chrono::high_resolution_clock::now(); }

  ///  Stop the timer.
  inline void stop() { stop_ = std::chrono::high_resolution_clock::now(); }

  ///  Retrieve the interval in seconds.
  double seconds() {
    duration diff = stop_ - start_;
    return diff.count();
  }
};
