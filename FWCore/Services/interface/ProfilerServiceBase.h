#ifndef FWCore_Services_ProfilerServiceBase_h__
#define FWCore_Services_ProfilerServiceBase_h__

/// @brief Base class for profiling services.
/// @note This class contains the undelying utility classes.
class ProfilerServiceBase {
public:

  enum class Color : std::size_t {
    Black = 0,
    Red,
    DarkGreen,
    Green,
    LightGreen,
    Blue,
    Amber,
    LightAmber,
    White
  };

  static size_t to_underlying(Color c) noexcept { return static_cast<std::size_t>(c); }
  /**
    * @brief Abstract color enumeration the derived classes can translate (or disregard).
    */
  class SpinLock {
  public:
    SpinLock() : flag_(ATOMIC_FLAG_INIT) {}

    void lock() {
      while (flag_.test_and_set(std::memory_order_acquire))
        ;
    }

    void unlock() { flag_.clear(std::memory_order_release); }

  private:
    std::atomic_flag flag_;
  };

};

#endif  // FWCore_Services_ProfilerServiceBase_h__