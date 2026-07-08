#ifndef FWCore_Services_ProfilerServiceBase_h__
#define FWCore_Services_ProfilerServiceBase_h__

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <cstdio>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

#include <boost/container_hash/hash.hpp>

#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_vector.h>

#include "FWCore/ServiceRegistry/interface/ESModuleCallingContext.h"

/// @brief Base class for profiling services.
/// @note This class contains the undelying utility classes.
class ProfilerServiceBase {
public:

  enum class Color : std::size_t {
    // Black, no variants
    Black = 0,
    // Red family (dark to light)
    Red_Dark2,
    Red_Dark1,
    Red,
    Red_Light1,
    Red_Light2,
    // Green family (dark to light)
    Green_Dark2,
    Green_Dark1,
    Green,
    Green_Light1,
    Green_Light2,
    // Blue family (dark to light)
    Blue_Dark2,
    Blue_Dark1,
    Blue,
    Blue_Light1,
    Blue_Light2,
    // Amber family (dark to light)
    Amber_Dark2,
    Amber_Dark1,
    Amber,
    Amber_Light1,
    Amber_Light2,
    // White, no variants
    White,
    // Grey family (dark to light)
    Grey_Dark2,
    Grey_Dark1,
    Grey,
    Grey_Light1,
    Grey_Light2,
    // Yellow family (dark to light)
    Yellow_Dark2,
    Yellow_Dark1,
    Yellow,
    Yellow_Light1,
    Yellow_Light2
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

  /// @brief Reader-writer spinlock.
  /// Compatible with std::lock_guard (exclusive) and std::shared_lock (shared).
  /// state_ == 0  : unlocked
  /// state_ == -1 : write-locked
  /// state_ >  0  : N concurrent readers
  class RWSpinLock {
  public:
    // Exclusive (write) access — use with std::lock_guard
    void lock() {
      int expected = 0;
      while (!state_.compare_exchange_weak(expected, kWriteLocked,
                                           std::memory_order_acquire,
                                           std::memory_order_relaxed)) {
        expected = 0;
      }
    }

    void unlock() { state_.store(0, std::memory_order_release); }

    // Shared (read) access — use with std::shared_lock
    void lock_shared() {
      while (true) {
        int val = state_.load(std::memory_order_relaxed);
        if (val >= 0 &&
            state_.compare_exchange_weak(val, val + 1,
                                         std::memory_order_acquire,
                                         std::memory_order_relaxed)) {
          return;
        }
      }
    }

    void unlock_shared() { state_.fetch_sub(1, std::memory_order_release); }

  private:
    static constexpr int kWriteLocked = -1;
    std::atomic<int> state_{0};
  };

  template <typename Range>
  class RangePool {
  public:
    RangePool() : next_allocation_size_(kInitialAllocationSize) { allocateUnlocked_(kInitialAllocationSize); }

    size_t acquireSlot() {
      size_t slot = 0;
      bool got_slot = free_slots_.try_pop(slot);
      while (not got_slot) {
        std::lock_guard<SpinLock> guard(mutex_);
        allocateUnlocked_(next_allocation_size_);
        next_allocation_size_ *= 2;
        got_slot = free_slots_.try_pop(slot);
      }
      return slot;
    }

    void releaseSlot(size_t slot) { free_slots_.push(slot); }

    Range& at(size_t slot) { return ranges_[slot]; }

  private:
    static constexpr size_t kInitialAllocationSize = 16;

    void allocateUnlocked_(size_t count) {
      auto const begin = ranges_.size();
      ranges_.grow_by(count);
      for (size_t index = begin; index < begin + count; ++index) {
        free_slots_.push(index);
      }
    }

    SpinLock mutex_;
    size_t next_allocation_size_;
    tbb::concurrent_vector<Range> ranges_;
    tbb::concurrent_queue<size_t> free_slots_;
  };

  template <typename Backend, typename Range, typename Domain, typename... KeyArgs>
  class InFlightRanges {
  public:
    using Key = std::tuple<std::decay_t<KeyArgs>...>;

    explicit InFlightRanges(RangePool<Range>& range_pool) : range_pool_(range_pool) {}

    void start(Domain& domain,
               std::string const& msg,
               Color color,
               char const* func,
               std::string_view signal,
               KeyArgs const&... keyArgs) {
      auto const key = makeKey_(keyArgs...);
      auto const slot = range_pool_.acquireSlot();
      auto [found, inserted] = [&]() {
        std::shared_lock<RWSpinLock> guard(mutex_);
        return in_flight_.emplace(std::piecewise_construct, std::forward_as_tuple(key), std::forward_as_tuple(slot));
      }();
      if (not inserted) {
        range_pool_.releaseSlot(slot);
        auto fullmsg = std::string("Warning: previous range not ended before starting a new one in ") + func +
                       " name=" + msg + " signal=" + std::string(signal);
        Backend::mark(domain, fullmsg.c_str(), Color::Red);
        std::cout << fullmsg << std::endl;
        return;
      }
      range_pool_.at(slot).startColorIn(domain, msg.c_str(), color, func);
    }

    void end(Domain& domain,
             std::string const& msg,
             char const* func,
             std::string_view signal,
             KeyArgs const&... keyArgs) {
      auto const key = makeKey_(keyArgs...);
      auto extracted = [&]() {
        std::lock_guard<RWSpinLock> guard(mutex_);
        return in_flight_.unsafe_extract(key);
      }();
      if (not extracted) {
        auto fullmsg = std::string("Warning: trying to end a range that is not started in ") + func + " name=" +
                       msg + " signal=" + std::string(signal);
        Backend::mark(domain, fullmsg.c_str(), Color::Red);
        std::cout << fullmsg << std::endl;
        return;
      }
      auto const slot = extracted.mapped();
      range_pool_.at(slot).endIn(domain, msg.c_str(), func);
      range_pool_.releaseSlot(slot);
    }

  private:
    static Key makeKey_(KeyArgs const&... keyArgs) { return Key{std::decay_t<KeyArgs>(keyArgs)...}; }

    RWSpinLock mutex_;
    RangePool<Range>& range_pool_;
    tbb::concurrent_unordered_map<Key, size_t, boost::hash<Key>> in_flight_;
  };

};

#endif  // FWCore_Services_ProfilerServiceBase_h__