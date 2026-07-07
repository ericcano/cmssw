#ifndef FWCore_Services_ProfilerServiceBase_h__
#define FWCore_Services_ProfilerServiceBase_h__

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <cstdio>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <boost/stacktrace.hpp>
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

  template <typename Range>
  class RangePool {
  public:
    RangePool() : next_allocation_size_(kInitialAllocationSize) { allocateUnlocked_(kInitialAllocationSize); }

    size_t acquireSlot() {
      size_t slot = 0;
      if (free_slots_.try_pop(slot)) {
        return slot;
      }

      std::lock_guard<SpinLock> guard(mutex_);
      bool got_slot = free_slots_.try_pop(slot);
      if (not got_slot) {
        allocateUnlocked_(next_allocation_size_);
        next_allocation_size_ *= 2;
        [[maybe_unused]] bool got_slot = free_slots_.try_pop(slot);
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

  template <typename Backend, typename Range, typename Domain>
  class TransformInFlightRanges {
  public:
    using Key = std::tuple<unsigned int, unsigned int, std::uintptr_t, std::string>;

    explicit TransformInFlightRanges(RangePool<Range>& range_pool) : range_pool_(range_pool) {}

    void start(Domain& domain,
           std::string const& msg,
           Color color,
           char const* func,
           unsigned int sid,
           unsigned int mid,
           std::uintptr_t callId,
           std::string_view signal) {
      size_t slot = 0;
      {
        std::lock_guard<SpinLock> guard(mutex_);
        auto key = makeKey_(sid, mid, callId, signal);
        auto found = in_flight_.find(key);
        if (found != in_flight_.end() and not found->second.empty()) {
          auto fullmsg = std::string("Warning: previous range not ended before starting a new one in ") + func +
                         " name=" + msg + " mid=" + std::to_string(mid) + " stream id=" + std::to_string(sid) +
                         " signal=" + std::string(signal);
          Backend::mark(domain, fullmsg.c_str(), Color::Red);
          std::cout << fullmsg << std::endl;
          return;
        }
        slot = range_pool_.acquireSlot();
        in_flight_[std::move(key)].push_back(slot);
      }
      range_pool_.at(slot).startColorIn(domain, msg.c_str(), color, func);
    }

    void end(Domain& domain,
         std::string const& msg,
         char const* func,
         unsigned int sid,
         unsigned int mid,
         std::uintptr_t callId,
         std::string_view signal) {
      std::optional<size_t> slot;
      {
        std::lock_guard<SpinLock> guard(mutex_);
        auto key = makeKey_(sid, mid, callId, signal);
        auto found = in_flight_.find(key);
        if (found != in_flight_.end() and not found->second.empty()) {
          slot = found->second.back();
          found->second.pop_back();
          if (found->second.empty()) {
            in_flight_.unsafe_erase(found);
          }
        }
      }
      if (not slot.has_value()) {
        auto fullmsg = std::string("Warning: trying to end a range that is not started in ") + func + " name=" +
                       msg + " mid=" + std::to_string(mid) + " stream id=" + std::to_string(sid) +
                       " signal=" + std::string(signal);
        Backend::mark(domain, fullmsg.c_str(), Color::Red);
        std::cout << fullmsg << std::endl;
        return;
      }
      range_pool_.at(*slot).endIn(domain, msg.c_str(), func);
      std::lock_guard<SpinLock> guard(mutex_);
      range_pool_.releaseSlot(*slot);
    }

  private:
    static Key makeKey_(unsigned int sid, unsigned int mid, std::uintptr_t callId, std::string_view signal) {
      return Key{sid, mid, callId, std::string(signal)};
    }

    SpinLock mutex_;
    RangePool<Range>& range_pool_;
    tbb::concurrent_unordered_map<Key, std::vector<size_t>, boost::hash<Key>> in_flight_;
  };

  template <typename Backend, typename Range, typename Domain>
  class GlobalESInFlightRanges {
  public:
    using Key = std::tuple<unsigned int, std::string, edm::ESModuleCallingContext::State, std::uintptr_t>;

    explicit GlobalESInFlightRanges(RangePool<Range>& range_pool) : range_pool_(range_pool) {}

    void start(Domain& domain,
           std::string const& msg,
           Color color,
           char const* func,
           unsigned int mid,
           std::string_view record,
           edm::ESModuleCallingContext::State const& state,
           std::uintptr_t callId) {
      size_t slot = 0;
      {
        std::lock_guard<SpinLock> guard(mutex_);
        auto key = makeKey_(mid, record, state, callId);
        auto found = in_flight_.find(key);
        if (found != in_flight_.end() and not found->second.empty()) {
          auto const& existingMsg = found->second.back().startMsg;
          auto const& existingStacktrace = found->second.back().stacktrace;
          auto fullmsg = std::string("\n\nWarning: previous range not ended before starting a new one in ") + func +
                         "\n  existing range: '" + existingMsg + "'" + "\n  existing backtrace: " +
                         existingStacktrace + "\n  new range: name=" + msg + " mid=" + std::to_string(mid) +
                         " record=" + std::string(record) + " state=" +
                         std::to_string(static_cast<int>(state)) + " callId=" + std::to_string(callId) +
                         "\n  new stacktrace: " + stacktraceString_();
          Backend::mark(domain, fullmsg.c_str(), Color::Red);
          std::cout << fullmsg << std::endl;
          return;
        }
        slot = range_pool_.acquireSlot();
        in_flight_[std::move(key)].push_back({slot, msg, stacktraceString_()});
      }
      range_pool_.at(slot).startColorIn(domain, msg.c_str(), color, func);
    }

    void end(Domain& domain,
         std::string const& msg,
         char const* func,
         unsigned int mid,
         std::string_view record,
         edm::ESModuleCallingContext::State const& state,
         std::uintptr_t callId) {
      std::optional<size_t> slot;
      {
        std::lock_guard<SpinLock> guard(mutex_);
        auto key = makeKey_(mid, record, state, callId);
        auto found = in_flight_.find(key);
        if (found != in_flight_.end() and not found->second.empty()) {
          slot = found->second.back().slot;
          found->second.pop_back();
          if (found->second.empty()) {
            in_flight_.unsafe_erase(found);
          }
        }
      }
      if (not slot.has_value()) {
        auto fullmsg = std::string("Warning: trying to end a range that is not started in ") + func + " name=" +
                       msg + " mid=" + std::to_string(mid) + " record=" + std::string(record) + " state=" +
                       std::to_string(static_cast<int>(state)) + " callId=" + std::to_string(callId);
        Backend::mark(domain, fullmsg.c_str(), Color::Red);
        std::cout << fullmsg << std::endl;
        return;
      }
      range_pool_.at(*slot).endIn(domain, msg.c_str(), func);
      std::lock_guard<SpinLock> guard(mutex_);
      range_pool_.releaseSlot(*slot);
    }

  private:
    struct InFlightEntry {
      size_t slot;
      std::string startMsg;
      std::string stacktrace;
    };

    static std::string stacktraceString_() {
      std::ostringstream out;
      out << boost::stacktrace::stacktrace{};
      return out.str();
    }

    static Key makeKey_(unsigned int mid,
                        std::string_view record,
                        edm::ESModuleCallingContext::State const& state,
                        std::uintptr_t callId) {
      return Key{mid, std::string(record), state, callId};
    }

    SpinLock mutex_;
    RangePool<Range>& range_pool_;
    tbb::concurrent_unordered_map<Key, std::vector<InFlightEntry>, boost::hash<Key>> in_flight_;
  };

};

#endif  // FWCore_Services_ProfilerServiceBase_h__