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
#include <unordered_map>
#include <vector>

#include <boost/stacktrace.hpp>

#include <tbb/concurrent_queue.h>
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
    explicit TransformInFlightRanges(RangePool<Range>& range_pool) : range_pool_(range_pool) {}

    void start(unsigned int sid,
               unsigned int mid,
               std::uintptr_t callId,
               std::string_view signal,
               Domain& domain,
               std::string const& msg,
               Color color,
               char const* func) {
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

    void end(unsigned int sid,
             unsigned int mid,
             std::uintptr_t callId,
             std::string_view signal,
             Domain& domain,
             std::string const& msg,
             char const* func) {
      std::optional<size_t> slot;
      {
        std::lock_guard<SpinLock> guard(mutex_);
        auto key = makeKey_(sid, mid, callId, signal);
        auto found = in_flight_.find(key);
        if (found != in_flight_.end() and not found->second.empty()) {
          slot = found->second.back();
          found->second.pop_back();
          if (found->second.empty()) {
            in_flight_.erase(found);
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
    static std::string makeKey_(unsigned int sid, unsigned int mid, std::uintptr_t callId, std::string_view signal) {
      std::string key;
      key.reserve(60 + signal.size());
      key += "T";
      key += std::to_string(sid);
      key += '|';
      key += std::to_string(mid);
      key += '|';
      key += std::to_string(callId);
      key += '|';
      key.append(signal.data(), signal.size());
      return key;
    }

    SpinLock mutex_;
    RangePool<Range>& range_pool_;
    std::unordered_map<std::string, std::vector<size_t>> in_flight_;
  };

  template <typename Backend, typename Range, typename Domain>
  class GlobalESInFlightRanges {
  public:
    explicit GlobalESInFlightRanges(RangePool<Range>& range_pool) : range_pool_(range_pool) {}

    void start(unsigned int mid,
               std::string_view record,
               std::string_view signal,
               std::string_view label,
               std::string_view type,
               std::size_t pid,
               edm::ESModuleCallingContext::State const& state,
               std::uintptr_t callId,
               Domain& domain,
               std::string const& msg,
               Color color,
               char const* func) {
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
                         " record=" + std::string(record) + " signal=" + std::string(signal) + " label=" +
                         std::string(label) + " type=" + std::string(type) + " pid=" + pidString_(pid) +
                         " callId=" + std::to_string(callId) +
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

    void end(unsigned int mid,
             std::string_view record,
             std::string_view signal,
             std::string_view label,
             std::string_view type,
             std::size_t pid,
             edm::ESModuleCallingContext::State const& state,
             std::uintptr_t callId,
             Domain& domain,
             std::string const& msg,
             char const* func) {
      std::optional<size_t> slot;
      {
        std::lock_guard<SpinLock> guard(mutex_);
        auto key = makeKey_(mid, record, state, callId);
        auto found = in_flight_.find(key);
        if (found != in_flight_.end() and not found->second.empty()) {
          slot = found->second.back().slot;
          found->second.pop_back();
          if (found->second.empty()) {
            in_flight_.erase(found);
          }
        }
      }
      if (not slot.has_value()) {
        auto fullmsg = std::string("Warning: trying to end a range that is not started in ") + func + " name=" +
                       msg + " mid=" + std::to_string(mid) + " record=" + std::string(record) + " signal=" +
                       std::string(signal) + " label=" + std::string(label) + " type=" + std::string(type) +
                       " pid=" + pidString_(pid);
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

    static std::string pidString_(std::size_t pid) {
      char buffer[32] = {0};
      std::snprintf(buffer, sizeof(buffer), "0x%zx", pid);
      return buffer;
    }

    static std::string stacktraceString_() {
      std::ostringstream out;
      out << boost::stacktrace::stacktrace{};
      return out.str();
    }

    static std::string makeKey_(unsigned int mid,
                                std::string_view record,
                                edm::ESModuleCallingContext::State const& state,
                                std::uintptr_t callId) {
      std::string key;
      key.reserve(60 + record.size());
      key += "S";
      key += std::to_string(mid);
      key += '|';
      key.append(record.data(), record.size());
      key += '|';
      key += std::to_string(static_cast<int>(state));
      key += '|';
      key += std::to_string(callId);
      return key;
    }

    SpinLock mutex_;
    RangePool<Range>& range_pool_;
    std::unordered_map<std::string, std::vector<InFlightEntry>> in_flight_;
  };

};

#endif  // FWCore_Services_ProfilerServiceBase_h__