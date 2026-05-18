#include <algorithm>
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <utility>
#include <mutex>

#include <oneapi/tbb/concurrent_vector.h>

#include <fmt/printf.h>

#include <ittnotify.h>

#include "DataFormats/Common/interface/HLTPathStatus.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ESModuleCallingContext.h"
#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "FWCore/ServiceRegistry/interface/PathsAndConsumesOfModulesBase.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"
#include "FWCore/Utilities/interface/TimeOfDay.h"

#include "FWCore/Services/interface/ProfilerService.h"

namespace {

  /**
   * \brief Spinlock mutex for thread safety without returning to kernel.
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

  /**
   * \brief Backend for Intel VTune Profiler using the ITT API.
   *
   * The ITT (Instrumentation and Tracing Technology) API annotates CMSSW tasks
   * and events for display in VTune's Threading and Hotspots timelines.
   *
   * VTune overlapped tasks are identified by a unique __itt_id and may begin/end
   * on different threads, making them suitable for CMSSW's pre/post signal model.
   * The task ID is generated from the Range object's address plus a monotonic counter.
   *
   * Typical usage:
   *   vtune -collect threading -- cmsRun config.py
   *   vtune -collect hotspots -knob enable-stack-collection=true -- cmsRun config.py
   *
   * If skipFirstEvent is True, start with collection paused:
   *   vtune -collect threading -start-paused -- cmsRun config.py
   */
  class VTuneBackend {
  public:
    using Color = ProfilerServiceColor;
    class Range;
    class Domain;

    static void mark(const Domain& domain, const char* message, Color color);

    /**
     * Resume / pause ITT data collection.
     * Equivalent to cudaProfilerStart/Stop for the CUDA backend.
     */
    static void profilerStart() { __itt_resume(); }
    static void profilerStop() { __itt_pause(); }

    /**
     * \brief Proxy type satisfying the Backend::EDMService concept.
     *
     * Unlike the CUDA/ROCm backends, VTune has no external GPU service
     * dependency. This proxy always reports itself as available and enabled.
     */
    struct ServiceState {
      bool enabled() const { return true; }
    };

    class EDMService {
      ServiceState state_;

    public:
      explicit operator bool() const { return true; }
      ServiceState* operator->() { return &state_; }
    };

    class Domain {
    public:
      friend class Range;
      friend void VTuneBackend::mark(const Domain&, const char*, Color);

      Domain() = default;
      ~Domain() = default;

      void create(const std::string& name) {
        assert(domain_ == nullptr);
        domain_ = __itt_domain_create(name.c_str());
      }

      /**
       * VTune ITT domains cannot be destroyed; disable instead by clearing flags.
       */
      void destroy() {
        if (domain_) {
          domain_->flags = 0;
          domain_ = nullptr;
        }
      }

      __itt_domain* nativeHandle() const { return domain_; }

    private:
      __itt_domain* domain_ = nullptr;
    };

    class Range {
    public:
      friend void VTuneBackend::mark(const Domain&, const char*, Color);

      Range() = default; 
      // copy constructor deleted
      Range(const Range&) = delete;
      /// Move copy constructor: we take a lock and move the contents
      /// We need it to resize vectors
      Range(Range&& o) noexcept {
        std::scoped_lock lock(o.mtx_);
        std::scoped_lock lock2(mtx_);
        domain_ = o.domain_;
        taskId_ = o.taskId_;
        active_ = o.active_;
        o.domain_ = nullptr;
        o.taskId_ = __itt_null;
        o.active_ = false;
      }

      ~Range() {
        std::scoped_lock lock(mtx_);
        if (active_ && domain_) {
          __itt_task_end_overlapped(domain_, taskId_);
          __itt_id_destroy(domain_, taskId_);
        }
      }

      void startColorIn(const Domain& domain, const char* message, Color /*color*/, const char* where) {
        std::scoped_lock lock(mtx_);
        if (active_ && domain_) {
          // Warn: previous task not ended before starting a new one.
          std::string fullmsg =
              fmt::sprintf("Warning: previous task not ended before starting a new one in %s for %s", where, message);
          __itt_marker(domain_, taskId_, __itt_string_handle_create(fullmsg.c_str()), __itt_marker_scope_task);
          __itt_task_end_overlapped(domain_, taskId_);
          __itt_id_destroy(domain_, taskId_);
        }
        domain_ = domain.nativeHandle();
        if (domain_) {
          taskId_ = __itt_id_make(this, itt_extra_counter_.fetch_add(1, std::memory_order_relaxed));
          __itt_id_create(domain_, taskId_);
          __itt_task_begin_overlapped(domain_, taskId_, __itt_null, __itt_string_handle_create(message));
          active_ = true;
        }
      }

      void endIn(const Domain& domain, const char* message, const char* where) {
        std::scoped_lock lock(mtx_);
        if (active_ && domain_) {
          __itt_task_end_overlapped(domain_, taskId_);
          __itt_id_destroy(domain_, taskId_);
          active_ = false;
          domain_ = nullptr;
          taskId_ = __itt_null;
        } else {
          // Warn: trying to end a task that was never started.
          if (__itt_domain* d = domain.nativeHandle()) {
            std::string fullmsg =
                fmt::sprintf("Warning: trying to end a task that is not started in %s for %s", where, message);
            __itt_marker(d, __itt_null, __itt_string_handle_create(fullmsg.c_str()), __itt_marker_scope_task);
          }
        }
      }

    private:
      __itt_domain* domain_ = nullptr;
      // The 'extra' field type is mandated by the ITT API (__itt_id uses unsigned long long).
      using Itt_Id_extra = unsigned long long;
      inline static std::atomic<Itt_Id_extra> itt_extra_counter_{0};
      static_assert(std::atomic<Itt_Id_extra>::is_always_lock_free, "Atomic counter must be lock free");
      __itt_id taskId_ = __itt_null;
      bool active_ = false;
      SpinLock mtx_;
    };

    static std::string shortName() { return "VTune"; }

    static std::string serviceComment() {
      return R"(This Service provides CMSSW-aware task annotations to Intel VTune Profiler via the ITT API.

Annotations appear in VTune's Threading and Hotspots timelines, labelled by EDM
module name and transition type (e.g. "hltPixelTracks ModuleEvent").

Typical usage:
  vtune -collect threading -- cmsRun config.py
  vtune -collect hotspots -knob enable-stack-collection=true -- cmsRun config.py

If skipFirstEvent is True, start with data collection paused:
  vtune -collect threading -start-paused -- cmsRun config.py

Notes:
  - Overlapped tasks are identified by a unique __itt_id (address + counter),
    so begin/end need not occur on the same thread.
  - TBB worker threads are automatically visible in VTune's Threading analysis.
  - Warning markers are emitted via __itt_marker when a frame is started before
    the previous one ends, or when an end is called without a matching begin.
  - Color information is not propagated (ITT API has no per-frame colour support).)";
    }
  };

  void VTuneBackend::mark(const VTuneBackend::Domain& domain, const char* message, Color /*color*/) {
    if (__itt_domain* d = domain.nativeHandle()) {
      __itt_string_handle* handle = __itt_string_handle_create(message);
      __itt_marker(d, __itt_null, handle, __itt_marker_scope_task);
    }
  }

}  // namespace

class VTuneProfilerService : public ProfilerService<VTuneBackend> {
public:
  using ProfilerService<VTuneBackend>::ProfilerService;
};

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
DEFINE_FWK_SERVICE(VTuneProfilerService);
