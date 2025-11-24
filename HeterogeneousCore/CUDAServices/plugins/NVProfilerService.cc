#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
// For scoped locks
#include <mutex>

#include <oneapi/tbb/concurrent_vector.h>

#include <fmt/printf.h>

#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvtx3.hpp>

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
#include "HeterogeneousCore/CUDAServices/interface/CUDAInterface.h"

using namespace std::string_literals;

namespace {
  nvtxRangeId_t nvtxDomainRangeStartColor(nvtxDomainHandle_t domain, const char* message, uint32_t color) {
    nvtxEventAttributes_t eventAttrib = {};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = color;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = message;
    return nvtxDomainRangeStartEx(domain, &eventAttrib);
  }

  __attribute__((unused)) void nvtxDomainMarkColor(nvtxDomainHandle_t domain, const char* message, uint32_t color) {
    nvtxEventAttributes_t eventAttrib = {};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = color;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = message;
    nvtxDomainMarkEx(domain, &eventAttrib);
  }

  enum {
    nvtxBlack = 0x00000000,
    nvtxRed = 0x00ff0000,
    nvtxDarkGreen = 0x00009900,
    nvtxGreen = 0x0000ff00,
    nvtxLightGreen = 0x00ccffcc,
    nvtxBlue = 0x000000ff,
    nvtxAmber = 0x00ffbf00,
    nvtxLightAmber = 0x00fff2cc,
    nvtxWhite = 0x00ffffff
  };

  constexpr nvtxRangeId_t nvtxInvalidRangeId = ~0ul;
  constexpr nvtxDomainHandle_t nvtxInvalidDomainId = nullptr;
  
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
   * \brief RAII helper class for unique NVTX ranges within a runtime defined domain.
   * (nvtx3::unique_range cannot be used as it requires a compile-time known domain)
   * 
   * Upon construction, does nmothing. Upon destruction, ends the range if set.
   * Also ends the range automatically and adds a mark if the previous range was not ended.
   * otherwise, domainStartRangeColor() starts the range, domainRangeEnd() ends the range.
   */
  class unique_range_in {
  public:
    unique_range_in() = default;
    // copy constructor deleted
    unique_range_in(const unique_range_in&) = delete;
    /// Move copy constructor: we take a lock and move the contents
    /// We need it to resize vectors of unique_range_in
    unique_range_in(unique_range_in&& o) noexcept {
      std::scoped_lock lock(o.mtx_);
      std::scoped_lock lock2(mtx_);
      domain_ = o.domain_;
      range_ = o.range_;
      o.domain_ = nvtxInvalidDomainId;
      o.range_ = nvtxInvalidRangeId;
    }
    ~unique_range_in() { 
      std::scoped_lock lock(mtx_); 
      if (range_ != nvtxInvalidRangeId) nvtxDomainRangeEnd(domain_, range_); 
    }

    void startColorIn(nvtxDomainHandle_t domain, const char* message, uint32_t color, const char * where) {
      std::scoped_lock lock(mtx_);
      if (range_ != nvtxInvalidRangeId) {
        std::string fullmsg = fmt::sprintf("Warning: previous range not ended before starting a new one in %s for %s", where, message);
        nvtxDomainMarkColor(domain_, fullmsg.c_str(), nvtxRed);
        nvtxDomainRangeEnd(domain_, range_);
      }
      domain_ = domain;
      range_ = nvtxDomainRangeStartColor(domain, message, color);
    }

    void endIn(nvtxDomainHandle_t domain, const char* message, const char * where) {
      std::scoped_lock lock(mtx_);
      if (range_ != nvtxInvalidRangeId) {
        nvtxDomainRangeEnd(domain_, range_);
        range_ = nvtxInvalidRangeId;
        domain_ = nvtxInvalidDomainId;
      } else {
        std::string fullmsg = fmt::sprintf("Warning: trying to end a range that is not started in %s for %s", where, message);
        nvtxDomainMarkColor(domain, fullmsg.c_str(), nvtxRed);
      }
    }
    private:
      nvtxRangeId_t range_ = nvtxInvalidRangeId; 
      nvtxDomainHandle_t domain_ = nvtxInvalidDomainId;
      SpinLock mtx_ = SpinLock{};
  };
}  // namespace

#define DECLARE_ES_SIGNAL_WATCHER(signal) \
  void pre##signal(edm::eventsetup::EventSetupRecordKey const& iKey, edm::ESModuleCallingContext const& mcc); \
  void post##signal(edm::eventsetup::EventSetupRecordKey const& iKey, edm::ESModuleCallingContext const& mcc);

// The global_es_modules_ vector is indexed by the ComponentDescription id_ field
#define DEFINE_ES_SIGNAL_WATCHER(signal) \
  void NVProfilerService::pre##signal(edm::eventsetup::EventSetupRecordKey const& iKey, edm::ESModuleCallingContext const& esmcc) { \
    auto mid = esmcc.componentDescription()->id_; \
    auto const& label = esmcc.componentDescription()->label_; \
    auto const& type = esmcc.componentDescription()->type_; \
    std::string msg; \
    if (label.size() == 0) { \
      /*Fallback on the type */ \
      msg = type + "(type) " + #signal ""; \
    } else { \
      msg = label + " " + #signal ""; \
    } \
    global_ES_modules_[mid].startColorIn(global_domain_, msg.c_str(), nvtxBlue, __func__); \
  } \
  void NVProfilerService::post##signal(edm::eventsetup::EventSetupRecordKey const& iKey,edm::ESModuleCallingContext const& esmcc) { \
    auto mid = esmcc.componentDescription()->id_; \
    auto const& label = esmcc.componentDescription()->label_; \
    auto const& type = esmcc.componentDescription()->type_; \
    std::string msg; \
    if (label.size() == 0) { \
      /* Fallback on the type */ \
      msg = type + "(type) " + #signal ""; \
    } else { \
      msg = label + " " + #signal ""; \
    } \
    global_ES_modules_[mid].endIn(global_domain_, msg.c_str(), __func__); \
  }

#define DECLARE_MODULE_STREAM_SIGNAL_WATCHER(signal) \
  void pre##signal(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc); \
  void post##signal(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc);

#define DEFINE_MODULE_STREAM_SIGNAL_WATCHER(signal) \
  void NVProfilerService::pre##signal(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) { \
    auto sid = sc.streamID(); \
    if (not skipFirstEvent_ or streamFirstEventDone_[sid]) { \
      auto mid = mcc.moduleDescription()->id(); \
      auto const& label = mcc.moduleDescription()->moduleLabel(); \
      auto const& msg = label + " " + #signal ""; \
      stream_modules_[sid][mid].startColorIn(stream_domain_[sid], msg.c_str(), labelColor(label) , __func__); \
    } \
  } \
  void NVProfilerService::post##signal(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) { \
    auto sid = sc.streamID(); \
    if (not skipFirstEvent_ or streamFirstEventDone_[sid]) { \
      auto mid = mcc.moduleDescription()->id(); \
      auto const& label = mcc.moduleDescription()->moduleLabel(); \
      auto const& msg = label + " " + #signal ""; \
      stream_modules_[sid][mid].endIn(stream_domain_[sid], msg.c_str(), __func__); \
    } \
  }

// This macro registers signal watchers pairs. Same for all.
#define REGISTER_SIGNAL_WATCHER(signal) \
  registry.watchPre##signal( \
      this, &NVProfilerService::pre##signal); \
  registry.watchPost##signal( \
      this, &NVProfilerService::post##signal);

class NVProfilerService {
public:
  NVProfilerService(const edm::ParameterSet&, edm::ActivityRegistry&);
  ~NVProfilerService();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void preallocate(edm::service::SystemBounds const&);

  // these signal pair are NOT guaranteed to be called by the same thread
  void preBeginJob(edm::ProcessContext const&);
  void postBeginJob();

  void lookupInitializationComplete(edm::PathsAndConsumesOfModulesBase const&, edm::ProcessContext const&);

  void preEndJob();
  void postEndJob();


/******* Global context signals  **********************************************/

  // these signal pair are NOT guaranteed to be called by the same thread
  void preGlobalBeginRun(edm::GlobalContext const&);
  void postGlobalBeginRun(edm::GlobalContext const&);

  // these signal pair are NOT guaranteed to be called by the same thread
  void preGlobalEndRun(edm::GlobalContext const&);
  void postGlobalEndRun(edm::GlobalContext const&);


/******* Stream context signals  **********************************************/

  // these signal pair are NOT guaranteed to be called by the same thread
  void preStreamBeginRun(edm::StreamContext const&);
  void postStreamBeginRun(edm::StreamContext const&);

  // these signal pair are NOT guaranteed to be called by the same thread
  void preStreamEndRun(edm::StreamContext const&);
  void postStreamEndRun(edm::StreamContext const&);

  /******** Global context lumi signals **********************************************/

  // these signal pair are NOT guaranteed to be called by the same thread
  void preGlobalBeginLumi(edm::GlobalContext const&);
  void postGlobalBeginLumi(edm::GlobalContext const&);

  // these signal pair are NOT guaranteed to be called by the same thread
  void preGlobalEndLumi(edm::GlobalContext const&);
  void postGlobalEndLumi(edm::GlobalContext const&);

  /******** Stream context lumi signals **********************************************/

  // these signal pair are NOT guaranteed to be called by the same thread
  void preStreamBeginLumi(edm::StreamContext const&);
  void postStreamBeginLumi(edm::StreamContext const&);

  // these signal pair are NOT guaranteed to be called by the same thread
  void preStreamEndLumi(edm::StreamContext const&);
  void postStreamEndLumi(edm::StreamContext const&);

  /******** Stream context events signal **********************************************/

  // these signal pair are NOT guaranteed to be called by the same thread
  void preEvent(edm::StreamContext const&);
  void postEvent(edm::StreamContext const&);

  /******** Path context event signals **********************************************/

  // these signal pair are NOT guaranteed to be called by the same thread
  void prePathEvent(edm::StreamContext const&, edm::PathContext const&);
  void postPathEvent(edm::StreamContext const&, edm::PathContext const&, edm::HLTPathStatus const&);

  /******** Module context signals *********************************************/

  // these signal pair are NOT guaranteed to be called by the same thread
  void preModuleEventPrefetching(edm::StreamContext const&, edm::ModuleCallingContext const&);
  void postModuleEventPrefetching(edm::StreamContext const&, edm::ModuleCallingContext const&);

  /******** File context signals **********************************************/

  // these signal pair are guaranteed to be called by the same thread
  void preOpenFile(std::string const&);
  void postOpenFile(std::string const&);

  // these signal pair are guaranteed to be called by the same thread
  void preCloseFile(std::string const&);
  void postCloseFile(std::string const&);

  /******** Source module context signals *************************************/

  // these signal pair are guaranteed to be called by the same thread
  void preSourceConstruction(edm::ModuleDescription const&);
  void postSourceConstruction(edm::ModuleDescription const&);

  /******** Source run context signals *************************************/

  // these signal pair are guaranteed to be called by the same thread
  void preSourceRun(edm::RunIndex);
  void postSourceRun(edm::RunIndex);

  /******** Source lumi context signals *************************************/

  // these signal pair are guaranteed to be called by the same thread
  void preSourceLumi(edm::LuminosityBlockIndex);
  void postSourceLumi(edm::LuminosityBlockIndex);

  /******** Source stream context signals *************************************/

  // these signal pair are guaranteed to be called by the same thread
  void preSourceEvent(edm::StreamID);
  void postSourceEvent(edm::StreamID);

  /******** Module no-context signals *********************************************/

  // these signal pair are guaranteed to be called by the same thread
  void preModuleConstruction(edm::ModuleDescription const&);
  void postModuleConstruction(edm::ModuleDescription const&);

  // these signal pair are guaranteed to be called by the same thread
  void preModuleDestruction(edm::ModuleDescription const&);
  void postModuleDestruction(edm::ModuleDescription const&);

  // these signal pair are guaranteed to be called by the same thread
  void preModuleBeginJob(edm::ModuleDescription const&);
  void postModuleBeginJob(edm::ModuleDescription const&);

  // these signal pair are guaranteed to be called by the same thread
  void preModuleEndJob(edm::ModuleDescription const&);
  void postModuleEndJob(edm::ModuleDescription const&);

  /******** Module global context signals *********************************************/

  // these signal pair are guaranteed to be called by the same thread
  void preModuleGlobalBeginRun(edm::GlobalContext const&, edm::ModuleCallingContext const&);
  void postModuleGlobalBeginRun(edm::GlobalContext const&, edm::ModuleCallingContext const&);

  // these signal pair are guaranteed to be called by the same thread
  void preModuleGlobalEndRun(edm::GlobalContext const&, edm::ModuleCallingContext const&);
  void postModuleGlobalEndRun(edm::GlobalContext const&, edm::ModuleCallingContext const&);

  // these signal pair are guaranteed to be called by the same thread
  void preModuleGlobalBeginLumi(edm::GlobalContext const&, edm::ModuleCallingContext const&);
  void postModuleGlobalBeginLumi(edm::GlobalContext const&, edm::ModuleCallingContext const&);

  // these signal pair are guaranteed to be called by the same thread
  void preModuleGlobalEndLumi(edm::GlobalContext const&, edm::ModuleCallingContext const&);
  void postModuleGlobalEndLumi(edm::GlobalContext const&, edm::ModuleCallingContext const&);

  /******** Module stream context signals *********************************************/

  DECLARE_MODULE_STREAM_SIGNAL_WATCHER(ModuleStreamBeginRun)
  DECLARE_MODULE_STREAM_SIGNAL_WATCHER(ModuleStreamEndRun)
  DECLARE_MODULE_STREAM_SIGNAL_WATCHER(ModuleBeginStream)
  DECLARE_MODULE_STREAM_SIGNAL_WATCHER(ModuleEndStream)
  DECLARE_MODULE_STREAM_SIGNAL_WATCHER(ModuleStreamBeginLumi)
  DECLARE_MODULE_STREAM_SIGNAL_WATCHER(ModuleStreamEndLumi)
  DECLARE_MODULE_STREAM_SIGNAL_WATCHER(ModuleEventAcquire)
  DECLARE_MODULE_STREAM_SIGNAL_WATCHER(ModuleEvent)
  DECLARE_MODULE_STREAM_SIGNAL_WATCHER(ModuleEventDelayedGet)
  DECLARE_MODULE_STREAM_SIGNAL_WATCHER(EventReadFromSource)
  DECLARE_MODULE_STREAM_SIGNAL_WATCHER(ModuleTransformPrefetching)
  DECLARE_MODULE_STREAM_SIGNAL_WATCHER(ModuleTransformAcquiring)
  DECLARE_MODULE_STREAM_SIGNAL_WATCHER(ModuleTransform)

  /******** ES module context signals *********************************************/
  // ES signal watchers
  void postESModuleRegistration(edm::eventsetup::ComponentDescription const&);
  // Prefetching is optionally watched
  // (see constructor)
  DECLARE_ES_SIGNAL_WATCHER(ESModulePrefetching)
  DECLARE_ES_SIGNAL_WATCHER(ESModule)
  DECLARE_ES_SIGNAL_WATCHER(ESModuleAcquire)

private:
  bool highlight(std::string const& label) const {
    return (std::binary_search(highlightModules_.begin(), highlightModules_.end(), label));
  }

  uint32_t labelColor(std::string const& label) const { return highlight(label) ? nvtxAmber : nvtxGreen; }

  uint32_t labelColorLight(std::string const& label) const {
    return highlight(label) ? nvtxLightAmber : nvtxLightGreen;
  }

  std::vector<std::string> highlightModules_;
  const bool showModulePrefetching_;
  const bool skipFirstEvent_;

  std::atomic<bool> globalFirstEventDone_ = false;
  std::vector<std::atomic<bool>> streamFirstEventDone_;
  unique_range_in globalRange_;                               // global event range
  std::vector<unique_range_in> event_;                        // per-stream event ranges
  std::vector<unique_range_in> source_;                       // per-stream source ranges TODO: it might be possible to merge this with event_
  std::vector<std::vector<unique_range_in>> path_;            // per-stream, per-path ranges
  std::vector<std::vector<unique_range_in>> endPath_;         // per-stream, per-endPath ranges
  std::vector<std::vector<unique_range_in>> stream_modules_;  // per-stream, per-module ranges
  std::vector<std::vector<unique_range_in>> stream_modules_acquire_;  // per-stream, per-module ranges for acquire, which can clash with produce
  // use a tbb::concurrent_vector rather than an std::vector because its final size is not known
  tbb::concurrent_vector<unique_range_in> global_modules_;  // global per-module events
  std::vector<std::vector<unique_range_in>> stream_ES_modules_;  // per-stream, per-ES-module ranges
  std::vector<std::vector<unique_range_in>> stream_ES_modules_acquire_;  // per-stream, per-ES-module ranges for acquire, which can clash with produce
  // use a tbb::concurrent_vector rather than an std::vector because its final size is not known
  tbb::concurrent_vector<unique_range_in> global_ES_modules_;  // global per-ES-module events

  nvtxDomainHandle_t global_domain_;               // NVTX domain for global EDM transitions
  std::vector<nvtxDomainHandle_t> stream_domain_;  // NVTX domains for per-EDM-stream transitions
};

NVProfilerService::NVProfilerService(edm::ParameterSet const& config, edm::ActivityRegistry& registry)
    : highlightModules_(config.getUntrackedParameter<std::vector<std::string>>("highlightModules")),
      showModulePrefetching_(config.getUntrackedParameter<bool>("showModulePrefetching")),
      skipFirstEvent_(config.getUntrackedParameter<bool>("skipFirstEvent")) {
  // make sure that CUDA is initialised, and that the CUDAInterface destructor is called after this service's destructor
  edm::Service<CUDAInterface> cuda;
  if (not cuda or not cuda->enabled())
    return;

  std::sort(highlightModules_.begin(), highlightModules_.end());

  // create the NVTX domain for global EDM transitions
  global_domain_ = nvtxDomainCreate("EDM Global");

  // enables profile collection; if profiling is already enabled it has no effect
  if (not skipFirstEvent_) {
    cudaProfilerStart();
  }

  registry.watchPreallocate(this, &NVProfilerService::preallocate);

  // these signal pair are NOT guaranteed to be called by the same thread
  registry.watchPreBeginJob(this, &NVProfilerService::preBeginJob);
  registry.watchPostBeginJob(this, &NVProfilerService::postBeginJob);

  registry.watchLookupInitializationComplete(this, &NVProfilerService::lookupInitializationComplete);

  registry.watchPreEndJob(this, &NVProfilerService::preEndJob);
  registry.watchPostEndJob(this, &NVProfilerService::postEndJob);

  // these signal pair are NOT guaranteed to be called by the same thread
  registry.watchPreGlobalBeginRun(this, &NVProfilerService::preGlobalBeginRun);
  registry.watchPostGlobalBeginRun(this, &NVProfilerService::postGlobalBeginRun);

  // these signal pair are NOT guaranteed to be called by the same thread
  registry.watchPreGlobalEndRun(this, &NVProfilerService::preGlobalEndRun);
  registry.watchPostGlobalEndRun(this, &NVProfilerService::postGlobalEndRun);

  // these signal pair are NOT guaranteed to be called by the same thread
  registry.watchPreStreamBeginRun(this, &NVProfilerService::preStreamBeginRun);
  registry.watchPostStreamBeginRun(this, &NVProfilerService::postStreamBeginRun);

  // these signal pair are NOT guaranteed to be called by the same thread
  registry.watchPreStreamEndRun(this, &NVProfilerService::preStreamEndRun);
  registry.watchPostStreamEndRun(this, &NVProfilerService::postStreamEndRun);

  // these signal pair are NOT guaranteed to be called by the same thread
  registry.watchPreGlobalBeginLumi(this, &NVProfilerService::preGlobalBeginLumi);
  registry.watchPostGlobalBeginLumi(this, &NVProfilerService::postGlobalBeginLumi);

  // these signal pair are NOT guaranteed to be called by the same thread
  registry.watchPreGlobalEndLumi(this, &NVProfilerService::preGlobalEndLumi);
  registry.watchPostGlobalEndLumi(this, &NVProfilerService::postGlobalEndLumi);

  // these signal pair are NOT guaranteed to be called by the same thread
  registry.watchPreStreamBeginLumi(this, &NVProfilerService::preStreamBeginLumi);
  registry.watchPostStreamBeginLumi(this, &NVProfilerService::postStreamBeginLumi);

  // these signal pair are NOT guaranteed to be called by the same thread
  registry.watchPreStreamEndLumi(this, &NVProfilerService::preStreamEndLumi);
  registry.watchPostStreamEndLumi(this, &NVProfilerService::postStreamEndLumi);

  // these signal pair are NOT guaranteed to be called by the same thread
  registry.watchPreEvent(this, &NVProfilerService::preEvent);
  registry.watchPostEvent(this, &NVProfilerService::postEvent);

  // these signal pair are NOT guaranteed to be called by the same thread
  registry.watchPrePathEvent(this, &NVProfilerService::prePathEvent);
  registry.watchPostPathEvent(this, &NVProfilerService::postPathEvent);

  if (showModulePrefetching_) {
    // these signal pair are NOT guaranteed to be called by the same thread
    registry.watchPreModuleEventPrefetching(this, &NVProfilerService::preModuleEventPrefetching);
    registry.watchPostModuleEventPrefetching(this, &NVProfilerService::postModuleEventPrefetching);
  }

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreOpenFile(this, &NVProfilerService::preOpenFile);
  registry.watchPostOpenFile(this, &NVProfilerService::postOpenFile);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreCloseFile(this, &NVProfilerService::preCloseFile);
  registry.watchPostCloseFile(this, &NVProfilerService::postCloseFile);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreSourceConstruction(this, &NVProfilerService::preSourceConstruction);
  registry.watchPostSourceConstruction(this, &NVProfilerService::postSourceConstruction);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreSourceRun(this, &NVProfilerService::preSourceRun);
  registry.watchPostSourceRun(this, &NVProfilerService::postSourceRun);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreSourceLumi(this, &NVProfilerService::preSourceLumi);
  registry.watchPostSourceLumi(this, &NVProfilerService::postSourceLumi);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreSourceEvent(this, &NVProfilerService::preSourceEvent);
  registry.watchPostSourceEvent(this, &NVProfilerService::postSourceEvent);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreModuleConstruction(this, &NVProfilerService::preModuleConstruction);
  registry.watchPostModuleConstruction(this, &NVProfilerService::postModuleConstruction);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreModuleDestruction(this, &NVProfilerService::preModuleDestruction);
  registry.watchPostModuleDestruction(this, &NVProfilerService::postModuleDestruction);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreModuleGlobalBeginRun(this, &NVProfilerService::preModuleGlobalBeginRun);
  registry.watchPostModuleGlobalBeginRun(this, &NVProfilerService::postModuleGlobalBeginRun);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreModuleGlobalEndRun(this, &NVProfilerService::preModuleGlobalEndRun);
  registry.watchPostModuleGlobalEndRun(this, &NVProfilerService::postModuleGlobalEndRun);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreModuleGlobalBeginLumi(this, &NVProfilerService::preModuleGlobalBeginLumi);
  registry.watchPostModuleGlobalBeginLumi(this, &NVProfilerService::postModuleGlobalBeginLumi);

  // these signal pair are guaranteed to be called by the same thread
  registry.watchPreModuleGlobalEndLumi(this, &NVProfilerService::preModuleGlobalEndLumi);
  registry.watchPostModuleGlobalEndLumi(this, &NVProfilerService::postModuleGlobalEndLumi);

  /******** Module stream context signals *********************************************/

  REGISTER_SIGNAL_WATCHER(ModuleBeginJob)
  REGISTER_SIGNAL_WATCHER(ModuleEndJob)
  REGISTER_SIGNAL_WATCHER(ModuleBeginStream)
  REGISTER_SIGNAL_WATCHER(ModuleEndStream)
  REGISTER_SIGNAL_WATCHER(ModuleStreamBeginRun)
  REGISTER_SIGNAL_WATCHER(ModuleStreamEndRun)
  REGISTER_SIGNAL_WATCHER(ModuleStreamBeginLumi)
  REGISTER_SIGNAL_WATCHER(ModuleStreamEndLumi)
  REGISTER_SIGNAL_WATCHER(ModuleEventAcquire)
  REGISTER_SIGNAL_WATCHER(ModuleEvent)
  REGISTER_SIGNAL_WATCHER(ModuleEventDelayedGet)
  REGISTER_SIGNAL_WATCHER(EventReadFromSource)
  if (showModulePrefetching_) {
    REGISTER_SIGNAL_WATCHER(ModuleTransformPrefetching)
  }
  REGISTER_SIGNAL_WATCHER(ModuleTransformAcquiring)
  REGISTER_SIGNAL_WATCHER(ModuleTransform)

  // ES signal watchers
  registry.watchPostESModuleRegistration(this, &NVProfilerService::postESModuleRegistration);
  if (showModulePrefetching_) {
    REGISTER_SIGNAL_WATCHER(ESModulePrefetching)
  }
  REGISTER_SIGNAL_WATCHER(ESModule)
  REGISTER_SIGNAL_WATCHER(ESModuleAcquire)
}

NVProfilerService::~NVProfilerService() {
  for (unsigned int sid = 0; sid < stream_domain_.size(); ++sid) {
    nvtxDomainDestroy(stream_domain_[sid]);
  }
  nvtxDomainDestroy(global_domain_);
  cudaProfilerStop();
}

void NVProfilerService::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::vector<std::string>>("highlightModules", {})->setComment("");
  desc.addUntracked<bool>("showModulePrefetching", false)
      ->setComment("Show the stack of dependencies that requested to run a module.");
  desc.addUntracked<bool>("skipFirstEvent", false)
      ->setComment(
          "Start profiling after the first event has completed.\nWith multiple streams, ignore transitions belonging "
          "to events started in parallel to the first event.\nRequires running nvprof with the '--profile-from-start "
          "off' option.");
  descriptions.add("NVProfilerService", desc);
  descriptions.setComment(R"(This Service provides CMSSW-aware annotations to nvprof/nvvm.

Notes on nvprof options:
  - the option '--profile-from-start off' should be used if skipFirstEvent is True.
  - the option '--cpu-profiling on' currently results in cmsRun being stuck at the beginning of the job.
  - the option '--cpu-thread-tracing on' is not compatible with jemalloc, and should only be used with cmsRunGlibC.)");
}

void NVProfilerService::preallocate(edm::service::SystemBounds const& bounds) {
  std::stringstream out;
  out << "preallocate: " << bounds.maxNumberOfConcurrentRuns() << " concurrent runs, "
      << bounds.maxNumberOfConcurrentLuminosityBlocks() << " luminosity sections, " << bounds.maxNumberOfStreams()
      << " streams\nrunning on " << bounds.maxNumberOfThreads() << " threads";
  nvtxDomainMarkColor(global_domain_, out.str().c_str(), nvtxAmber);

  auto concurrentStreams = bounds.maxNumberOfStreams();
  // create the NVTX domains for per-EDM-stream transitions
  stream_domain_.resize(concurrentStreams);
  for (unsigned int sid = 0; sid < concurrentStreams; ++sid) {
    stream_domain_[sid] = nvtxDomainCreate(fmt::sprintf("EDM Stream %d", sid).c_str());
  }

  event_.resize(concurrentStreams);
  path_.resize(concurrentStreams);
  endPath_.resize(concurrentStreams);
  source_.resize(concurrentStreams);
  // per stream path and end path arrays will be resized in lookupInitializationComplete()
  stream_modules_.resize(concurrentStreams);
  for (auto& modulesForOneStream : stream_modules_) {
    modulesForOneStream.resize(global_modules_.size());
  }
  stream_modules_acquire_.resize(concurrentStreams);
  for (auto& modulesForOneStream : stream_modules_acquire_) {
    modulesForOneStream.resize(global_modules_.size());
  }

  if (skipFirstEvent_) {
    globalFirstEventDone_ = false;
    std::vector<std::atomic<bool>> tmp(concurrentStreams);
    for (auto& element : tmp)
      std::atomic_init(&element, false);
    streamFirstEventDone_ = std::move(tmp);
  }
}

void NVProfilerService::preBeginJob(edm::ProcessContext const& context) {
  globalRange_.startColorIn(global_domain_, "preBeginJob", nvtxAmber, __func__);
}

void NVProfilerService::postBeginJob() {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, "postBeginJob", __func__);
  }
}

void NVProfilerService::lookupInitializationComplete(edm::PathsAndConsumesOfModulesBase const& pathsAndConsumes,
                                                     edm::ProcessContext const&) {
  nvtxDomainMarkColor(global_domain_, "lookupInitializationComplete", nvtxAmber);
  // We could potentially get all we want from pathsAndConsumes...
  assert(path_.size() > 0 and endPath_.size() > 0);
  for (auto& streamPaths : path_) {
    streamPaths.resize(pathsAndConsumes.paths().size());
  }
  for (auto& streamEndPaths : endPath_) {
    streamEndPaths.resize(pathsAndConsumes.endPaths().size());
  }
}

void NVProfilerService::preEndJob() {
  globalRange_.startColorIn(global_domain_, "EndJob", nvtxAmber, __func__);
}

void NVProfilerService::postEndJob() {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, "EndJob", __func__);
  }
}

void NVProfilerService::preSourceEvent(edm::StreamID sid) {
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    source_[sid].startColorIn(stream_domain_[sid], "source", nvtxAmber, __func__);
  }
}

void NVProfilerService::postSourceEvent(edm::StreamID sid) {
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    source_[sid].endIn(stream_domain_[sid], "source", __func__);
  }
}

void NVProfilerService::preSourceLumi(edm::LuminosityBlockIndex index) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.startColorIn(global_domain_, "source lumi", nvtxAmber, __func__ );
  }
}

void NVProfilerService::postSourceLumi(edm::LuminosityBlockIndex index) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, "source lumi", __func__);
  }
}

void NVProfilerService::preSourceRun(edm::RunIndex index) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.startColorIn(global_domain_, "source run", nvtxAmber, __func__);
  }
}

void NVProfilerService::postSourceRun(edm::RunIndex index) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, "source run", __func__);
  }
}

void NVProfilerService::preOpenFile(std::string const& lfn) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.startColorIn(global_domain_, ("open file "s + lfn).c_str(), nvtxAmber, __func__);
  }
}

void NVProfilerService::postOpenFile(std::string const& lfn) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, ("open file "s + lfn).c_str(), __func__);
  }
}

void NVProfilerService::preCloseFile(std::string const& lfn) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.startColorIn(global_domain_, ("close file "s + lfn).c_str(), nvtxAmber, __func__);
  }
}

void NVProfilerService::postCloseFile(std::string const& lfn) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, ("close file "s + lfn).c_str(), __func__);
  }
}

void NVProfilerService::preGlobalBeginRun(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.startColorIn(global_domain_, "global begin run", nvtxAmber, __func__);
  }
}

void NVProfilerService::postGlobalBeginRun(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, "global begin run", __func__);
  }
}

void NVProfilerService::preGlobalEndRun(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.startColorIn(global_domain_, "global end run", nvtxAmber, __func__);
  }
}

void NVProfilerService::postGlobalEndRun(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, "global end run", __func__);
  }
}

void NVProfilerService::preStreamBeginRun(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].startColorIn(stream_domain_[sid], "stream begin run", nvtxAmber, __func__);
  }
}

void NVProfilerService::postStreamBeginRun(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].endIn(stream_domain_[sid], "stream begin run", __func__);
  }
}

void NVProfilerService::preStreamEndRun(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].startColorIn(stream_domain_[sid], "stream end run", nvtxAmber, __func__);
  }
}

void NVProfilerService::postStreamEndRun(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].endIn(stream_domain_[sid], "stream end run", __func__);
  }
}

void NVProfilerService::preGlobalBeginLumi(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.startColorIn(global_domain_, "global begin lumi", nvtxAmber, __func__);
  }
}

void NVProfilerService::postGlobalBeginLumi(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, "global begin lumi", __func__);
  }
}

void NVProfilerService::preGlobalEndLumi(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.startColorIn(global_domain_, "global end lumi", nvtxAmber, __func__);
  }
}

void NVProfilerService::postGlobalEndLumi(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, "global end lumi", __func__);
  }
}

void NVProfilerService::preStreamBeginLumi(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].startColorIn(stream_domain_[sid], "stream begin lumi", nvtxAmber, __func__);
  }
}

  void NVProfilerService::postStreamBeginLumi(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].endIn(stream_domain_[sid], "stream begin lumi", __func__);
  }
}

void NVProfilerService::preStreamEndLumi(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].startColorIn(stream_domain_[sid], "stream end lumi", nvtxAmber, __func__);
  }
}

void NVProfilerService::postStreamEndLumi(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].endIn(stream_domain_[sid], "stream end lumi", __func__);
  }
}

void NVProfilerService::preEvent(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    std::string msg = fmt::sprintf("event run = %d event = %d", sc.eventID().run(), sc.eventID().event());
    event_[sid].startColorIn(stream_domain_[sid], "event", nvtxDarkGreen, __func__);
  }
}

void NVProfilerService::postEvent(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].endIn(stream_domain_[sid], "event", __func__);
  } else {
    streamFirstEventDone_[sid] = true;
    auto identity = [](bool x) { return x; };
    if (std::all_of(streamFirstEventDone_.begin(), streamFirstEventDone_.end(), identity)) {
      bool expected = false;
      if (globalFirstEventDone_.compare_exchange_strong(expected, true))
        cudaProfilerStart();
    }
  }
}

void NVProfilerService::prePathEvent(edm::StreamContext const& sc, edm::PathContext const& pc) {
  auto sid = sc.streamID();
  auto pid = pc.pathID();
  auto& pathOrEndPath = pc.isEndPath() ? endPath_[sid][pid] : path_[sid][pid];
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    pathOrEndPath.startColorIn(stream_domain_[sid], ("path " + pc.pathName()).c_str(), nvtxDarkGreen, __func__);
  }
}

void NVProfilerService::postPathEvent(edm::StreamContext const& sc,
                                      edm::PathContext const& pc,
                                      edm::HLTPathStatus const& hlts) {
  auto sid = sc.streamID();
  auto pid = pc.pathID();
  auto& pathOrEndPath = pc.isEndPath() ? endPath_[sid][pid] : path_[sid][pid];
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    pathOrEndPath.endIn(stream_domain_[sid], ("path " + pc.pathName()).c_str(), __func__);
  }
}

void NVProfilerService::preModuleConstruction(edm::ModuleDescription const& desc) {
  auto mid = desc.id();
  global_modules_.grow_to_at_least(mid + 1);
  std::cout << "NVProfilerService::preModuleConstruction: module id " << mid
            << ", label: " << desc.moduleLabel() << "\n";

  // This normally does nothing because stream_modules_ is empty when
  // called. But there is a rare case when a looper is used that replacement
  // modules can be constructed at end of loop. I'm not sure if that feature
  // is ever actually used but just to be safe...
  for (auto& modulesForOneStream : stream_modules_) {
    modulesForOneStream.resize(global_modules_.size());
  }

  if (not skipFirstEvent_) {
    auto const& label = desc.moduleLabel();
    auto const& msg = label + " construction";
    global_modules_[mid].startColorIn(global_domain_, msg.c_str(), labelColor(label), __func__);
  }
}

void NVProfilerService::postModuleConstruction(edm::ModuleDescription const& desc) {
  if (not skipFirstEvent_) {
    auto mid = desc.id();
    auto const& label = desc.moduleLabel();
    auto const& msg = label + " construction";
    global_modules_[mid].endIn(global_domain_, msg.c_str(), __func__);
  }
}

void NVProfilerService::preModuleDestruction(edm::ModuleDescription const& desc) {
  if (not skipFirstEvent_) {
    auto mid = desc.id();
    auto const& label = desc.moduleLabel();
    auto const& msg = label + " destruction";
    global_modules_[mid].startColorIn(global_domain_, msg.c_str(), labelColor(label), __func__);
  }
}

void NVProfilerService::postModuleDestruction(edm::ModuleDescription const& desc) {
  if (not skipFirstEvent_) {
    auto mid = desc.id();
    auto const& label = desc.moduleLabel();
    auto const& msg = label + " destruction";
    global_modules_[mid].endIn(global_domain_, msg.c_str(), __func__);
  }
}

void NVProfilerService::preModuleBeginJob(edm::ModuleDescription const& desc) {
  if (not skipFirstEvent_) {
    auto mid = desc.id();
    auto const& label = desc.moduleLabel();
    auto const& msg = label + " begin job";
    global_modules_[mid].startColorIn(global_domain_, msg.c_str(), labelColor(label), __func__);
  }
}

void NVProfilerService::postModuleBeginJob(edm::ModuleDescription const& desc) {
  if (not skipFirstEvent_) {
    auto mid = desc.id();
    auto const& label = desc.moduleLabel();
    auto const& msg = label + " begin job";
    global_modules_[mid].endIn(global_domain_, msg.c_str(), __func__);
  }
}

void NVProfilerService::preModuleEndJob(edm::ModuleDescription const& desc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = desc.id();
    auto const& label = desc.moduleLabel();
    auto const& msg = label + " end job";
    global_modules_[mid].startColorIn(global_domain_, msg.c_str(), labelColor(label), __func__);
  }
}

void NVProfilerService::postModuleEndJob(edm::ModuleDescription const& desc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = desc.id();
    auto const& label = desc.moduleLabel();
    auto const& msg = label + " end job";
    global_modules_[mid].endIn(global_domain_, msg.c_str(), __func__);
  }
}

/******** Module stream context signals *********************************************/

DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleBeginStream)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleEndStream)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleStreamBeginRun)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleStreamEndRun)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleStreamBeginLumi)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleStreamEndLumi)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleEventPrefetching)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleEventAcquire)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleEvent)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleEventDelayedGet)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(EventReadFromSource)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleTransformPrefetching)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleTransformAcquiring)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleTransform)

void NVProfilerService::preModuleGlobalBeginRun(edm::GlobalContext const& gc, edm::ModuleCallingContext const& mcc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = mcc.moduleDescription()->id();
    auto const& label = mcc.moduleDescription()->moduleLabel();
    auto const& msg = label + " global begin run";
    global_modules_[mid].startColorIn(global_domain_, msg.c_str(), labelColor(label), __func__);
  }
}

void NVProfilerService::postModuleGlobalBeginRun(edm::GlobalContext const& gc, edm::ModuleCallingContext const& mcc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = mcc.moduleDescription()->id();
    auto const& label = mcc.moduleDescription()->moduleLabel();
    auto const& msg = label + " global begin run";
    global_modules_[mid].endIn(global_domain_, "", __func__);
  }
}

void NVProfilerService::preModuleGlobalEndRun(edm::GlobalContext const& gc, edm::ModuleCallingContext const& mcc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = mcc.moduleDescription()->id();
    auto const& label = mcc.moduleDescription()->moduleLabel();
    auto const& msg = label + " global end run";
    global_modules_[mid].startColorIn(global_domain_, msg.c_str(), labelColor(label), __func__);
  }
}

void NVProfilerService::postModuleGlobalEndRun(edm::GlobalContext const& gc, edm::ModuleCallingContext const& mcc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = mcc.moduleDescription()->id();
    auto const& label = mcc.moduleDescription()->moduleLabel();
    auto const& msg = label + " global end run";
    global_modules_[mid].endIn(global_domain_, msg.c_str(), __func__);
  }
}

void NVProfilerService::preModuleGlobalBeginLumi(edm::GlobalContext const& gc, edm::ModuleCallingContext const& mcc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = mcc.moduleDescription()->id();
    auto const& label = mcc.moduleDescription()->moduleLabel();
    auto const& msg = label + " global begin lumi";
    global_modules_[mid].startColorIn(global_domain_, msg.c_str(), labelColor(label), __func__);
  }
}

void NVProfilerService::postModuleGlobalBeginLumi(edm::GlobalContext const& gc, edm::ModuleCallingContext const& mcc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = mcc.moduleDescription()->id();
    auto const& label = mcc.moduleDescription()->moduleLabel();
    auto const& msg = label + " global begin lumi";
    global_modules_[mid].endIn(global_domain_, msg.c_str(), __func__);
  }
}

void NVProfilerService::preModuleGlobalEndLumi(edm::GlobalContext const& gc, edm::ModuleCallingContext const& mcc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = mcc.moduleDescription()->id();
    auto const& label = mcc.moduleDescription()->moduleLabel();
    auto const& msg = label + " global end lumi";
    global_modules_[mid].startColorIn(global_domain_, msg.c_str(), labelColor(label), __func__);
  }
}

void NVProfilerService::postModuleGlobalEndLumi(edm::GlobalContext const& gc, edm::ModuleCallingContext const& mcc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = mcc.moduleDescription()->id();
    auto const& label = mcc.moduleDescription()->moduleLabel();
    auto const& msg = label + " global end lumi";
    global_modules_[mid].endIn(global_domain_, msg.c_str(), __func__);
  }
}

void NVProfilerService::preSourceConstruction(edm::ModuleDescription const& desc) {
  auto mid = desc.id();
  global_modules_.grow_to_at_least(mid + 1);

  if (not skipFirstEvent_) {
    auto const& label = desc.moduleLabel();
    auto const& msg = label + " construction";
    global_modules_[mid].startColorIn(global_domain_, msg.c_str(), labelColor(label), __func__);
  }
}

void NVProfilerService::postSourceConstruction(edm::ModuleDescription const& desc) {
  if (not skipFirstEvent_) {
    auto mid = desc.id();
    auto const& label = desc.moduleLabel();
    auto const& msg = label + " construction";
    global_modules_[mid].endIn(global_domain_, msg.c_str(), __func__);
  }
}

void NVProfilerService::postESModuleRegistration(edm::eventsetup::ComponentDescription const& componentDescription) {
  auto mid = componentDescription.id_;
  auto const& label = componentDescription.label_;
  auto const& msg = label + " " + "ESModuleReRegistration";
  global_ES_modules_.grow_to_at_least(mid + 1);
  nvtxDomainMarkColor(global_domain_, msg.c_str(), nvtxAmber);
}

void NVProfilerService::preESModulePrefetching(edm::eventsetup::EventSetupRecordKey const& iKey, edm::ESModuleCallingContext const& esmcc) { 
  auto mid = esmcc.componentDescription()->id_; 
  auto const& label = esmcc.componentDescription()->label_;
  auto const& type = esmcc.componentDescription()->type_;
  std::string msg;
  if (label.size() == 0) {
    // Fallback on the type
    msg = type + "(type) " + "ES prefetch" " acquire";
  } else {
    msg = label + " " + "ES prefetch" " acquire";
  }
  global_ES_modules_[mid].startColorIn(global_domain_, msg.c_str(), nvtxBlue, __func__); 
} 
  
void NVProfilerService::postESModulePrefetching(edm::eventsetup::EventSetupRecordKey const& iKey,edm::ESModuleCallingContext const& esmcc) { 
  auto mid = esmcc.componentDescription()->id_;
  auto const& label = esmcc.componentDescription()->label_;
  auto const& type = esmcc.componentDescription()->type_;
  std::string msg;
  if (label.size() == 0) {
    // Fallback on the type
    msg = type + "(type) " + "ES prefetch" " acquire";
  } else {
    msg = label + " " + "ES prefetch" " acquire";
  }
  global_ES_modules_[mid].endIn(global_domain_, msg.c_str(), __func__); 
}

/*DEFINE_ES_SIGNAL_WATCHER(ESModule)*/
void NVProfilerService::preESModule(edm::eventsetup::EventSetupRecordKey const& iKey,
                                    edm::ESModuleCallingContext const& esmcc) {
  auto mid = esmcc.componentDescription()->id_;
  auto const& label = esmcc.componentDescription()->label_;
  auto const& type = esmcc.componentDescription()->type_;
  auto const& context = iKey.name();
  std::string msg = "ESModule: label = '" + label + "', type = '" + type + "', record = '" + context + "'";
  global_ES_modules_[mid].startColorIn(global_domain_, msg.c_str(), nvtxBlue, __func__);
}

void NVProfilerService::postESModule(edm::eventsetup::EventSetupRecordKey const& iKey,
                                     edm::ESModuleCallingContext const& esmcc) {
  auto mid = esmcc.componentDescription()->id_;
  auto const& label = esmcc.componentDescription()->label_;
  auto const& type = esmcc.componentDescription()->type_;
  auto const& context = iKey.name();
  std::string msg = "ESModule: label = '" + label + "', type = '" + type + "', record = '" + context + "'";
  global_ES_modules_[mid].endIn(global_domain_, msg.c_str(), __func__);
}

DEFINE_ES_SIGNAL_WATCHER(ESModuleAcquire)

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
DEFINE_FWK_SERVICE(NVProfilerService);
