#ifndef __FWCore_Services_ProfilerService_h__
#define __FWCore_Services_ProfilerService_h__

#include "FWCore/Services/interface/ProfilerServiceBase.h"

#include <atomic>
#include <iostream>
#include <mutex>
#include <optional>
#include <string_view>
#include <unordered_map>

#include <fmt/printf.h>

#include <boost/stacktrace.hpp>

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
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"
#include "FWCore/Utilities/interface/TimeOfDay.h"

/**
 * Based template class for range/mark based profiling services, targeting
 * NVidia NVTX, AMP ROCmTX, or VTune.
 */

/**
 * Helper macros to declare signal handler pairs by parameter signature.
 */
#define DECLARE_SIGNAL_WATCHER_NOARGS(signal) \
  void pre##signal();                          \
  void post##signal();

#define DECLARE_SIGNAL_WATCHER_PROCESS_CONTEXT(signal) \
  void pre##signal(edm::ProcessContext const&);        \
  void post##signal();

#define DECLARE_SIGNAL_WATCHER_SOURCE_PROCESS_BLOCK(signal)   \
  void pre##signal();                                         \
  void post##signal(std::string const&);

#define DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT(signal)                  \
  void pre##signal(edm::StreamContext const&);                        \
  void post##signal(edm::StreamContext const&);

#define DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT(signal)                  \
  void pre##signal(edm::GlobalContext const&);                        \
  void post##signal(edm::GlobalContext const&);

#define DECLARE_SIGNAL_WATCHER_STREAM_ID(signal)                     \
  void pre##signal(edm::StreamID);                                   \
  void post##signal(edm::StreamID);

#define DECLARE_SIGNAL_WATCHER_LUMIBLOCK_INDEX(signal)                \
  void pre##signal(edm::LuminosityBlockIndex);                      \
  void post##signal(edm::LuminosityBlockIndex);

#define DECLARE_SIGNAL_WATCHER_RUN_INDEX(signal)                      \
  void pre##signal(edm::RunIndex);                                  \
  void post##signal(edm::RunIndex);

#define DECLARE_SIGNAL_WATCHER_STRING(signal)                         \
  void pre##signal(std::string const&);                             \
  void post##signal(std::string const&);

#define DECLARE_SIGNAL_WATCHER_MODULE_DESCRIPTION(signal)             \
  void pre##signal(edm::ModuleDescription const&);                   \
  void post##signal(edm::ModuleDescription const&);

#define DECLARE_SIGNAL_WATCHER_COMPONENT_DESCRIPTION(signal)          \
  void pre##signal(edm::eventsetup::ComponentDescription const&);   \
  void post##signal(edm::eventsetup::ComponentDescription const&);

#define DECLARE_SIGNAL_WATCHER_IOV_SYNC_VALUE(signal)                 \
  void pre##signal(edm::IOVSyncValue const&);                       \
  void post##signal(edm::IOVSyncValue const&);

#define DECLARE_SIGNAL_WATCHER_EVENT_SETUP_RECORD_KEY_ES_MODULE_CALLING_CONTEXT(signal) \
  void pre##signal(edm::eventsetup::EventSetupRecordKey const&, edm::ESModuleCallingContext const&); \
  void post##signal(edm::eventsetup::EventSetupRecordKey const&, edm::ESModuleCallingContext const&);

#define DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_PATH_CONTEXT(signal)     \
  void pre##signal(edm::StreamContext const&, edm::PathContext const&);

#define DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_PATH_CONTEXT_HLT_STATUS(signal) \
  void post##signal(edm::StreamContext const&, edm::PathContext const&, edm::HLTPathStatus const&);

#define DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_MODULE_CALLING_CONTEXT(signal) \
  void pre##signal(edm::StreamContext const&, edm::ModuleCallingContext const&); \
  void post##signal(edm::StreamContext const&, edm::ModuleCallingContext const&);

#define DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT_MODULE_CALLING_CONTEXT(signal) \
  void pre##signal(edm::GlobalContext const&, edm::ModuleCallingContext const&); \
  void post##signal(edm::GlobalContext const&, edm::ModuleCallingContext const&);

#define DECLARE_SIGNAL_WATCHER_TERMINATION_ORIGIN_STREAM(signal) \
  void pre##signal(edm::StreamContext const&, edm::TerminationOrigin);

#define DECLARE_SIGNAL_WATCHER_TERMINATION_ORIGIN_GLOBAL(signal) \
  void pre##signal(edm::GlobalContext const&, edm::TerminationOrigin);

#define DECLARE_SIGNAL_WATCHER_TERMINATION_ORIGIN_SOURCE(signal) \
  void pre##signal(edm::TerminationOrigin);

// ES module signal ranges are keyed dynamically to avoid collisions from overlapping calls.
#define DEFINE_ES_SIGNAL_WATCHER(signal)                                                        \
  template <class Backend>                                                                      \
  void ProfilerService<Backend>::pre##signal(edm::eventsetup::EventSetupRecordKey const& iKey,  \
                                             edm::ESModuleCallingContext const& esmcc) {        \
    auto mid = esmcc.componentDescription()->id_;                                               \
    auto const& record = iKey.name();                                                           \
    auto const& label = esmcc.componentDescription()->label_;                                   \
    auto const& type = esmcc.componentDescription()->type_;                                     \
    auto const& state = esmcc.state();                                                            \
    auto const callId = esmcc.callID();                                                         \
    std::string msg;                                                                            \
    if (label.size() == 0) {                                                                    \
      /*Fallback on the type */                                                                 \
      msg = type + "(type) " + #signal " record=" + record + " state=" + (state == edm::ESModuleCallingContext::State::kRunning ? "running" : "prefetching");                                    \
    } else {                                                                                    \
      msg = label + " " + #signal " record=" + record + " state=" + (state == edm::ESModuleCallingContext::State::kRunning ? "running" : "prefetching");                                         \
    }                                                                                           \
    global_es_in_flight_ranges_.start(global_domain_, msg, Color::Blue, __func__, #signal, mid, iKey.name(), state, callId); \
  }                                                                                             \
  template <class Backend>                                                                      \
  void ProfilerService<Backend>::post##signal(edm::eventsetup::EventSetupRecordKey const& iKey, \
                                              edm::ESModuleCallingContext const& esmcc) {       \
    auto mid = esmcc.componentDescription()->id_;                                               \
    auto const& record = iKey.name();                                                           \
    auto const& label = esmcc.componentDescription()->label_;                                   \
    auto const& type = esmcc.componentDescription()->type_;                                     \
    auto const& state = esmcc.state();                                                            \
    auto const callId = esmcc.callID();                                                         \
    std::string msg;                                                                            \
    if (label.size() == 0) {                                                                    \
      /* Fallback on the type */                                                                \
      msg = type + "(type) " + #signal " record=" + record + " state=" + (state == edm::ESModuleCallingContext::State::kRunning ? "running" : "prefetching");                                    \
    } else {                                                                                    \
      msg = label + " " + #signal " record=" + record + " state=" + (state == edm::ESModuleCallingContext::State::kRunning ? "running" : "prefetching");                                         \
    }                                                                                           \
    global_es_in_flight_ranges_.end(global_domain_, msg, __func__, #signal, mid, iKey.name(), state, callId); \
  }

#define DEFINE_MODULE_STREAM_SIGNAL_WATCHER(signal, inFlightRanges)                                                \
  template <class Backend>                                                                                          \
  void ProfilerService<Backend>::pre##signal(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {  \
    auto sid = sc.streamID();                                                                                       \
    if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {                                                        \
      auto mid = mcc.moduleDescription()->id();                                                                     \
      auto const& label = mcc.moduleDescription()->moduleLabel();                                                   \
      auto const& msg = label + " " + #signal "";                                                                   \
      inFlightRanges.start(stream_domain_[sid], msg, labelColor(label), __func__, #signal, sid, mid);             \
    }                                                                                                               \
  }                                                                                                                 \
  template <class Backend>                                                                                          \
  void ProfilerService<Backend>::post##signal(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) { \
    auto sid = sc.streamID();                                                                                       \
    if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {                                                        \
      auto mid = mcc.moduleDescription()->id();                                                                     \
      auto const& label = mcc.moduleDescription()->moduleLabel();                                                   \
      auto const& msg = label + " " + #signal "";                                                                   \
      inFlightRanges.end(stream_domain_[sid], msg, __func__, #signal, sid, mid);                                   \
    }                                                                                                               \
  }

#define DEFINE_MODULE_TRANSFORM_SIGNAL_WATCHER(signal)                                                              \
  template <class Backend>                                                                                          \
  void ProfilerService<Backend>::pre##signal(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {  \
    auto sid = sc.streamID();                                                                                       \
    if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {                                                        \
      auto mid = mcc.moduleDescription()->id();                                                                     \
      auto const callId = mcc.callID();                                                                                       \
      auto const msg = transformMessage_(mcc, #signal);                                                            \
      transform_in_flight_ranges_.start(stream_domain_[sid], msg, Color::Blue, __func__, #signal, sid, mid, callId);            \
    }                                                                                                               \
  }                                                                                                                 \
  template <class Backend>                                                                                          \
  void ProfilerService<Backend>::post##signal(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) { \
    auto sid = sc.streamID();                                                                                       \
    if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {                                                        \
      auto mid = mcc.moduleDescription()->id();                                                                     \
      auto const callId = mcc.callID();                                                                                       \
      auto const msg = transformMessage_(mcc, #signal);                                                            \
      transform_in_flight_ranges_.end(stream_domain_[sid], msg, __func__, #signal, sid, mid, callId);                            \
    }                                                                                                               \
  }

// This macro registers signal watchers pairs. Same for all.
#define REGISTER_SIGNAL_WATCHER(signal)                           \
  registry.watchPre##signal(this, &ProfilerService::pre##signal); \
  registry.watchPost##signal(this, &ProfilerService::post##signal);

// Macro for global-module (GlobalContext, ModuleCallingContext) signal pairs, using global_modules_
#define DEFINE_GLOBAL_MODULE_SIGNAL_WATCHER(signal)                                                                   \
  template <class Backend>                                                                                            \
  void ProfilerService<Backend>::pre##signal(edm::GlobalContext const& gc, edm::ModuleCallingContext const& mcc) {    \
    if (not skipFirstEvent_ or globalFirstEventDone_) {                                                               \
      auto mid = mcc.moduleDescription()->id();                                                                       \
      auto const& label = mcc.moduleDescription()->moduleLabel();                                                     \
      auto const& msg = label + " " + #signal "";                                                                     \
      global_modules_[mid].startColorIn(global_domain_, msg.c_str(), labelColor(label), __func__);                    \
    }                                                                                                                 \
  }                                                                                                                   \
  template <class Backend>                                                                                            \
  void ProfilerService<Backend>::post##signal(edm::GlobalContext const& gc, edm::ModuleCallingContext const& mcc) {   \
    if (not skipFirstEvent_ or globalFirstEventDone_) {                                                               \
      auto mid = mcc.moduleDescription()->id();                                                                       \
      auto const& label = mcc.moduleDescription()->moduleLabel();                                                     \
      auto const& msg = label + " " + #signal "";                                                                     \
      global_modules_[mid].endIn(global_domain_, msg.c_str(), __func__);                                              \
    }                                                                                                                 \
  }



// Useful for starting constructs using std::string::operator+() with a litteral string.
using namespace std::string_literals;

/**
 * @brief Base class for profiling services.
 * @tparam Backend The backend implementation class.
 * The backend will have to implement the actual range/mark operations, plus
 * capture and domains management.
 * Current expected classes and functions are:
 * - Range class with:
 *   - startColorIn(domain, message, color, func)
 *   - endIn(domain, message, func)
 * - markColorIn(domain, message, color, func)
 * - Domain management class with:
 *   - domainCreate(name)
 *   - domainDestroy(domain) (maybe destructor will be enough)
 * - Start/stop the underlying EDM service.
 * - profilerStart()
 * - profilerStop() (maybe wrapped into a class with the previous function)
 */
template <typename Backend>
class ProfilerService: public ProfilerServiceBase {
public:
  using Range = typename Backend::Range;
  using Domain = typename Backend::Domain;

  ProfilerService(const edm::ParameterSet&, edm::ActivityRegistry&);
  ~ProfilerService();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  /******** Infrastructure/setup signal pairs *************************************/

  void postServicesConstruction();

  void preEventSetupModulesConstruction();
  void postEventSetupModulesConstruction();

  void preModulesAndSourceConstruction();
  void postModulesAndSourceConstruction();

  void preFinishSchedule();
  void postFinishSchedule();

  void prePrincipalsCreation();
  void postPrincipalsCreation();

  void preScheduleConsistencyCheck();
  void postScheduleConsistencyCheck();

  void preallocate(edm::service::SystemBounds const&);

  void preEventSetupConfigurationFinalized();
  void postEventSetupConfigurationFinalized();

  void eventSetupConfiguration(edm::eventsetup::ESRecordsToProductResolverIndices const&,
                               edm::ProcessContext const&);

  void preModulesInitializationFinalized();
  void postModulesInitializationFinalized();

  DECLARE_SIGNAL_WATCHER_PROCESS_CONTEXT(BeginJob)

  DECLARE_SIGNAL_WATCHER_NOARGS(EndJob)

  void lookupInitializationComplete(edm::PathsAndConsumesOfModulesBase const&, edm::ProcessContext const&);

  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT(BeginStream)
  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT(EndStream)

  void jobFailure();

  /******** Source transition signals *********************************************/

  void preSourceNextTransition();
  void postSourceNextTransition();

  /******** Source stream context signals *************************************/

  DECLARE_SIGNAL_WATCHER_STREAM_ID(SourceEvent)

  /******** Source lumi context signals *************************************/

  DECLARE_SIGNAL_WATCHER_LUMIBLOCK_INDEX(SourceLumi)

  /******** Source run context signals *************************************/

  DECLARE_SIGNAL_WATCHER_RUN_INDEX(SourceRun)

  /******** Source process block signals *************************************/

  DECLARE_SIGNAL_WATCHER_SOURCE_PROCESS_BLOCK(SourceProcessBlock)

  /******** File context signals **********************************************/

  DECLARE_SIGNAL_WATCHER_STRING(OpenFile)
  DECLARE_SIGNAL_WATCHER_STRING(CloseFile)

  /******** Output file signals **********************************************/

  DECLARE_SIGNAL_WATCHER_NOARGS(OpenOutputFiles)
  DECLARE_SIGNAL_WATCHER_NOARGS(CloseOutputFiles)

  /******** Module stream context signals *********************************************/

  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_MODULE_CALLING_CONTEXT(ModuleBeginStream)
  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_MODULE_CALLING_CONTEXT(ModuleEndStream)

  /******** Process block signals **********************************************/

  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT(BeginProcessBlock)
  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT(AccessInputProcessBlock)
  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT(EndProcessBlock)

  /******** Job-level single signals *********************************************/

  void beginProcessing();
  void endProcessing();

  /******* Global context signals  **********************************************/

  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT(GlobalBeginRun)
  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT(GlobalEndRun)

  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT(WriteProcessBlock)

  /******** Global write signals **********************************************/

  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT(GlobalWriteRun)

  /******* Stream context signals  **********************************************/

  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT(StreamBeginRun)
  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT(StreamEndRun)

  /******** Global context lumi signals **********************************************/

  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT(GlobalBeginLumi)
  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT(GlobalEndLumi)

  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT(GlobalWriteLumi)

  /******** Stream context lumi signals **********************************************/

  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT(StreamBeginLumi)
  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT(StreamEndLumi)

  /******** Stream context events signal **********************************************/

  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT(Event)
  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT(ClearEvent)

  /******** Path context event signals **********************************************/

  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_PATH_CONTEXT(PathEvent)
  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_PATH_CONTEXT_HLT_STATUS(PathEvent)

  /******** Early termination signals (Pre only, no Post) *****************************/

  DECLARE_SIGNAL_WATCHER_TERMINATION_ORIGIN_STREAM(StreamEarlyTermination)
  DECLARE_SIGNAL_WATCHER_TERMINATION_ORIGIN_GLOBAL(GlobalEarlyTermination)
  DECLARE_SIGNAL_WATCHER_TERMINATION_ORIGIN_SOURCE(SourceEarlyTermination)

  /******** ES module construction signals **********************************************/

  DECLARE_SIGNAL_WATCHER_COMPONENT_DESCRIPTION(ESModuleConstruction)

  /******** ES module context signals *********************************************/

  void postESModuleRegistration(edm::eventsetup::ComponentDescription const&);

  /******** ES IOV sync signals **********************************************/

  void esSyncIOVQueuing(edm::IOVSyncValue const&);

  DECLARE_SIGNAL_WATCHER_IOV_SYNC_VALUE(ESSyncIOV)

  // Prefetching is optionally watched
  // (see constructor)
  DECLARE_SIGNAL_WATCHER_EVENT_SETUP_RECORD_KEY_ES_MODULE_CALLING_CONTEXT(ESModulePrefetching)
  DECLARE_SIGNAL_WATCHER_EVENT_SETUP_RECORD_KEY_ES_MODULE_CALLING_CONTEXT(ESModule)
  DECLARE_SIGNAL_WATCHER_EVENT_SETUP_RECORD_KEY_ES_MODULE_CALLING_CONTEXT(ESModuleAcquire)

  /******** Module no-context signals *********************************************/

  DECLARE_SIGNAL_WATCHER_MODULE_DESCRIPTION(ModuleConstruction)
  DECLARE_SIGNAL_WATCHER_MODULE_DESCRIPTION(ModuleDestruction)
  DECLARE_SIGNAL_WATCHER_MODULE_DESCRIPTION(ModuleBeginJob)
  DECLARE_SIGNAL_WATCHER_MODULE_DESCRIPTION(ModuleEndJob)

  /******** Module context signals *********************************************/

  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_MODULE_CALLING_CONTEXT(ModuleEventPrefetching)

  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_MODULE_CALLING_CONTEXT(ModuleEvent)
  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_MODULE_CALLING_CONTEXT(ModuleEventAcquire)
  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_MODULE_CALLING_CONTEXT(ModuleTransformPrefetching)
  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_MODULE_CALLING_CONTEXT(ModuleTransform)
  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_MODULE_CALLING_CONTEXT(ModuleTransformAcquiring)
  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_MODULE_CALLING_CONTEXT(ModuleEventDelayedGet)
  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_MODULE_CALLING_CONTEXT(EventReadFromSource)

  /******** Module stream prefetching signals **********************************************/

  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_MODULE_CALLING_CONTEXT(ModuleStreamPrefetching)
  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_MODULE_CALLING_CONTEXT(ModuleStreamBeginRun)
  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_MODULE_CALLING_CONTEXT(ModuleStreamEndRun)
  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_MODULE_CALLING_CONTEXT(ModuleStreamBeginLumi)
  DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_MODULE_CALLING_CONTEXT(ModuleStreamEndLumi)

  /******** Module global/process block context signals *************************************/

  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT_MODULE_CALLING_CONTEXT(ModuleBeginProcessBlock)
  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT_MODULE_CALLING_CONTEXT(ModuleAccessInputProcessBlock)
  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT_MODULE_CALLING_CONTEXT(ModuleEndProcessBlock)

  /******** Module global prefetching and process block signals **********************************************/

  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT_MODULE_CALLING_CONTEXT(ModuleGlobalPrefetching)
  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT_MODULE_CALLING_CONTEXT(ModuleGlobalBeginRun)
  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT_MODULE_CALLING_CONTEXT(ModuleGlobalEndRun)
  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT_MODULE_CALLING_CONTEXT(ModuleGlobalBeginLumi)
  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT_MODULE_CALLING_CONTEXT(ModuleGlobalEndLumi)

  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT_MODULE_CALLING_CONTEXT(ModuleWriteProcessBlock)
  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT_MODULE_CALLING_CONTEXT(ModuleWriteRun)
  DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT_MODULE_CALLING_CONTEXT(ModuleWriteLumi)

  /******** Source module context signals *************************************/

  DECLARE_SIGNAL_WATCHER_MODULE_DESCRIPTION(SourceConstruction)

private:
  using SharedRangePool = ProfilerServiceBase::RangePool<Range>;
  using GlobalESInFlightRanges =
      ProfilerServiceBase::InFlightRanges<Backend, Range, Domain, unsigned int, std::string, edm::ESModuleCallingContext::State, std::uintptr_t>;
  using StreamModuleInFlightRanges =
      ProfilerServiceBase::InFlightRanges<Backend, Range, Domain, unsigned int, unsigned int>;
  using TransformInFlightRanges =
      ProfilerServiceBase::InFlightRanges<Backend, Range, Domain, unsigned int, unsigned int, std::uintptr_t>;

  std::string transformMessage_(edm::ModuleCallingContext const& mcc, char const* signal) const {
    auto const& label = mcc.moduleDescription()->moduleLabel();
    return label + " " + signal;
  }

  bool highlight(std::string const& label) const {
    return (std::binary_search(highlightModules_.begin(), highlightModules_.end(), label));
  }

  Color labelColor(std::string const& label) const { return highlight(label) ? Color::Amber : Color::Green; }

  Color labelColorLight(std::string const& label) const {
    return highlight(label) ? Color::Amber_Light1 : Color::Green_Light1;
  }

  std::vector<std::string> highlightModules_;
  const bool showModulePrefetching_;
  const bool skipFirstEvent_;

  std::atomic<bool> globalFirstEventDone_ = false;
  std::vector<std::atomic<bool>> streamFirstEventDone_;
  Range globalRange_;          // global event range
  std::vector<Range> event_;   // per-stream event ranges
  std::vector<Range> source_;  // per-stream source ranges TODO: it might be possible to merge this with event_
  std::vector<std::vector<Range>> path_;            // per-stream, per-path ranges
  std::vector<std::vector<Range>> endPath_;         // per-stream, per-endPath ranges
  SharedRangePool range_pool_;
  GlobalESInFlightRanges global_es_in_flight_ranges_;
  StreamModuleInFlightRanges stream_modules_in_flight_ranges_;
  StreamModuleInFlightRanges stream_modules_event_in_flight_ranges_;
  StreamModuleInFlightRanges stream_modules_event_acquire_in_flight_ranges_;
  TransformInFlightRanges transform_in_flight_ranges_;
  // use a tbb::concurrent_vector rather than an std::vector because its final size is not known
  tbb::concurrent_vector<Range> global_modules_;       // global per-module events
  std::vector<std::vector<Range>> stream_ES_modules_;  // per-stream, per-ES-module ranges
  std::vector<std::vector<Range>>
      stream_ES_modules_acquire_;  // per-stream, per-ES-module ranges for acquire, which can clash with produce
  // use a tbb::concurrent_vector rather than an std::vector because its final size is not known
  tbb::concurrent_vector<Range> global_ES_modules_;  // global per-ES-module events

  Domain global_domain_;               // NVTX domain for global EDM transitions
  std::vector<Domain> stream_domain_;  // NVTX domains for per-EDM-stream transitions
};

template <typename Backend>
ProfilerService<Backend>::ProfilerService(edm::ParameterSet const& config, edm::ActivityRegistry& registry)
    : highlightModules_(config.getUntrackedParameter<std::vector<std::string>>("highlightModules")),
      showModulePrefetching_(config.getUntrackedParameter<bool>("showModulePrefetching")),
      skipFirstEvent_(config.getUntrackedParameter<bool>("skipFirstEvent")),
      range_pool_(),
      global_es_in_flight_ranges_(range_pool_),
      stream_modules_in_flight_ranges_(range_pool_),
      stream_modules_event_in_flight_ranges_(range_pool_),
      stream_modules_event_acquire_in_flight_ranges_(range_pool_),
      transform_in_flight_ranges_(range_pool_) {
  // make sure that CUDA is initialised, and that the CUDAInterface destructor is called after this service's destructor
  typename Backend::EDMService service;
  std::cout << Backend::shortName() << "ProfilerService: initializing..." << std::endl;
  if (not service) {
    std::cout << Backend::shortName() << "ProfilerService: EDM service not available, disabling profiling service"
              << std::endl;
    return;
  }
  if (not service or not service->enabled()) {
    std::cout << Backend::shortName()
              << "ProfilerService: EDM service failed to be enabled, disabling profiling service" << std::endl;
    return;
  }
  std::cout << Backend::shortName()
            << "ProfilerService: EDM service initialized successfully. Registering watchers to EDM." << std::endl;

  std::sort(highlightModules_.begin(), highlightModules_.end());

  // create the NVTX domain for global EDM transitions
  global_domain_.create("EDM Global");

  // enables profile collection; if profiling is already enabled it has no effect
  // otherwise, make sure it is stopped.
  if (not skipFirstEvent_) {
    Backend::profilerStart();
  } else {
    Backend::profilerStop();
  }

  // Keep watcher registration order aligned with ActivityRegistry::watch* declarations.

  registry.watchPostServicesConstruction(this, &ProfilerService::postServicesConstruction);

  REGISTER_SIGNAL_WATCHER(EventSetupModulesConstruction)

  REGISTER_SIGNAL_WATCHER(ModulesAndSourceConstruction)

  REGISTER_SIGNAL_WATCHER(FinishSchedule)

  REGISTER_SIGNAL_WATCHER(PrincipalsCreation)

  REGISTER_SIGNAL_WATCHER(ScheduleConsistencyCheck)

  registry.watchPreallocate(this, &ProfilerService::preallocate);

  REGISTER_SIGNAL_WATCHER(EventSetupConfigurationFinalized)
  registry.watchEventSetupConfiguration(this, &ProfilerService::eventSetupConfiguration);

  REGISTER_SIGNAL_WATCHER(ModulesInitializationFinalized)

  // these signal pair are NOT guaranteed to be called by the same thread
  REGISTER_SIGNAL_WATCHER(BeginJob)
  REGISTER_SIGNAL_WATCHER(EndJob)

  registry.watchLookupInitializationComplete(this, &ProfilerService::lookupInitializationComplete);

  REGISTER_SIGNAL_WATCHER(BeginStream)
  REGISTER_SIGNAL_WATCHER(EndStream)

  registry.watchJobFailure(this, &ProfilerService::jobFailure);

  REGISTER_SIGNAL_WATCHER(SourceNextTransition)

  // these signal pair are guaranteed to be called by the same thread
  REGISTER_SIGNAL_WATCHER(SourceEvent)
  REGISTER_SIGNAL_WATCHER(SourceLumi)
  REGISTER_SIGNAL_WATCHER(SourceRun)
  REGISTER_SIGNAL_WATCHER(SourceProcessBlock)

  REGISTER_SIGNAL_WATCHER(OpenFile)
  REGISTER_SIGNAL_WATCHER(CloseFile)
  REGISTER_SIGNAL_WATCHER(OpenOutputFiles)
  REGISTER_SIGNAL_WATCHER(CloseOutputFiles)
  /******** Module stream context signals *********************************************/
  REGISTER_SIGNAL_WATCHER(ModuleBeginStream)
  REGISTER_SIGNAL_WATCHER(ModuleEndStream)

  // Process block signal pairs
  REGISTER_SIGNAL_WATCHER(BeginProcessBlock)
  REGISTER_SIGNAL_WATCHER(AccessInputProcessBlock)
  REGISTER_SIGNAL_WATCHER(EndProcessBlock)

  // Job-level single signals
  registry.watchBeginProcessing(this, &ProfilerService::beginProcessing);
  registry.watchEndProcessing(this, &ProfilerService::endProcessing);

  // these signal pair are NOT guaranteed to be called by the same thread
  REGISTER_SIGNAL_WATCHER(GlobalBeginRun)
  REGISTER_SIGNAL_WATCHER(GlobalEndRun)

  REGISTER_SIGNAL_WATCHER(WriteProcessBlock)

  // Global write signal pairs
  REGISTER_SIGNAL_WATCHER(GlobalWriteRun)

  // these signal pair are NOT guaranteed to be called by the same thread
  REGISTER_SIGNAL_WATCHER(StreamBeginRun)
  REGISTER_SIGNAL_WATCHER(StreamEndRun)

  // these signal pair are NOT guaranteed to be called by the same thread
  REGISTER_SIGNAL_WATCHER(GlobalBeginLumi)
  REGISTER_SIGNAL_WATCHER(GlobalEndLumi)

  REGISTER_SIGNAL_WATCHER(GlobalWriteLumi)

  REGISTER_SIGNAL_WATCHER(StreamBeginLumi)
  REGISTER_SIGNAL_WATCHER(StreamEndLumi)

  // these signal pair are NOT guaranteed to be called by the same thread
  REGISTER_SIGNAL_WATCHER(Event)

  REGISTER_SIGNAL_WATCHER(ClearEvent)

  // these signal pair are NOT guaranteed to be called by the same thread
  REGISTER_SIGNAL_WATCHER(PathEvent)

  // Early termination signals (Pre only)
  registry.watchPreStreamEarlyTermination(this, &ProfilerService::preStreamEarlyTermination);
  registry.watchPreGlobalEarlyTermination(this, &ProfilerService::preGlobalEarlyTermination);
  registry.watchPreSourceEarlyTermination(this, &ProfilerService::preSourceEarlyTermination);

  // these signal pair are guaranteed to be called by the same thread
  REGISTER_SIGNAL_WATCHER(ESModuleConstruction)

  // ES signal watchers
  registry.watchPostESModuleRegistration(this, &ProfilerService::postESModuleRegistration);

  // ES IOV sync signals
  registry.watchESSyncIOVQueuing(this, &ProfilerService::esSyncIOVQueuing);
  REGISTER_SIGNAL_WATCHER(ESSyncIOV)

  if (showModulePrefetching_) {
    REGISTER_SIGNAL_WATCHER(ESModulePrefetching)
  }
  REGISTER_SIGNAL_WATCHER(ESModule)
  REGISTER_SIGNAL_WATCHER(ESModuleAcquire)

  REGISTER_SIGNAL_WATCHER(ModuleConstruction)
  REGISTER_SIGNAL_WATCHER(ModuleDestruction)

  REGISTER_SIGNAL_WATCHER(ModuleBeginJob)
  REGISTER_SIGNAL_WATCHER(ModuleEndJob)

  if (showModulePrefetching_) {
    // these signal pair are NOT guaranteed to be called by the same thread
    REGISTER_SIGNAL_WATCHER(ModuleEventPrefetching)
  }
  REGISTER_SIGNAL_WATCHER(ModuleEvent)
  REGISTER_SIGNAL_WATCHER(ModuleEventAcquire)
  if (showModulePrefetching_) {
    REGISTER_SIGNAL_WATCHER(ModuleTransformPrefetching)
  }
  REGISTER_SIGNAL_WATCHER(ModuleTransform)
  REGISTER_SIGNAL_WATCHER(ModuleTransformAcquiring)
  REGISTER_SIGNAL_WATCHER(ModuleEventDelayedGet)
  REGISTER_SIGNAL_WATCHER(EventReadFromSource)

  // Module stream prefetching signal pair
  if (showModulePrefetching_) {
    REGISTER_SIGNAL_WATCHER(ModuleStreamPrefetching)
  }
  REGISTER_SIGNAL_WATCHER(ModuleStreamBeginRun)
  REGISTER_SIGNAL_WATCHER(ModuleStreamEndRun)
  REGISTER_SIGNAL_WATCHER(ModuleStreamBeginLumi)
  REGISTER_SIGNAL_WATCHER(ModuleStreamEndLumi)

  REGISTER_SIGNAL_WATCHER(ModuleBeginProcessBlock)
  REGISTER_SIGNAL_WATCHER(ModuleAccessInputProcessBlock)
  REGISTER_SIGNAL_WATCHER(ModuleEndProcessBlock)

  // Module global prefetching and process block signal pairs
  if (showModulePrefetching_) {
    REGISTER_SIGNAL_WATCHER(ModuleGlobalPrefetching)
  }

  // these signal pair are guaranteed to be called by the same thread
  REGISTER_SIGNAL_WATCHER(ModuleGlobalBeginRun)
  REGISTER_SIGNAL_WATCHER(ModuleGlobalEndRun)
  REGISTER_SIGNAL_WATCHER(ModuleGlobalBeginLumi)
  REGISTER_SIGNAL_WATCHER(ModuleGlobalEndLumi)

  REGISTER_SIGNAL_WATCHER(ModuleWriteProcessBlock)
  REGISTER_SIGNAL_WATCHER(ModuleWriteRun)
  REGISTER_SIGNAL_WATCHER(ModuleWriteLumi)

  // these signal pair are guaranteed to be called by the same thread
  REGISTER_SIGNAL_WATCHER(SourceConstruction)
}

template <typename Backend>
ProfilerService<Backend>::~ProfilerService() {
  for (unsigned int sid = 0; sid < stream_domain_.size(); ++sid) {
    stream_domain_[sid].destroy();
  }
  global_domain_.destroy();
  Backend::profilerStop();
}

template <typename Backend>
void ProfilerService<Backend>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::vector<std::string>>("highlightModules", {})->setComment("");
  desc.addUntracked<bool>("showModulePrefetching", false)
      ->setComment("Show the stack of dependencies that requested to run a module.");
  desc.addUntracked<bool>("skipFirstEvent", false)
      ->setComment(
          "Start profiling after the first event has completed.\nWith multiple streams, ignore transitions belonging "
          "to events started in parallel to the first event.\nRequires running nvprof with the '--profile-from-start "
          "off' option.");
  descriptions.add(Backend::shortName() + "ProfilerService", desc);
  descriptions.setComment(Backend::serviceComment());
  // For reference, here is a possible extended comment for nvprof/nvvm backends:
  //   descriptions.setComment(R"(This Service provides CMSSW-aware annotations to nvprof/nvvm.

  // Notes on nvprof options:
  //   - the option '--profile-from-start off' should be used if skipFirstEvent is True.
  //   - the option '--cpu-profiling on' currently results in cmsRun being stuck at the beginning of the job.
  //   - the option '--cpu-thread-tracing on' is not compatible with jemalloc, and should only be used with cmsRunGlibC.)");
}

template <typename Backend>
void ProfilerService<Backend>::preallocate(edm::service::SystemBounds const& bounds) {
  std::stringstream out;
  out << "preallocate: " << bounds.maxNumberOfConcurrentRuns() << " concurrent runs, "
      << bounds.maxNumberOfConcurrentLuminosityBlocks() << " luminosity sections, " << bounds.maxNumberOfStreams()
      << " streams\nrunning on " << bounds.maxNumberOfThreads() << " threads";
  Backend::mark(global_domain_, out.str().c_str(), Color::Amber);

  auto concurrentStreams = bounds.maxNumberOfStreams();
  // create the NVTX domains for per-EDM-stream transitions
  stream_domain_.resize(concurrentStreams);
  for (unsigned int sid = 0; sid < concurrentStreams; ++sid) {
    stream_domain_[sid].create(fmt::sprintf("EDM Stream %d", sid).c_str());
  }

  event_.resize(concurrentStreams);
  path_.resize(concurrentStreams);
  endPath_.resize(concurrentStreams);
  source_.resize(concurrentStreams);

  if (skipFirstEvent_) {
    globalFirstEventDone_ = false;
    std::vector<std::atomic<bool>> tmp(concurrentStreams);
    for (auto& element : tmp)
      std::atomic_init(&element, false);
    streamFirstEventDone_ = std::move(tmp);
  }
}

template <class Backend>
void ProfilerService<Backend>::postServicesConstruction() {
  Backend::mark(global_domain_, "postServicesConstruction", Color::Amber);
}

template <class Backend>
void ProfilerService<Backend>::preEventSetupConfigurationFinalized() {
  if (not skipFirstEvent_) {
    Backend::mark(global_domain_, "preEventSetupConfigurationFinalized", Color::Amber);
  }
}

template <class Backend>
void ProfilerService<Backend>::postEventSetupConfigurationFinalized() {
  if (not skipFirstEvent_) {
    Backend::mark(global_domain_, "postEventSetupConfigurationFinalized", Color::Amber);
  }
}

template <class Backend>
void ProfilerService<Backend>::eventSetupConfiguration(
    edm::eventsetup::ESRecordsToProductResolverIndices const&, edm::ProcessContext const&) {
  if (not skipFirstEvent_) {
    Backend::mark(global_domain_, "eventSetupConfiguration", Color::Amber);
  }
}

template <class Backend>
void ProfilerService<Backend>::preEventSetupModulesConstruction() {
  if (not skipFirstEvent_) {
    Backend::mark(global_domain_, "preEventSetupModulesConstruction", Color::Amber);
  }
}

template <class Backend>
void ProfilerService<Backend>::postEventSetupModulesConstruction() {
  if (not skipFirstEvent_) {
    Backend::mark(global_domain_, "postEventSetupModulesConstruction", Color::Amber);
  }
}

template <class Backend>
void ProfilerService<Backend>::preModulesAndSourceConstruction() {
  if (not skipFirstEvent_) {
    Backend::mark(global_domain_, "preModulesAndSourceConstruction", Color::Amber);
  }
}

template <class Backend>
void ProfilerService<Backend>::postModulesAndSourceConstruction() {
  if (not skipFirstEvent_) {
    Backend::mark(global_domain_, "postModulesAndSourceConstruction", Color::Amber);
  }
}

template <typename Backend>
void ProfilerService<Backend>::preBeginJob(edm::ProcessContext const& context) {
  globalRange_.startColorIn(global_domain_, "preBeginJob", Color::Amber, __func__);
}

template <typename Backend>
void ProfilerService<Backend>::postBeginJob() {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, "postBeginJob", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preEndJob() {
  globalRange_.startColorIn(global_domain_, "EndJob", Color::Amber, __func__);
}

template <class Backend>
void ProfilerService<Backend>::postEndJob() {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, "EndJob", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::lookupInitializationComplete(edm::PathsAndConsumesOfModulesBase const& pathsAndConsumes,
                                                            edm::ProcessContext const&) {
  Backend::mark(global_domain_, "lookupInitializationComplete", Color::Amber);
  // We could potentially get all we want from pathsAndConsumes...
  assert(path_.size() > 0 and endPath_.size() > 0);
  for (auto& streamPaths : path_) {
    streamPaths.resize(pathsAndConsumes.paths().size());
  }
  for (auto& streamEndPaths : endPath_) {
    streamEndPaths.resize(pathsAndConsumes.endPaths().size());
  }
}

template <class Backend>
void ProfilerService<Backend>::preBeginStream(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].startColorIn(stream_domain_[sid], "begin stream", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postBeginStream(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].endIn(stream_domain_[sid], "begin stream", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preEndStream(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].startColorIn(stream_domain_[sid], "end stream", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postEndStream(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].endIn(stream_domain_[sid], "end stream", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::jobFailure() {
  Backend::mark(global_domain_, "jobFailure", Color::Red);
}

template <class Backend>
void ProfilerService<Backend>::preSourceNextTransition() {
  globalRange_.startColorIn(global_domain_, "source transition", Color::Amber, __func__);
}

template <class Backend>
void ProfilerService<Backend>::postSourceNextTransition() {
  globalRange_.endIn(global_domain_, "source transition", __func__);
}

template <class Backend>
void ProfilerService<Backend>::preSourceEvent(edm::StreamID sid) {
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    source_[sid].startColorIn(stream_domain_[sid], "source", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postSourceEvent(edm::StreamID sid) {
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    source_[sid].endIn(stream_domain_[sid], "source", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preSourceLumi(edm::LuminosityBlockIndex index) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.startColorIn(global_domain_, "source lumi", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postSourceLumi(edm::LuminosityBlockIndex index) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, "source lumi", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preSourceRun(edm::RunIndex index) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.startColorIn(global_domain_, "source run", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postSourceRun(edm::RunIndex index) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, "source run", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preOpenFile(std::string const& lfn) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.startColorIn(global_domain_, ("open file "s + lfn).c_str(), Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postOpenFile(std::string const& lfn) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, ("open file "s + lfn).c_str(), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preCloseFile(std::string const& lfn) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.startColorIn(global_domain_, ("close file "s + lfn).c_str(), Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postCloseFile(std::string const& lfn) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, ("close file "s + lfn).c_str(), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preGlobalBeginRun(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.startColorIn(global_domain_, "global begin run", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postGlobalBeginRun(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, "global begin run", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preGlobalEndRun(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.startColorIn(global_domain_, "global end run", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postGlobalEndRun(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, "global end run", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preStreamBeginRun(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].startColorIn(stream_domain_[sid], "stream begin run", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postStreamBeginRun(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].endIn(stream_domain_[sid], "stream begin run", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preStreamEndRun(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].startColorIn(stream_domain_[sid], "stream end run", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postStreamEndRun(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].endIn(stream_domain_[sid], "stream end run", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preGlobalBeginLumi(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.startColorIn(global_domain_, "global begin lumi", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postGlobalBeginLumi(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, "global begin lumi", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preGlobalEndLumi(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.startColorIn(global_domain_, "global end lumi", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postGlobalEndLumi(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, "global end lumi", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preStreamBeginLumi(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].startColorIn(stream_domain_[sid], "stream begin lumi", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postStreamBeginLumi(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].endIn(stream_domain_[sid], "stream begin lumi", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preStreamEndLumi(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].startColorIn(stream_domain_[sid], "stream end lumi", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postStreamEndLumi(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].endIn(stream_domain_[sid], "stream end lumi", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preEvent(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    std::string msg = fmt::sprintf("event run = %d event = %d", sc.eventID().run(), sc.eventID().event());
    event_[sid].startColorIn(stream_domain_[sid], "event", Color::Green_Dark1, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postEvent(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].endIn(stream_domain_[sid], "event", __func__);
  } else {
    streamFirstEventDone_[sid] = true;
    auto identity = [](bool x) { return x; };
    if (std::all_of(streamFirstEventDone_.begin(), streamFirstEventDone_.end(), identity)) {
      bool expected = false;
      if (globalFirstEventDone_.compare_exchange_strong(expected, true))
        Backend::profilerStart();
    }
  }
}

template <class Backend>
void ProfilerService<Backend>::preClearEvent(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].startColorIn(stream_domain_[sid], "clear event", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postClearEvent(edm::StreamContext const& sc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    event_[sid].endIn(stream_domain_[sid], "clear event", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::prePathEvent(edm::StreamContext const& sc, edm::PathContext const& pc) {
  auto sid = sc.streamID();
  auto pid = pc.pathID();
  auto& pathOrEndPath = pc.isEndPath() ? endPath_[sid][pid] : path_[sid][pid];
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    pathOrEndPath.startColorIn(stream_domain_[sid], ("path " + pc.pathName()).c_str(), Color::Green_Dark1, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postPathEvent(edm::StreamContext const& sc,
                                             edm::PathContext const& pc,
                                             edm::HLTPathStatus const& hlts) {
  auto sid = sc.streamID();
  auto pid = pc.pathID();
  auto& pathOrEndPath = pc.isEndPath() ? endPath_[sid][pid] : path_[sid][pid];
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    pathOrEndPath.endIn(stream_domain_[sid], ("path " + pc.pathName()).c_str(), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preModuleConstruction(edm::ModuleDescription const& desc) {
  auto mid = desc.id();
  global_modules_.grow_to_at_least(mid + 1);

  if (not skipFirstEvent_) {
    auto const& label = desc.moduleLabel();
    auto const& msg = label + " construction";
    global_modules_[mid].startColorIn(global_domain_, msg.c_str(), labelColor(label), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postModuleConstruction(edm::ModuleDescription const& desc) {
  if (not skipFirstEvent_) {
    auto mid = desc.id();
    auto const& label = desc.moduleLabel();
    auto const& msg = label + " construction";
    global_modules_[mid].endIn(global_domain_, msg.c_str(), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preModuleDestruction(edm::ModuleDescription const& desc) {
  if (not skipFirstEvent_) {
    auto mid = desc.id();
    auto const& label = desc.moduleLabel();
    auto const& msg = label + " destruction";
    global_modules_[mid].startColorIn(global_domain_, msg.c_str(), labelColor(label), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postModuleDestruction(edm::ModuleDescription const& desc) {
  if (not skipFirstEvent_) {
    auto mid = desc.id();
    auto const& label = desc.moduleLabel();
    auto const& msg = label + " destruction";
    global_modules_[mid].endIn(global_domain_, msg.c_str(), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preModuleBeginJob(edm::ModuleDescription const& desc) {
  if (not skipFirstEvent_) {
    auto mid = desc.id();
    auto const& label = desc.moduleLabel();
    auto const& msg = label + " begin job";
    global_modules_[mid].startColorIn(global_domain_, msg.c_str(), labelColor(label), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postModuleBeginJob(edm::ModuleDescription const& desc) {
  if (not skipFirstEvent_) {
    auto mid = desc.id();
    auto const& label = desc.moduleLabel();
    auto const& msg = label + " begin job";
    global_modules_[mid].endIn(global_domain_, msg.c_str(), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preModuleEndJob(edm::ModuleDescription const& desc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = desc.id();
    auto const& label = desc.moduleLabel();
    auto const& msg = label + " end job";
    global_modules_[mid].startColorIn(global_domain_, msg.c_str(), labelColor(label), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postModuleEndJob(edm::ModuleDescription const& desc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = desc.id();
    auto const& label = desc.moduleLabel();
    auto const& msg = label + " end job";
    global_modules_[mid].endIn(global_domain_, msg.c_str(), __func__);
  }
}

/******** Module stream context signals *********************************************/

DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleBeginStream, stream_modules_in_flight_ranges_)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleEndStream, stream_modules_in_flight_ranges_)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleEventPrefetching, stream_modules_in_flight_ranges_)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleEvent, stream_modules_event_in_flight_ranges_)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleEventAcquire, stream_modules_event_acquire_in_flight_ranges_)
DEFINE_MODULE_TRANSFORM_SIGNAL_WATCHER(ModuleTransformPrefetching)
DEFINE_MODULE_TRANSFORM_SIGNAL_WATCHER(ModuleTransform)
DEFINE_MODULE_TRANSFORM_SIGNAL_WATCHER(ModuleTransformAcquiring)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleEventDelayedGet, stream_modules_in_flight_ranges_)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(EventReadFromSource, stream_modules_in_flight_ranges_)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleStreamPrefetching, stream_modules_in_flight_ranges_)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleStreamBeginRun, stream_modules_in_flight_ranges_)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleStreamEndRun, stream_modules_in_flight_ranges_)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleStreamBeginLumi, stream_modules_in_flight_ranges_)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleStreamEndLumi, stream_modules_in_flight_ranges_)

template <class Backend>
void ProfilerService<Backend>::preModuleGlobalBeginRun(edm::GlobalContext const& gc,
                                                       edm::ModuleCallingContext const& mcc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = mcc.moduleDescription()->id();
    auto const& label = mcc.moduleDescription()->moduleLabel();
    auto const& msg = label + " global begin run";
    global_modules_[mid].startColorIn(global_domain_, msg.c_str(), labelColor(label), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postModuleGlobalBeginRun(edm::GlobalContext const& gc,
                                                        edm::ModuleCallingContext const& mcc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = mcc.moduleDescription()->id();
    auto const& label = mcc.moduleDescription()->moduleLabel();
    auto const& msg = label + " global begin run";
    global_modules_[mid].endIn(global_domain_, "", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preModuleGlobalEndRun(edm::GlobalContext const& gc,
                                                     edm::ModuleCallingContext const& mcc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = mcc.moduleDescription()->id();
    auto const& label = mcc.moduleDescription()->moduleLabel();
    auto const& msg = label + " global end run";
    global_modules_[mid].startColorIn(global_domain_, msg.c_str(), labelColor(label), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postModuleGlobalEndRun(edm::GlobalContext const& gc,
                                                      edm::ModuleCallingContext const& mcc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = mcc.moduleDescription()->id();
    auto const& label = mcc.moduleDescription()->moduleLabel();
    auto const& msg = label + " global end run";
    global_modules_[mid].endIn(global_domain_, msg.c_str(), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preModuleGlobalBeginLumi(edm::GlobalContext const& gc,
                                                        edm::ModuleCallingContext const& mcc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = mcc.moduleDescription()->id();
    auto const& label = mcc.moduleDescription()->moduleLabel();
    auto const& msg = label + " global begin lumi";
    global_modules_[mid].startColorIn(global_domain_, msg.c_str(), labelColor(label), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postModuleGlobalBeginLumi(edm::GlobalContext const& gc,
                                                         edm::ModuleCallingContext const& mcc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = mcc.moduleDescription()->id();
    auto const& label = mcc.moduleDescription()->moduleLabel();
    auto const& msg = label + " global begin lumi";
    global_modules_[mid].endIn(global_domain_, msg.c_str(), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preModuleGlobalEndLumi(edm::GlobalContext const& gc,
                                                      edm::ModuleCallingContext const& mcc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = mcc.moduleDescription()->id();
    auto const& label = mcc.moduleDescription()->moduleLabel();
    auto const& msg = label + " global end lumi";
    global_modules_[mid].startColorIn(global_domain_, msg.c_str(), labelColor(label), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postModuleGlobalEndLumi(edm::GlobalContext const& gc,
                                                       edm::ModuleCallingContext const& mcc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    auto mid = mcc.moduleDescription()->id();
    auto const& label = mcc.moduleDescription()->moduleLabel();
    auto const& msg = label + " global end lumi";
    global_modules_[mid].endIn(global_domain_, msg.c_str(), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preESModuleConstruction(edm::eventsetup::ComponentDescription const& desc) {
  auto mid = desc.id_;
  global_ES_modules_.grow_to_at_least(mid + 1);
  if (not skipFirstEvent_) {
    auto const& label = desc.label_;
    auto const& type = desc.type_;
    std::string msg;
    if (label.empty()) {
      msg = type + "(type) construction";
    } else {
      msg = label + " construction";
    }
    global_ES_modules_[mid].startColorIn(global_domain_, msg.c_str(), Color::Blue, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postESModuleConstruction(edm::eventsetup::ComponentDescription const& desc) {
  if (not skipFirstEvent_) {
    auto mid = desc.id_;
    auto const& label = desc.label_;
    auto const& type = desc.type_;
    std::string msg;
    if (label.empty()) {
      msg = type + "(type) construction";
    } else {
      msg = label + " construction";
    }
    global_ES_modules_[mid].endIn(global_domain_, msg.c_str(), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postESModuleRegistration(
    edm::eventsetup::ComponentDescription const& componentDescription) {
  auto mid = componentDescription.id_;
  auto const& label = componentDescription.label_;
  auto const& msg = label + " " + "ESModuleReRegistration";
  global_ES_modules_.grow_to_at_least(mid + 1);
  Backend::mark(global_domain_, msg.c_str(), Color::Amber);
}

template <class Backend>
void ProfilerService<Backend>::preESModulePrefetching(edm::eventsetup::EventSetupRecordKey const& iKey,
                                                      edm::ESModuleCallingContext const& esmcc) {
  preESModuleAcquire(iKey, esmcc);
}

template <class Backend>
void ProfilerService<Backend>::postESModulePrefetching(edm::eventsetup::EventSetupRecordKey const& iKey,
                                                       edm::ESModuleCallingContext const& esmcc) {
  postESModuleAcquire(iKey, esmcc);
}

DEFINE_ES_SIGNAL_WATCHER(ESModule)
DEFINE_ES_SIGNAL_WATCHER(ESModuleAcquire)

/******** Job-level single signal implementations *************************************/

/******** Infrastructure/setup signal implementations *************************************/

template <class Backend>
void ProfilerService<Backend>::preFinishSchedule() {
  if (not skipFirstEvent_) {
    globalRange_.startColorIn(global_domain_, "finish schedule", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postFinishSchedule() {
  if (not skipFirstEvent_) {
    globalRange_.endIn(global_domain_, "finish schedule", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::prePrincipalsCreation() {
  if (not skipFirstEvent_) {
    Backend::mark(global_domain_, "prePrincipalsCreation", Color::Amber);
  }
}

template <class Backend>
void ProfilerService<Backend>::postPrincipalsCreation() {
  if (not skipFirstEvent_) {
    Backend::mark(global_domain_, "postPrincipalsCreation", Color::Amber);
  }
}

template <class Backend>
void ProfilerService<Backend>::preScheduleConsistencyCheck() {
  if (not skipFirstEvent_) {
    Backend::mark(global_domain_, "preScheduleConsistencyCheck", Color::Amber);
  }
}

template <class Backend>
void ProfilerService<Backend>::postScheduleConsistencyCheck() {
  if (not skipFirstEvent_) {
    Backend::mark(global_domain_, "postScheduleConsistencyCheck", Color::Amber);
  }
}

template <class Backend>
void ProfilerService<Backend>::preModulesInitializationFinalized() {
  if (not skipFirstEvent_) {
    Backend::mark(global_domain_, "preModulesInitializationFinalized", Color::Amber);
  }
}

template <class Backend>
void ProfilerService<Backend>::postModulesInitializationFinalized() {
  if (not skipFirstEvent_) {
    Backend::mark(global_domain_, "postModulesInitializationFinalized", Color::Amber);
  }
}

/******** Process block signal implementations *************************************/

template <class Backend>
void ProfilerService<Backend>::preBeginProcessBlock(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.startColorIn(global_domain_, "begin process block", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postBeginProcessBlock(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, "begin process block", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preEndProcessBlock(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.startColorIn(global_domain_, "end process block", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postEndProcessBlock(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, "end process block", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preAccessInputProcessBlock(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.startColorIn(global_domain_, "access input process block", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postAccessInputProcessBlock(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, "access input process block", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preWriteProcessBlock(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.startColorIn(global_domain_, "write process block", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postWriteProcessBlock(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, "write process block", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::beginProcessing() {
  Backend::mark(global_domain_, "beginProcessing", Color::Amber);
}

template <class Backend>
void ProfilerService<Backend>::endProcessing() {
  Backend::mark(global_domain_, "endProcessing", Color::Amber);
}

/******** Global write signal implementations *************************************/

template <class Backend>
void ProfilerService<Backend>::preGlobalWriteRun(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.startColorIn(global_domain_, "global write run", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postGlobalWriteRun(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, "global write run", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preGlobalWriteLumi(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.startColorIn(global_domain_, "global write lumi", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postGlobalWriteLumi(edm::GlobalContext const& gc) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, "global write lumi", __func__);
  }
}

/******** Output file signal implementations *************************************/

template <class Backend>
void ProfilerService<Backend>::preOpenOutputFiles() {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.startColorIn(global_domain_, "open output files", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postOpenOutputFiles() {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, "open output files", __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::preCloseOutputFiles() {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.startColorIn(global_domain_, "close output files", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postCloseOutputFiles() {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, "close output files", __func__);
  }
}

/******** Source process block signal implementations *************************************/

template <class Backend>
void ProfilerService<Backend>::preSourceProcessBlock() {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.startColorIn(global_domain_, "source process block", Color::Amber, __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postSourceProcessBlock(std::string const& processName) {
  if (not skipFirstEvent_ or globalFirstEventDone_) {
    globalRange_.endIn(global_domain_, "source process block", __func__);
  }
}

/******** ES IOV sync signal implementations *************************************/

template <class Backend>
void ProfilerService<Backend>::esSyncIOVQueuing(edm::IOVSyncValue const&) {
  Backend::mark(global_domain_, "esSyncIOVQueuing", Color::Blue);
}

template <class Backend>
void ProfilerService<Backend>::preESSyncIOV(edm::IOVSyncValue const&) {
  globalRange_.startColorIn(global_domain_, "ES sync IOV", Color::Blue, __func__);
}

template <class Backend>
void ProfilerService<Backend>::postESSyncIOV(edm::IOVSyncValue const&) {
  globalRange_.endIn(global_domain_, "ES sync IOV", __func__);
}

/******** Early termination signal implementations *****************************/

template <class Backend>
void ProfilerService<Backend>::preStreamEarlyTermination(edm::StreamContext const& sc,
                                                         edm::TerminationOrigin origin) {
  auto sid = sc.streamID();
  Backend::mark(stream_domain_[sid], "early termination", Color::Red);
}

template <class Backend>
void ProfilerService<Backend>::preGlobalEarlyTermination(edm::GlobalContext const& gc,
                                                         edm::TerminationOrigin origin) {
  Backend::mark(global_domain_, "global early termination", Color::Red);
}

template <class Backend>
void ProfilerService<Backend>::preSourceEarlyTermination(edm::TerminationOrigin origin) {
  Backend::mark(global_domain_, "source early termination", Color::Red);
}

/******** ES module construction signal implementations *****************************/

/******** Module global prefetching and process block signal implementations ***********/

DEFINE_GLOBAL_MODULE_SIGNAL_WATCHER(ModuleBeginProcessBlock)
DEFINE_GLOBAL_MODULE_SIGNAL_WATCHER(ModuleAccessInputProcessBlock)
DEFINE_GLOBAL_MODULE_SIGNAL_WATCHER(ModuleEndProcessBlock)
DEFINE_GLOBAL_MODULE_SIGNAL_WATCHER(ModuleGlobalPrefetching)
DEFINE_GLOBAL_MODULE_SIGNAL_WATCHER(ModuleWriteProcessBlock)
DEFINE_GLOBAL_MODULE_SIGNAL_WATCHER(ModuleWriteRun)
DEFINE_GLOBAL_MODULE_SIGNAL_WATCHER(ModuleWriteLumi)

template <class Backend>
void ProfilerService<Backend>::preSourceConstruction(edm::ModuleDescription const& desc) {
  auto mid = desc.id();
  global_modules_.grow_to_at_least(mid + 1);

  if (not skipFirstEvent_) {
    auto const& label = desc.moduleLabel();
    auto const& msg = label + " construction";
    global_modules_[mid].startColorIn(global_domain_, msg.c_str(), labelColor(label), __func__);
  }
}

template <class Backend>
void ProfilerService<Backend>::postSourceConstruction(edm::ModuleDescription const& desc) {
  if (not skipFirstEvent_) {
    auto mid = desc.id();
    auto const& label = desc.moduleLabel();
    auto const& msg = label + " construction";
    global_modules_[mid].endIn(global_domain_, msg.c_str(), __func__);
  }
}

#undef DECLARE_SIGNAL_WATCHER_NOARGS
#undef DECLARE_SIGNAL_WATCHER_PROCESS_CONTEXT
#undef DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT
#undef DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT
#undef DECLARE_SIGNAL_WATCHER_STREAM_ID
#undef DECLARE_SIGNAL_WATCHER_LUMIBLOCK_INDEX
#undef DECLARE_SIGNAL_WATCHER_RUN_INDEX
#undef DECLARE_SIGNAL_WATCHER_STRING
#undef DECLARE_SIGNAL_WATCHER_MODULE_DESCRIPTION
#undef DECLARE_SIGNAL_WATCHER_COMPONENT_DESCRIPTION
#undef DECLARE_SIGNAL_WATCHER_IOV_SYNC_VALUE
#undef DECLARE_SIGNAL_WATCHER_EVENT_SETUP_RECORD_KEY_ES_MODULE_CALLING_CONTEXT
#undef DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_PATH_CONTEXT
#undef DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_PATH_CONTEXT_HLT_STATUS
#undef DECLARE_SIGNAL_WATCHER_STREAM_CONTEXT_MODULE_CALLING_CONTEXT
#undef DECLARE_SIGNAL_WATCHER_GLOBAL_CONTEXT_MODULE_CALLING_CONTEXT
#undef DECLARE_SIGNAL_WATCHER_TERMINATION_ORIGIN_STREAM
#undef DECLARE_SIGNAL_WATCHER_TERMINATION_ORIGIN_GLOBAL
#undef DECLARE_SIGNAL_WATCHER_TERMINATION_ORIGIN_SOURCE
#undef DEFINE_ES_SIGNAL_WATCHER
#undef DEFINE_MODULE_STREAM_SIGNAL_WATCHER
#undef DEFINE_GLOBAL_MODULE_SIGNAL_WATCHER
#undef REGISTER_SIGNAL_WATCHER

#endif  // __FWCore_Services_ProfilerService_h__
