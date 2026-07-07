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
 * Helper marcos to declare similar functions (for pre/post couples).
 */
#define DECLARE_ES_SIGNAL_WATCHER(signal)                                                                     \
  void pre##signal(edm::eventsetup::EventSetupRecordKey const& iKey, edm::ESModuleCallingContext const& mcc); \
  void post##signal(edm::eventsetup::EventSetupRecordKey const& iKey, edm::ESModuleCallingContext const& mcc);

// ES module signal ranges are keyed dynamically to avoid collisions from overlapping calls.
#define DEFINE_ES_SIGNAL_WATCHER(signal)                                                        \
  template <class Backend>                                                                      \
  void ProfilerService<Backend>::pre##signal(edm::eventsetup::EventSetupRecordKey const& iKey,  \
                                             edm::ESModuleCallingContext const& esmcc) {        \
    auto mid = esmcc.componentDescription()->id_;                                               \
    auto const& record = iKey.name();                                                           \
    auto const& label = esmcc.componentDescription()->label_;                                   \
    auto const& type = esmcc.componentDescription()->type_;                                     \
    auto const pid = esmcc.componentDescription()->pid_.smallHash();                           \
    auto const& state = esmcc.state();                                                            \
    auto const callId = esmcc.callID();                                                         \
    std::string msg;                                                                            \
    if (label.size() == 0) {                                                                    \
      /*Fallback on the type */                                                                 \
      msg = type + "(type) " + #signal " record=" + record + " state=" + (state == edm::ESModuleCallingContext::State::kRunning ? "running" : "prefetching");                                    \
    } else {                                                                                    \
      msg = label + " " + #signal " record=" + record + " state=" + (state == edm::ESModuleCallingContext::State::kRunning ? "running" : "prefetching");                                         \
    }                                                                                           \
    global_es_in_flight_ranges_.start(mid, iKey.name(), #signal, label, type, pid, state, callId, global_domain_, msg, Color::Blue, __func__); \
  }                                                                                             \
  template <class Backend>                                                                      \
  void ProfilerService<Backend>::post##signal(edm::eventsetup::EventSetupRecordKey const& iKey, \
                                              edm::ESModuleCallingContext const& esmcc) {       \
    auto mid = esmcc.componentDescription()->id_;                                               \
    auto const& record = iKey.name();                                                           \
    auto const& label = esmcc.componentDescription()->label_;                                   \
    auto const& type = esmcc.componentDescription()->type_;                                     \
    auto const pid = esmcc.componentDescription()->pid_.smallHash();                           \
    auto const& state = esmcc.state();                                                            \
    auto const callId = esmcc.callID();                                                         \
    std::string msg;                                                                            \
    if (label.size() == 0) {                                                                    \
      /* Fallback on the type */                                                                \
      msg = type + "(type) " + #signal " record=" + record + " state=" + (state == edm::ESModuleCallingContext::State::kRunning ? "running" : "prefetching");                                    \
    } else {                                                                                    \
      msg = label + " " + #signal " record=" + record + " state=" + (state == edm::ESModuleCallingContext::State::kRunning ? "running" : "prefetching");                                         \
    }                                                                                           \
    global_es_in_flight_ranges_.end(mid, iKey.name(), #signal, label, type, pid, state, callId, global_domain_, msg, __func__); \
  }

#define DECLARE_MODULE_STREAM_SIGNAL_WATCHER(signal)                                    \
  void pre##signal(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc); \
  void post##signal(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc);

#define DEFINE_MODULE_STREAM_SIGNAL_WATCHER(signal, streamModules)                                                  \
  template <class Backend>                                                                                          \
  void ProfilerService<Backend>::pre##signal(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) {  \
    auto sid = sc.streamID();                                                                                       \
    if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {                                                        \
      auto mid = mcc.moduleDescription()->id();                                                                     \
      auto const& label = mcc.moduleDescription()->moduleLabel();                                                   \
      auto const& msg = label + " " + #signal "";                                                                   \
      startStreamModuleRange_(streamModules, sid, mid, msg, labelColor(label), __func__);                         \
    }                                                                                                               \
  }                                                                                                                 \
  template <class Backend>                                                                                          \
  void ProfilerService<Backend>::post##signal(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) { \
    auto sid = sc.streamID();                                                                                       \
    if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {                                                        \
      auto mid = mcc.moduleDescription()->id();                                                                     \
      auto const& label = mcc.moduleDescription()->moduleLabel();                                                   \
      auto const& msg = label + " " + #signal "";                                                                   \
      endStreamModuleRange_(streamModules, sid, mid, msg, __func__);                                               \
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
      transform_in_flight_ranges_.start(sid, mid, callId ,#signal, stream_domain_[sid], msg, Color::Blue, __func__);            \
    }                                                                                                               \
  }                                                                                                                 \
  template <class Backend>                                                                                          \
  void ProfilerService<Backend>::post##signal(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc) { \
    auto sid = sc.streamID();                                                                                       \
    if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {                                                        \
      auto mid = mcc.moduleDescription()->id();                                                                     \
      auto const callId = mcc.callID();                                                                                       \
      auto const msg = transformMessage_(mcc, #signal);                                                            \
      transform_in_flight_ranges_.end(sid, mid, callId, #signal, stream_domain_[sid], msg, __func__);                            \
    }                                                                                                               \
  }

// This macro registers signal watchers pairs. Same for all.
#define REGISTER_SIGNAL_WATCHER(signal)                           \
  registry.watchPre##signal(this, &ProfilerService::pre##signal); \
  registry.watchPost##signal(this, &ProfilerService::post##signal);

// Macro for global-module (GlobalContext, ModuleCallingContext) signal pairs, using global_modules_
#define DECLARE_GLOBAL_MODULE_SIGNAL_WATCHER(signal)                                      \
  void pre##signal(edm::GlobalContext const& gc, edm::ModuleCallingContext const& mcc);   \
  void post##signal(edm::GlobalContext const& gc, edm::ModuleCallingContext const& mcc);

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

  void preClearEvent(edm::StreamContext const&);
  void postClearEvent(edm::StreamContext const&);

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

  /******** Source transition signals *********************************************/

  void preSourceNextTransition();
  void postSourceNextTransition();

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

  /******** Job-level single signals *********************************************/

  void beginProcessing();
  void endProcessing();
  void jobFailure();
  void postServicesConstruction();

  /******** Infrastructure/setup signal pairs *************************************/

  void preBeginStream(edm::StreamContext const&);
  void postBeginStream(edm::StreamContext const&);

  void preEndStream(edm::StreamContext const&);
  void postEndStream(edm::StreamContext const&);

  void preEventSetupConfigurationFinalized();
  void postEventSetupConfigurationFinalized();

  void eventSetupConfiguration(edm::eventsetup::ESRecordsToProductResolverIndices const&,
                                edm::ProcessContext const&);

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

  void preModulesInitializationFinalized();
  void postModulesInitializationFinalized();

  /******** Process block signals **********************************************/

  void preBeginProcessBlock(edm::GlobalContext const&);
  void postBeginProcessBlock(edm::GlobalContext const&);

  void preEndProcessBlock(edm::GlobalContext const&);
  void postEndProcessBlock(edm::GlobalContext const&);

  void preAccessInputProcessBlock(edm::GlobalContext const&);
  void postAccessInputProcessBlock(edm::GlobalContext const&);

  void preWriteProcessBlock(edm::GlobalContext const&);
  void postWriteProcessBlock(edm::GlobalContext const&);

  /******** Global write signals **********************************************/

  void preGlobalWriteRun(edm::GlobalContext const&);
  void postGlobalWriteRun(edm::GlobalContext const&);

  void preGlobalWriteLumi(edm::GlobalContext const&);
  void postGlobalWriteLumi(edm::GlobalContext const&);

  /******** Output file signals **********************************************/

  void preOpenOutputFiles();
  void postOpenOutputFiles();

  void preCloseOutputFiles();
  void postCloseOutputFiles();

  /******** Source process block signals *************************************/

  void preSourceProcessBlock();
  void postSourceProcessBlock(std::string const&);

  /******** ES IOV sync signals **********************************************/

  void esSyncIOVQueuing(edm::IOVSyncValue const&);

  void preESSyncIOV(edm::IOVSyncValue const&);
  void postESSyncIOV(edm::IOVSyncValue const&);

  /******** ES module construction signals **********************************************/

  void preESModuleConstruction(edm::eventsetup::ComponentDescription const&);
  void postESModuleConstruction(edm::eventsetup::ComponentDescription const&);

  /******** Early termination signals (Pre only, no Post) *****************************/

  void preStreamEarlyTermination(edm::StreamContext const&, edm::TerminationOrigin);
  void preGlobalEarlyTermination(edm::GlobalContext const&, edm::TerminationOrigin);
  void preSourceEarlyTermination(edm::TerminationOrigin);

  /******** Module stream prefetching signals **********************************************/

  DECLARE_MODULE_STREAM_SIGNAL_WATCHER(ModuleStreamPrefetching)

  /******** Module global prefetching and process block signals **********************************************/

  DECLARE_GLOBAL_MODULE_SIGNAL_WATCHER(ModuleGlobalPrefetching)
  DECLARE_GLOBAL_MODULE_SIGNAL_WATCHER(ModuleBeginProcessBlock)
  DECLARE_GLOBAL_MODULE_SIGNAL_WATCHER(ModuleEndProcessBlock)
  DECLARE_GLOBAL_MODULE_SIGNAL_WATCHER(ModuleAccessInputProcessBlock)
  DECLARE_GLOBAL_MODULE_SIGNAL_WATCHER(ModuleWriteProcessBlock)
  DECLARE_GLOBAL_MODULE_SIGNAL_WATCHER(ModuleWriteRun)
  DECLARE_GLOBAL_MODULE_SIGNAL_WATCHER(ModuleWriteLumi)

  /******** ES module context signals *********************************************/
  // ES signal watchers
  void postESModuleRegistration(edm::eventsetup::ComponentDescription const&);
  // Prefetching is optionally watched
  // (see constructor)
  DECLARE_ES_SIGNAL_WATCHER(ESModulePrefetching)
  DECLARE_ES_SIGNAL_WATCHER(ESModule)
  DECLARE_ES_SIGNAL_WATCHER(ESModuleAcquire)

private:
  using StreamModuleRangeStacks = std::vector<std::vector<std::vector<Range>>>;

  class TransformInFlightRanges {
  public:
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
          auto fullmsg = "Warning: previous range not ended before starting a new one in "s + func +
                          " name=" + msg + " mid=" + std::to_string(mid) +
                          " stream id=" + std::to_string(sid) + " signal=" + std::string(signal);
          Backend::mark(domain, fullmsg.c_str(), Color::Red);
          std::cout << fullmsg << std::endl;
          return;
        }
        slot = acquireSlot_();
        in_flight_[std::move(key)].push_back(slot);
      }
      ranges_[slot].startColorIn(domain, msg.c_str(), color, func);
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
        auto fullmsg = "Warning: trying to end a range that is not started in "s + func + " name=" + msg +
                        " mid=" + std::to_string(mid) + " stream id=" + std::to_string(sid) +
                        " signal=" + std::string(signal);
        Backend::mark(domain, fullmsg.c_str(), Color::Red);
        std::cout << fullmsg << std::endl;
        return;
      }
      ranges_[*slot].endIn(domain, msg.c_str(), func);
      std::lock_guard<SpinLock> guard(mutex_);
      releaseSlot_(*slot);
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

    size_t acquireSlot_() {
      if (not free_slots_.empty()) {
        auto slot = free_slots_.back();
        free_slots_.pop_back();
        return slot;
      }
      ranges_.emplace_back();
      return ranges_.size() - 1;
    }

    void releaseSlot_(size_t slot) { free_slots_.push_back(slot); }

    SpinLock mutex_;
    std::vector<Range> ranges_;
    std::vector<size_t> free_slots_;
    std::unordered_map<std::string, std::vector<size_t>> in_flight_;
  };

  void startStreamModuleRange_(StreamModuleRangeStacks& streamModules,
                               unsigned int sid,
                               unsigned int mid,
                               std::string const& msg,
                               Color color,
                               char const* func) {
    std::lock_guard<SpinLock> guard(stream_modules_mutex_);
    auto& ranges = streamModules[sid][mid];
    if (not ranges.empty()) {
      auto fullmsg = "Warning: previous range not ended before starting a new one in "s + func +
                      " name=" + msg + " mid=" + std::to_string(mid) +
                      " stream id=" + std::to_string(sid);
      Backend::mark(stream_domain_[sid], fullmsg.c_str(), Color::Red);
      std::cout << fullmsg << std::endl;
    }
    ranges.emplace_back();
    ranges.back().startColorIn(stream_domain_[sid], msg.c_str(), color, func);
  }

  void endStreamModuleRange_(StreamModuleRangeStacks& streamModules,
                             unsigned int sid,
                             unsigned int mid,
                             std::string const& msg,
                             char const* func) {
    std::lock_guard<SpinLock> guard(stream_modules_mutex_);
    auto& ranges = streamModules[sid][mid];
    if (ranges.empty()) {
      auto fullmsg = "Warning: trying to end a range that is not started in "s + func + " name=" + msg +
                      " mid=" + std::to_string(mid) + " stream id=" + std::to_string(sid);
      Backend::mark(stream_domain_[sid], fullmsg.c_str(), Color::Red);
      std::cout << fullmsg << std::endl;
      return;
    }
    ranges.back().endIn(stream_domain_[sid], msg.c_str(), func);
    ranges.pop_back();
  }

  std::string transformMessage_(edm::ModuleCallingContext const& mcc, char const* signal) const {
    auto const& label = mcc.moduleDescription()->moduleLabel();
    return label + " " + signal;
  }

  class GlobalESInFlightRanges {
  public:
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
        [[maybe_unused]] bool alreadyInFlight = (found != in_flight_.end() and not found->second.empty());
        [[maybe_unused]] bool target = (mid == 78 and record ==  "HcalPedestalWidthsRcd");
        if (found != in_flight_.end() and not found->second.empty()) {
          auto const& existingMsg = found->second.back().startMsg;
          auto const& existingStacktrace = found->second.back().stacktrace;
          auto fullmsg = "\n\nWarning: previous range not ended before starting a new one in "s + func +
                          "\n  existing range: '" + existingMsg + "'" +
                          "\n  existing backtrace: " + existingStacktrace +
                          "\n  new range: name=" + msg + " mid=" + std::to_string(mid) +
                          " record=" + std::string(record) + " signal=" + std::string(signal) +
                          " label=" + std::string(label) + " type=" + std::string(type) +
                          " pid=" + pidString_(pid) + " callId=" + std::to_string(callId) +
                          "\n  new stacktrace: " + to_string(boost::stacktrace::stacktrace{});
          Backend::mark(domain, fullmsg.c_str(), Color::Red);
          std::cout << fullmsg << std::endl;
          return;
        }
        slot = acquireSlot_();
        in_flight_[std::move(key)].push_back({slot, msg, to_string(boost::stacktrace::stacktrace{})});
      }
      ranges_[slot].startColorIn(domain, msg.c_str(), color, func);
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
        auto fullmsg = "Warning: trying to end a range that is not started in "s + func + " name=" + msg +
                        " mid=" + std::to_string(mid) + " record=" + std::string(record) +
                        " signal=" + std::string(signal) + " label=" + std::string(label) +
                        " type=" + std::string(type) + " pid=" + pidString_(pid);
        Backend::mark(domain, fullmsg.c_str(), Color::Red);
        std::cout << fullmsg << std::endl;
        return;
      }
      ranges_[*slot].endIn(domain, msg.c_str(), func);
      std::lock_guard<SpinLock> guard(mutex_);
      releaseSlot_(*slot);
    }

  private:
    struct InFlightEntry {
      size_t slot;
      std::string startMsg;
      std::string stacktrace;
    };

    static std::string pidString_(std::size_t pid) { return fmt::sprintf("0x%zx", pid); }

    static std::string makeKey_(unsigned int mid, std::string_view record, edm::ESModuleCallingContext::State const& state, std::uintptr_t callId) {
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

    size_t acquireSlot_() {
      if (not free_slots_.empty()) {
        auto slot = free_slots_.back();
        free_slots_.pop_back();
        return slot;
      }
      ranges_.emplace_back();
      return ranges_.size() - 1;
    }

    void releaseSlot_(size_t slot) { free_slots_.push_back(slot); }

    SpinLock mutex_;
    std::vector<Range> ranges_;
    std::vector<size_t> free_slots_;
    std::unordered_map<std::string, std::vector<InFlightEntry>> in_flight_;
  };

  bool highlight(std::string const& label) const {
    return (std::binary_search(highlightModules_.begin(), highlightModules_.end(), label));
  }

  Color labelColor(std::string const& label) const { return highlight(label) ? Color::Amber : Color::Green; }

  Color labelColorLight(std::string const& label) const {
    return highlight(label) ? Color::LightAmber : Color::LightGreen;
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
  StreamModuleRangeStacks stream_modules_;  // generic per-stream, per-module stacks of ranges
  StreamModuleRangeStacks stream_modules_event_;
  StreamModuleRangeStacks stream_modules_event_acquire_;
  TransformInFlightRanges transform_in_flight_ranges_;
  SpinLock stream_modules_mutex_;
  // use a tbb::concurrent_vector rather than an std::vector because its final size is not known
  tbb::concurrent_vector<Range> global_modules_;       // global per-module events
  std::vector<std::vector<Range>> stream_ES_modules_;  // per-stream, per-ES-module ranges
  std::vector<std::vector<Range>>
      stream_ES_modules_acquire_;  // per-stream, per-ES-module ranges for acquire, which can clash with produce
  // use a tbb::concurrent_vector rather than an std::vector because its final size is not known
  tbb::concurrent_vector<Range> global_ES_modules_;  // global per-ES-module events
  GlobalESInFlightRanges global_es_in_flight_ranges_;

  Domain global_domain_;               // NVTX domain for global EDM transitions
  std::vector<Domain> stream_domain_;  // NVTX domains for per-EDM-stream transitions
};

template <typename Backend>
ProfilerService<Backend>::ProfilerService(edm::ParameterSet const& config, edm::ActivityRegistry& registry)
    : highlightModules_(config.getUntrackedParameter<std::vector<std::string>>("highlightModules")),
      showModulePrefetching_(config.getUntrackedParameter<bool>("showModulePrefetching")),
      skipFirstEvent_(config.getUntrackedParameter<bool>("skipFirstEvent")) {
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
  REGISTER_SIGNAL_WATCHER(EndProcessBlock)
  REGISTER_SIGNAL_WATCHER(AccessInputProcessBlock)
  REGISTER_SIGNAL_WATCHER(WriteProcessBlock)

  // Job-level single signals
  registry.watchBeginProcessing(this, &ProfilerService::beginProcessing);
  registry.watchEndProcessing(this, &ProfilerService::endProcessing);

  // these signal pair are NOT guaranteed to be called by the same thread
  REGISTER_SIGNAL_WATCHER(GlobalBeginRun)
  REGISTER_SIGNAL_WATCHER(GlobalEndRun)

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

  // ES module construction signal pair
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

  // these signal pair are guaranteed to be called by the same thread
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

  // Module global prefetching and process block signal pairs
  if (showModulePrefetching_) {
    REGISTER_SIGNAL_WATCHER(ModuleGlobalPrefetching)
  }

  // these signal pair are guaranteed to be called by the same thread
  REGISTER_SIGNAL_WATCHER(ModuleGlobalBeginRun)
  REGISTER_SIGNAL_WATCHER(ModuleGlobalEndRun)
  REGISTER_SIGNAL_WATCHER(ModuleGlobalBeginLumi)
  REGISTER_SIGNAL_WATCHER(ModuleGlobalEndLumi)

  REGISTER_SIGNAL_WATCHER(ModuleBeginProcessBlock)
  REGISTER_SIGNAL_WATCHER(ModuleEndProcessBlock)
  REGISTER_SIGNAL_WATCHER(ModuleAccessInputProcessBlock)
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
  // per stream path and end path arrays will be resized in lookupInitializationComplete()
  stream_modules_.resize(concurrentStreams);
  for (auto& modulesForOneStream : stream_modules_) {
    modulesForOneStream.resize(global_modules_.size());
  }
  for (auto& modulesForOneStream : stream_modules_event_) {
    modulesForOneStream.resize(global_modules_.size());
  }
  for (auto& modulesForOneStream : stream_modules_event_acquire_) {
    modulesForOneStream.resize(global_modules_.size());
  }
  stream_modules_event_.resize(concurrentStreams);
  for (auto& modulesForOneStream : stream_modules_event_) {
    modulesForOneStream.resize(global_modules_.size());
  }
  stream_modules_event_acquire_.resize(concurrentStreams);
  for (auto& modulesForOneStream : stream_modules_event_acquire_) {
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
    event_[sid].startColorIn(stream_domain_[sid], "event", Color::DarkGreen, __func__);
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
    pathOrEndPath.startColorIn(stream_domain_[sid], ("path " + pc.pathName()).c_str(), Color::DarkGreen, __func__);
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
  std::cout << "ProfilerService::preModuleConstruction: module id " << mid << ", label: " << desc.moduleLabel() << "\n";

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

DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleBeginStream, stream_modules_)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleEndStream, stream_modules_)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleStreamBeginRun, stream_modules_)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleStreamEndRun, stream_modules_)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleStreamBeginLumi, stream_modules_)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleStreamEndLumi, stream_modules_)
// DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleEventPrefetching)
template <class Backend>
void ProfilerService<Backend>::preModuleEventPrefetching(edm::StreamContext const& sc,
                                                         edm::ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    auto mid = mcc.moduleDescription()->id();
    auto const& label = mcc.moduleDescription()->moduleLabel();
    auto const& msg = label + " " +
                      "ModuleEventPrefetching"
                      "";
    startStreamModuleRange_(stream_modules_, sid, mid, msg, labelColor(label), __func__);
  }
}
template <class Backend>
void ProfilerService<Backend>::postModuleEventPrefetching(edm::StreamContext const& sc,
                                                          edm::ModuleCallingContext const& mcc) {
  auto sid = sc.streamID();
  if (not skipFirstEvent_ or streamFirstEventDone_[sid]) {
    auto mid = mcc.moduleDescription()->id();
    auto const& label = mcc.moduleDescription()->moduleLabel();
    auto const& msg = label + " " +
                      "ModuleEventPrefetching"
                      "";
    endStreamModuleRange_(stream_modules_, sid, mid, msg, __func__);
  }
}
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleEventAcquire, stream_modules_event_acquire_)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleEvent, stream_modules_event_)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleEventDelayedGet, stream_modules_)
DEFINE_MODULE_STREAM_SIGNAL_WATCHER(EventReadFromSource, stream_modules_)
DEFINE_MODULE_TRANSFORM_SIGNAL_WATCHER(ModuleTransformPrefetching)
DEFINE_MODULE_TRANSFORM_SIGNAL_WATCHER(ModuleTransformAcquiring)
DEFINE_MODULE_TRANSFORM_SIGNAL_WATCHER(ModuleTransform)

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
void ProfilerService<Backend>::preSourceNextTransition() {
  globalRange_.startColorIn(global_domain_, "source transition", Color::Amber, __func__);
}

template <class Backend>
void ProfilerService<Backend>::postSourceNextTransition() {
  globalRange_.endIn(global_domain_, "source transition", __func__);
}

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
  auto mid = esmcc.componentDescription()->id_;
  auto const& record = iKey.name();
  auto const& cd = esmcc.componentDescription();
  auto const& label = cd->label_;
  auto const& type = cd->type_;
  auto const& pid = cd->pid_.smallHash();
  auto const& state = esmcc.state();
  auto const& callId = esmcc.callID();
  std::string msg;
  if (label.size() == 0) {
    // Fallback on the type
    msg = type + "(type) " +
          "ES prefetch"
          " acquire"
          " mid=" + std::to_string(mid) +
          " signal=" + std::string("ESModulePrefetching") +
          " record=" +
          record +
          // Print pid as hex
          " pid=" + [](auto pid) {
            std::stringstream ss;
            ss << std::hex << pid;
            return ss.str();
          }(pid) +
          " state=" + (state == edm::ESModuleCallingContext::State::kRunning ? "running" : "prefetching") +
          " callId=" + std::to_string(callId);
  } else {
    msg = label + " " +
          "ES prefetch"
          " acquire"
          " mid=" + std::to_string(mid) +
          " signal=" + std::string("ESModulePrefetching") +
          " record=" +
          record +
          " pid=" + 
          [](auto pid) {
            std::stringstream ss;
            ss << std::hex << pid;
            return ss.str();
          }(pid) +
          " type=" + type +
          " state=" + (state == edm::ESModuleCallingContext::State::kRunning ? "running" : "prefetching") +
          " callId=" + std::to_string(callId);
  }
  global_es_in_flight_ranges_.start(mid, iKey.name(), "ESModulePrefetching", label, type, pid, state, callId, global_domain_, msg, Color::Blue,
                                    __func__);
}

template <class Backend>
void ProfilerService<Backend>::postESModulePrefetching(edm::eventsetup::EventSetupRecordKey const& iKey,
                                                       edm::ESModuleCallingContext const& esmcc) {
  auto mid = esmcc.componentDescription()->id_;
  auto const& record = iKey.name();
  auto const& cd_post = esmcc.componentDescription();
  auto const& label = cd_post->label_;
  auto const& type = cd_post->type_;
  auto const pid = cd_post->pid_.smallHash();
  auto const& state = esmcc.state();
  auto const callId = esmcc.callID();
  std::string msg;
  if (label.size() == 0) {
    // Fallback on the type
    msg = type + "(type) " +
          "ES prefetch"
          " acquire"
          " mid=" + std::to_string(mid) +
          " signal=" + std::string("ESModulePrefetching") +
          " record=" +
          record +
          " state=" + (state == edm::ESModuleCallingContext::State::kRunning ? "running" : "prefetching");
  } else {
    msg = label + " " +
          "ES prefetch"
          " acquire"
          " record=" +
          record +
          " state=" + (state == edm::ESModuleCallingContext::State::kRunning ? "running" : "prefetching");
  }
  global_es_in_flight_ranges_.end(mid, iKey.name(), "ESModulePrefetching", label, type, pid, state, callId, global_domain_, msg, __func__);
}

/*DEFINE_ES_SIGNAL_WATCHER(ESModule)*/
template <class Backend>
void ProfilerService<Backend>::preESModule(edm::eventsetup::EventSetupRecordKey const& iKey,
                                           edm::ESModuleCallingContext const& esmcc) {
  auto mid = esmcc.componentDescription()->id_;
  auto const& cd_pre = esmcc.componentDescription();
  auto const& label = cd_pre->label_;
  auto const& type = cd_pre->type_;
  auto const pid = cd_pre->pid_.smallHash();
  auto const& state = esmcc.state();
  auto const& context = iKey.name();
  auto const& callId = esmcc.callID();
  std::string msg = "ESModule: label = '" + label + "', type = '" + type + "', record = '" + context + "' mid=" +
                    std::to_string(mid) + 
                    " state=" +  ( state == edm::ESModuleCallingContext::State::kRunning ? "running" : "prefetching" ) + 
                    " pid=" + std::to_string(pid) + " context=" + context + " callId=" + std::to_string(callId);
  global_es_in_flight_ranges_.start(mid, context, "ESModule", label, type, pid, state, callId, global_domain_, msg, Color::Blue, __func__);
}

template <class Backend>
void ProfilerService<Backend>::postESModule(edm::eventsetup::EventSetupRecordKey const& iKey,
                                            edm::ESModuleCallingContext const& esmcc) {
  auto mid = esmcc.componentDescription()->id_;
  auto const& cd_post2 = esmcc.componentDescription();
  auto const& label = cd_post2->label_;
  auto const& type = cd_post2->type_;
  auto const pid = cd_post2->pid_.smallHash();
  auto const& context = iKey.name();
  auto const& state = esmcc.state();
  auto const& callId = esmcc.callID();
  std::string msg = "ESModule: label = '" + label + "', type = '" + type + "', record = '" + context + "' state=" + 
     (state == edm::ESModuleCallingContext::State::kRunning ? "running" : "prefetching") + " pid=" + std::to_string(pid) + " context=" + context + " callId=" + std::to_string(callId);
  global_es_in_flight_ranges_.end(mid, context, "ESModule", label, type, pid, state, callId, global_domain_, msg, __func__);
}

DEFINE_ES_SIGNAL_WATCHER(ESModuleAcquire)

/******** Job-level single signal implementations *************************************/

template <class Backend>
void ProfilerService<Backend>::beginProcessing() {
  Backend::mark(global_domain_, "beginProcessing", Color::Amber);
}

template <class Backend>
void ProfilerService<Backend>::endProcessing() {
  Backend::mark(global_domain_, "endProcessing", Color::Amber);
}

template <class Backend>
void ProfilerService<Backend>::jobFailure() {
  Backend::mark(global_domain_, "jobFailure", Color::Red);
}

template <class Backend>
void ProfilerService<Backend>::postServicesConstruction() {
  Backend::mark(global_domain_, "postServicesConstruction", Color::Amber);
}

/******** Infrastructure/setup signal implementations *************************************/

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

/******** ES module construction signal implementations *****************************/

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

/******** Module stream prefetching signal implementations *****************************/

DEFINE_MODULE_STREAM_SIGNAL_WATCHER(ModuleStreamPrefetching, stream_modules_)

/******** Module global prefetching and process block signal implementations ***********/

DEFINE_GLOBAL_MODULE_SIGNAL_WATCHER(ModuleGlobalPrefetching)
DEFINE_GLOBAL_MODULE_SIGNAL_WATCHER(ModuleBeginProcessBlock)
DEFINE_GLOBAL_MODULE_SIGNAL_WATCHER(ModuleEndProcessBlock)
DEFINE_GLOBAL_MODULE_SIGNAL_WATCHER(ModuleAccessInputProcessBlock)
DEFINE_GLOBAL_MODULE_SIGNAL_WATCHER(ModuleWriteProcessBlock)
DEFINE_GLOBAL_MODULE_SIGNAL_WATCHER(ModuleWriteRun)
DEFINE_GLOBAL_MODULE_SIGNAL_WATCHER(ModuleWriteLumi)

#undef DECLARE_ES_SIGNAL_WATCHER
#undef DEFINE_ES_SIGNAL_WATCHER
#undef DECLARE_MODULE_STREAM_SIGNAL_WATCHER
#undef DEFINE_MODULE_STREAM_SIGNAL_WATCHER
#undef DECLARE_GLOBAL_MODULE_SIGNAL_WATCHER
#undef DEFINE_GLOBAL_MODULE_SIGNAL_WATCHER
#undef REGISTER_SIGNAL_WATCHER

#endif  // __FWCore_Services_ProfilerService_h__
