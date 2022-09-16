#ifndef HeterogeneousCore_CUDAUtilities_interface_ScopedNVTXRange_h
#define HeterogeneousCore_CUDAUtilities_interface_ScopedNVTXRange_h 

#include <nvToolsExt.h>
#include <string>

class ScopedNVTXRange {
public:
  ScopedNVTXRange(const std::string& label) : nvtxRange_(nvtxRangeStartA(label.c_str())) {}
  void end() { if(!ended_) { nvtxRangeEnd(nvtxRange_);  ended_ = true; } }
  ~ScopedNVTXRange() { end(); }

private:
  nvtxRangeId_t nvtxRange_;
  bool ended_ = false;
};

#endif