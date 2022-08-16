#ifndef HeterogeneousCore_CUDAUtilities_interface_ScopedNVTXRange_h

#include <nvToolsExt.h>
#include <string>

class ScopedNVTXRange {
public:
  ScopedNVTXRange(const std::string& label) : nvtxRange_(nvtxRangeStartA(label.c_str())) {}
  ~ScopedNVTXRange() { nvtxRangeEnd(nvtxRange_); }

private:
  nvtxRangeId_t nvtxRange_;
};

#endif