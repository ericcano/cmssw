#include "HeterogeneousCore/CUDAUtilities/interface/ScopedNVTXRange.h"

// Validation with:
// nsys profile ../test/el8_amd64_gcc10/ScopedNVTXRange_t
// nsys stats report<n>.nsys-rep

int main(void) {
  ScopedNVTXRange sr1("ScopedRangeTest1");
  ScopedNVTXRange sr2("ScopedRangeTest2");
}
