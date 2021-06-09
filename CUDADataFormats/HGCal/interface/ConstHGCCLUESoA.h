#ifndef CUDADataFormats_HGCal_ConstHGCCLUESoA_h
#define CUDADataFormats_HGCal_ConstHGCCLUESoA_h

#include <cstdint>

class ConstHGCCLUESoA {  //const version of the HGCRecHit class (data in the event should be immutable)
public:
  float const *rho; //energy density of the calibrated rechit
  float const *delta; //closest distance to a rechit with a higher density
  int32_t const *nearestHigher; //index of the nearest rechit with a higher density
  int32_t const *clusterIndex;  //cluster index the rechit belongs to
  bool const *isSeed; // is the rechit a cluster seed?
  //Note: isSeed is of type int in the CPU version to to std::vector optimizations
};

#endif  //CUDADataFormats_HGCal_ConstHGCCLUESoA_h
