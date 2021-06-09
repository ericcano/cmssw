#ifndef CUDADataFormats_HGCal_HGCCLUESoA_h
#define CUDADataFormats_HGCal_HGCCLUESoA_h

#include <cstdint>

class HGCCLUESoA {
public:
  float *rho; //energy density of the calibrated rechit
  float *delta; //closest distance to a rechit with a higher density
  int32_t *nearestHigher; //index of the nearest rechit with a higher density
  int32_t *clusterIndex;  //cluster index the rechit belongs to
  bool *isSeed; // is the rechit a cluster seed?
  //Note: isSeed is of type int in the CPU version to to std::vector optimizations

  uint32_t nbytes;  //number of bytes of the SoA
  uint32_t nhits;   //number of hits stored in the SoA
  uint32_t pad;     //pad of memory block (used for warp alignment, slightly larger than 'nhits_')
};

namespace memory {
  namespace npointers {
    //number of float pointers in the rechits SoA
    constexpr unsigned float_hgcclue_soa = 2;
    //number of int32 pointers in the rechits SoA
    constexpr unsigned int32_hgcclue_soa = 2;
    //number of bool pointers in the rechits SoA
    constexpr unsigned bool_hgcclue_soa = 1;
    //number of different pointer types in the rechits SoA
    constexpr unsigned ntypes_hgcclue_soa = 3;
  } // namespace npointers
} // namespace memory

#endif  //CUDADataFormats_HGCal_HGCCLUESoA_h
