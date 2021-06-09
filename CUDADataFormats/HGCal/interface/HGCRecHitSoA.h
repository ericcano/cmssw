#ifndef CUDADataFormats_HGCal_HGCRecHitSoA_h
#define CUDADataFormats_HGCal_HGCRecHitSoA_h

#include <cstdint>

class HGCRecHitSoA {
public:
  float *energy;       //calibrated energy of the rechit
  float *time;         //time jitter of the UncalibRecHit
  float *timeError;    //time resolution
  float *sigmaNoise;   //cell noise
  uint32_t *id;        //rechit detId
  uint32_t *flagBits;  //rechit flags describing its status (DataFormats/HGCRecHit/interface/HGCRecHit.h)

  uint32_t nbytes;  //number of bytes of the SoA
  uint32_t nhits;   //number of hits stored in the SoA
  uint32_t pad;     //pad of memory block (used for warp alignment, slightly larger than 'nhits_')
};

namespace memory {
  namespace npointers {
    constexpr unsigned float_hgcrechits_soa = 4;   //number of float pointers in the rechits SoA
    constexpr unsigned uint32_hgcrechits_soa = 2;  //number of uint32_t pointers in the rechits SoA
    constexpr unsigned ntypes_hgcrechits_soa = 2;  //number of different pointer types in the rechits SoA
  }                                                // namespace npointers
}  // namespace memory

#endif  //CUDADataFormats_HGCal_HGCRecHitSoA_h
