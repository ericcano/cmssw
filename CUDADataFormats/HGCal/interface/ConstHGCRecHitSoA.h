#ifndef CUDADataFormats_HGCal_ConstHGCRecHitSoA_h
#define CUDADataFormats_HGCal_ConstHGCRecHitSoA_h

#include <cstdint>

class ConstHGCRecHitSoA {  //const version of the HGCRecHit class (data in the event should be immutable)
public:
  float const *energy;       //calibrated energy of the rechit
  float const *time;         //time jitter of the UncalibRecHit
  float const *timeError;    //time resolution
  float const *sigmaNoise;   //cell noise
  uint32_t const *id;        //rechit detId
  uint32_t const *flagBits;  //rechit flags describing its status (DataFormats/HGCRecHit/interface/HGCRecHit.h)

  uint32_t nhits;   //number of hits stored in the SoA
};

#endif  //CUDADataFormats_HGCal_ConstHGCRecHitSoA_h
