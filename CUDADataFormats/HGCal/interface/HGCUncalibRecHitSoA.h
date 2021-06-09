#ifndef CUDADataFormats_HGCal_HGCUncalibRecHitSoA_h
#define CUDADataFormats_HGCal_HGCUncalibRecHitSoA_h

#include <cstdint>

class HGCUncalibRecHitSoA {
public:
  float *amplitude;     //uncalib rechit amplitude, i.e., the average number of MIPs
  float *pedestal;      //reconstructed pedestal
  float *jitter;        //reconstructed time jitter
  float *chi2;          //chi2 of the pulse
  float *OOTamplitude;  //out-of-time reconstructed amplitude
  float *OOTchi2;       //out-of-time chi2
  uint32_t *
      flags;  //uncalibrechit flags describing its status (DataFormats/HGCRecHit/interface/HGCUncalibRecHit.h); to be propagated to the rechits
  uint32_t *aux;  //aux word; first 8 bits contain time (jitter) error
  uint32_t *id;   //uncalibrechit detector id

  uint32_t nbytes;  //number of bytes of the SoA
  uint32_t nhits;   //number of hits stored in the SoA
  uint32_t pad;     //pad of memory block (used for warp alignment, slighlty larger than 'nhits_')
};

namespace memory {
  namespace npointers {
    constexpr unsigned float_hgcuncalibrechits_soa = 6;   //number of float pointers in the uncalibrated rechits SoA
    constexpr unsigned uint32_hgcuncalibrechits_soa = 3;  //number of uint32_t pointers in the uncalibrated rechits SoA
    constexpr unsigned ntypes_hgcuncalibrechits_soa =
        2;  //number of different pointer types in the uncalibrated rechits SoA
  }         // namespace npointers
}  // namespace memory

#endif
