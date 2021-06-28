#ifndef CUDADataFormats_HGCal_HGCCLUESoA_h
#define CUDADataFormats_HGCal_HGCCLUESoA_h

#include <cstdint>
#include "CUDADataFormats/Common/interface/SoAmacros.h"

declare_SoA_template(HGCCLUESoADescriptor,
  SoA_column(float, rho),             /* energy density of the calibrated rechit */
  SoA_column(float, delta),           /* closest distance to a rechit with a higher density */
  SoA_column(int32_t, nearestHigher), /* index of the nearest rechit with a higher density */
  SoA_column(int32_t, clusterIndex),  /* cluster index the rechit belongs to */
  SoA_column(bool, isSeed)            /* is the rechit a cluster seed? */
  /* Note: isSeed is of type int in the CPU version to to std::vector optimizations */      
);


class HGCCLUESoA: public HGCCLUESoADescriptor {
public:
  HGCCLUESoA(std::byte* mem, size_t nElements): 
    HGCCLUESoADescriptor(mem, nElements) {}

  uint32_t nbytes;  //number of bytes of the SoA
  uint32_t nhits;   //number of hits stored in the SoA
  uint32_t pad;     //pad of memory block (used for warp alignment, slightly larger than 'nhits_')
};

#endif  //CUDADataFormats_HGCal_HGCCLUESoA_h
