#ifndef HGCalCLUEAlgoGPUHAD_h
#define HGCalCLUEAlgoGPUHAD_h

#include <math.h>
#include <limits>
#include <numeric>

#include <cuda_runtime.h>
#include <cuda.h>

#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalCLUEAlgoGPUBase.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/LayerTilesGPU.h"
#include "CUDADataFormats/HGCal/interface/HGCRecHitSoA.h"


namespace clue_gpu {
  class HGCCLUEInputSoAHAD {
  public:
    float *eta; //x position of the calibrated rechit
    float *phi; //y position of the calibrated rechit
    float *energy; //calibrated energy of the rechit
    float *sigmaNoise; //calibrated noise of the rechit cell
    int32_t *layer; //layer position of the calibrated rechit
    bool *isSi; //whether the rechit was detected in silicon or scintillator
    
    uint32_t pad_; //pad of memory block (used for warp alignment, slightly larger than 'nhits_')
  };

  //number of float pointers in the CLUE HAD SoA
  constexpr unsigned float_hgcclue_inhadsoa = 4;
  constexpr unsigned int32_hgcclue_inhadsoa = 1;
  constexpr unsigned bool_hgcclue_inhadsoa = 1;
  //number of different pointer types in the CLUE HAD SoA
  constexpr unsigned ntypes_hgcclue_inhadsoa = 3;
} // namespace npointers

class HGCalCLUEAlgoGPUHAD final: public HGCalCLUEAlgoGPUBase {
 public:
  HGCalCLUEAlgoGPUHAD(float, float, float, float,
		      const HGCRecHitSoA&);
  ~HGCalCLUEAlgoGPUHAD();

  void populate(const ConstHGCRecHitSoA&,
		const hgcal_conditions::HeterogeneousPositionsConditionsESProduct*,
		const cudaStream_t&) override;
  void make_clusters(const unsigned, const cudaStream_t&) override;

 private:
  static constexpr unsigned mNThreadsHAD = 1024;
  clue_gpu::HGCCLUEInputSoAEM mDevPointsEM;
  clue_gpu::HGCCLUEInputSoAHAD mDevPointsHAD;

  void init_device(uint32_t, const cudaStream_t&);
  void free_device();
};

#endif // HGCalCLUEAlgoGPUHAD_h
