#ifndef RecoLocalCalo_HGCalRecProducer_HGCalCLUEAlgoGPUEM_h
#define RecoLocalCalo_HGCalRecProducer_HGCalCLUEAlgoGPUEM_h

#include <math.h>
#include <limits>
#include <numeric>

#include <cuda_runtime.h>
#include <cuda.h>

#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalCLUEAlgoGPUBase.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/LayerTilesGPU.h"
#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalCLUEAlgoGPUEMKernelImpl.cuh"

#include "CUDADataFormats/HGCal/interface/HGCRecHitSoA.h"
#include "CUDADataFormats/HGCal/interface/ConstHGCRecHitSoA.h"
#include "CUDADataFormats/HGCal/interface/HGCCLUESoA.h"
#include "CUDADataFormats/HGCal/interface/HGCConditions.h"

namespace clue_gpu {
  //number of float pointers in the CLUE EM SoA
  constexpr unsigned float_hgcclue_inemsoa = 4;
  constexpr unsigned int32_hgcclue_inemsoa = 1;
  //number of different pointer types in the CLUE EM SoA
  constexpr unsigned ntypes_hgcclue_inemsoa = 2;
} // namespace npointers

class HGCalCLUEAlgoGPUEM final: public HGCalCLUEAlgoGPUBase {
public:
  HGCalCLUEAlgoGPUEM(float, float, float, float,
		     const HGCCLUESoA&);
  ~HGCalCLUEAlgoGPUEM() = default;

  void populate(const ConstHGCRecHitSoA&,
		const hgcal_conditions::HeterogeneousPositionsConditionsESProduct*,
		const cudaStream_t&) override;
  void make_clusters(const unsigned, const cudaStream_t&) override;

private:
  static constexpr unsigned mNThreadsEM = 1024;
  clue_gpu::HGCCLUEInputSoAEM mDevPoints;

  void set_input_SoA_layout(const uint32_t, const cudaStream_t&);
};

#endif // RecoLocalCalo_HGCalRecProducer_HGCalCLUEAlgoGPUEM_h
