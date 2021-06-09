#include <cuda.h>
#include <cuda_runtime.h>

#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalCLUEAlgoGPUHAD.h"

void HGCalCLUEAlgoGPUHAD::init_device(uint32_t nhits, const cudaStream_t &stream) {
  const std::array<uint32_t, clue_gpu::ntypes_hgcclue_inhadsoa> sizes_ = {
	   {clue_gpu::float_hgcclue_inhadsoa * sizeof(float),
	    clue_gpu::bool_hgcclue_inhadsoa * sizeof(bool)}
  };
  const uint32_t size_tot = std::accumulate(sizes_.begin(), sizes_.end(), 0);
  const uint32_t pad = calculate_padding(nhits);
  // mMem = allocate_soa_memory_block(size_tot, nhits, stream);

  // //set input SoA memory view
  // d_points.eta_        = reinterpret_cast<float *>(mMem.get());
  // d_points.phi_        = d_points.eta_ + pad;
  // d_points.layer_      = d_points.phi_ + pad;
  // d_points.energy_     = d_points.layer_ + pad;
  // d_points.sigmaNoise_ = d_points.energy_ + pad;
  // d_points.isSi_       = reinterpret_cast<bool *>(d_points.sigmaNoise_ + pad);

  // allocate_common_memory_blocks(nhits);
}

void HGCalCLUEAlgoGPUHAD::populate(const ConstHGCRecHitSoA& hits,
				   const hgcal_conditions::HeterogeneousPositionsConditionsESProduct* conds,
				   const cudaStream_t &stream) {

}

void HGCalCLUEAlgoGPUHAD::make_clusters(const unsigned nhits,
					const cudaStream_t &stream) {
  
}
