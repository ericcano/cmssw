#include <cuda.h>
#include <cuda_runtime.h>

#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalCLUEAlgoGPUBase.h"
#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalCLUEAlgoGPUEM.h"

HGCalCLUEAlgoGPUEM::HGCalCLUEAlgoGPUEM(float dc, float kappa, float ecut, float outlierDeltaFactor,
				       const HGCCLUESoA& hits_soa)
  : HGCalCLUEAlgoGPUBase(dc, kappa, ecut, outlierDeltaFactor, hits_soa)
{}

void HGCalCLUEAlgoGPUEM::set_input_SoA_layout(const uint32_t nhits, const cudaStream_t &stream) {
  const std::array<uint32_t, clue_gpu::ntypes_hgcclue_inemsoa> sizes_ = {
		{clue_gpu::float_hgcclue_inemsoa * sizeof(float),
		 clue_gpu::int32_hgcclue_inemsoa * sizeof(int32_t)}
  };
  const uint32_t size_tot = std::accumulate(sizes_.begin(), sizes_.end(), 0);
  mMem = allocate_soa_memory_block(size_tot, nhits, stream);
  const uint32_t pad = calculate_padding(nhits);
  
  //set input SoA memory view
  mDevPoints.x          = reinterpret_cast<float *>(mMem.get());
  mDevPoints.y          = mDevPoints.x      + pad;
  mDevPoints.energy     = mDevPoints.y      + pad;
  mDevPoints.sigmaNoise = mDevPoints.energy + pad;
  mDevPoints.layer      = reinterpret_cast<int32_t *>(mDevPoints.sigmaNoise + pad);

  mDevPoints.pad = pad;
}

void HGCalCLUEAlgoGPUEM::populate(const ConstHGCRecHitSoA& hits,
				  const hgcal_conditions::HeterogeneousPositionsConditionsESProduct* conds,
				  const cudaStream_t& stream) {
  const unsigned nhits = hits.nhits;
  set_input_SoA_layout(nhits, stream);
  allocate_common_memory_blocks(nhits);

  const dim3 blockSize(mNThreadsEM,1,1);
  const dim3 gridSize(ceil(nhits/static_cast<float>(blockSize.x)),1,1);

  kernel_fill_input_soa<<<gridSize,blockSize>>>(hits, mDevPoints, conds, mEcut);
}

void HGCalCLUEAlgoGPUEM::make_clusters(const unsigned nhits,
				       const cudaStream_t &stream) {
  const dim3 blockSize(mNThreadsEM,1,1);
  const dim3 gridSize(ceil(nhits/static_cast<float>(blockSize.x)),1,1);

  ////////////////////////////////////////////
  // calculate rho, delta and find seeds
  // 1 point per thread
  ////////////////////////////////////////////
  kernel_compute_histogram<<<gridSize,blockSize,0,stream>>>(mDevHist, mDevPoints, nhits);
  kernel_calculate_density<<<gridSize,blockSize,0,stream>>>(mDevHist, mDevPoints, mCLUESoA,
							    mDc, nhits);
  kernel_calculate_distanceToHigher<<<gridSize,blockSize,0,stream>>>(mDevHist, mDevPoints, mCLUESoA,
								     mOutlierDeltaFactor, mDc,
								     nhits);
  kernel_find_clusters<<<gridSize,blockSize,0,stream>>>(mDevSeeds, mDevFollowers,
					       mDevPoints, mCLUESoA,
					       mOutlierDeltaFactor, mDc, mKappa,
					       nhits);
  
  ////////////////////////////////////////////
  // assign clusters
  // 1 point per seeds
  ////////////////////////////////////////////
  const dim3 gridSize_nseeds(ceil(clue_gpu::maxNSeeds/1024.0),1,1);
  kernel_assign_clusters<<<gridSize_nseeds,blockSize,0,stream>>>(mDevSeeds, mDevFollowers, mCLUESoA,
								 nhits);
}
