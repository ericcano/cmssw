#include <cuda.h>
#include <cuda_runtime.h>

#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalCLUEAlgoGPUBase.h"

HGCalCLUEAlgoGPUBase::HGCalCLUEAlgoGPUBase(float pDc, float pKappa, float pEcut,
					   float pOutlierDeltaFactor,
					   const HGCCLUESoA& pCLUESoA)
  : mDc(pDc), mKappa(pKappa), mEcut(pEcut), mOutlierDeltaFactor(pOutlierDeltaFactor), mCLUESoA(pCLUESoA)
{}

HGCalCLUEAlgoGPUBase::~HGCalCLUEAlgoGPUBase() { free_device(); }
    
void HGCalCLUEAlgoGPUBase::free_device() {
  // algorithm internal variables
  cudaFree(mDevHist);
  cudaFree(mDevSeeds);
  cudaFree(mDevFollowers);
}

void HGCalCLUEAlgoGPUBase::allocate_common_memory_blocks(uint32_t nhits) {
  cudaMalloc(&mDevHist, sizeof(LayerTilesGPU) * NLAYERS);
  cudaMalloc(&mDevSeeds, sizeof(cms::cuda::VecArray<int,clue_gpu::maxNSeeds>) );
  cudaMalloc(&mDevFollowers, sizeof(cms::cuda::VecArray<int,clue_gpu::maxNFollowers>)*nhits);
}

uint32_t HGCalCLUEAlgoGPUBase::calculate_padding(uint32_t nhits) {
  //align to warp boundary (assumption: warpSize = 32)
  return ((nhits - 1) / 32 + 1) * 32;
}

float HGCalCLUEAlgoGPUBase::calculate_block_multiplicity(unsigned nelements, unsigned nthreads) {
  return ceil(nelements/static_cast<float>(nthreads));
}

cms::cuda::device::unique_ptr<std::byte[]>
HGCalCLUEAlgoGPUBase::allocate_soa_memory_block(uint32_t st, uint32_t nhits, const cudaStream_t &stream) {
  const uint32_t pad = calculate_padding(nhits);
  return cms::cuda::make_device_unique<std::byte[]>(pad * st, stream);
}
