#ifndef RecoLocalCalo_HGCalRecProducers_HGCalCLUEAlgoGPUEMKernelImpl_cuh
#define RecoLocalCalo_HGCalRecProducers_HGCalCLUEAlgoGPUEMKernelImpl_cuh

#include <cuda_runtime.h>
#include <cuda.h>

#include "CUDADataFormats/HGCal/interface/HGCRecHitSoA.h"
#include "CUDADataFormats/HGCal/interface/ConstHGCRecHitSoA.h"
#include "CUDADataFormats/HGCal/interface/HGCCLUESoA.h"
#include "CUDADataFormats/HGCal/interface/HGCConditions.h"

#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalCLUEAlgoGPUBase.h"
#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalCLUEAlgoGPUEM.h"

#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"

__global__
void kernel_fill_input_soa(ConstHGCRecHitSoA hits,
			   clue_gpu::HGCCLUEInputSoAEM in,
			   const hgcal_conditions::HeterogeneousPositionsConditionsESProduct* conds,
			   float ecut);

__global__
void kernel_compute_histogram( LayerTilesGPU *hist,
			       clue_gpu::HGCCLUEInputSoAEM in,
			       int numberOfPoints
			       );

__global__
void kernel_calculate_density( LayerTilesGPU *hist, 
			       clue_gpu::HGCCLUEInputSoAEM in,
			       HGCCLUESoA out,
			       float dc,
			       int numberOfPoints
			       );

__global__
void kernel_calculate_distanceToHigher(LayerTilesGPU* hist, 
				       clue_gpu::HGCCLUEInputSoAEM in,
				       HGCCLUESoA out,
				       float outlierDeltaFactor,
				       float dc,
				       int numberOfPoints
				       );

__global__
void kernel_find_clusters( cms::cuda::VecArray<int,clue_gpu::maxNSeeds>* d_seeds,
			   cms::cuda::VecArray<int,clue_gpu::maxNFollowers>* d_followers,
			   clue_gpu::HGCCLUEInputSoAEM in,
			   HGCCLUESoA out,
			   float outlierDeltaFactor, float dc, float kappa,
			   int numberOfPoints
			   );

__global__
void kernel_assign_clusters( const cms::cuda::VecArray<int,clue_gpu::maxNSeeds>* d_seeds, 
			     const cms::cuda::VecArray<int,clue_gpu::maxNFollowers>* d_followers,
			     HGCCLUESoA out, int numberOfPoints);

#endif //RecoLocalCalo_HGCalRecProducers_HGCalCLUEAlgoGPUEMKernelImpl_cuh
