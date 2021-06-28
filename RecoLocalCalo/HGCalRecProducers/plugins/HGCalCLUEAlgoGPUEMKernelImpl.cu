#include <cuda.h>
#include <cuda_runtime.h>

#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalCellPositionsKernelImpl.cuh"
#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalCLUEAlgoGPUEMKernelImpl.cuh"

#include "HeterogeneousCore/CUDAUtilities/interface/VecArray.h"

__device__
bool is_energy_valid(float en) {
  return en != clue_gpu::unphysicalEnergy;
} // kernel

__global__
void kernel_fill_input_soa(ConstHGCRecHitSoA hits,
			   clue_gpu::HGCCLUEInputSoAEM in,
			   const hgcal_conditions::HeterogeneousPositionsConditionsESProduct* conds,
			   float ecut)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned i = tid; i < hits.nhits; i += blockDim.x * gridDim.x) {
    in.sigmaNoise[i] = hits.sigmaNoise[i];
    in.energy[i] = (hits.energy[i]<ecut*in.sigmaNoise[i]) ? clue_gpu::unphysicalEnergy : hits.energy[i];

    //logic in https://github.com/cms-sw/cmssw/blob/master/RecoLocalCalo/HGCalRecProducers/plugins/HGCalCellPositionsKernelImpl.cu
    const unsigned shift = hash_function(hits.id[i], conds);

    in.x[i] = conds->posmap.x[shift];
    in.y[i] = conds->posmap.y[shift];

    if(shift<static_cast<unsigned>(conds->posmap.nCellsTot)) { //silicon
      HeterogeneousHGCSiliconDetId did(hits.id[i]);
      in.layer[i] = abs(did.layer());  //remove abs if both endcaps are considered for x and y
    }
    else { //scintillator
      HeterogeneousHGCScintillatorDetId did(hits.id[i]);
      in.layer[i] = abs(did.layer());  //remove abs if both endcaps are considered for x and y
    }
  }
} // kernel


__global__
void kernel_compute_histogram( LayerTilesGPU *hist,
			       clue_gpu::HGCCLUEInputSoAEM in,
			       int numberOfPoints
			       )
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < numberOfPoints) {
    if( is_energy_valid(in.energy[i]) )
      // push index of points into tiles
      hist[in.layer[i]].fill(in.x[i], in.y[i], i);
  }
} // kernel

__global__
void kernel_calculate_density( LayerTilesGPU *hist, 
			       clue_gpu::HGCCLUEInputSoAEM in,
			       HGCCLUESoA out,
			       float dc,
			       int numberOfPoints
			       ) 
{ 
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < numberOfPoints){
    double rhoi{0.};

    if( is_energy_valid(in.energy[i]) ) {
      
      int layeri = in.layer[i];
      float xi = in.x[i];
      float yi = in.y[i];
      
      // get search box 
      int4 search_box = hist[layeri].searchBox(xi-dc, xi+dc, yi-dc, yi+dc);

      // loop over bins in the search box
      for(int xBin = search_box.x; xBin < search_box.y+1; ++xBin) {
	for(int yBin = search_box.z; yBin < search_box.w+1; ++yBin) {

	  // get the id of this bin
	  int binId = hist[layeri].getGlobalBinByBin(xBin,yBin);
	  // get the size of this bin
	  int binSize  = hist[layeri][binId].size();

	  // interate inside this bin
	  for (int binIter = 0; binIter < binSize; binIter++) {
	    int j = hist[layeri][binId][binIter];

	    if( is_energy_valid(in.energy[j]) ) {
	      // query N_{mDc}(i)
	      float xj = in.x[j];
	      float yj = in.y[j];
	      float dist_ij = std::sqrt((xi-xj)*(xi-xj) + (yi-yj)*(yi-yj));
	      if(dist_ij <= dc) { 
		// sum weights within N_{mDc}(i)
		rhoi += (i == j ? 1.f : 0.5f) * in.energy[j];
	      }
	    }
	  
	  } // end of interate inside this bin
	}
      } // end of loop over bins in search box
    }
    
    out[i].rho = (float)rhoi;
  }
} //kernel


__global__
void kernel_calculate_distanceToHigher(LayerTilesGPU* hist, 
				       clue_gpu::HGCCLUEInputSoAEM in,
				       HGCCLUESoA out,
				       float outlierDeltaFactor,
				       float dc,
				       int numberOfPoints
				       )
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  float dm = outlierDeltaFactor * dc;

  if (i < numberOfPoints){

    float deltai = std::numeric_limits<float>::max();
    int nearestHigheri = -1;

    if( is_energy_valid(in.energy[i]) ) {
      
      int layeri = in.layer[i];
      float xi = in.x[i];
      float yi = in.y[i];
      float rhoi = out[i].rho;

      // get search box 
      int4 search_box = hist[layeri].searchBox(xi-dm, xi+dm, yi-dm, yi+dm);

      // loop over all bins in the search box
      for(int xBin = search_box.x; xBin < search_box.y+1; ++xBin) {
	for(int yBin = search_box.z; yBin < search_box.w+1; ++yBin) {
	  // get the id of this bin
	  int binId = hist[layeri].getGlobalBinByBin(xBin,yBin);
	  // get the size of this bin
	  int binSize  = hist[layeri][binId].size();

	  // interate inside this bin
	  for (int binIter = 0; binIter < binSize; binIter++) {
	    int j = hist[layeri][binId][binIter];

	    if( is_energy_valid(in.energy[j]) ) {
	      // query N'_{dm}(i)
	      float xj = in.x[j];
	      float yj = in.y[j];
	      float dist_ij = std::sqrt((xi-xj)*(xi-xj) + (yi-yj)*(yi-yj));
	      bool foundHigher = (out[j].rho > rhoi);
	      // in the rare case where rho is the same, use detid
	      foundHigher = foundHigher || ( (out[j].rho == rhoi) && (j>i));
	      if(foundHigher && dist_ij <= dm) { // definition of N'_{dm}(i)
		// find the nearest point within N'_{dm}(i)
		if (dist_ij<deltai) {
		  // update deltai and nearestHigheri
		  deltai = dist_ij;
		  nearestHigheri = j;
		}
	      }
	    }
	  } // end of interate inside this bin
	}
      } // end of loop over bins in search box

    }
    
    out[i].delta = deltai;
    out[i].nearestHigher = nearestHigheri;
  }
} //kernel



__global__
void kernel_find_clusters( cms::cuda::VecArray<int,clue_gpu::maxNSeeds>* d_seeds,
			   cms::cuda::VecArray<int,clue_gpu::maxNFollowers>* d_followers,
			   clue_gpu::HGCCLUEInputSoAEM in,
			   HGCCLUESoA out,
			   float outlierDeltaFactor, float dc, float kappa,
			   int numberOfPoints
			   ) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;


  if (i < numberOfPoints and is_energy_valid(in.energy[i])) {
    // initialize clusterIndex
    out[i].clusterIndex = -1;
    // determine seed or outlier
    float deltai = out[i].delta;
    float rhoi = out[i].rho;
    float rhoc = kappa * in.sigmaNoise[i];
    bool isSeed = (deltai > dc) && (rhoi >= rhoc);
    bool isOutlier = (deltai > outlierDeltaFactor * dc) && (rhoi < rhoc);

    if (isSeed) {
      // set isSeed as 1
      out[i].isSeed = 1;
      d_seeds[0].push_back(i); // head of d_seeds
    } else {
      if (!isOutlier) {
        assert(out[i].nearestHigher < numberOfPoints);
        // register as follower of its nearest higher
        d_followers[out[i].nearestHigher].push_back(i);  
      }
    }
  }
} //kernel


__global__
void kernel_assign_clusters( const cms::cuda::VecArray<int,clue_gpu::maxNSeeds>* d_seeds, 
			     const cms::cuda::VecArray<int,clue_gpu::maxNFollowers>* d_followers,
			     HGCCLUESoA out, int numberOfPoints)
{
  
  int idxCls = blockIdx.x * blockDim.x + threadIdx.x;
  const auto& seeds = d_seeds[0];
  const auto nSeeds = seeds.size();
  if (idxCls < nSeeds){

    int localStack[clue_gpu::localStackSizePerSeed] = {-1};
    int localStackSize = 0;

    // asgine cluster to seed[idxCls]
    int idxThisSeed = seeds[idxCls];
    out[idxThisSeed].clusterIndex = idxCls;
    // push_back idThisSeed to localStack
    localStack[localStackSize] = idxThisSeed;
    localStackSize++;
    // process all elements in localStack
    while (localStackSize>0){
      // get last element of localStack
      int idxEndOflocalStack = localStack[localStackSize-1];

      int temp_clusterIndex = out[idxEndOflocalStack].clusterIndex;
      // pop_back last element of localStack
      localStack[localStackSize-1] = -1;
      localStackSize--;
      
      // loop over followers of last element of localStack
      for( int j : d_followers[idxEndOflocalStack]){
        // // pass id to follower
        out[j].clusterIndex = temp_clusterIndex;
        // push_back follower to localStack
        //localStack[localStackSize] = j;
        localStackSize++;
      }
    }
  }
} // kernel
