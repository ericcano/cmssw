#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include "DataFormats/DetId/interface/DetId.h"
#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalCellPositionsKernelImpl.cuh"

__global__ void fill_positions_from_detids(
    const hgcal_conditions::HeterogeneousPositionsConditionsESProduct* conds) {
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  for (unsigned int i = tid; i < conds->nelems_posmap; i += blockDim.x * gridDim.x) {

    if(i<conds->posmap.nCellsTot) { //this branch divergence will only affect one warp
      
      HeterogeneousHGCSiliconDetId did(conds->posmap.detid[i]);
      const float cU = static_cast<float>(did.cellU());
      const float cV = static_cast<float>(did.cellV());
      const float wU = static_cast<float>(did.waferU());
      const float wV = static_cast<float>(did.waferV());
      const float ncells = static_cast<float>(did.nCellsSide());
      const int32_t layer = did.layer();

      //based on `std::pair<float, float> HGCalDDDConstants::locateCell(const HGCSiliconDetId&, bool)
      const float r_x2 = conds->posmap.waferSize + conds->posmap.sensorSeparation;
      const float r = 0.5f * r_x2;
      const float sqrt3 = __fsqrt_rn(3.f);
      const float rsqrt3 = __frsqrt_rn(3.f);  //rsqrt: 1 / sqrt
      const float R = r_x2 * rsqrt3;
      const float n2 = ncells / 2.f;
      const float yoff_abs = rsqrt3 * r_x2;
      const float yoff = (layer % 2 == 1) ? yoff_abs : -1.f * yoff_abs;  //CHANGE according to Sunanda's reply
      float xpos = (-2.f * wU + wV) * r;
      float ypos = yoff + (1.5f * wV * R);
      const float R1 = __fdividef(conds->posmap.waferSize, 3.f * ncells);
      const float r1_x2 = R1 * sqrt3;
      xpos += (1.5f * (cV - ncells) + 1.f) * R1;
      ypos += (cU - 0.5f * cV - n2) * r1_x2;
      conds->posmap.x[i] =
        xpos;  // times side; multiply by -1 if one wants to obtain the position from the opposite endcap. CAREFUL WITH LATER DETECTOR ALIGNMENT!!!
      conds->posmap.y[i] = ypos;
    }

    else {

      HeterogeneousHGCScintillatorDetId did(conds->posmap.detid[i]);
      // const int32_t lay  = did.layer();
      // const int32_t iphi = did.iphi();
      // const int32_t ir   = did.iradiusAbs();
      
      // const double  phi  = (iphi - 0.5) * conds->scintCellSize(lay);
      // const int32_t type = (id.type() > 0) ? 1 : 0;
      // const double r = (((ir + 1) < static_cast<int32_t>(conds->radiusLayer_[type].size()))
      // 			? (0.5 * (conds->radiusLayer_[type][ir] + conds->radiusLayer_[type][ir - 1]))
      // 			: (1.5 * conds->radiusLayer_[type][ir] - 0.5 * conds->radiusLayer_[type][ir - 1]));

      // // times side; multiply by -1 if one wants to obtain the position from the opposite endcap. CAREFUL WITH LATER DETECTOR ALIGNMENT!!!
      // conds->posmap.x[i] = r*cos(phi);
      
      // conds->posmap.y[i] = r*sin(phi);
    }
  }
}

__global__ void print_positions_from_detids(
    const hgcal_conditions::HeterogeneousPositionsConditionsESProduct* conds) {
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int32_t layer;
  
  for (unsigned int i = tid; i < conds->nelems_posmap; i += blockDim.x * gridDim.x) {

    if(i<conds->posmap.nCellsTot) { //this branch divergence will only affect one warp
      
      HeterogeneousHGCSiliconDetId did(conds->posmap.detid[i]);
      layer = did.layer();

    }

    else {

      HeterogeneousHGCScintillatorDetId did(conds->posmap.detid[i]);
      layer = did.layer();

    }

    float posz = conds->posmap.zLayer[layer - 1];
    printf("PosX: %lf\t PosY: %lf\t Posz: %lf\n", conds->posmap.x[i], conds->posmap.y[i], posz);
  }
}

//eventually this can also be written in parallel
__device__ unsigned map_cell_index(const float& cu, const float& cv, const unsigned& ncells_side) {
  unsigned counter = 0;
  //left side of wafer
  for (int cellUmax = ncells_side, icellV = 0; cellUmax < 2 * ncells_side && icellV < ncells_side;
       ++cellUmax, ++icellV) {
    for (int icellU = 0; icellU <= cellUmax; ++icellU) {
      if (cu == icellU and cv == icellV)
        return counter;
      else
        counter += 1;
    }
  }
  //right side of wafer
  for (int cellUmin = 1, icellV = ncells_side; cellUmin <= ncells_side && icellV < 2 * ncells_side;
       ++cellUmin, ++icellV) {
    for (int icellU = cellUmin; icellU < 2 * ncells_side; ++icellU) {
      if (cu == icellU and cv == icellV)
        return counter;
      else
        counter += 1;
    }
  }
  printf("ERROR: The cell was not found!");
  return 99;
}

__device__
unsigned get_nlayers_ee(const hgcal_conditions::HeterogeneousPositionsConditionsESProduct* conds) {
  return conds->posmap.lastLayerEE - conds->posmap.firstLayerEE;
}

//returns the index of the positions of a specific cell
//performs several geometry-related shifts, and adds them at the end:
//   1) number of cells up to the layer being inspected
//   2) number of cells up to the waferUchunk in question, only in the layer being inspected
//   3) number of cells up to the waferV in question, only in the layer and waferUchunk being inspected
//   4) cell index within this layer, waferUchunk and waferV
//Note: a 'waferUchunk' represents the first dimension of a 2D squared grid of wafers, and includes multiple waferV
__device__ unsigned hash_function(const uint32_t& detid,
                                  const hgcal_conditions::HeterogeneousPositionsConditionsESProduct* conds) {
  unsigned shift_total = 0;

  DetId basedid(detid);
  if(basedid.det()==DetId::HGCalEE or basedid.det()==DetId::HGCalHSi)
    {
      HeterogeneousHGCSiliconDetId did(detid);
      const bool isEE = basedid.det()==DetId::HGCalEE ? true : false;
      
      const int32_t wU = did.waferU();
      const int32_t wV = did.waferV();
      const int32_t l = abs(did.layer());  //remove abs in case both endcaps are considered for x and y

      const unsigned thislayer = isEE ? l - conds->posmap.firstLayerEE
	: l - conds->posmap.firstLayerHEF; //refers to HEF only, starting one layer after conds->posmap.lastLayerEE
      const unsigned thisUwafer = isEE ? wU - conds->posmap.waferMinEE : wU - conds->posmap.waferMinHEF;
      const unsigned thisVwafer = isEE ? wV - conds->posmap.waferMinEE : wV - conds->posmap.waferMinEE;

      const unsigned nwafers1DEE =  conds->posmap.waferMaxEE  - conds->posmap.waferMinEE;
      const unsigned nwafers1DHEF = conds->posmap.waferMaxHEF - conds->posmap.waferMinHEF;

      //layer shift in terms of cell number
      unsigned ncells_up_to_thislayer = 0;
      for (unsigned q = 0; q < thislayer; ++q)
	ncells_up_to_thislayer += conds->posmap.nCellsLayer[q];

      //waferU shift in terms of cell number
      unsigned ncells_up_to_thisUwafer = 0;
      unsigned nwaferUchunks_up_to_this_layer = isEE ? thislayer * nwafers1DEE
	: get_nlayers_ee(conds) * nwafers1DEE //#waferU chunks in EE
	+ thislayer * nwafers1DHEF;           //#waferU chunks in HEF
      for (unsigned q = 0; q < thisUwafer; ++q)
	ncells_up_to_thisUwafer += conds->posmap.nCellsWaferUChunk[nwaferUchunks_up_to_this_layer + q];

      //waferV shift in terms of cell number
      unsigned ncells_up_to_thisVwafer = 0;
      const unsigned nwafers_up_to_thisLayer = isEE ? thislayer * nwafers1DEE * nwafers1DEE
	: get_nlayers_ee(conds) * nwafers1DEE * nwafers1DEE
	+ thislayer * nwafers1DHEF * nwafers1DHEF;
      const unsigned nwafers_up_to_thisUwafer = thisUwafer * (isEE ? nwafers1DEE : nwafers1DHEF);
      
      for (unsigned q = 0; q < thisVwafer; ++q)
	ncells_up_to_thisVwafer += conds->posmap.nCellsHexagon[nwafers_up_to_thisLayer + nwafers_up_to_thisUwafer + q];

      //cell shift in terms of cell number
      const unsigned cell_shift = map_cell_index(did.cellU(), did.cellV(), did.nCellsSide());
      shift_total = ncells_up_to_thislayer + ncells_up_to_thisUwafer + ncells_up_to_thisVwafer + cell_shift;
    }
  else if(basedid.det() == DetId::HGCalHSc)
    {
      const int32_t shift_silicon = conds->posmap.nCellsTot;
      
      HeterogeneousHGCScintillatorDetId did(detid);
      const int32_t layer = abs(did.layer());  //remove abs in case both endcaps are considered for x and y
      shift_total = shift_silicon + 0; //CALCULATE HAD SHIFT
    }
  else
    //nose: else if (det == DetId::Forward and DetId(detid).subdetId() == ForwardSubdetector::HFNose) {
    printf("WRONG detector!!!!!!!!!!\n");
  return shift_total;
}

__global__ void test(uint32_t detid_test,
                     const hgcal_conditions::HeterogeneousPositionsConditionsESProduct* conds) {
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid == 0) {
    //printf("Nelems: %u\n", static_cast<unsigned>(conds->nelems_posmap));
    for (unsigned i = 0; i < 1; ++i) {
      const unsigned shift = hash_function(detid_test, conds);  // 2416969935, 2552165379, ...
      //printf("id: cu: %d, cv: %d, wu: %d, wv: %d, ncells: %d, layer: %d\n", cU, cV, wU, wV, ncs, layer);
      printf("id: %u | shift: %u | x: %lf y: %lf\n",
             conds->posmap.detid[shift],
             shift,
             conds->posmap.x[shift],
             conds->posmap.y[shift]);
    }
  }
}
