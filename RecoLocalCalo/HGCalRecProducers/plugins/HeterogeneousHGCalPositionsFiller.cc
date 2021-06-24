/////////////////////////////////////////////////////////////////////////////////
// The geometry is ordered in a predefined way
// The ordering must be followed when 
//    retrieving data directly in the GPUs
//    using the detids filled in this class
//
// Geometry ordering rule:
//
// - Consider the subdetectors (EE, HEF and HEB) as separate blocks
// - For each subdetector loop over all its layers
// - If the subdetectors is made of silicon:
//     - Loop over wafer U and V coordinates to get a single wafer
//       (waferU memory "chunks" contain waferV coordinates)
//     - Loop over cell U and V coordinates to get a single cell
//     - Fill detid using HGCSiliconDetId class
// - If the subdetectors is made of scintillator material:
//     - Loop over ieta and iphi !!!NOT TRIED YET!!!
//
// - Currently we are assuming both endcaps to be perfectly symmetric
//     and so are allocating half the memory
//
// All this information is stored in a single memory block.
// The CPU-to-GPU transfer is done in the HeterogeneousPositionsConditions class.
/////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <memory>
#include <numeric>
#include <cuda_runtime.h>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"

#include "CondFormats/HGCalObjects/interface/HeterogeneousHGCalPositionsConditions.h"
#include "CondFormats/DataRecord/interface/HeterogeneousHGCalPositionsConditionsRecord.h"
#include "RecoLocalCalo/HGCalRecProducers/plugins/KernelManagerHGCalCellPositions.h"
#include "CUDADataFormats/HGCal/interface/HGCConditions.h"

class HeterogeneousHGCalPositionsFiller : public edm::ESProducer {
public:
  explicit HeterogeneousHGCalPositionsFiller(const edm::ParameterSet&);
  ~HeterogeneousHGCalPositionsFiller() override;
  std::unique_ptr<HeterogeneousHGCalPositionsConditions> produce(
      const HeterogeneousHGCalPositionsConditionsRecord&);

private:
  edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> mGeomTokEE;
  edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> mGeomTokHEF;
  edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> mGeomTokHEB;

  void clear_conditions_();
  void reserve_conditions_(const HGCalDDDConstants*, const HGCalParameters*,
			   const HGCalDDDConstants*, const HGCalParameters*,
			   const HGCalDDDConstants*, const HGCalParameters*);
  void fill_conditions_silicon_(const HGCalDDDConstants*, const HGCalParameters*,
				const HGCalDDDConstants*, const HGCalParameters*);
  void fill_conditions_scintillator_(const HGCalDDDConstants*, const HGCalParameters*);

  hgcal_conditions::positions::HGCalPositionsMapping* mPosmap;

  const HGCalDDDConstants* mDDDEE  = nullptr;
  const HGCalDDDConstants* mDDDHEF = nullptr;
  const HGCalDDDConstants* mDDDHEB = nullptr;
};

HeterogeneousHGCalPositionsFiller::HeterogeneousHGCalPositionsFiller(const edm::ParameterSet& ps) {
  auto cc = setWhatProduced(this, "");
  mGeomTokEE = cc.consumesFrom<HGCalGeometry, IdealGeometryRecord>(
      edm::ESInputTag{"", "HGCalEESensitive"});
  mGeomTokHEF = cc.consumesFrom<HGCalGeometry, IdealGeometryRecord>(
      edm::ESInputTag{"", "HGCalHESiliconSensitive"});
  mGeomTokHEB = cc.consumesFrom<HGCalGeometry, IdealGeometryRecord>(
      edm::ESInputTag{"", "HGCalHEScintillatorSensitive"});

  mPosmap = new hgcal_conditions::positions::HGCalPositionsMapping();
}

HeterogeneousHGCalPositionsFiller::~HeterogeneousHGCalPositionsFiller() { delete mPosmap; }

void HeterogeneousHGCalPositionsFiller::clear_conditions_() {
  mPosmap->zLayer.clear();
  mPosmap->nCellsLayer.clear();
  mPosmap->nCellsWaferUChunk.clear();
  mPosmap->nCellsHexagon.clear();
  mPosmap->detid.clear();
}

void HeterogeneousHGCalPositionsFiller::reserve_conditions_(const HGCalDDDConstants* dEE,  const HGCalParameters* pEE,
							    const HGCalDDDConstants* dHEF, const HGCalParameters* pHEF,
							    const HGCalDDDConstants* dHEB, const HGCalParameters* pHEB) {

  //define help variables
  enum mDet { EE, HEF, /*Nose,*/ HEB, NDETS };
  const std::array<const HGCalDDDConstants*,NDETS> d = {{ dEE, dHEF, /*dNose*,*/ dHEB }};
  const std::array<const HGCalParameters*,NDETS> p   = {{ pEE, pHEF, /*pNose*,*/ pHEB }};
  const unsigned nsubSi = mDet::NDETS-1; //number of subdetectors made of Silicon (all except HEB)
  
  const std::array<int, mDet::NDETS> nlayers = {{ d[mDet::EE]->lastLayer(true)-d[mDet::EE]->firstLayer()+1,
						  d[mDet::HEF]->lastLayer(true)-d[mDet::HEF]->firstLayer()+1,
						  d[mDet::HEB]->lastLayer(true)-d[mDet::HEB]->firstLayer()+1 }};

  //store upper estimates for wafer and cell numbers for each subdetector
  std::array<int, nsubSi> upper_estimate_wafer_number_1D;
  for(int i=mDet::EE; i!=mDet::HEB; ++i) //skips HEB
    upper_estimate_wafer_number_1D[i] = nlayers[i]*(d[i]->waferMax()-d[i]->waferMin());

  std::array<int, nsubSi> upper_estimate_wafer_number_2D;
  for(int i=mDet::EE; i!=mDet::HEB; ++i) //skips HEB
    upper_estimate_wafer_number_2D[i] = upper_estimate_wafer_number_1D[i]*(d[i]->waferMax()-d[i]->waferMin());
  
  std::array<int, nsubSi> upper_estimate_cell_number;
  for(int i=mDet::EE; i!=mDet::HEB; ++i) //skips HEB
    upper_estimate_cell_number[i] = upper_estimate_wafer_number_2D[i] * 3 * 12 * 12;

  //Sum upper estimates for all silicon detectors and reserve sizes for positions conditions
  //The estimates for the number of cells:
  //  1) only take into account one endcap
  //  2) are very conservative
  auto acc = [](const auto& arr) noexcept -> int {
	       return std::accumulate(arr.begin(), arr.end(), 0);
	     };

  const int nLayersSi = acc(nlayers) - nlayers[mDet::HEB];
  mPosmap->zLayer.resize(nLayersSi);
  mPosmap->nCellsLayer.reserve(nLayersSi);

  const int nCellsUTot = acc(upper_estimate_wafer_number_1D);
  mPosmap->nCellsWaferUChunk.reserve(nCellsUTot);

  const int nCellsHexTot = acc(upper_estimate_wafer_number_2D);
  mPosmap->nCellsHexagon.reserve(nCellsHexTot);

  const int nCellsTot = acc(upper_estimate_cell_number);
  mPosmap->detid.reserve(nCellsTot);
  
  //set variables
  mPosmap->waferSize = static_cast<float>(p[mDet::EE]->waferSize_);
  assert(p[mDet::EE]->waferSize_ == p[mDet::HEF]->waferSize_);

  mPosmap->sensorSeparation = static_cast<float>(p[mDet::HEF]->sensorSeparation_);
  assert(p[mDet::EE]->sensorSeparation_ == p[mDet::HEF]->sensorSeparation_);
    
  mPosmap->firstLayerEE  = d[mDet::EE]->firstLayer();
  mPosmap->firstLayerHEF = d[mDet::HEF]->firstLayer();
  mPosmap->firstLayerHEB = d[mDet::HEB]->firstLayer();
  
  mPosmap->lastLayerEE  = d[mDet::EE]->lastLayer(true);
  mPosmap->lastLayerHEF = d[mDet::HEF]->lastLayer(true);
  mPosmap->lastLayerHEB = d[mDet::HEB]->lastLayer(true);
  
  mPosmap->waferMinEE  = d[mDet::EE]->waferMin();
  mPosmap->waferMinHEF = d[mDet::HEF]->waferMin();

  mPosmap->waferMaxEE  = d[mDet::EE]->waferMax();
  mPosmap->waferMaxHEF = d[mDet::HEF]->waferMax();

  //scintillator additional variables missing...
  // nlayers[mDet::HEB]
}

void HeterogeneousHGCalPositionsFiller::fill_conditions_silicon_(const HGCalDDDConstants* dEE, const HGCalParameters* pEE,
								 const HGCalDDDConstants* dHEF, const HGCalParameters* pHEF) {
  //define help variables
  enum mDet { EE, HEF, /*Nose,*/ NDETS };
  const std::array<const HGCalDDDConstants*,NDETS> d = {{ dEE, dHEF /*, dNose*/}};

  auto ee_layer_to_full_layer = [dEE](const auto& i) noexcept -> int {
				  return i - dEE->firstLayer();
				};
  auto hef_layer_to_full_layer = [dEE,dHEF](const auto& i) noexcept -> int {
				   return i - dHEF->firstLayer() + dEE->lastLayer(true);
				 };

  //fill the CPU position structure from the geometry
  unsigned sumCellsTot=0, sumCellsLayer=0, sumCellsWaferUChunk=0;

  //store detids following a geometry ordering
  for(unsigned x=0; x<d.size(); ++x) {
    
    for (int ilayer=d[x]->firstLayer(); ilayer<=d[x]->lastLayer(true); ++ilayer) {
      sumCellsLayer = 0;
      int layeridx = x==mDet::EE ? ee_layer_to_full_layer(ilayer) : hef_layer_to_full_layer(ilayer);
      mPosmap->zLayer[layeridx] =
	static_cast<float>(d[x]->waferZ(ilayer, true)); //originally a double
    
      for (int iwaferU=d[x]->waferMin(); iwaferU<d[x]->waferMax(); ++iwaferU) {
	sumCellsWaferUChunk = 0;

	for (int iwaferV=d[x]->waferMin(); iwaferV<d[x]->waferMax(); ++iwaferV) {
	  //0: fine; 1: coarseThin; 2: coarseThick (as defined in DataFormats/ForwardDetId/interface/HGCSiliconDetId.h)
	  int type = d[x]->waferType(ilayer, iwaferU, iwaferV);
	  int nCellsHexSide =
	    d[x]->numberCellsHexagon(ilayer, iwaferU, iwaferV, false);
	  int nCellsHexTotal =
	    d[x]->numberCellsHexagon(ilayer, iwaferU, iwaferV, true);
	
	  sumCellsLayer += nCellsHexTotal;
	  sumCellsWaferUChunk += nCellsHexTotal;
	  mPosmap->nCellsHexagon.push_back(nCellsHexTotal);

	  //left side of wafer
	  for (int cellUmax=nCellsHexSide, icellV=0; cellUmax<2*nCellsHexSide and icellV<nCellsHexSide;
	       ++cellUmax, ++icellV) {
	    for (int icellU = 0; icellU <= cellUmax; ++icellU) {
	      HGCSiliconDetId detid_(DetId::HGCalHSi, 1, type, ilayer, iwaferU, iwaferV, icellU, icellV);
	      mPosmap->detid.push_back(detid_.rawId());
	    }
	  }
	  //right side of wafer
	  for (int cellUmin=1, icellV=nCellsHexSide; cellUmin<=nCellsHexSide and icellV<2*nCellsHexSide;
	       ++cellUmin, ++icellV) {
	    for (int icellU = cellUmin; icellU < 2 * nCellsHexSide; ++icellU) {
	      HGCSiliconDetId detid_(DetId::HGCalHSi, 1, type, ilayer, iwaferU, iwaferV, icellU, icellV);
	      mPosmap->detid.push_back(detid_.rawId());
	    }
	  }
	}
	mPosmap->nCellsWaferUChunk.push_back(sumCellsWaferUChunk);
      }
      sumCellsTot += sumCellsLayer;
      mPosmap->nCellsLayer.push_back(sumCellsLayer);
    }

  } // for(auto &&x : d)

  mPosmap->nCellsTot = sumCellsTot;
}

void HeterogeneousHGCalPositionsFiller::fill_conditions_scintillator_(const HGCalDDDConstants* d, const HGCalParameters* p) {
  
  auto subfirst = [this](const auto& i) noexcept -> int {
  		    return i - mPosmap->firstLayerHEB;
  		  };

  //store detids following a geometry ordering
  for (int ilayer=mPosmap->firstLayerHEB; ilayer<=mPosmap->lastLayerHEB; ++ilayer) {
    subfirst(ilayer);
  }

}

std::unique_ptr<HeterogeneousHGCalPositionsConditions> HeterogeneousHGCalPositionsFiller::produce(
    const HeterogeneousHGCalPositionsConditionsRecord& iRecord) {

  clear_conditions_();
  
  auto geomEE = iRecord.getTransientHandle(mGeomTokEE);
  mDDDEE = &(geomEE->topology().dddConstants());
  
  auto geomHEF = iRecord.getTransientHandle(mGeomTokHEF);
  mDDDHEF = &(geomHEF->topology().dddConstants());

  auto geomHEB = iRecord.getTransientHandle(mGeomTokHEB);
  mDDDHEB = &(geomHEB->topology().dddConstants());

  reserve_conditions_( mDDDEE,  mDDDEE->getParameter(),
		       mDDDHEF, mDDDHEF->getParameter(),
		       mDDDHEB, mDDDHEB->getParameter() );

  fill_conditions_silicon_( mDDDEE, mDDDEE->getParameter(),
			    mDDDHEF, mDDDHEF->getParameter() );
  //fill_conditions_scintillator( mDDDHEB, mDDDHEB->getParameter() );

  std::unique_ptr<HeterogeneousHGCalPositionsConditions> up =
      std::make_unique<HeterogeneousHGCalPositionsConditions>(mPosmap);
  return up;
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"
DEFINE_FWK_EVENTSETUP_MODULE(HeterogeneousHGCalPositionsFiller);
