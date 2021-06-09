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

#include "RecoLocalCalo/HGCalRecProducers/plugins/HeterogeneousHGCalPositionsFiller.h"

HeterogeneousHGCalPositionsFiller::HeterogeneousHGCalPositionsFiller(const edm::ParameterSet& ps) {
  geomTokEE_ = setWhatProduced(this).consumesFrom<HGCalGeometry, IdealGeometryRecord>(
      edm::ESInputTag{"", "HGCalEESensitive"});
  geomTokHEF_ = setWhatProduced(this).consumesFrom<HGCalGeometry, IdealGeometryRecord>(
      edm::ESInputTag{"", "HGCalHESiliconSensitive"});
  geomTokHEB_ = setWhatProduced(this).consumesFrom<HGCalGeometry, IdealGeometryRecord>(
      edm::ESInputTag{"", "HGCalHEScintillatorSensitive"});

  mPosmap = new hgcal_conditions::positions::HGCalPositionsMapping();
}

HeterogeneousHGCalPositionsFiller::~HeterogeneousHGCalPositionsFiller() { delete mPosmap; }

void HeterogeneousHGCalPositionsFiller::clear_conditions_() {
  mPosmap->zLayer.clear();
  mPosmap->nCellsSubDet.clear();
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

  const int nLayersTot = acc(nlayers);
  mPosmap->zLayer.resize(nLayersTot);
  mPosmap->nCellsSubDet.reserve(nsubSi);
  mPosmap->nCellsLayer.reserve(nLayersTot);

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
    
  mPosmap->firstLayerSi = d[mDet::EE]->firstLayer();
  assert(mPosmap->firstLayerSi == 1);  //otherwise the loop over the layers has to be changed
  mPosmap->firstLayerSci = d[mDet::HEB]->firstLayer();
  assert(mPosmap->firstLayerSci == 1);  //otherwise the loop over the layers has to be changed
  
  mPosmap->lastLayerSi = d[mDet::HEF]->lastLayer(true);
  mPosmap->lastLayerSci = d[mDet::HEB]->lastLayer(true);
  
  mPosmap->waferMin = d[mDet::EE]->waferMin();
  assert(d[mDet::EE]->waferMin() == d[mDet::HEF]->waferMin());
  mPosmap->waferMax = d[mDet::EE]->waferMax();
  assert(d[mDet::EE]->waferMax() == d[mDet::HEF]->waferMax());

  //scintillator additional variables missing...
}

void HeterogeneousHGCalPositionsFiller::fill_conditions_silicon_(const HGCalDDDConstants* dEE, const HGCalParameters* pEE,
								 const HGCalDDDConstants* dHEF, const HGCalParameters* pHEF) {
  //define help variables
  enum mDet { EE, HEF, /*Nose,*/ NDETS };
  const std::array<const HGCalDDDConstants*,NDETS> d = {{ dEE, dHEF /*, dNose*/}};

  auto subfirst = [this](const auto& i) noexcept -> int {
  		    return i - this->mPosmap->firstLayerSi;
  		  };

  //fill the CPU position structure from the geometry
  unsigned sumCellsTot=0, sumCellsSubDet=0, sumCellsLayer, sumCellsWaferUChunk;

  //store detids following a geometry ordering
  for (int ilayer=mPosmap->firstLayerSi; ilayer<=mPosmap->lastLayerSi; ++ilayer) {
    sumCellsLayer = 0;
    mPosmap->zLayer[subfirst(ilayer)] =
      static_cast<float>(d[mDet::EE]->waferZ(ilayer, true)); //originally a double
    assert(d[mDet::EE]->waferZ(ilayer, true) == d[mDet::HEF]->waferZ(ilayer, true));
    
    for (int iwaferU=mPosmap->waferMin; iwaferU<mPosmap->waferMax; ++iwaferU) {
      sumCellsWaferUChunk = 0;

      for (int iwaferV=mPosmap->waferMin; iwaferV<mPosmap->waferMax; ++iwaferV) {
        //0: fine; 1: coarseThin; 2: coarseThick (as defined in DataFormats/ForwardDetId/interface/HGCSiliconDetId.h)
        int type = d[mDet::EE]->waferType(ilayer, iwaferU, iwaferV);
	assert(type == d[mDet::HEF]->waferType(ilayer, iwaferU, iwaferV));

        int nCellsHexSide =
	  d[mDet::EE]->numberCellsHexagon(ilayer, iwaferU, iwaferV, false);
	assert(nCellsHexSide ==
	       d[mDet::HEF]->numberCellsHexagon(ilayer, iwaferU, iwaferV, false));
	
        int nCellsHexTotal =
	  d[mDet::EE]->numberCellsHexagon(ilayer, iwaferU, iwaferV, true);
	assert(nCellsHexTotal ==
	       d[mDet::HEF]->numberCellsHexagon(ilayer, iwaferU, iwaferV, true));
	
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
    
    if( ilayer == subfirst(d[mDet::EE]->lastLayer(true)) or
	ilayer == subfirst(d[mDet::HEF]->lastLayer(true)) ) {
      mPosmap->nCellsSubDet.push_back(sumCellsSubDet);
      sumCellsSubDet=0;
    }
  }

  mPosmap->nCellsTot = std::accumulate(mPosmap->nCellsSubDet.begin(), mPosmap->nCellsSubDet.end(), 0);
}

void HeterogeneousHGCalPositionsFiller::fill_conditions_scintillator_(const HGCalDDDConstants* d, const HGCalParameters* p) {
}

std::unique_ptr<HeterogeneousHGCalPositionsConditions> HeterogeneousHGCalPositionsFiller::produce(
    const HeterogeneousHGCalPositionsConditionsRecord& iRecord) {

  clear_conditions_();
  
  auto geomEE = iRecord.getTransientHandle(geomTokEE_);
  mDDDEE = &(geomEE->topology().dddConstants());
  
  auto geomHEF = iRecord.getTransientHandle(geomTokHEF_);
  mDDDHEF = &(geomHEF->topology().dddConstants());

  auto geomHEB = iRecord.getTransientHandle(geomTokHEB_);
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
