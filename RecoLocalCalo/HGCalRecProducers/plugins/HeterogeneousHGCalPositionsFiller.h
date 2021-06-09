#ifndef RecoLocalCalo_HGCalESProducers_HeterogeneousHGCalPositionsFiller_h
#define RecoLocalCalo_HGCalESProducers_HeterogeneousHGCalPositionsFiller_h

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
  edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> geomTokEE_, geomTokHEF_, geomTokHEB_;

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

#endif  //RecoLocalCalo_HGCalESProducers_HeterogeneousHGCalPositionsFiller_h
