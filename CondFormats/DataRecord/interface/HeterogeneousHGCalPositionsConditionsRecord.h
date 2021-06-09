#ifndef CondFormats_DataRecord_HeterogeneousHGCalPositionsConditionsRecord_h
#define CondFormats_DataRecord_HeterogeneousHGCalPositionsConditionsRecord_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/Utilities/interface/mplVector.h"

class HeterogeneousHGCalPositionsConditionsRecord
    : public edm::eventsetup::DependentRecordImplementation<HeterogeneousHGCalPositionsConditionsRecord,
                                                            edm::mpl::Vector<IdealGeometryRecord>> {};

#endif  //CondFormats_DataRecord_HeterogeneousHGCalPositionsConditionsRecord_h
