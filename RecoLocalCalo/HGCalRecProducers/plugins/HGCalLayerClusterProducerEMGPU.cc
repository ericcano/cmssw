#ifndef __RecoLocalCalo_HGCRecProducers_HGCalLayerClusterProducerEMGPU_H__
#define __RecoLocalCalo_HGCRecProducers_HGCalLayerClusterProducerEMGPU_H__

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/PluginDescription.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalCLUEAlgoGPUEM.h"

#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDACore/interface/ContextState.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "CUDADataFormats/HGCal/interface/HGCRecHitSoA.h"
#include "CUDADataFormats/HGCal/interface/HGCRecHitGPUProduct.h"
#include "CUDADataFormats/HGCal/interface/HGCCLUEGPUProduct.h"

#include "CondFormats/HGCalObjects/interface/HeterogeneousHGCalPositionsConditions.h"
#include "CondFormats/DataRecord/interface/HeterogeneousHGCalPositionsConditionsRecord.h"


class HGCalLayerClusterProducerEMGPU : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  HGCalLayerClusterProducerEMGPU(const edm::ParameterSet&);
  ~HGCalLayerClusterProducerEMGPU() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  float mDc, mKappa, mEcut, mOutlierDeltaFactor;
  edm::ESGetToken<HeterogeneousHGCalPositionsConditions,
		  HeterogeneousHGCalPositionsConditionsRecord> gpuPositionsTok_;
  edm::EDGetTokenT<cms::cuda::Product<HGCRecHitGPUProduct>> InEEToken;
  edm::EDPutTokenT<cms::cuda::Product<HGCCLUEGPUProduct>> OutEEToken;
  cms::cuda::ContextState ctxState_;
  HGCCLUEGPUProduct mClusters;
};

DEFINE_FWK_MODULE(HGCalLayerClusterProducerEMGPU);

HGCalLayerClusterProducerEMGPU::HGCalLayerClusterProducerEMGPU(const edm::ParameterSet& ps)
  : mDc(ps.getParameter<double>("dc")),
    mKappa(ps.getParameter<double>("kappa")),
    mEcut(ps.getParameter<double>("ecut")),
    mOutlierDeltaFactor(ps.getParameter<double>("outlierdeltafactor")),
    gpuPositionsTok_(esConsumes<HeterogeneousHGCalPositionsConditions,
		     HeterogeneousHGCalPositionsConditionsRecord>()),
    InEEToken{consumes<cms::cuda::Product<HGCRecHitGPUProduct>>(ps.getParameter<edm::InputTag>("HGCEEInputGPU"))},
    OutEEToken{produces<cms::cuda::Product<HGCCLUEGPUProduct>>()}
{}

void HGCalLayerClusterProducerEMGPU::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // hgcalLayerClusters
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("HGCEEInputGPU", edm::InputTag("EERecHitGPUProd"));
  descriptions.add("hgcalLayerClustersGPU", desc);
}

void HGCalLayerClusterProducerEMGPU::acquire(edm::Event const& event,
					   edm::EventSetup const& setup,
					   edm::WaitingTaskWithArenaHolder w) {
  cms::cuda::ScopedContextAcquire ctx{event.streamID(), std::move(w), ctxState_};
  const auto& eeHits = ctx.get(event, InEEToken);
  const unsigned nhits(eeHits.nHits());
  
  mClusters = HGCCLUEGPUProduct(nhits, ctx.stream());

  //retrieve HGCAL positions conditions data
  auto hPosConds = setup.getHandle(gpuPositionsTok_);
  const auto* gpuPositionsConds = hPosConds->getHeterogeneousConditionsESProductAsync(ctx.stream());
  
  HGCalCLUEAlgoGPUEM algo(mDc, mKappa, mEcut, mOutlierDeltaFactor,
			  mClusters.get());
  algo.populate(eeHits.get(), gpuPositionsConds, ctx.stream());
  algo.make_clusters(nhits, ctx.stream());
}

void HGCalLayerClusterProducerEMGPU::produce(edm::Event& event,
					     const edm::EventSetup& es) {
  cms::cuda::ScopedContextProduce ctx{ctxState_};
  ctx.emplace(event, OutEEToken, std::move(mClusters));
}
#endif  //__RecoLocalCalo_HGCRecProducers_HGCalLayerClusterProducerEMGPU_H__
