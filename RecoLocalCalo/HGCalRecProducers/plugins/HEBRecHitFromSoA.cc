#include <memory>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

#include "CUDADataFormats/HGCal/interface/HGCRecHitCPUProduct.h"
#include "CUDADataFormats/HGCal/interface/ConstHGCRecHitSoA.h"

class HEBRecHitFromSoA : public edm::stream::EDProducer<> {
public:
  explicit HEBRecHitFromSoA(const edm::ParameterSet& ps);
  ~HEBRecHitFromSoA() override;

  void produce(edm::Event&, const edm::EventSetup&) override;
  void convert_soa_data_to_collection_(uint32_t, HGCRecHitCollection&, ConstHGCRecHitSoA*);

private:
  std::unique_ptr<HGChefRecHitCollection> rechits_;
  edm::EDGetTokenT<HGCRecHitCPUProduct> recHitSoAToken_;
  edm::EDPutTokenT<HGChefRecHitCollection> recHitCollectionToken_;
};

HEBRecHitFromSoA::HEBRecHitFromSoA(const edm::ParameterSet& ps) {
  recHitSoAToken_ = consumes<HGCRecHitCPUProduct>(ps.getParameter<edm::InputTag>("HEBRecHitSoATok"));
  recHitCollectionToken_ = produces<HGChebRecHitCollection>();
}

HEBRecHitFromSoA::~HEBRecHitFromSoA() {}

void HEBRecHitFromSoA::produce(edm::Event& event, const edm::EventSetup& setup) {
  const HGCRecHitCPUProduct& recHits = event.get(recHitSoAToken_);
  ConstHGCRecHitSoA recHitsSoA = recHits.get();
  rechits_ = std::make_unique<HGCRecHitCollection>();
  convert_soa_data_to_collection_(recHits.nHits(), *rechits_, &recHitsSoA);
  event.put(std::move(rechits_));
}

void HEBRecHitFromSoA::convert_soa_data_to_collection_(uint32_t nhits,
                                                       HGCRecHitCollection& rechits,
                                                       ConstHGCRecHitSoA* h_calibSoA) {
  rechits.reserve(nhits);
  for (unsigned i=0; i<nhits; ++i) {
    DetId id_converted(h_calibSoA->id[i]);
    float son = h_calibSoA->energy[i]/h_calibSoA->sigmaNoise[i];
    rechits.emplace_back(id_converted,
                         h_calibSoA->energy[i],
                         h_calibSoA->time[i],
                         0,
                         h_calibSoA->flagBits[i],
			 son,
                         h_calibSoA->timeError[i]);
    rechits[i].setSignalOverSigmaNoise(son);
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HEBRecHitFromSoA);
