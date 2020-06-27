// system include files
#include <cmath>
#include <memory>
#include <vector>

// CMSSW include files
#include "DataFormats/Math/interface/Vector3D.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "cudavectors.h"

class ConvertToCartesianVectorsCUDA : public edm::stream::EDProducer<> {
public:
  explicit ConvertToCartesianVectorsCUDA(const edm::ParameterSet&);
  ~ConvertToCartesianVectorsCUDA() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  using CartesianVectors = std::vector<math::XYZVectorF>;
  using CylindricalVectors = std::vector<math::RhoEtaPhiVectorF>;

  virtual void produce(edm::Event&, const edm::EventSetup&) override;

  edm::EDGetTokenT<CylindricalVectors> input_;
  edm::EDPutTokenT<CartesianVectors> output_;
};

ConvertToCartesianVectorsCUDA::ConvertToCartesianVectorsCUDA(const edm::ParameterSet& config)
    : input_(consumes<CylindricalVectors>(config.getParameter<edm::InputTag>("input"))) {
  output_ = produces<CartesianVectors>();
}

void ConvertToCartesianVectorsCUDA::produce(edm::Event& event, const edm::EventSetup& setup) {
  auto const& input = event.get(input_);
  auto elements = input.size();
  auto product = std::make_unique<CartesianVectors>(elements);
  // A very brutal implementation based on the fact that math::XYZVectorF is, in the end,
  // a pod in memory (no vtable), like cudavectors::CartesianVector, and likewise in cylindrical
  static_assert(sizeof(CylindricalVectors::value_type) == sizeof(cudavectors::CylindricalVector),
          "Size mismatch between CylindricalVectors::value_type and CylindricalVector");
  static_assert(sizeof(CartesianVectors::value_type) == sizeof(cudavectors::CartesianVector),
          "Size mismatch between CylindricalVectors::value_type and CylindricalVector");
  cudavectors::convertWrapper(
    reinterpret_cast<const cudavectors::CylindricalVector*>(input.data()),
    reinterpret_cast<cudavectors::CartesianVector*>(product->data()),
    elements);
  //std::cout << "CUDA converted from: rho=" << input[0].rho() << " eta=" << input[0].eta() << " phi=" << input[0].phi() << std::endl;
  //std::cout << "CUDA converted to: x=" << (*product)[0].x() << " y=" << (*product)[0].y() << " z=" << (*product)[0].z() << std::endl;

  event.put(output_, std::move(product));
}

void ConvertToCartesianVectorsCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("input", edm::InputTag("cylindricalVectors"));
  descriptions.addWithDefaultLabel(desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(ConvertToCartesianVectorsCUDA);
