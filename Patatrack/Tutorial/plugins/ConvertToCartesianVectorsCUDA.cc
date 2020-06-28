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

// CMSSW include files
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_noncached_unique_ptr.h"

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
  using cudavectors::CartesianVector;
  using cudavectors::CylindricalVector;

  auto const& input = event.get(input_);
  auto elements = input.size();
  auto product = std::make_unique<CartesianVectors>(elements);

  // allocate memory on the GPU for the cylindrical and cartesian vectors
  //auto deviceInput = cms::cuda::device::make_device_unique<CylindricalVector[]>(size, nullptr);
  //auto deviceOutput = cms::cuda::device::make_device_unique<CartesianVector[]>(size, nullptr);
  cms::cuda::device::unique_ptr<CylindricalVector[]> deviceInput =
          cms::cuda::make_device_unique<CylindricalVector[]>(elements, cudaStreamDefault);
//  cms::cuda::host::noncached::unique_ptr<CylindricalVector[]> deviceInput =
//          cms::cuda::make_host_noncached_unique<CylindricalVector[]>(elements);
  cms::cuda::device::unique_ptr<CartesianVector[]> deviceOutput =
          cms::cuda::make_device_unique<CartesianVector[]>(elements, cudaStreamDefault);

  //cudaCheck(cudaMalloc(&deviceInput, sizeof(CylindricalVector)*elements));
  //cudaCheck(cudaMalloc(&deviceOutput, sizeof(CartesianVector)*elements));

  // A very brutal implementation based on the fact that math::XYZVectorF is, in the end,
  // a pod in memory (no vtable), like cudavectors::CartesianVector, and likewise in cylindrical
  static_assert(sizeof(CylindricalVectors::value_type) == sizeof(cudavectors::CylindricalVector),
          "Size mismatch between CylindricalVectors::value_type and CylindricalVector");
  static_assert(sizeof(CartesianVectors::value_type) == sizeof(cudavectors::CartesianVector),
          "Size mismatch between CylindricalVectors::value_type and CylindricalVector");
  // copy the input data to the GPU. This is in practice a reinterpret cast
  cudaCheck(cudaMemcpy(deviceInput.get(), input.data(),
    sizeof(CylindricalVector) * elements, cudaMemcpyHostToDevice));
//  cudaCheck(cudaMemcpy(deviceInput.get(), input.data(),
//    sizeof(CylindricalVector) * elements, cudaMemcpyHostToHost));

  cudavectors::convertWrapper(deviceInput.get(), deviceOutput.get(), elements);

  // copy the result from the GPU
  cudaCheck(cudaMemcpy(product->data(), deviceOutput.get(),
    sizeof(CartesianVector) * elements, cudaMemcpyDeviceToHost));

  // free the GPU memory;
  //cudaCheck(cudaFree(deviceInput));
  //cudaCheck(cudaFree(deviceOutput));
  cudaCheck(cudaGetLastError());

  event.put(output_, std::move(product));
}

void ConvertToCartesianVectorsCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("input", edm::InputTag("cylindricalVectors"));
  descriptions.addWithDefaultLabel(desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(ConvertToCartesianVectorsCUDA);
