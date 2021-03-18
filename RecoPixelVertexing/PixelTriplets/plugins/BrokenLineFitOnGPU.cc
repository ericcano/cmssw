#include "BrokenLineFitOnGPU.h"

void HelixFitOnGPU::launchBrokenLineKernelsOnCPU(HitsView const* hv, uint32_t hitsInFit, uint32_t maxNumberOfTuples) {
  assert(tuples_);

  //  Fit internals
  auto hitsGPU_ = std::make_unique<double[]>(maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix3xNd<4>) / sizeof(double));
  auto hits_geGPU_ = std::make_unique<float[]>(maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix6x4f) / sizeof(float));
  auto fast_fit_resultsGPU_ =
      std::make_unique<double[]>(maxNumberOfConcurrentFits_ * sizeof(riemannFit::Vector4d) / sizeof(double));

  for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
    // fit triplets
    kernelBLFastFit<3>(
        tuples_, tupleMultiplicity_, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), 3, offset);

    kernelBLFit<3>(tupleMultiplicity_,
                   bField_,
                   outputSoa_,
                   hitsGPU_.get(),
                   hits_geGPU_.get(),
                   fast_fit_resultsGPU_.get(),
                   3,
                   offset);

    // fit quads
    kernelBLFastFit<4>(
        tuples_, tupleMultiplicity_, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), 4, offset);

    kernelBLFit<4>(tupleMultiplicity_,
                   bField_,
                   outputSoa_,
                   hitsGPU_.get(),
                   hits_geGPU_.get(),
                   fast_fit_resultsGPU_.get(),
                   4,
                   offset);

    if (fit5as4_) {
      // fit penta (only first 4)
      kernelBLFastFit<4>(
          tuples_, tupleMultiplicity_, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), 5, offset);

      kernelBLFit<4>(tupleMultiplicity_,
                     bField_,
                     outputSoa_,
                     hitsGPU_.get(),
                     hits_geGPU_.get(),
                     fast_fit_resultsGPU_.get(),
                     5,
                     offset);
    } else {
      // fit penta (all 5)
      kernelBLFastFit<5>(
          tuples_, tupleMultiplicity_, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), 5, offset);

      kernelBLFit<5>(tupleMultiplicity_,
                     bField_,
                     outputSoa_,
                     hitsGPU_.get(),
                     hits_geGPU_.get(),
                     fast_fit_resultsGPU_.get(),
                     5,
                     offset);
    }

  }  // loop on concurrent fits
}
