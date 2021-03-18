#include "RiemannFitOnGPU.h"

void HelixFitOnGPU::launchRiemannKernelsOnCPU(HitsView const *hv, uint32_t nhits, uint32_t maxNumberOfTuples) {
  assert(tuples_);

  //  Fit internals
  auto hitsGPU = std::make_unique<double[]>(maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix3xNd<4>) / sizeof(double));
  auto hits_geGPU = std::make_unique<float[]>(maxNumberOfConcurrentFits_ * sizeof(riemannFit::Matrix6x4f) / sizeof(float));
  auto fast_fit_resultsGPU =
      std::make_unique<double[]>(maxNumberOfConcurrentFits_ * sizeof(riemannFit::Vector4d) / sizeof(double));
  auto circle_fit_resultsGPU_holder = std::make_unique<char[]>(maxNumberOfConcurrentFits_ * sizeof(riemannFit::CircleFit));
  riemannFit::CircleFit *circle_fit_resultsGPU = (riemannFit::CircleFit *)(circle_fit_resultsGPU_holder.get());

  for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
    // triplets
    kernelFastFit<3>(
        tuples_, tupleMultiplicity_, 3, hv, hitsGPU.get(), hits_geGPU.get(), fast_fit_resultsGPU.get(), offset);

    kernelCircleFit<3>(tupleMultiplicity_,
                       3,
                       bField_,
                       hitsGPU.get(),
                       hits_geGPU.get(),
                       fast_fit_resultsGPU.get(),
                       circle_fit_resultsGPU,
                       offset);

    kernelLineFit<3>(tupleMultiplicity_,
                     3,
                     bField_,
                     outputSoa_,
                     hitsGPU.get(),
                     hits_geGPU.get(),
                     fast_fit_resultsGPU.get(),
                     circle_fit_resultsGPU,
                     offset);

    // quads
    kernelFastFit<4>(
        tuples_, tupleMultiplicity_, 4, hv, hitsGPU.get(), hits_geGPU.get(), fast_fit_resultsGPU.get(), offset);

    kernelCircleFit<4>(tupleMultiplicity_,
                       4,
                       bField_,
                       hitsGPU.get(),
                       hits_geGPU.get(),
                       fast_fit_resultsGPU.get(),
                       circle_fit_resultsGPU,
                       offset);

    kernelLineFit<4>(tupleMultiplicity_,
                     4,
                     bField_,
                     outputSoa_,
                     hitsGPU.get(),
                     hits_geGPU.get(),
                     fast_fit_resultsGPU.get(),
                     circle_fit_resultsGPU,
                     offset);

    if (fit5as4_) {
      // penta
      kernelFastFit<4>(
          tuples_, tupleMultiplicity_, 5, hv, hitsGPU.get(), hits_geGPU.get(), fast_fit_resultsGPU.get(), offset);

      kernelCircleFit<4>(tupleMultiplicity_,
                         5,
                         bField_,
                         hitsGPU.get(),
                         hits_geGPU.get(),
                         fast_fit_resultsGPU.get(),
                         circle_fit_resultsGPU,
                         offset);

      kernelLineFit<4>(tupleMultiplicity_,
                       5,
                       bField_,
                       outputSoa_,
                       hitsGPU.get(),
                       hits_geGPU.get(),
                       fast_fit_resultsGPU.get(),
                       circle_fit_resultsGPU,
                       offset);

    } else {
      // penta all 5
      kernelFastFit<5>(
          tuples_, tupleMultiplicity_, 5, hv, hitsGPU.get(), hits_geGPU.get(), fast_fit_resultsGPU.get(), offset);

      kernelCircleFit<5>(tupleMultiplicity_,
                         5,
                         bField_,
                         hitsGPU.get(),
                         hits_geGPU.get(),
                         fast_fit_resultsGPU.get(),
                         circle_fit_resultsGPU,
                         offset);

      kernelLineFit<5>(tupleMultiplicity_,
                       5,
                       bField_,
                       outputSoa_,
                       hitsGPU.get(),
                       hits_geGPU.get(),
                       fast_fit_resultsGPU.get(),
                       circle_fit_resultsGPU,
                       offset);
    }
  }
}
