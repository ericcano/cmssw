#ifndef DataFormats_SiPixelMappingSoA_SiPixelClustersDevice_h
#define DataFormats_SiPixelMappingSoA_SiPixelClustersDevice_h

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelMappingLayout.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using SiPixelMappingDevice = PortableCollection<SiPixelMappingLayout<>>;
  using SiPixelMappingHost = PortableHostCollection<SiPixelMappingLayout<>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_SiPixelMappingSoA_SiPixelClustersDevice_h
