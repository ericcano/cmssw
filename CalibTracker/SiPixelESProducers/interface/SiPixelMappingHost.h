#ifndef CalibTracker_SiPixelESProducers_SiPixelMappingHost_h
#define CalibTracker_SiPixelESProducers_SiPixelMappingHost_h

#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "SiPixelMappingLayout.h"

using SiPixelMappingHost = PortableHostCollection<SiPixelMappingLayout<>>;

#endif  // CalibTracker_SiPixelESProducers_SiPixelMappingHost_h
