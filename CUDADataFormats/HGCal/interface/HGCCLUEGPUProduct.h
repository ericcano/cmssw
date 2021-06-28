#ifndef CUDADAtaFormats_HGCal_HGCCLUEGPUProduct_H
#define CUDADAtaFormats_HGCal_HGCCLUEGPUProduct_H

#include <numeric>

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#include "CUDADataFormats/HGCal/interface/HGCCLUESoA.h"
#include "CUDADataFormats/HGCal/interface/HGCUncalibRecHitSoA.h"

class HGCCLUEGPUProduct {
public:
  HGCCLUEGPUProduct() = default;
  explicit HGCCLUEGPUProduct(uint32_t nhits, const cudaStream_t &stream) : nhits_(nhits) {
    /* CUDA allocations are already aligned */
    mMemCLUEDev = cms::cuda::make_device_unique<std::byte[]>(
        HGCCLUESoADescriptor::computeDataSize(nhits), stream);
  }
  ~HGCCLUEGPUProduct() = default;

  HGCCLUEGPUProduct(const HGCCLUEGPUProduct &) = delete;
  HGCCLUEGPUProduct &operator=(const HGCCLUEGPUProduct &) = delete;
  HGCCLUEGPUProduct(HGCCLUEGPUProduct &&) = default;
  HGCCLUEGPUProduct &operator=(HGCCLUEGPUProduct &&) = default;

  HGCCLUESoA get() {
    HGCCLUESoA soa(mMemCLUEDev.get(), nhits_);
    soa.nhits = nhits_;
    return soa;
  }

  const HGCCLUESoA get() const {
    HGCCLUESoA soa(mMemCLUEDev.get(), nhits_);
    soa.nhits = nhits_;
    return soa;
  }

  //number of hits stored in the SoA
  uint32_t nHits() const { return nhits_; }

private:
  cms::cuda::device::unique_ptr<std::byte[]> mMemCLUEDev;
  uint32_t nhits_;
};

#endif  //CUDADAtaFormats_HGCal_HGCCLUEGPUProduct_H
