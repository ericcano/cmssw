#ifndef CUDADAtaFormats_HGCal_HGCCLUECPUProduct_H
#define CUDADAtaFormats_HGCal_HGCCLUECPUProduct_H

#include <numeric>

#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

#include "CUDADataFormats/HGCal/interface/HGCCLUESoA.h"
#include "CUDADataFormats/HGCal/interface/HGCUncalibRecHitSoA.h"

class HGCCLUECPUProduct {
public:
  HGCCLUECPUProduct() = default;
  explicit HGCCLUECPUProduct(uint32_t nhits, const cudaStream_t &stream) : nhits_(nhits) {
    mMemCLUEHost = cms::cuda::make_host_unique<std::byte[]>(
            HGCCLUESoADescriptor::computeDataSize(nhits), stream);
  }
  ~HGCCLUECPUProduct() = default;

  HGCCLUECPUProduct(const HGCCLUECPUProduct &) = delete;
  HGCCLUECPUProduct &operator=(const HGCCLUECPUProduct &) = delete;
  HGCCLUECPUProduct(HGCCLUECPUProduct &&) = default;
  HGCCLUECPUProduct &operator=(HGCCLUECPUProduct &&) = default;

  HGCCLUESoA get() {
    HGCCLUESoA soa(mMemCLUEHost.get(), nhits_);
    soa.nhits = nhits_;
    return soa;
  }

  const HGCCLUESoA get() const {
    HGCCLUESoA soa(mMemCLUEHost.get(), nhits_);
    soa.nhits = nhits_;
    return soa;
  }

  //number of hits stored in the SoA
  uint32_t nHits() const { return nhits_; }

private:
  cms::cuda::host::unique_ptr<std::byte[]> mMemCLUEHost;
  uint32_t nhits_;
};

#endif  //CUDADAtaFormats_HGCal_HGCCLUECPUProduct_H
