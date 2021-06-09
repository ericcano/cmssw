#ifndef CUDADAtaFormats_HGCal_HGCCLUECPUProduct_H
#define CUDADAtaFormats_HGCal_HGCCLUECPUProduct_H

#include <numeric>

#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

#include "CUDADataFormats/HGCal/interface/HGCCLUESoA.h"
#include "CUDADataFormats/HGCal/interface/ConstHGCCLUESoA.h"
#include "CUDADataFormats/HGCal/interface/HGCUncalibRecHitSoA.h"

class HGCCLUECPUProduct {
public:
  HGCCLUECPUProduct() = default;
  explicit HGCCLUECPUProduct(uint32_t nhits, const cudaStream_t &stream) : nhits_(nhits) {
    size_tot_ = std::accumulate(sizes_.begin(), sizes_.end(), 0);
    pad_ = ((nhits - 1) / 32 + 1) * 32; //align to warp boundary (assumption: warpSize = 32)
    mem_ = cms::cuda::make_host_unique<std::byte[]>(pad_ * size_tot_, stream);
  }
  ~HGCCLUECPUProduct() = default;

  HGCCLUECPUProduct(const HGCCLUECPUProduct &) = delete;
  HGCCLUECPUProduct &operator=(const HGCCLUECPUProduct &) = delete;
  HGCCLUECPUProduct(HGCCLUECPUProduct &&) = default;
  HGCCLUECPUProduct &operator=(HGCCLUECPUProduct &&) = default;

  HGCCLUESoA get() {
    HGCCLUESoA soa;
    soa.rho = reinterpret_cast<float *>(mem_.get());
    soa.delta = soa.rho + pad_;
    soa.nearestHigher = reinterpret_cast<int32_t *>(soa.delta + pad_);
    soa.clusterIndex = soa.nearestHigher + pad_;
    soa.isSeed = reinterpret_cast<bool *>(soa.clusterIndex + pad_);
    soa.nbytes = size_tot_;
    soa.nhits = nhits_;
    soa.pad = pad_;
    return soa;
  }

  ConstHGCCLUESoA get() const {
    ConstHGCCLUESoA soa;
    soa.rho = reinterpret_cast<float const*>(mem_.get());
    soa.delta = soa.rho + pad_;
    soa.nearestHigher = reinterpret_cast<int32_t const*>(soa.delta + pad_);
    soa.clusterIndex = soa.nearestHigher + pad_;
    soa.isSeed = reinterpret_cast<bool const*>(soa.clusterIndex + pad_);
    return soa;
  }

  //number of hits stored in the SoA
  uint32_t nHits() const { return nhits_; }
  //pad of memory block (used for warp alignment, slighlty larger than 'nhits_')
  uint32_t pad() const { return pad_; }
  //number of bytes of the SoA
  uint32_t nBytes() const { return size_tot_; }

private:
  cms::cuda::host::unique_ptr<std::byte[]> mem_;
  static constexpr std::array<uint32_t, memory::npointers::ntypes_hgcclue_soa> sizes_ = {
      {memory::npointers::float_hgcclue_soa * sizeof(float),
       memory::npointers::int32_hgcclue_soa * sizeof(uint32_t),
       memory::npointers::bool_hgcclue_soa * sizeof(bool)}};
  uint32_t pad_;
  uint32_t nhits_;
  uint32_t size_tot_;
};

#endif  //CUDADAtaFormats_HGCal_HGCCLUECPUProduct_H
