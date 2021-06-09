#ifndef CUDADAtaFormats_HGCal_HGCRecHitGPUProduct_H
#define CUDADAtaFormats_HGCal_HGCRecHitGPUProduct_H

#include <numeric>

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"

#include "CUDADataFormats/HGCal/interface/HGCRecHitSoA.h"
#include "CUDADataFormats/HGCal/interface/ConstHGCRecHitSoA.h"
#include "CUDADataFormats/HGCal/interface/HGCUncalibRecHitSoA.h"

class HGCRecHitGPUProduct {
public:
  HGCRecHitGPUProduct() = default;
  explicit HGCRecHitGPUProduct(uint32_t nhits, const cudaStream_t &stream) : nhits_(nhits) {
    size_tot_ = std::accumulate(sizes_.begin(), sizes_.end(), 0);  //this might be done at compile time
    pad_ = ((nhits - 1) / 32 + 1) * 32;                            //align to warp boundary (assumption: warpSize = 32)
    mem_ = cms::cuda::make_device_unique<std::byte[]>(pad_ * size_tot_, stream);
  }
  ~HGCRecHitGPUProduct() = default;

  HGCRecHitGPUProduct(const HGCRecHitGPUProduct &) = delete;
  HGCRecHitGPUProduct &operator=(const HGCRecHitGPUProduct &) = delete;
  HGCRecHitGPUProduct(HGCRecHitGPUProduct &&) = default;
  HGCRecHitGPUProduct &operator=(HGCRecHitGPUProduct &&) = default;

  HGCRecHitSoA get() {
    HGCRecHitSoA soa;
    soa.energy = reinterpret_cast<float *>(mem_.get());
    soa.time = soa.energy + pad_;
    soa.timeError = soa.time + pad_;
    soa.sigmaNoise = soa.timeError + pad_;
    soa.id = reinterpret_cast<uint32_t *>(soa.sigmaNoise + pad_);
    soa.flagBits = soa.id + pad_;
    soa.nbytes = size_tot_;
    soa.nhits = nhits_;
    soa.pad = pad_;
    return soa;
  }
  ConstHGCRecHitSoA get() const {
    ConstHGCRecHitSoA soa;
    soa.energy = reinterpret_cast<float const *>(mem_.get());
    soa.time = soa.energy + pad_;
    soa.timeError = soa.time + pad_;
    soa.sigmaNoise = soa.timeError + pad_;
    soa.id = reinterpret_cast<uint32_t const *>(soa.sigmaNoise + pad_);
    soa.flagBits = soa.id + pad_;
    soa.nhits = nhits_;
    return soa;
  }

  //number of hits stored in the SoA
  uint32_t nHits() const { return nhits_; }
  //pad of memory block (used for warp alignment, slighlty larger than 'nhits_')
  uint32_t pad() const { return pad_; }
  //number of bytes of the SoA
  uint32_t nBytes() const { return size_tot_; }

private:
  cms::cuda::device::unique_ptr<std::byte[]> mem_;
  static constexpr std::array<int, memory::npointers::ntypes_hgcrechits_soa> sizes_ = {
      {memory::npointers::float_hgcrechits_soa * sizeof(float),
       memory::npointers::uint32_hgcrechits_soa * sizeof(uint32_t)}};
  uint32_t pad_;
  uint32_t nhits_;
  uint32_t size_tot_;
};

#endif  //CUDADAtaFormats_HGCal_HGCRecHitGPUProduct_H
