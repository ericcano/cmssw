#ifndef CUDADAtaFormats_HGCal_HGCUncalibRecHitHost_H
#define CUDADAtaFormats_HGCal_HGCUncalibRecHitHost_H

#include <cassert>
#include <numeric>

#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "CUDADataFormats/HGCal/interface/HGCRecHitSoA.h"
#include "CUDADataFormats/HGCal/interface/HGCUncalibRecHitSoA.h"

template <class T>
class HGCUncalibRecHitHost {
public:
  HGCUncalibRecHitHost() = default;
  explicit HGCUncalibRecHitHost(uint32_t nhits, const T &hits, const cudaStream_t &stream) : nhits_(nhits) {
    size_tot_ = std::accumulate(sizes_.begin(), sizes_.end(), 0);  //this might be done at compile time
    pad_ = ((nhits - 1) / 32 + 1) * 32;                            //align to warp boundary (assumption: warpSize = 32)
    ptr_ = cms::cuda::make_host_unique<std::byte[]>(pad_ * size_tot_, stream);

    defineSoAMemoryLayout_();
    fillSoA_(hits);
  }
  ~HGCUncalibRecHitHost() = default;

  HGCUncalibRecHitHost(const HGCUncalibRecHitHost &) = delete;
  HGCUncalibRecHitHost &operator=(const HGCUncalibRecHitHost &) = delete;
  HGCUncalibRecHitHost(HGCUncalibRecHitHost &&) = default;
  HGCUncalibRecHitHost &operator=(HGCUncalibRecHitHost &&) = default;

  void defineSoAMemoryLayout_() {
    soa_.amplitude = reinterpret_cast<float *>(ptr_.get());
    soa_.pedestal = soa_.amplitude + pad_;
    soa_.jitter = soa_.pedestal + pad_;
    soa_.chi2 = soa_.jitter + pad_;
    soa_.OOTamplitude = soa_.chi2 + pad_;
    soa_.OOTchi2 = soa_.OOTamplitude + pad_;
    soa_.flags = reinterpret_cast<uint32_t *>(soa_.OOTchi2 + pad_);
    soa_.aux = soa_.flags + pad_;
    soa_.id = soa_.aux + pad_;
    soa_.aux = soa_.flags + pad_;
    soa_.id = soa_.aux + pad_;

    soa_.nbytes = size_tot_;
    soa_.nhits = nhits_;
    soa_.pad = pad_;
  }

  void fillSoA_(const T &c) {
    for (unsigned i(0); i < nhits_; ++i) {
      soa_.amplitude[i] = c[i].amplitude();
      soa_.pedestal[i] = c[i].pedestal();
      soa_.jitter[i] = c[i].jitter();
      soa_.chi2[i] = c[i].chi2();
      soa_.OOTamplitude[i] = c[i].outOfTimeEnergy();
      soa_.OOTchi2[i] = c[i].outOfTimeChi2();
      soa_.flags[i] = c[i].flags();
      soa_.aux[i] = 0;
      soa_.id[i] = c[i].id().rawId();
    }
  }

  HGCUncalibRecHitSoA get() const { return soa_; }
  uint32_t nHits() const { return nhits_; }
  uint32_t pad() const { return pad_; }
  uint32_t nBytes() const { return size_tot_; }

private:
  cms::cuda::host::unique_ptr<std::byte[]> ptr_;
  HGCUncalibRecHitSoA soa_;
  static constexpr std::array<int, memory::npointers::ntypes_hgcuncalibrechits_soa> sizes_ = {
      {memory::npointers::float_hgcuncalibrechits_soa * sizeof(float),
       memory::npointers::uint32_hgcuncalibrechits_soa * sizeof(uint32_t)}};
  uint32_t size_tot_;
  uint32_t pad_;
  uint32_t nhits_;
};

#endif  //CUDADAtaFormats_HGCal_HGCUncalibRecHitHost_H
