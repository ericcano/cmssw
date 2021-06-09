#include "CUDADataFormats/HGCal/interface/HGCUncalibRecHitDevice.h"

HGCUncalibRecHitDevice::HGCUncalibRecHitDevice(uint32_t nhits, const cudaStream_t& stream) : nhits_(nhits) {
  size_tot_ = std::accumulate(sizes_.begin(), sizes_.end(), 0);  //this might be done at compile time
  pad_ = ((nhits - 1) / 32 + 1) * 32;                            //align to warp boundary (assumption: warpSize = 32)
  ptr_ = cms::cuda::make_device_unique<std::byte[]>(pad_ * size_tot_, stream);

  defineSoAMemoryLayout_();
}

void HGCUncalibRecHitDevice::defineSoAMemoryLayout_() {
  soa_.amplitude = reinterpret_cast<float*>(ptr_.get());
  soa_.pedestal = soa_.amplitude + pad_;
  soa_.jitter = soa_.pedestal + pad_;
  soa_.chi2 = soa_.jitter + pad_;
  soa_.OOTamplitude = soa_.chi2 + pad_;
  soa_.OOTchi2 = soa_.OOTamplitude + pad_;
  soa_.flags = reinterpret_cast<uint32_t*>(soa_.OOTchi2 + pad_);
  soa_.aux = soa_.flags + pad_;
  soa_.id = soa_.aux + pad_;

  soa_.nbytes = size_tot_;
  soa_.nhits = nhits_;
  soa_.pad = pad_;
}
