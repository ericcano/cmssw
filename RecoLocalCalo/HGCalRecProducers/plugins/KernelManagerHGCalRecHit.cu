#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include "RecoLocalCalo/HGCalRecProducers/plugins/KernelManagerHGCalRecHit.h"
#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalRecHitKernelImpl.cuh"

namespace {  //kernel parameters
  dim3 nb_rechits_;
  constexpr dim3 nt_rechits_(1024);
}  // namespace

KernelManagerHGCalRecHit::KernelManagerHGCalRecHit(const HGCUncalibRecHitSoA& pUncalibSoAHost,
                                                   const HGCUncalibRecHitSoA& pUncalibSoADev,
                                                   const HGCRecHitSoA& pCalibSoADev)
    : mUncalibSoAHost(pUncalibSoAHost), mUncalibSoADev(pUncalibSoADev), mCalibSoADev(pCalibSoADev) {
  mNHits = mUncalibSoAHost.nhits;
  mPad = mUncalibSoAHost.pad;
  ::nb_rechits_ = (mPad + ::nt_rechits_.x - 1) / ::nt_rechits_.x;
  mNBytesDev = mUncalibSoADev.nbytes * mPad;
}

KernelManagerHGCalRecHit::KernelManagerHGCalRecHit(const HGCRecHitSoA& pCalibSoAHost,
                                                   const ConstHGCRecHitSoA& pCalibConstSoADev)
    : mCalibSoAHost(pCalibSoAHost), mCalibConstSoADev(pCalibConstSoADev) {
  mNHits = mCalibSoAHost.nhits;
  mPad = mCalibSoAHost.pad;
  ::nb_rechits_ = (mPad + ::nt_rechits_.x - 1) / ::nt_rechits_.x;
  mNBytesHost = mCalibSoAHost.nbytes * mPad;
}

KernelManagerHGCalRecHit::~KernelManagerHGCalRecHit() {}

void KernelManagerHGCalRecHit::transfer_soa_to_device_(const cudaStream_t& stream) {
  cudaCheck(cudaMemcpyAsync(
      mUncalibSoADev.amplitude, mUncalibSoAHost.amplitude, mNBytesDev, cudaMemcpyHostToDevice, stream));
}

void KernelManagerHGCalRecHit::transfer_soa_to_host(const cudaStream_t& stream) {
  cudaCheck(
      cudaMemcpyAsync(mCalibSoAHost.energy, mCalibConstSoADev.energy, mNBytesHost, cudaMemcpyDeviceToHost, stream));
}

void KernelManagerHGCalRecHit::run_kernels(const KernelConstantData<HGCeeUncalibRecHitConstantData>* kcdata,
                                           const cudaStream_t& stream) {
  transfer_soa_to_device_(stream);
  ee_to_rechit<<<::nb_rechits_, ::nt_rechits_, 0, stream>>>(mCalibSoADev, mUncalibSoADev, kcdata->data_, mNHits);
  cudaCheck(cudaGetLastError());
}

void KernelManagerHGCalRecHit::run_kernels(const KernelConstantData<HGChefUncalibRecHitConstantData>* kcdata,
                                           const cudaStream_t& stream) {
  transfer_soa_to_device_(stream);
  hef_to_rechit<<<::nb_rechits_, ::nt_rechits_, 0, stream>>>(mCalibSoADev, mUncalibSoADev, kcdata->data_, mNHits);
  cudaCheck(cudaGetLastError());
}

void KernelManagerHGCalRecHit::run_kernels(const KernelConstantData<HGChebUncalibRecHitConstantData>* kcdata,
                                           const cudaStream_t& stream) {
  transfer_soa_to_device_(stream);
  heb_to_rechit<<<::nb_rechits_, ::nt_rechits_, 0, stream>>>(mCalibSoADev, mUncalibSoADev, kcdata->data_, mNHits);
  cudaCheck(cudaGetLastError());
}
