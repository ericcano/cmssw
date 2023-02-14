#ifndef DataFormats_Portable_interface_alpaka_PortableDeviceCollection_h
#define DataFormats_Portable_interface_alpaka_PortableDeviceCollection_h

#include <optional>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  // ... or any other CPU-based accelerators

  // generic SoA-based product in host memory
  template <typename T>
  using PortableCollection = ::PortableHostCollection<T>;

#else

  // generic SoA-based product in device memory
  template <typename T>
  using PortableCollection = ::PortableDeviceCollection<T, Device>;

#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace traits {

  // specialise the trait for the device provided by the ALPAKA_ACCELERATOR_NAMESPACE
  template <typename T>
  class PortableCollectionTrait<T, ALPAKA_ACCELERATOR_NAMESPACE::Device> {
    using CollectionType = ALPAKA_ACCELERATOR_NAMESPACE::PortableCollection<T>;
  };

}  // namespace traits

namespace cms::alpakatools {
  template <typename TLayout, typename TDevice>
  struct CopyToHost<PortableDeviceCollection<TLayout, TDevice>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, PortableDeviceCollection<TLayout, TDevice> const& srcData) {
      PortableHostCollection<TLayout> dstData(srcData->metadata().size(), queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      return dstData;
    }
  };

  template <typename TLayout>
  struct CopyToDevice<PortableHostCollection<TLayout>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, PortableHostCollection<TLayout> const& srcData) {
      using TDevice = typename alpaka::trait::DevType<TQueue>::type;
      PortableDeviceCollection<TLayout, TDevice> dstData(srcData->metadata().size(), queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      return dstData;
    }
  };
}  // namespace cms::alpakatools

namespace ALPAKA_ACCELERATOR_NAMESPACE {

#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  // ... or any other CPU-based accelerators

  //  template <typename T0>
  //  struct PortableCollection: public ::PortableHostCollectionImpl<T0> {
  //    using ::PortableHostCollectionImpl<T0>::PortableHostCollectionImpl;
  //   };

  //  template <typename T0>
  //  using PortableCollection = ::PortableHostCollectionImpl<T0>;

  template <typename T0, typename T1>
  using PortableCollection2 = ::PortableHostMultiCollectionImpl<T0, T1>;

  template <typename T0, typename T1, typename T2>
  using PortableCollection3 = ::PortableHostMultiCollectionImpl<T0, T1, T2>;

#else
  //template <typename T0>
  //struct PortableCollection: public ::PortableDeviceCollection<Device, T0> {
  //  using ::PortableDeviceCollection<Device, T0>::PortableDeviceCollection;
  ///};

  //  template <typename T0>
  //  using PortableCollection = ::PortableDeviceCollection<Device, T0>;

  template <typename T0, typename T1>
  using PortableCollection2 = ::PortableDeviceMultiCollection<Device, T0, T1>;

  template <typename T0, typename T1, typename T2>
  using PortableCollection3 = ::PortableDeviceMultiCollection<Device, T0, T1, T2>;

#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace traits {
// specialise the trait for the device provided by the ALPAKA_ACCELERATOR_NAMESPACE
#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  template <typename T0, typename... Args>
  class PortableMultiCollectionTrait<ALPAKA_ACCELERATOR_NAMESPACE::Device, T0, Args...> {
    using CollectionType = ::PortableHostMultiCollectionImpl<T0, Args...>;
  };
#else
  template <typename T0, typename... Args>
  class PortableMultiCollectionTrait<ALPAKA_ACCELERATOR_NAMESPACE::Device, T0, Args...> {
    using CollectionType = ::PortableDeviceMultiCollection<ALPAKA_ACCELERATOR_NAMESPACE::Device, T0, Args...>;
  };
#endif

}  // namespace traits

namespace cms::alpakatools {
  template <typename TDevice, typename T0, typename... Args>
  struct CopyToHost<PortableDeviceMultiCollection<TDevice, T0, Args...>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, PortableDeviceMultiCollection<TDevice, T0, Args...> const& srcData) {
      PortableHostMultiCollectionImpl<T0, Args...> dstData(srcData.sizes(), queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      return dstData;
    }
  };

  template <typename T0, typename... Args>
  struct CopyToDevice<PortableHostMultiCollectionImpl<T0, Args...>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, PortableHostMultiCollectionImpl<T0, Args...> const& srcData) {
      using TDevice = typename alpaka::trait::DevType<TQueue>::type;
      PortableDeviceMultiCollection<TDevice, T0, Args...> dstData(srcData.sizes(), queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      return dstData;
    }
  };

#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  /*template <typename T0> 
  struct CopyToDevice<ALPAKA_ACCELERATOR_NAMESPACE::PortableCollection<T0>>: public CopyToDevice<PortableHostCollectionImpl<T0>>
  {};    */
#else
  /* template <typename T0> 
  struct CopyToHost<ALPAKA_ACCELERATOR_NAMESPACE::PortableCollection<T0>>: 
  public CopyToHost<::PortableDeviceCollection<ALPAKA_ACCELERATOR_NAMESPACE::Device, T0>> {
    using CopyToHost<::PortableDeviceCollection<ALPAKA_ACCELERATOR_NAMESPACE::Device, T0>>::CopyToHost;
  };*//*
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, PortableDeviceCollection<TDevice, T0> const& srcData) {
      PortableHostCollectionImpl<T0, Args...> dstData(srcData.sizes(), queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      return dstData;
  };*/
#endif

}  // namespace cms::alpakatools

#endif  // DataFormats_Portable_interface_alpaka_PortableDeviceCollection_h
