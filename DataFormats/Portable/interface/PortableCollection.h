#ifndef DataFormats_Portable_interface_PortableCollection_h
#define DataFormats_Portable_interface_PortableCollection_h

#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"

namespace traits {

  // trait for a generic SoA-based product
  template <typename T, typename TDev, typename = std::enable_if_t<alpaka::isDevice<TDev>>>
  class PortableCollectionTrait;

}  // namespace traits

// type alias for a generic SoA-based product
template <typename T, typename TDev, typename = std::enable_if_t<alpaka::isDevice<TDev>>>
using PortableCollection = typename traits::PortableCollectionTrait<T, TDev>::CollectionType;

namespace traits {
  // trait for a generic SoA-based product
  template <typename TDev, typename T0, typename... Args>
  class PortableMultiCollectionTrait;

}  // namespace traits

// type alias for a generic SoA-based product
template <typename TDev, typename T0, typename... Args>
using PortableMultiCollection = typename traits::PortableMultiCollectionTrait<TDev, T0, Args...>::CollectionType;

#endif  // DataFormats_Portable_interface_PortableCollection_h
