#ifndef DataFormats_Portable_interface_PortableCollection_h
#define DataFormats_Portable_interface_PortableCollection_h

#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"

namespace traits {

  // trait for a generic SoA-based product
  template <typename TDev,
            typename T0,
            typename... Args>
  class PortableCollectionTrait;

}  // namespace traits

// type alias for a generic SoA-based product
template <typename TDev,
          typename T0,
          typename... Args>
using PortableCollection = typename traits::PortableCollectionTrait<TDev, T0, Args...>::CollectionType;

#endif  // DataFormats_Portable_interface_PortableCollection_h
