#ifndef DataFormats_Portable_interface_PortableCollection_h
#define DataFormats_Portable_interface_PortableCollection_h

#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"

namespace traits {

  // trait for a generic SoA-based product
  template <typename TDev,
            typename T0, 
            typename T1 = void,
            typename T2 = void,
            typename T3 = void,
            typename T4 = void,
            typename = std::enable_if_t<alpaka::isDevice<TDev>>>
  class PortableCollectionTrait;

}  // namespace traits

// type alias for a generic SoA-based product
template <typename TDev,
          typename T0,
          typename T1 = void,
          typename T2 = void,
          typename T3 = void,
          typename T4 = void,
          typename = std::enable_if_t<alpaka::isDevice<TDev>>>
using PortableCollection = typename traits::PortableCollectionTrait<TDev, T0, T1, T2, T3, T4>::CollectionType;

#endif  // DataFormats_Portable_interface_PortableCollection_h
