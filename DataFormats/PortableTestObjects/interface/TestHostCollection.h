#ifndef DataFormats_PortableTestObjects_interface_TestHostCollection_h
#define DataFormats_PortableTestObjects_interface_TestHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/TestSoA.h"

#include <RootMetaSelection.h>

namespace portabletest {

  // SoA with x, y, z, id fields in host memory
  using TestHostCollection = PortableHostCollection<TestSoA>;

  using TestHostMultiCollection = PortableHostCollection<TestSoA, TestSoA2>;

}  // namespace portabletest

namespace ROOT {
  namespace Meta {
    namespace Selection {
      namespace portablecollection {
        template <std::size_t Idx, typename T, typename... Args> 
        // This is enough to cover all the cases of CollectionImpl which has no members...
        class CollectionImpl: KeepFirstTemplateArguments<0> {};
        class c0: CollectionImpl<0, void> {};

        template <std::size_t Idx, typename T>
        class CollectionLeaf {};

        class CL0_TestHostCollection: CollectionLeaf<0, portabletest::TestSoALayout<128,false>> {};
        class CL1_TestHostMultiCollection: CollectionLeaf<1, portabletest::TestSoALayout2<128,false>> {};
      }
      namespace portabletest {
        class TestHostCollection: KeepFirstTemplateArguments<1> {};
        class TestHostMultiCollection: KeepFirstTemplateArguments<2> {};
      }
    }
  }
}

#endif  // DataFormats_PortableTestObjects_interface_TestHostCollection_h
