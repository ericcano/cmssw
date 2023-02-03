#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Portable/interface/Product.h"
#include "DataFormats/PortableTestObjects/interface/TestHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/TestSoA.h"
#include "DataFormats/Portable/interface/PortableCollectionCommon.h"

namespace ROOT::Meta::Selection {
  namespace portablecollection {
    class c0 : CollectionImpl<0, void> {};

    class CL0_TestHostCollection : CollectionLeaf<0, portabletest::TestSoALayout<128, false>> {};
    class CL1_TestHostMultiCollection : CollectionLeaf<1, portabletest::TestSoALayout2<128, false>> {};
  }  // namespace portablecollection
  namespace portabletest {
    class TestHostCollection : KeepFirstTemplateArguments<1> {};
    class TestHostMultiCollection : KeepFirstTemplateArguments<2> {};
  }  // namespace portabletest
}    // namespace Selection
