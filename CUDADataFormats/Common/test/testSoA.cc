#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <stdexcept>
#include <stdlib.h>
#include "cppunit/extensions/HelperMacros.h"

/*
 * Unit tests skeleton based on DataFormats/Common/test/ (testDataFormatsCommon)
 */

#include "CUDADataFormats/Common/interface/SoAmacros.h"

class testSoA: public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testSoA);
  CPPUNIT_TEST(initialTest);
  CPPUNIT_TEST(checkAlignment);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}
  void initialTest();
  void checkAlignment();
};


CPPUNIT_TEST_SUITE_REGISTRATION(testSoA);


#define check(X) \
  do { std::cout << #X " is " << (X) << std::endl; } while(false)

// declare a statically-sized SoA, templated on the column size and (optional) alignment

declare_SoA_template(SoA,
  // predefined static scalars
  // size_t size;
  // size_t alignment;

  // columns: one value per element
  SoA_column(double, x),
  SoA_column(double, y),
  SoA_column(double, z),
  SoA_column(uint16_t, colour),
  SoA_column(int32_t, value),
  SoA_column(const char *, name),

  // scalars: one value for the whole structure
  SoA_scalar(const char *, description)
);

void testSoA::initialTest() {
  std::cout << std::boolalpha;



  SoA::dump(10, 8);
  // CPPUNIT_ASSERT_EQUAL_MESSAGE("Size of SoA<1>", 
  //     3 * sizeof(double)
  //     + (sizeof(uint16_t) / sizeof(int32_t) + 1) * sizeof(int32_t) // Take into account the padding to align the next element
  //     + sizeof(int32_t)
  //     + 2 * sizeof(const char *), 
  //   sizeof(SoA<1>));
  //check(sizeof(SoA<1>));
  std::cout << std::endl;

  SoA::dump(10);
  SoA::dump(31);
  SoA::dump(32);
  std::cout << std::endl;

  SoA::dump(1, 64);
  SoA::dump(10, 64);
  SoA::dump(31, 64);
  SoA::dump(32, 64);
  std::cout << std::endl;

  SoA soa(new std::byte[SoA::computeDataSize(10,32)], 10, 32);
  check(& soa.z()[7] == & (soa[7].z));

  soa[7].x = 0.;
  soa[7].y = 3.1416;
  soa[7].z = -1.;
  soa[7].colour = 42;
  soa[7].value = 9999;
  soa[7].name = "element";

  soa[9] = soa[7];

  CPPUNIT_ASSERT(& soa.z()[7] == & (soa[7].z));
}

template <typename T>
void checkValuesAlignment(SoA &soa, T SoA::element::*member, const std::string & memberName, uint32_t bitAlignment) {
  if (bitAlignment % 8) CPPUNIT_FAIL("bitAlignment not byte aligned.");
  size_t byteAlignment = bitAlignment / 8;
  for (size_t i=0; i<soa.nElements(); i++) {
    // Check that each value is aligned
    if (reinterpret_cast<std::uintptr_t>(&(soa[i].*member)) % byteAlignment
            != (i * T::valueSize) %byteAlignment ) {
      std::stringstream err;
      err << "Misaligned value: " <<  memberName << " at index=" << i
              << " address=" << &(soa[i].*member) << " byteAlignment=" << byteAlignment
              << " address lower part: " << reinterpret_cast<std::uintptr_t>(&(soa[i].*member)) % byteAlignment
              << " expected address lower part: " << ((i * T::valueSize) % byteAlignment)
              << " size=" << soa.nElements() << " align=" << soa.byteAlignment();
      CPPUNIT_FAIL(err.str());
    }
    // Check that all values except the first-in rows (address 0 modulo alignment)
    // are contiguous to their predecessors in memory (this will detect cutting
    // memory/cache/etc... lines in unexpected places (for blocked SoA like AoSoA)
    if ((reinterpret_cast<std::uintptr_t>(&(soa[i].*member)) % byteAlignment)
         && (reinterpret_cast<std::uintptr_t>(&(soa[i - 1].*member)) + T::valueSize
              != reinterpret_cast<std::uintptr_t>(&(soa[i].*member)))) {
      std::stringstream err;
      err << "Unexpected non-contiguity: " <<  memberName << " at index=" << i
              << " address=" << &(soa[i].*member) << " is not contiguous to "
              << memberName << " at index=" << i - 1 << "address=" << &(soa[i - 1].*member)
              << " size=" << soa.nElements() << " align=" << soa.byteAlignment() << " valueSize=" << T::valueSize;
      CPPUNIT_FAIL(err.str());
    }
  }
}

void checkSoAAlignment(size_t nElements, size_t byteAlignment, uint32_t bitAlignment) {
  // Allocate a block which is aligned (we need to pass free as the custom free function (to replace delete).
  auto malloc_deleter = [](void * p){free(p);};
  std::unique_ptr<std::byte, decltype(malloc_deleter)> dataBlock(
    reinterpret_cast<std::byte*>(aligned_alloc(byteAlignment, SoA::computeDataSize(nElements,byteAlignment))),
    malloc_deleter
  );
  SoA soa(dataBlock.get(), nElements, byteAlignment);
  checkValuesAlignment(soa, &SoA::element::x, "x", bitAlignment);
  checkValuesAlignment(soa, &SoA::element::y, "y", bitAlignment);
  checkValuesAlignment(soa, &SoA::element::z, "z", bitAlignment);
  checkValuesAlignment(soa, &SoA::element::colour, "colour", bitAlignment);
  checkValuesAlignment(soa, &SoA::element::value, "value", bitAlignment);
  checkValuesAlignment(soa, &SoA::element::name, "name", bitAlignment);
}

void testSoA::checkAlignment() {
  checkSoAAlignment(1, 1, 8);
  checkSoAAlignment(10, 1, 8);
  checkSoAAlignment(31, 1, 8);
  checkSoAAlignment(32, 1, 8);

  checkSoAAlignment(1,64, 8*64);
  checkSoAAlignment(10,64, 8*64);
  checkSoAAlignment(31,64, 8*64);
  checkSoAAlignment(32,64, 8*64);
}

