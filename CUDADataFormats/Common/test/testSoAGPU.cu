#include "cppunit/extensions/HelperMacros.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

/*
 * Unit tests for SoA running on GPU
 */

#include "CUDADataFormats/Common/interface/SoAmacros.h"

#include <cuda_runtime_api.h>

class testSoAGPU: public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testSoAGPU);
  CPPUNIT_TEST(fill);
  CPPUNIT_TEST(crossProduct);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}
  void fill();
  void crossProduct();
  
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
    SoA_column(double *, py),

    // scalars: one value for the whole structure
    SoA_scalar(const char *, description)
  );

  // declare equivalent struct
  struct AoSelement {
    double x;
    double y;
    double z;
    uint16_t colour;
    int32_t value;
    double * py;
  };
private:
  static constexpr int defaultDevice = 0;
  static constexpr size_t elementsCount = 100;
  
  // Check we find what we wanted to initialize.
  // Pass should be initialized to true.
  template <class T>
  __host__ __device__ static void checkSoAelement(T soa, size_t i, bool & pass) {
    if (i >= soa.nElements() || !pass) return;
    if (soa[i].x != 11.0 * i) { pass = false; return; }
    if (soa[i].y != 22.0 * i) { pass = false; return; }
    if (soa[i].z != 33.0 * i) { pass = false; return; }
    if (soa[i].colour != i) { pass = false; return; }
    if (soa[i].value != static_cast<int32_t>(0x10001 * i)) { pass = false; return; }
  }

  // Check r[i].{x, y, z} are close enough to zero compared to a[i].{x,y,z} and b[i].{x,y,z}
  // to validate a cross product of a vector with itself produced a zero (enough) result.
  template <class T>
  __host__ __device__ static void checkSoAelementNullityRealtiveToSquare(T resSoA, T referenceSoA, size_t i, bool & pass) {
    if (i >= resSoA.nElements() || !pass) return;
    auto ref = max (abs(referenceSoA[i].x), 
                    max(abs(referenceSoA[i].y), 
                        abs(referenceSoA[i].z)));
    ref *= ref;
    ref *= std::numeric_limits<double>::epsilon();
    if (abs(resSoA[i].x) > ref) { pass = false; return; }
    if (abs(resSoA[i].y) > ref) { pass = false; return; }
    if (abs(resSoA[i].z) > ref) { pass = false; return; }    
  }
};

CPPUNIT_TEST_SUITE_REGISTRATION(testSoAGPU);

#define CUDA_UNIT_CHECK(A) CPPUNIT_ASSERT_NO_THROW(cudaCheck(A))

namespace {
  // fill element
  template <class T>
  __host__ __device__ void fillElement(T & e, size_t i) {
    e.x = 11.0 * i;
    e.y = 22.0 * i;
    e.z = 33.0 * i;
    e.colour = i;
    e.value = 0x10001 * i;
    e.py = &e.y;
  }

  // Fill up the elements of the SoA
  __global__ void fillSoA(testSoAGPU::SoA soa) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= soa.nElements()) return;
    // compiler does not belive we can use a temporary soa[i] to store results.
    // So make an lvalue.
    auto e = soa[i];
    fillElement(e, i);
  }

  __global__ void fillAoS(testSoAGPU::AoSelement *aos, size_t nElements) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nElements) return;
    fillElement(aos[i], i);
  }

  // Simple cross product for elements
  template <typename T>
  __host__ __device__ void crossProduct(T & r, const T & a, const T & b) {
    r.x = a.y * b.z - a.z * b.y;
    r.y = a.z * b.x - a.x * b.z;
    r.z = a.x * b.y - a.y * b.x;
  }

  // Simple cross product (SoA)
  __global__ void indirectCrossProductSoA(testSoAGPU::SoA r, const testSoAGPU::SoA a, const testSoAGPU::SoA b) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= min(r.nElements(), min(a.nElements(), b.nElements()))) return;
    // C++ does not allow creating non-const references to temporary variables
    // this workaround makes the temporary variable 
    auto ri = r[i];
    // Print addresses for a few samples
    if (not i%10) {
      // TODO: the output of this is fishy, we expect equality. The rest seems to work though (ongoing).
      printf ("i=%zd &r[i].x=%p &ri.x=%p\n", i, &r[i].x, &ri.x);
    }
    crossProduct(ri, a[i], b[i]);
  }

  // Simple cross product (SoA on CPU)
  __host__ void indirectCPUcrossProductSoA(testSoAGPU::SoA r, const testSoAGPU::SoA a, const testSoAGPU::SoA b) {
    for (size_t i =0; i< min(r.nElements(), min(a.nElements(), b.nElements())); ++i) {
      // This vesion is also affected.
      auto ri = r[i];
      crossProduct(ri, a[i], b[i]);
    }
  }

  // Simple cross product (AoS)
  __global__ void crossProductAoS(testSoAGPU::AoSelement *r,
          testSoAGPU::AoSelement *a, testSoAGPU::AoSelement *b, size_t nElements) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nElements) return;
    crossProduct(r[i], a[i], b[i]);
  }

  void hexdump(void *ptr, int buflen) {
    /* From https://stackoverflow.com/a/29865 with adaptations */
    unsigned char *buf = (unsigned char*)ptr;
    int i, j;
    for (i=0; i<buflen; i+=16) {
      printf("%06x: ", i);
      for (j=0; j<16; j++) {
        if (i+j < buflen)
          printf("%02x ", buf[i+j]);
        else
          printf("   ");
        if ((i+j) % 4 == 3) printf (" ");
      }
      printf(" ");
  //  for (j=0; j<16; j++)
  //    if (i+j < buflen)
  //      printf("%c", isprint(buf[i+j]) ? buf[i+j] : '.');
      printf("\n");
    }
  }
}

void testSoAGPU::fill() {
  // Get device, stream, memory
  cudaDeviceProp deviceProperties;
  CPPUNIT_ASSERT(cms::cudatest::testDevices());
  CUDA_UNIT_CHECK(cudaGetDeviceProperties(&deviceProperties, defaultDevice));
  cudaStream_t stream;
  CUDA_UNIT_CHECK(cudaStreamCreate(&stream));
  
  // Allocate memory and populate SoA descriptors
  auto deviceSoABlock = cms::cuda::make_device_unique<std::byte[]>(SoA::computeDataSize(elementsCount), stream);
  auto hostSoABlock = cms::cuda::make_host_unique<std::byte[]>(SoA::computeDataSize(elementsCount), stream);
  SoA deviceSoA(deviceSoABlock.get(), elementsCount);
  SoA hostSoA(hostSoABlock.get(), elementsCount);
  
  // Call kernel, get result
  fillSoA<<<(elementsCount - 1)/deviceProperties.warpSize + 1, deviceProperties.warpSize, 0, stream>>>(deviceSoA);
  CUDA_UNIT_CHECK(cudaMemcpyAsync(hostSoABlock.get(), deviceSoABlock.get(), SoA::computeDataSize(hostSoA.nElements()), cudaMemcpyDeviceToHost, stream));
  CUDA_UNIT_CHECK(cudaStreamSynchronize(stream));

  // Validate result
  bool pass = true;
  size_t i = 0;
  for (; pass && i< hostSoA.nElements(); i++) checkSoAelement(hostSoA, i, pass);
  if (!pass) {
    std::cout << "In " << typeid(*this).name() << " check failed at i= " << i << ")" << std::endl;
    hexdump(hostSoABlock.get(), SoA::computeDataSize(hostSoA.nElements()));
    printf("base=%p, &y=%p\n", deviceSoABlock.get(), deviceSoA.y());
  }
  CPPUNIT_ASSERT(pass);
}

void testSoAGPU::crossProduct() {
  // Get device, stream, memory
  cudaDeviceProp deviceProperties;
  CPPUNIT_ASSERT(cms::cudatest::testDevices());
  CUDA_UNIT_CHECK(cudaGetDeviceProperties(&deviceProperties, defaultDevice));
  cudaStream_t stream;
  CUDA_UNIT_CHECK(cudaStreamCreate(&stream));
  
  // Allocate memory and populate SoA descriptors (device A as source and R as result of cross product)
  auto deviceSoABlockA = cms::cuda::make_device_unique<std::byte[]>(SoA::computeDataSize(elementsCount), stream);
  auto deviceSoABlockR = cms::cuda::make_device_unique<std::byte[]>(SoA::computeDataSize(elementsCount), stream);
  auto hostSoABlockA = cms::cuda::make_host_unique<std::byte[]>(SoA::computeDataSize(elementsCount), stream);
  auto hostSoABlockR = cms::cuda::make_host_unique<std::byte[]>(SoA::computeDataSize(elementsCount), stream);
  SoA deviceSoAA(deviceSoABlockA.get(), elementsCount);
  SoA deviceSoAR(deviceSoABlockR.get(), elementsCount);
  SoA hostSoAA(hostSoABlockA.get(), elementsCount);
  SoA hostSoAR(hostSoABlockR.get(), elementsCount);
  
  // Call kernels, get result. Also fill up result SoA to ensure the results go in the right place.
  fillSoA<<<(elementsCount - 1)/deviceProperties.warpSize + 1, deviceProperties.warpSize, 0, stream>>>(deviceSoAA);
  fillSoA<<<(elementsCount - 1)/deviceProperties.warpSize + 1, deviceProperties.warpSize, 0, stream>>>(deviceSoAR);
  indirectCrossProductSoA<<<(elementsCount - 1)/deviceProperties.warpSize + 1, deviceProperties.warpSize, 0, stream>>>(deviceSoAR, deviceSoAA, deviceSoAA);
  CUDA_UNIT_CHECK(cudaMemcpyAsync(hostSoABlockA.get(), deviceSoABlockA.get(), SoA::computeDataSize(hostSoAA.nElements()), cudaMemcpyDeviceToHost, stream));
  CUDA_UNIT_CHECK(cudaMemcpyAsync(hostSoABlockR.get(), deviceSoABlockR.get(), SoA::computeDataSize(hostSoAR.nElements()), cudaMemcpyDeviceToHost, stream));
  CUDA_UNIT_CHECK(cudaStreamSynchronize(stream));

  // Validate result
  bool pass = true;
  size_t i = 0;
  for (; pass && i< hostSoAR.nElements(); i++) checkSoAelementNullityRealtiveToSquare(hostSoAR, hostSoAA, i, pass);
  if (!pass) {
    std::cout << "In " << typeid(*this).name() << " check failed at i= " << i << ")" << std::endl;
    std::cout << "result[" << i << "].x=" << hostSoAR[i].x << " .y=" << hostSoAR[i].y << " .z=" << hostSoAR[i].z << std::endl;
  } else {
    std::cout << std::endl;
    for (size_t j=0; j<10 ; ++j) {
      std::cout << "result[" << j << "]].x=" << hostSoAR[j].x << " .y=" << hostSoAR[j].y << " .z=" << hostSoAR[j].z << std::endl;
      std::cout << "A[" << j << "]].x=" << hostSoAA[j].x << " .y=" << hostSoAA[j].y << " .z=" << hostSoAA[j].z << std::endl;
    }
  }
  CPPUNIT_ASSERT(pass);
}