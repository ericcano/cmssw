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
  CPPUNIT_TEST(runInGPU);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}
  void runInGPU();
  
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
  __host__ __device__ static void checkSoAelement(SoA soa, size_t i, bool * pass) {
    if (i >= soa.nElements() || !*pass) return;
    if (soa[i].x != 11.0 * i) { *pass = false; return; }
    if (soa[i].y != 22.0 * i) { *pass = false; return; }
    if (soa[i].z != 33.0 * i) { *pass = false; return; }
    if (soa[i].colour != i) { *pass = false; return; }
    if (soa[i].value != static_cast<int32_t>(0x10001 * i)) { *pass = false; return; }
  }

};

CPPUNIT_TEST_SUITE_REGISTRATION(testSoAGPU);

#define CUDA_UNIT_CHECK(A) CPPUNIT_ASSERT_NO_THROW(cudaCheck(A))

namespace {
  // Fill up the elements of the SoA
  __global__ void fillSoA(testSoAGPU::SoA soa) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= soa.nElements()) return;
    soa[i].x = 11.0 * i;
    soa[i].y = 22.0 * i;
    soa[i].z = 33.0 * i;
    soa[i].colour = i;
    soa[i].value = 0x10001 * i;
    soa[i].py = &soa[i].y;
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

void testSoAGPU::runInGPU() {
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
  for (; pass && i< hostSoA.nElements(); i++) checkSoAelement(hostSoA, i, &pass);
  if (!pass) {
    std::cout << "In " << typeid(*this).name() << " check failed at i= " << i << ")" << std::endl;
    hexdump(hostSoABlock.get(), SoA::computeDataSize(hostSoA.nElements()));
    printf("base=%p, &y=%p\n", deviceSoABlock.get(), deviceSoA.y());
  }
  CPPUNIT_ASSERT(pass);
}