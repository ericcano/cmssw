// system include files
#include <cmath>
#include <type_traits>
#include <stdio.h>

// CUDA include files
#include <cuda_runtime.h>

// CMSSW include files
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "cudavectors.h"

namespace cudavectors {

  __host__ __device__ inline void convert(CylindricalVector const& cylindrical, CartesianVector & cartesian) {
      cartesian.x = cylindrical.rho * std::cos(cylindrical.phi);
      cartesian.y = cylindrical.rho * std::sin(cylindrical.phi);
      cartesian.z = cylindrical.rho * std::sinh(cylindrical.eta);

  }

  __global__ void convertKernel(CylindricalVector const* cylindrical, CartesianVector* cartesian, size_t size) {
    __shared__ CylindricalVector sInput[threadsPerBlock];
    __shared__ CartesianVector sOutput[threadsPerBlock];

    // Weed out the non-needed threads;
    auto blockStart = blockIdx.x * blockDim.x;
    if (blockStart + threadIdx.x >= size) return;

    // Collaboratively load the input by lines into the shared memory.
    // The load working by batch (as many batches as there are floats in the struct.

    // We have to take into account the incomplete blocks.

    //size_t thisBlockSize = std::min(size - blockStart, blockDim.x); // <- did not compile oO
    size_t thisBlockSize = size - blockStart;
    if (thisBlockSize > blockDim.x) thisBlockSize = blockDim.x;
    {
      // Limited scope for early release of registers variables.

      // We need to load as many lines as there are floats (or padding) in the struct.
      // If the struct is not float-aligned, we bailout at compile time.
      static_assert(!(sizeof(CylindricalVector) % sizeof(float)),
              "Cannot load this way: struct not float aligned.");
      size_t firstFloat = blockStart * sizeof(CylindricalVector) / sizeof(float);
      const auto *pInput = reinterpret_cast<const float *>(cylindrical);
      auto *psInput = reinterpret_cast<float *>(sInput);
      pInput += (firstFloat + threadIdx.x);
      psInput += threadIdx.x;
      #pragma unroll
      for (size_t i=0; i< sizeof(CylindricalVector) / sizeof(float); i++) {
        *psInput = *pInput; psInput += thisBlockSize; pInput += thisBlockSize;
      }
    }
    __syncthreads();
    convert (sInput[threadIdx.x], sOutput[threadIdx.x]);
    //if (!(threadIdx.x || blockIdx.x)) {
    //  printf("threadIdx.x=%d, blockIdx.x=%d ", threadIdx.x, blockIdx.x);
    //  printf("In:rho=%f, eta=%f, phi=%f ", sInput[0].rho, sInput[0].eta,sInput[0].phi);
    //  printf("Out:x=%f, y=%f, z=%f\n", sOutput[0].x, sOutput[0].y,sOutput[0].z);
    //}

    __syncthreads();
    // Collaboratively copy the output.
    {
      static_assert(!(sizeof(CartesianVector) % sizeof(float)),
              "Cannot store this way: struct not float aligned.");
      size_t firstFloat = blockStart * sizeof(CartesianVector) / sizeof(float);
      auto *pOutput = reinterpret_cast<float *>(cartesian);
      auto *psOutput = reinterpret_cast<float *>(sOutput);
      pOutput += (firstFloat + threadIdx.x);
      psOutput += threadIdx.x;
      // We need to store as many lines as there are floats (or padding) in the struct.
      // If the struct is not float-aligned, we bailout at compile time.
      #pragma unroll
      for (size_t i=0; i< sizeof(CartesianVector) / sizeof(float); i++) {
        *pOutput = *psOutput; pOutput += thisBlockSize; psOutput += thisBlockSize;
      }
    }
  }

  void convertWrapper(CylindricalVector const* cylindrical, CartesianVector* cartesian, size_t size) {
    // allocate memory on the GPU for the cylindrical and cartesian vectors
    //auto deviceInput = cms::cuda::device::make_device_unique<CylindricalVector[]>(size, nullptr);
    //auto deviceOutput = cms::cuda::device::make_device_unique<CartesianVector[]>(size, nullptr);
    CylindricalVector * deviceInput;
    CartesianVector * deviceOutput;
    cudaCheck(cudaMalloc(&deviceInput, sizeof(CylindricalVector)*size));
    cudaCheck(cudaMalloc(&deviceOutput, sizeof(CartesianVector)*size));

    // copy the input data to the GPU
    cudaCheck(cudaMemcpy(deviceInput, cylindrical,
      sizeof(CylindricalVector) * size, cudaMemcpyHostToDevice));

    // convert the vectors from cylindrical to cartesian coordinates, on the GPU
    //std::cout << "Kernel launch size=" << size << " blocks=" << size/threadsPerBlock+1 << " threadsPerBlock=" << threadsPerBlock << std::endl;
    convertKernel<<<size/threadsPerBlock+1, threadsPerBlock>>>(deviceInput, deviceOutput, size);

    // copy the result from the GPU
    cudaCheck(cudaMemcpy(cartesian, deviceOutput,
      sizeof(CartesianVector) * size, cudaMemcpyDeviceToHost));

    // free the GPU memory;
    cudaCheck(cudaFree(deviceInput));
    cudaCheck(cudaFree(deviceOutput));
    cudaCheck(cudaGetLastError());
  }

}  // namespace cudavectors
