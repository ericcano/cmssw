#ifndef HeterogeneousCore_AlpakaInterface_interface_prefixScan_h
#define HeterogeneousCore_AlpakaInterface_interface_prefixScan_h
// Active version

#include <algorithm>
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "FWCore/Utilities/interface/CMSUnrollLoop.h"

namespace cms { 
  namespace alpakatools {

    // FIXME warpSize should be device-dependent
    constexpr uint32_t warpSizeHardcodedFixme = 32;
    constexpr uint64_t warpMask = ~(~0ull << warpSizeHardcodedFixme);

#if (defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDA_ARCH__)) || \
    (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && defined(__HIP_DEVICE_COMPILE__))

    template <typename T>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void warpPrefixScan(uint32_t laneId, T const* ci, T* co, uint32_t i, uint32_t mask) {
#if defined(__HIP_DEVICE_COMPILE__)
      ALPAKA_ASSERT_OFFLOAD(mask == warpMask);
#endif
      // ci and co may be the same
      auto x = ci[i];
      CMS_UNROLL_LOOP
      for (uint32_t offset = 1; offset < warpSize; offset <<= 1) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
        auto y = __shfl_up_sync(mask, x, offset);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
        auto y = __shfl_up(x, offset);
#endif
        if (laneId >= offset)
          x += y;
      }
      co[i] = x;
    }

    template <typename T>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void warpPrefixScan(uint32_t laneId, T* c, uint32_t i, uint32_t mask) {
#if defined(__HIP_DEVICE_COMPILE__)
      ALPAKA_ASSERT_OFFLOAD(mask == warpMask);
#endif
      auto x = c[i];
      CMS_UNROLL_LOOP
      for (uint32_t offset = 1; offset < warpSize; offset <<= 1) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
        auto y = __shfl_up_sync(mask, x, offset);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
        auto y = __shfl_up(x, offset);
#endif
        if (laneId >= offset)
          x += y;
      }
      c[i] = x;
    }

#endif  // (defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDA_ARCH__)) || (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && defined(__HIP_DEVICE_COMPILE__))

    // limited to warpSize² elements
    template <typename TAcc, typename T>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void blockPrefixScan(
        const TAcc& acc, T const* ci, T* co, uint32_t size, T* ws = nullptr) {
#if (defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDA_ARCH__)) || \
    (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && defined(__HIP_DEVICE_COMPILE__))
      uint32_t const blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);
      uint32_t const blockThreadIdx(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
      ALPAKA_ASSERT_OFFLOAD(ws);
      ALPAKA_ASSERT_OFFLOAD(size <= warpSize * warpSize);
      ALPAKA_ASSERT_OFFLOAD(0 == blockDimension % warpSize);
      auto first = blockThreadIdx;
#if defined(__CUDA_ARCH__)
      auto mask = __ballot_sync(warpMask, first < size);
#elif defined(__HIP_DEVICE_COMPILE__)
      auto mask = warpMask;
#endif
      auto laneId = blockThreadIdx & (warpSize - 1);

      for (auto i = first; i < size; i += blockDimension) {
        warpPrefixScan(laneId, ci, co, i, mask);
        auto warpId = i / warpSize;
        // FIXME test ?
        ALPAKA_ASSERT_OFFLOAD(warpId < warpSize);
        if ((warpSize - 1) == laneId)
          ws[warpId] = co[i];
#if defined(__CUDA_ARCH__)
        mask = __ballot_sync(mask, i + blockDimension < size);
#endif
      }
      alpaka::syncBlockThreads(acc);
      if (size <= warpSize)
        return;
      if (blockThreadIdx < warpSize) {
        warpPrefixScan(laneId, ws, blockThreadIdx, warpMask);
      }
      alpaka::syncBlockThreads(acc);
      for (auto i = first + warpSize; i < size; i += blockDimension) {
        uint32_t warpId = i / warpSize;
        co[i] += ws[warpId - 1];
      }
      alpaka::syncBlockThreads(acc);
#else
      co[0] = ci[0];
      for (uint32_t i = 1; i < size; ++i)
        co[i] = ci[i] + co[i - 1];
#endif  // (defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDA_ARCH__)) || (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && defined(__HIP_DEVICE_COMPILE__))
    }
    
    template <typename TAcc, typename T>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void blockPrefixScan(const TAcc& acc,
                                                             T* __restrict__ c,
                                                             uint32_t size,
                                                             T* __restrict__ ws = nullptr) {
#if (defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDA_ARCH__)) || \
    (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && defined(__HIP_DEVICE_COMPILE__))
      uint32_t const blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);
      uint32_t const blockThreadIdx(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
      ALPAKA_ASSERT_OFFLOAD(ws);
      ALPAKA_ASSERT_OFFLOAD(size <= warpSize * warpSize);
      ALPAKA_ASSERT_OFFLOAD(0 == blockDimension % warpSize);
      auto first = blockThreadIdx;
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
      auto mask = __ballot_sync(warpMask, first < size);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
      auto mask = warpMask;
#endif
      auto laneId = blockThreadIdx & (warpSize - 1);

      for (auto i = first; i < size; i += blockDimension) {
        warpPrefixScan(laneId, c, i, mask);
        auto warpId = i / warpSize;
        ALPAKA_ASSERT_OFFLOAD(warpId < warpSize);
        if ((warpSize - 1) == laneId)
          ws[warpId] = c[i];
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
        mask = __ballot_sync(mask, i + blockDimension < size);
#endif
      }
      alpaka::syncBlockThreads(acc);
      if (size <= warpSize)
        return;
      if (blockThreadIdx < warpSize) {
        warpPrefixScan(laneId, ws, blockThreadIdx, warpMask);
      }
      alpaka::syncBlockThreads(acc);
      for (auto i = first + warpSize; i < size; i += blockDimension) {
        auto warpId = i / warpSize;
        c[i] += ws[warpId - 1];
      }
      alpaka::syncBlockThreads(acc);
#else
      for (uint32_t i = 1; i < size; ++i)
        c[i] += c[i - 1];
#endif  // (defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDA_ARCH__)) || (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && defined(__HIP_DEVICE_COMPILE__))
    }

    // in principle not limited....
    template <typename T>
    struct multiBlockPrefixScan {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc& acc, T const* ci, T* co, uint32_t size, int32_t numBlocks, int32_t* pc) const {
        //volatile T const* ci = ici;
        //volatile T* co = ico;
        // Get shared variable.
        auto& ws = alpaka::declareSharedVar<T[warpSizeHardcodedFixme], __COUNTER__>(acc);
    #ifdef __CUDA_ARCH__
       // TODO assert(sizeof(T) * gridDim.x <= dynamic_smem_size());  // size of psum below
    #endif
        const auto threadsPerGrid = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0u];
        const auto threadsPerBlock = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
        const auto blocksPerGrid = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u];
        const auto blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];
        const auto threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
        assert(threadsPerGrid >= size);
        // first each block does a scan
        [[maybe_unused]] int off = threadsPerBlock * blockIdx;
        if (size - off > 0)
          blockPrefixScan(acc, ci + off, co + off, std::min(threadsPerBlock, size - off), ws);

        // count blocks that finished
        auto& isLastBlockDone = alpaka::declareSharedVar<bool, __COUNTER__>(acc);
        //__shared__ bool isLastBlockDone;
        if (0 == threadIdx) {
          alpaka::mem_fence(acc, alpaka::memory_scope::Device{});
          auto value = alpaka::atomicAdd(acc, pc, 1, alpaka::hierarchy::Blocks{});  // block counter
          isLastBlockDone = (value == (int(blocksPerGrid) - 1));
        }

        alpaka::syncBlockThreads(acc);

        if (!isLastBlockDone)
          return;

        assert(int(blocksPerGrid) == *pc);

        // good each block has done its work and now we are left in last block

        // let's get the partial sums from each block
        T* psum = alpaka::getDynSharedMem<T>(acc);
        for (uint32_t i = threadIdx, ni = blocksPerGrid; i < ni; i += threadsPerBlock) {
          auto j = threadsPerBlock * i + threadsPerBlock - 1;
          psum[i] = (j < size) ? co[j] : T(0);
        }
        alpaka::syncBlockThreads(acc);
        blockPrefixScan(acc, psum, psum, blocksPerGrid, ws);

        // now it would have been handy to have the other blocks around...
        for (uint32_t i = threadIdx + threadsPerBlock, k = 0; i < size; i += threadsPerBlock, ++k) {
          co[i] += psum[k];
        }
      }
    };  

  }  // namespace alpakatools
}  // namespace cms

// declare the amount of block shared memory used by the multiBlockPrefixScanSecondStep kernel
namespace alpaka::trait {
  // Variable size shared mem   
  template <typename TAcc, typename T>
  struct BlockSharedMemDynSizeBytes<cms::alpakatools::multiBlockPrefixScan<T>, TAcc> {
    template <typename TVec>
    ALPAKA_FN_HOST_ACC static std::size_t getBlockSharedMemDynSizeBytes(
        cms::alpakatools::multiBlockPrefixScan<T> const& /* kernel */,
        TVec const& /* blockThreadExtent */,
        TVec const& /* threadElemExtent */,
        T const* /* ci */,
        T const* /* co */,
        int32_t /* size */,
        int32_t numBlocks,
        int32_t const* /* pc */) {
      std::size_t ret = sizeof(int32_t) * numBlocks;
      return ret;
    }
  };

}  // namespace alpaka::trait

#endif  // HeterogeneousCore_AlpakaInterface_interface_prefixScan_h
