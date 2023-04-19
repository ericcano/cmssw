#ifndef RecoPixelVertexing_PixelVertexFinding_clusterTracksByDensityAlpaka_h
#define RecoPixelVertexing_PixelVertexFinding_clusterTracksByDensityAlpaka_h

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <alpaka/alpaka.hpp>
#include "DataFormats/Vertex/interface/ZVertexLayout.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaUtilities/interface/HistoContainer.h"
#include "../PixelVertexWorkSpaceLayout.h"
#include "vertexFinder.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace vertexFinder {
    using VtxSoAView = ::zVertex::ZVertexSoAView;
    using WsSoAView = ::vertexFinder::workSpace::PixelVertexWorkSpaceSoAView;
    // this algo does not really scale as it works in a single block...
    // enough for <10K tracks we have
    //
    // based on Rodrighez&Laio algo
    //
    template <typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void __attribute__((always_inline))
    clusterTracksByDensity(const TAcc& acc,
                           VtxSoAView& pdata,
                           WsSoAView& pws,
                           int minT,      // min number of neighbours to be "seed"
                           float eps,     // max absolute distance to cluster
                           float errmax,  // max error to be "seed"
                           float chi2max, // max normalized distance to cluster
                           uint32_t nBins,// number of bins
                           int32_t size   // maximum number of elements
    ) {
      using namespace vertexFinder;
      constexpr bool verbose = false;  // in principle the compiler should optmize out if false
      const uint32_t threadIdxLocal(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);

      if (verbose && 0 == threadIdxLocal)
        printf("params %d %f %f %f\n", minT, eps, errmax, chi2max);

      auto er2mx = errmax * errmax;

      auto& __restrict__ data = pdata;
      auto& __restrict__ ws = pws;
      auto nt = ws.ntrks();
      float const* __restrict__ zt = ws.zt();
      float const* __restrict__ ezt2 = ws.ezt2();

      uint32_t& nvFinal = data.nvFinal();
      uint32_t& nvIntermediate = ws.nvIntermediate();

      uint8_t* __restrict__ izt = ws.izt();
      int32_t* __restrict__ nn = data.ndof();
      int32_t* __restrict__ iv = ws.iv();

      ALPAKA_ASSERT_OFFLOAD(zt);
      ALPAKA_ASSERT_OFFLOAD(ezt2);
      ALPAKA_ASSERT_OFFLOAD(izt);
      ALPAKA_ASSERT_OFFLOAD(nn);
      ALPAKA_ASSERT_OFFLOAD(iv);

#if 1
      // using Hist = cms::alpakatools::HistoContainerFixedSize<T= uint8_t,  256, uint32_t NBINS =  16000,
      //    int32_t SIZE =   8, uint32_t S = sizeof(T) * 8, I = uint16_t(, uint32_t NHISTS = (default) 1)>;
      constexpr uint32_t nHists = 1;
      using Hist = cms::alpakatools::HistoContainerRuntimeSized<uint8_t, 8, uint16_t>;
      // The buffer will  contain:
      // - The histo container structure
      // - The SoA(s?) containing the data storage for it, and for the hws array (Hist::Counter[32])
      // (32 being warp or block size?)
      // - Compile time sizes are:
      // NBINS = 256
      // SIZE =  16000
      // NHISTS = (default) 1
      // totbins() = NHISTS(=1) * NBINS + 1
      // capacity() = SIZE
      // Leading to storage of: 
      // Counter (=uint32_t) off[totbins()]
      // index_type (=uint16_t) bins[capacity() = SIZE]
      // 
      // Make sure we are properly aligned for any scalar type
      std::byte* staticBuffer = reinterpret_cast<std::byte *>(alpaka::getDynSharedMem<std::max_align_t>(acc));
      // Allocate hist in shared memory
      auto& hist = *reinterpret_cast<Hist *>(staticBuffer);
      // Allocate other buffers: hist stores (off[totbins() = NHISTS * NBINS + 1] and bins[capacity() = SIZE])
      constexpr size_t histAlignedSize = ((sizeof(Hist) - 1) / sizeof(std::max_align_t) + 1) * sizeof(std::max_align_t);
      Hist::Counter* histOffs = reinterpret_cast<Hist::Counter*>(staticBuffer + histAlignedSize);
      std::size_t histOffsSize = nHists /* = 1 */  * nBins + 1;
      std::size_t histOffsAlignedByteSize = ((histOffsSize - 1) / sizeof(std::max_align_t) + 1) * sizeof(std::max_align_t);
      Hist::index_type* histBins = reinterpret_cast<Hist::index_type*>(reinterpret_cast<std::byte*>(histOffs) + histOffsAlignedByteSize);
      std::size_t histBinsSize = size;
      std::size_t histBinsAlignedByteSize = ((histBinsSize - 1) / sizeof(std::max_align_t) + 1) * sizeof(std::max_align_t);
      Hist::Counter* hws=reinterpret_cast<Hist::Counter*>(reinterpret_cast<std::byte*>(histBins) + histBinsAlignedByteSize);
      std::size_t hwsSize = 32 * sizeof(Hist::Counter);
      std::size_t hwsAlignedByteSize = ((hwsSize - 1) / sizeof(std::max_align_t) + 1) * sizeof(std::max_align_t);
      // Initialize the hist struct (one thread per block)
      if (0 == threadIdxLocal) {
        new(&hist)Hist(nBins, size, histOffs, histBins);
        printf("Initialized hist (@%p) with nBins=%d, size=%d, histAlignedSize=%lu, histOffs@%p, histOffsSize=%lu, histOffsAlignedByteSize=%lu\n",
                &hist, nBins, size, histAlignedSize, histOffs, histOffsSize, histOffsAlignedByteSize);
        printf("    histBins@%p, histBinsSize=%lu, histBinsAlignedByteSize=%lu, hws@%p, hwsSize=%lu, hwsAlignedByteSize=%lu\n",
                histBins, histBinsSize, histBinsAlignedByteSize, hws, hwsSize, hwsAlignedByteSize);
      }
      alpaka::syncBlockThreads(acc);
//      auto& hist = alpaka::declareSharedVar<Hist, __COUNTER__>(acc);
//      auto& hws = alpaka::declareSharedVar<Hist::Counter[32], __COUNTER__>(acc);
      for (auto j : cms::alpakatools::elements_with_stride(acc, hist.totbins())) {
        hist.off_[j] = 0;
      }
#else
      using Hist = cms::alpakatools::HistoContainerFixedSize<uint8_t, 256, 16000, 8, uint16_t>;
      auto& hist = alpaka::declareSharedVar<Hist, __COUNTER__>(acc);
      auto& hws = alpaka::declareSharedVar<Hist::Counter[32], __COUNTER__>(acc);

      for (auto j : cms::alpakatools::elements_with_stride(acc, Hist::totbins())) {
        hist.off[j] = 0;
      }
#endif
      alpaka::syncBlockThreads(acc);
      
      
      if (0 == threadIdxLocal) {
        printf ("hist.totbins()=%d\n", hist.totbins());
        for (std::size_t i=0; i<hist.totbins(); i++) {
          if (hist.off_[i]) {
            printf("After offset initialization: hist.off_[%ld]=%d\n", i, hist.off_[i]);
          }
        }
      }
      alpaka::syncBlockThreads(acc);
      if (verbose && 0 == threadIdxLocal)
        printf("booked hist with %d bins, size %d for %d tracks\n", hist.nbins(), hist.capacity(), nt);

      ALPAKA_ASSERT_OFFLOAD(nt <= hist.capacity());

      // fill hist  (bin shall be wider than "eps")
      for (auto i : cms::alpakatools::elements_with_stride(acc, nt)) {
        ALPAKA_ASSERT_OFFLOAD(i < ::zVertex::MAXTRACKS);
        int iz = int(zt[i] * 10.);  // valid if eps<=0.1
        // iz = std::clamp(iz, INT8_MIN, INT8_MAX);  // sorry c++17 only
        iz = std::min(std::max(iz, INT8_MIN), INT8_MAX);
        izt[i] = iz - INT8_MIN;
        ALPAKA_ASSERT_OFFLOAD(iz - INT8_MIN >= 0);
        ALPAKA_ASSERT_OFFLOAD(iz - INT8_MIN < 256);
        hist.count(acc, izt[i]);
        iv[i] = i;
        nn[i] = 0;
      }
      alpaka::syncBlockThreads(acc);
      if (threadIdxLocal < 32)
        hws[threadIdxLocal] = 0;  // used by prefix scan...
      alpaka::syncBlockThreads(acc);
      hist.finalize(acc, hws);
      alpaka::syncBlockThreads(acc);
      ALPAKA_ASSERT_OFFLOAD(hist.size() == nt);
      for (auto i : cms::alpakatools::elements_with_stride(acc, nt)) {
        hist.fill(acc, izt[i], uint16_t(i));
      }
      alpaka::syncBlockThreads(acc);

      // count neighbours
      if (0 == threadIdxLocal)
        printf("zt@%p, ezt2@%p, nn@%p, izt@%p\n", zt, ezt2, nn, izt);
      alpaka::syncBlockThreads(acc);
      for (auto i : cms::alpakatools::elements_with_stride(acc, nt)) {
        if (ezt2[i] > er2mx)
          continue;
        auto loop = [&](uint32_t j) {
          printf("chk 1\n");
          if (i == j)
            return;
          printf("chk 2\n");
          auto dist = std::abs(zt[i] - zt[j]);
          printf("chk 3\n");
          if (dist > eps)
            return;
          printf("chk 4\n");
          if (dist * dist > chi2max * (ezt2[i] + ezt2[j]))
            return;
          printf("chk 5\n");
          nn[i]++;
          printf("chk 6\n");
        };

        printf("chk B1, hist@%p, threadIdxLocal=%d\n", &hist, threadIdxLocal);
        cms::alpakatools::forEachInBins(hist, izt[i], 1, loop);
        printf("chk B2, threadIdxLocal=%d\n", threadIdxLocal);
      }

      alpaka::syncBlockThreads(acc);

      // find closest above me .... (we ignore the possibility of two j at same distance from i)
      for (auto i : cms::alpakatools::elements_with_stride(acc, nt)) {
        float mdist = eps;
        auto loop = [&](uint32_t j) {
          if (nn[j] < nn[i])
            return;
          if (nn[j] == nn[i] && zt[j] >= zt[i])
            return;  // if equal use natural order...
          auto dist = std::abs(zt[i] - zt[j]);
          if (dist > mdist)
            return;
          if (dist * dist > chi2max * (ezt2[i] + ezt2[j]))
            return;  // (break natural order???)
          mdist = dist;
          iv[i] = j;  // assign to cluster (better be unique??)
        };
        cms::alpakatools::forEachInBins(hist, izt[i], 1, loop);
      }

      alpaka::syncBlockThreads(acc);

#ifdef GPU_DEBUG
      //  mini verification
      for (auto i : cms::alpakatools::elements_with_stride(acc, nt)) {
        if (iv[i] != int(i))
          ALPAKA_ASSERT_OFFLOAD(iv[iv[i]] != int(i));
      }
      alpaka::syncBlockThreads(acc);
#endif

      // consolidate graph (percolate index of seed)
      for (auto i : cms::alpakatools::elements_with_stride(acc, nt)) {
        auto m = iv[i];
        while (m != iv[m])
          m = iv[m];
        iv[i] = m;
      }

#ifdef GPU_DEBUG
      alpaka::syncBlockThreads(acc);
      //  mini verification
      for (auto i : cms::alpakatools::elements_with_stride(acc, nt)) {
        if (iv[i] != int(i))
          ALPAKA_ASSERT_OFFLOAD(iv[iv[i]] != int(i));
      }
#endif

#ifdef GPU_DEBUG
      // and verify that we did not spit any cluster...
      for (auto i : cms::alpakatools::elements_with_stride(acc, nt)) {
        auto minJ = i;
        auto mdist = eps;
        auto loop = [&](uint32_t j) {
          if (nn[j] < nn[i])
            return;
          if (nn[j] == nn[i] && zt[j] >= zt[i])
            return;  // if equal use natural order...
          auto dist = std::abs(zt[i] - zt[j]);
          if (dist > mdist)
            return;
          if (dist * dist > chi2max * (ezt2[i] + ezt2[j]))
            return;
          mdist = dist;
          minJ = j;
        };
        cms::alpakatools::forEachInBins(hist, izt[i], 1, loop);
        // should belong to the same cluster...
        ALPAKA_ASSERT_OFFLOAD(iv[i] == iv[minJ]);
        ALPAKA_ASSERT_OFFLOAD(nn[i] <= nn[iv[i]]);
      }
      alpaka::syncBlockThreads(acc);
#endif

      // TODO: other shared variable!
      auto& foundClusters = alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);
      foundClusters = 0;
      alpaka::syncBlockThreads(acc);

      // find the number of different clusters, identified by a tracks with clus[i] == i and density larger than threshold;
      // mark these tracks with a negative id.
      for (auto i : cms::alpakatools::elements_with_stride(acc, nt)) {
        if (iv[i] == int(i)) {
          if (nn[i] >= minT) {
            auto old = alpaka::atomicInc(acc, &foundClusters, 0xffffffff, alpaka::hierarchy::Threads{});
            iv[i] = -(old + 1);
          } else {  // noise
            iv[i] = -9998;
          }
        }
      }
      alpaka::syncBlockThreads(acc);

      ALPAKA_ASSERT_OFFLOAD(foundClusters < ::zVertex::MAXVTX);

      // propagate the negative id to all the tracks in the cluster.
      for (auto i : cms::alpakatools::elements_with_stride(acc, nt)) {
        if (iv[i] >= 0) {
          // mark each track in a cluster with the same id as the first one
          iv[i] = iv[iv[i]];
        }
      }
      alpaka::syncBlockThreads(acc);

      // adjust the cluster id to be a positive value starting from 0
      for (auto i : cms::alpakatools::elements_with_stride(acc, nt)) {
        iv[i] = -iv[i] - 1;
      }

      nvIntermediate = nvFinal = foundClusters;
      if (verbose && 0 == threadIdxLocal)
        printf("found %d proto vertices\n", foundClusters);
    }
    class clusterTracksByDensityKernel {
    public:
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    VtxSoAView pdata,
                                    WsSoAView pws,
                                    int minT,      // min number of neighbours to be "seed"
                                    float eps,     // max absolute distance to cluster
                                    float errmax,  // max error to be "seed"
                                    float chi2max, // max normalized distance to cluster
                                    uint32_t nBins,// number of bins
                                    int32_t size   // maximum number of elements
      ) const {
        clusterTracksByDensity(acc, pdata, pws, minT, eps, errmax, chi2max, nBins, size);
      }
    };
  }  // namespace vertexFinder
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace alpaka::trait {
  template<typename TAcc, typename TSfinae>
  struct BlockSharedMemDynSizeBytes<ALPAKA_ACCELERATOR_NAMESPACE::vertexFinder::clusterTracksByDensityKernel, TAcc, TSfinae>
  {
#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored                                                                                  \
        "-Wdocumentation" // clang does not support the syntax for variadic template arguments "args,..."
#endif
    //! \param kernelFnObj The kernel object for which the block shared memory size should be calculated.
    //! \param blockThreadExtent The block thread extent.
    //! \param threadElemExtent The thread element extent.
    //! \tparam TArgs The kernel invocation argument types pack.
    //! \param args,... The kernel invocation arguments.
    //! \return The size of the shared memory allocated for a block in bytes.
    //! The default version always returns zero.
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TDim, typename... TArgs>
    ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
        [[maybe_unused]] ALPAKA_ACCELERATOR_NAMESPACE::vertexFinder::clusterTracksByDensityKernel const& kernelFnObj,
        [[maybe_unused]] Vec<TDim, Idx<TAcc>> const& blockThreadExtent,
        [[maybe_unused]] Vec<TDim, Idx<TAcc>> const& threadElemExtent,
        //[[maybe_unused]] const TAcc& acc,
        [[maybe_unused]] ALPAKA_ACCELERATOR_NAMESPACE::vertexFinder::VtxSoAView pdata,
        [[maybe_unused]] ALPAKA_ACCELERATOR_NAMESPACE::vertexFinder::WsSoAView pws,
        [[maybe_unused]] int minT,      // min number of neighbours to be "seed"
        [[maybe_unused]] float eps,     // max absolute distance to cluster
        [[maybe_unused]] float errmax,  // max error to be "seed"
        [[maybe_unused]] float chi2max,
        uint32_t nBins,// number of bins
        int32_t size)   // maximum number of elements) -> std::size_t
    {
      // The shared memory contains:
      // - The (fixed size) HistoContainerRuntimeSized object
      // - The offsets array
      // - The bins array
      // - The hws (hist work space?) array.
      // On top of that, things should be aligned to 
      using Hist = cms::alpakatools::HistoContainerRuntimeSized<uint8_t, 8, uint16_t>;
      constexpr uint32_t nHists = 1;
      constexpr std::size_t histAlignedSize = ((sizeof(Hist) - 1) / sizeof(std::max_align_t) + 1) * sizeof(std::max_align_t);
      std::size_t histOffsSize = nHists /* = 1 */  * nBins + 1;
      std::size_t histOffsAlignedByteSize = ((histOffsSize - 1) / sizeof(std::max_align_t) + 1) * sizeof(std::max_align_t);
      std::size_t histBinsSize = size;
      std::size_t histBinsAlignedByteSize = ((histBinsSize - 1) / sizeof(std::max_align_t) + 1) * sizeof(std::max_align_t);
      std::size_t hwsSize = 32 * sizeof(Hist::Counter);
      std::size_t hwsAlignedByteSize = ((hwsSize - 1) / sizeof(std::max_align_t) + 1) * sizeof(std::max_align_t);
      return  histAlignedSize + histOffsAlignedByteSize + histBinsAlignedByteSize + hwsAlignedByteSize;
    }
  };
} // namespace alpaka::trait

#endif  // RecoPixelVertexing_PixelVertexFinding_clusterTracksByDensityAlpaka_h
