  #ifndef DataFormats_Portable_interface_PortableHostCollection_h
#define DataFormats_Portable_interface_PortableHostCollection_h

#include <cassert>
#include <optional>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
//#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "DataFormats/Portable/interface/PortableCollectionCommon.h"

// generic SoA-based product in host memory
template <typename T0, typename... Args>
class PortableHostCollectionImpl {

  template <typename T>
  static constexpr std::size_t count_t_ = portablecollection::typeCount<T, T0, Args...>;

  template <typename T>
  static constexpr std::size_t index_t_ = portablecollection::typeIndex<T, T0, Args...>;

  static constexpr std::size_t members_ = portablecollection::membersCount<T0, Args...>;

public:
  using Buffer = cms::alpakatools::host_buffer<std::byte[]>;
  using ConstBuffer = cms::alpakatools::const_host_buffer<std::byte[]>;
  using Implementation = portablecollection::CollectionImpl<0, T0, Args...>;

  using SizesArray = std::array<int32_t, members_>;

  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  using Layout = portablecollection::TypeResolver<Idx, T0, Args...>;
  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  using View = typename Layout<Idx>::View;
  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  using ConstView = typename Layout<Idx>::ConstView;

private:
  template <std::size_t Idx>
  using Leaf = portablecollection::CollectionLeaf<Idx, Layout<Idx>>;

  template <std::size_t Idx>
  Leaf<Idx>& get() {
    return static_cast<Leaf<Idx>&>(impl_);
  }

  template <std::size_t Idx>
  Leaf<Idx> const& get() const {
    return static_cast<Leaf<Idx> const&>(impl_);
  }

  template <typename T>
  portablecollection::CollectionLeaf<index_t_<T>, T>& get() {
    return static_cast<portablecollection::CollectionLeaf<index_t_<T>, T>&>(impl_);
  }

  template <typename T>
  const portablecollection::CollectionLeaf<index_t_<T>, T>& get() const {
    return static_cast<const portablecollection::CollectionLeaf<index_t_<T>, T>&>(impl_);
  }

  static int32_t computeDataSize(const std::array<int32_t, members_>& sizes) {
    int32_t ret = 0;
    constexpr_for<0, members_>([&sizes, &ret](auto i) { ret += Layout<i>::computeDataSize(sizes[i]); });
    return ret;
  }

public:
  PortableHostCollectionImpl() = default;

  PortableHostCollectionImpl(int32_t elements, alpaka_common::DevHost const& host)
      // allocate pageable host memory
      : buffer_{cms::alpakatools::make_host_buffer<std::byte[]>(Layout<>::computeDataSize(elements))},
        impl_{buffer_->data(), elements} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout<>::alignment == 0);
    static_assert(members_ == 1);
  }

  template <typename TQueue, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
  PortableHostCollectionImpl(int32_t elements, TQueue const& queue)
      // allocate pinned host memory associated to the given work queue, accessible by the queue's device
      : buffer_{cms::alpakatools::make_host_buffer<std::byte[]>(queue, Layout<>::computeDataSize(elements))},
        impl_{buffer_->data(), elements} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout<>::alignment == 0);
    static_assert(members_ == 1);
  }

  PortableHostCollectionImpl(const std::array<int32_t, members_>& sizes, alpaka_common::DevHost const& host)
      // allocate pinned host memory associated to the given work queue, accessible by the queue's device
      : buffer_{cms::alpakatools::make_host_buffer<std::byte[]>(computeDataSize(sizes))}, impl_{buffer_->data(), sizes} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    constexpr_for<0, members_>(
        [&](auto i) { assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout<i>::alignment == 0); });
    constexpr auto alignment = Layout<0>::alignment;
    constexpr_for<1, members_>([&alignment](auto i) { static_assert(alignment == Layout<i>::alignment); });
  }

  template <typename TQueue, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
  PortableHostCollectionImpl(const std::array<int32_t, members_>& sizes, TQueue const& queue)
      // allocate pinned host memory associated to the given work queue, accessible by the queue's device
      : buffer_{cms::alpakatools::make_host_buffer<std::byte[]>(queue, computeDataSize(sizes))},
        impl_{buffer_->data(), sizes} {
    // Alpaka set to a default alignment of 128 bytes defining ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
    constexpr_for<0, members_>(
        [&](auto i) { assert(reinterpret_cast<uintptr_t>(buffer_->data()) % Layout<i>::alignment == 0); });
    constexpr auto alignment = Layout<0>::alignment;
    constexpr_for<1, members_>([&alignment](auto i) { static_assert(alignment == Layout<i>::alignment); });
  }

  // non-copyable
  PortableHostCollectionImpl(PortableHostCollectionImpl const&) = delete;
  PortableHostCollectionImpl& operator=(PortableHostCollectionImpl const&) = delete;

  // movable
  PortableHostCollectionImpl(PortableHostCollectionImpl&&) = default;
  PortableHostCollectionImpl& operator=(PortableHostCollectionImpl&&) = default;

  // default destructor
  ~PortableHostCollectionImpl() = default;

  // access the View by index
  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  View<Idx>& view() {
    return get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  ConstView<Idx> const& view() const {
    return get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  ConstView<Idx> const& const_view() const {
    return get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  View<Idx>& operator*() {
    return get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  ConstView<Idx> const& operator*() const {
    return get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  View<Idx>* operator->() {
    return &get<Idx>().view_;
  }

  template <std::size_t Idx = 0, typename = std::enable_if_t<(members_ > Idx)>>
  ConstView<Idx> const* operator->() const {
    return &get<Idx>().view_;
  }

  // access the View by type
  template <typename T>
  typename T::View& view() {
    return get<T>().view_;
  }

  template <typename T>
  typename T::ConstView const& view() const {
    return get<T>().view_;
  }

  template <typename T>
  typename T::ConstView const& const_view() const {
    return get<T>().view_;
  }

  template <typename T>
  typename T::View& operator*() {
    return get<T>().view_;
  }

  template <typename T>
  typename T::ConstView const& operator*() const {
    return get<T>().view_;
  }

  template <typename T>
  typename T::View* operator->() {
    return &get<T>().view_;
  }

  template <typename T>
  typename T::ConstView const* operator->() const {
    return &get<T>().view_;
  }

  // access the Buffer
  Buffer buffer() { return *buffer_; }
  ConstBuffer buffer() const { return *buffer_; }
  ConstBuffer const_buffer() const { return *buffer_; }

  // Extract the sizes array
  SizesArray sizes() const {
    SizesArray ret;
    constexpr_for<0, members_>([&](auto i) { ret[i] = get<i>().layout_.metadata().size(); });
    return ret;
  }
  // part of the ROOT read streamer
  static void ROOTReadStreamer(PortableHostCollectionImpl* newObj, Implementation const& onfileImpl) {
    newObj->~PortableHostCollectionImpl();
    // use the global "host" object returned by cms::alpakatools::host()
    std::array<int32_t, members_> sizes;
    constexpr_for<0, members_>([&sizes, &onfileImpl](auto i) {
      sizes[i] = static_cast<Leaf<i> const&>(onfileImpl).layout_.metadata().size();
    });
    new (newObj) PortableHostCollectionImpl(sizes, cms::alpakatools::host());
    [[maybe_unused]] auto * newObjCopy = newObj;
    constexpr_for<0, members_>([&newObj, &onfileImpl](auto i) {
      std::cout << "Calling streamer for leaf<" << i << "> : " << typeid(typename Leaf<i>::Layout).name() << std::endl;
      static_cast<Leaf<i>&>(newObj->impl_).layout_.ROOTReadStreamer(static_cast<Leaf<i> const&>(onfileImpl).layout_);
    });
    // TODO: freeing of onfileImpl (as will be part of:
    //  Fix memory leaks in PortableHostCollectionImpl deserialization #40492
    // )
  }

private:
  std::optional<Buffer> buffer_;  //!
  Implementation impl_;           // (serialized: this is where the layouts live)
};

template <typename T0>
using PortableHostCollection = ::PortableHostCollectionImpl<T0>;

template <typename T0, typename T1>
using PortableHostCollection2 = ::PortableHostCollectionImpl<T0, T1>;

template <typename T0, typename T1, typename T2>
using PortableHostCollection3 = ::PortableHostCollectionImpl<T0, T1, T2>;

#endif  // DataFormats_Portable_interface_PortableHostCollection_h
