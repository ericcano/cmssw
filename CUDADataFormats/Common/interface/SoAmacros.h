#ifndef CUDADataFormats_Common_SoAmacros_h
#define CUDADataFormats_Common_SoAmacros_h

#include "boost/preprocessor.hpp"

// CUDA attributes
#ifdef __CUDACC__
#define SOA_HOST_ONLY __host__
#define SOA_HOST_DEVICE __host__ __device__
#else
#define SOA_HOST_ONLY
#define SOA_HOST_DEVICE
#endif

namespace cms::cuda {
  // Helper template managing the value within its column
  template<typename T>
  class SoAValue {
  public:
    SOA_HOST_DEVICE SoAValue(size_t i, T * col): idx_(i), col_(col) {}
    SOA_HOST_DEVICE operator T&() { return col_[idx_]; }
    SOA_HOST_DEVICE operator const T&() const { return col_[idx_]; }
    SOA_HOST_DEVICE T* operator& () { return &col_[idx_]; }
    SOA_HOST_DEVICE const T* operator& () const { return &col_[idx_]; }
    template <typename T2>
    SOA_HOST_DEVICE T& operator= (const T2& v) { col_[idx_] = v; return col_[idx_]; }
    typedef T valueType;
    static constexpr auto valueSize = sizeof(T);
  private:
    size_t idx_;
    T *col_;
  };
} // namespace cms::cuda;

/* declare "scalars" (one value shared across the whole SoA) and "columns" (one value per element) */
#define _VALUE_TYPE_SCALAR 0
#define _VALUE_TYPE_COLUMN 1

#define SoA_scalar(TYPE, NAME) (_VALUE_TYPE_SCALAR, TYPE, NAME)
#define SoA_column(TYPE, NAME) (_VALUE_TYPE_COLUMN, TYPE, NAME)

/* General helper macros for iterating on columns or scalars reparately */
/* Predicate for Boost PP filters. ELEMENT is expected to be SoA_scalar or Soa_column */
#define _IS_VALUE_TYPE_PREDICATE(S, DATA, ELEM)                                                                                     \
  BOOST_PP_EQUAL(BOOST_PP_TUPLE_ELEM(0, ELEM), DATA)

/* Iterate on the macro MACRO chosen type of elements */
#define _ITERATE_ON_VALUE_TYPE(MACRO, DATA, VALUE_TYPE, ...) \
  BOOST_PP_SEQ_FOR_EACH(MACRO, DATA,\
    BOOST_PP_SEQ_FILTER(_IS_VALUE_TYPE_PREDICATE, VALUE_TYPE, \
      BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__) \
    )\
  )

/* Iterate on the macro MACRO chosen type of elements and return the result as a comma separated list */
#define _ITERATE_ON_VALUE_TYPE_COMMA(MACRO, DATA, VALUE_TYPE, ...) \
  BOOST_PP_TUPLE_ENUM(\
    BOOST_PP_SEQ_TO_TUPLE(\
      _ITERATE_ON_VALUE_TYPE(MACRO, DATA, VALUE_TYPE, __VA_ARGS__)\
    )\
  )

/* Iterate MACRO on all elements */
#define _ITERATE_ON_ALL(MACRO, DATA, ...) \
  BOOST_PP_SEQ_FOR_EACH(MACRO, DATA,\
    BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__) \
  )

/* dump SoA fields information; these should expand to, for columns:
 *
 *   std::cout << "  x_[" << SoA::size << "] at " 
 *             << offsetof(SoA, SoA::x_) << " has size " << sizeof(SoA::x_) << std::endl;
 *
 * and for scalars:
 *
 *   std::cout << "  x_ at " 
 *             << offsetof(SoA, SoA::x_) << " has size " << sizeof(SoA::x_) << std::endl;
 *
 */

#define _DECLARE_SOA_DUMP_INFO_IMPL(IS_COLUMN, TYPE, NAME)                                                                          \
  BOOST_PP_IIF(IS_COLUMN,                                                                                                           \
    std::cout << "  " BOOST_PP_STRINGIZE(NAME) "_ at offset " << offset                                                             \
              <<  " has size " << sizeof(TYPE) * nElements << " and padding "                                                       \
              << ((nElements * sizeof(TYPE) / byteAlignment) + 1) * byteAlignment - (sizeof(TYPE) * nElements) <<  std::endl;       \
  ,                                                                                                                                 \
    std::cout << "  " BOOST_PP_STRINGIZE(NAME) "_ at offset " << offset                                                             \
              <<  " has size " << sizeof(TYPE) << " and padding "                                                                   \
              << (sizeof(TYPE) / byteAlignment + 1) * byteAlignment - sizeof(TYPE) <<  std::endl;                                   \
  )

#define _DECLARE_SOA_DUMP_INFO(R, DATA, TYPE_NAME)                                                                                  \
  BOOST_PP_EXPAND(_DECLARE_SOA_DUMP_INFO_IMPL TYPE_NAME)

#define _ASSIGN_SOA_COLUMN_OR_SCALAR_IMPL(IS_COLUMN, TYPE, NAME)                                                                    \
  BOOST_PP_CAT(NAME, _) = reinterpret_cast<TYPE *>(curMem);                                                                         \
    BOOST_PP_IIF(IS_COLUMN,                                                                                                         \
    curMem += (((nElements_ * sizeof(TYPE) - 1) / byteAlignment_) + 1) * byteAlignment_;                                            \
  ,                                                                                                                                 \
    curMem += (((sizeof(TYPE) - 1) / byteAlignment_) + 1) * byteAlignment_;                                                         \
  )

#define _ASSIGN_SOA_COLUMN_OR_SCALAR(R, DATA, TYPE_NAME)                                                                            \
  _ASSIGN_SOA_COLUMN_OR_SCALAR_IMPL TYPE_NAME

#define _ACCUMULATE_SOA_ELEMENT_IMPL(IS_COLUMN, TYPE, NAME)                                                                         \
  BOOST_PP_IIF(IS_COLUMN,                                                                                                           \
    ret += (((nElements * sizeof(TYPE) - 1) / byteAlignment) + 1) * byteAlignment;                                                  \
  ,                                                                                                                                 \
    ret += (((sizeof(TYPE) - 1) / byteAlignment) + 1) * byteAlignment;                                                              \
  )

#define _ACCUMULATE_SOA_ELEMENT(R, DATA, TYPE_NAME)                                                                                 \
  _ACCUMULATE_SOA_ELEMENT_IMPL TYPE_NAME

#define _DECLARE_SOA_CONST_ELEMENT_ACCESSOR_IMPL(IS_COLUMN, TYPE, NAME)                                                             \
  SOA_HOST_DEVICE                                                                                                                   \
  BOOST_PP_IIF(IS_COLUMN,                                                                                                           \
    TYPE const & NAME() { return * (soa_. NAME () + index_); }                                                                      \
  ,                                                                                                                                 \
    TYPE const & NAME() { return soa_. NAME (); }                                                                                   \
  )

#define _DECLARE_SOA_CONST_ELEMENT_ACCESSOR(R, DATA, TYPE_NAME)                                                                     \
  BOOST_PP_EXPAND(_DECLARE_SOA_CONST_ELEMENT_ACCESSOR_IMPL TYPE_NAME)

/* declare AoS-like element value aregs for contructor; these should expand,for columns only */
#define _DECLARE_ELEMENT_VALUE_ARG_IMPL(IS_COLUMN, TYPE, NAME)                                                                      \
  BOOST_PP_IIF(IS_COLUMN,  \
    (TYPE *NAME)\
  ,\
    BOOST_PP_EMPTY()\
  )

#define _DECLARE_ELEMENT_VALUE_ARG(R, DATA, TYPE_NAME)                                                                              \
  BOOST_PP_EXPAND(_DECLARE_ELEMENT_VALUE_ARG_IMPL TYPE_NAME)

/* declare AoS-like element value members; these should expand,for columns only */
/* We filter the value list beforehand to avoid having a comma inside a macro parameter */
#define _DECLARE_ELEMENT_VALUE_COPY_IMPL(IS_COLUMN, TYPE, NAME)                                                                     \
  static_cast<TYPE &>(NAME) = static_cast<std::add_const<TYPE>::type &>(other.NAME);

#define _DECLARE_ELEMENT_VALUE_COPY(R, DATA, TYPE_NAME)                                                                             \
  _DECLARE_ELEMENT_VALUE_COPY_IMPL TYPE_NAME

/* declare AoS-like element value members; these should expand,for columns only */
/* We filter the value list beforehand to avoid having a comma inside a macro parameter */
#define _DECLARE_ELEMENT_VALUE_MEMBER_IMPL(IS_COLUMN, TYPE, NAME)                                                                   \
  cms::cuda::SoAValue<TYPE> NAME;

#define _DECLARE_ELEMENT_VALUE_MEMBER(R, DATA, TYPE_NAME)                                                                           \
  _DECLARE_ELEMENT_VALUE_MEMBER_IMPL TYPE_NAME

/* declare AoS-like element value args for contructor; these should expand,for columns only */
#define _DECLARE_ELEMENT_VALUE_MEMBER_INITIALISATION(R, DATA, TYPE_NAME)                                                            \
  (BOOST_PP_TUPLE_ELEM(2, TYPE_NAME)(DATA, BOOST_PP_TUPLE_ELEM(2, TYPE_NAME)))

/* declare AoS-like element value args for contructor; these should expand,for columns only */
#define _DECLARE_ELEMENT_CONSTR_CALL_IMPL(IS_COLUMN, TYPE, NAME)                                                                    \
  (BOOST_PP_CAT(NAME, _))

#define _DECLARE_ELEMENT_CONSTR_CALL(R, DATA, TYPE_NAME)                                                                            \
  BOOST_PP_EXPAND(_DECLARE_ELEMENT_CONSTR_CALL_IMPL TYPE_NAME)

#define _DECLARE_SOA_ACCESSOR_IMPL(IS_COLUMN, TYPE, NAME)                                                                           \
  SOA_HOST_DEVICE                                                                                                                   \
  BOOST_PP_IIF(IS_COLUMN,                                                                                                           \
    TYPE* NAME() { return BOOST_PP_CAT(NAME, _); }                                                                                  \
  ,                                                                                                                                 \
    TYPE& NAME() { return * BOOST_PP_CAT(NAME, _); }                                                                                \
  )

#define _DECLARE_SOA_ACCESSOR(R, DATA, TYPE_NAME)                                                                                   \
  BOOST_PP_EXPAND(_DECLARE_SOA_ACCESSOR_IMPL TYPE_NAME)

#define _DECLARE_SOA_CONST_ACCESSOR_IMPL(IS_COLUMN, TYPE, NAME)                                                                     \
  SOA_HOST_DEVICE                                                                                                                   \
  BOOST_PP_IIF(IS_COLUMN,                                                                                                           \
    TYPE const* NAME() const { return BOOST_PP_CAT(NAME, _); }                                                                      \
  ,                                                                                                                                 \
    TYPE const& NAME() const { return * BOOST_PP_CAT(NAME, _); }                                                                    \
  )

#define _DECLARE_SOA_CONST_ACCESSOR(R, DATA, TYPE_NAME)                                                                             \
  BOOST_PP_EXPAND(_DECLARE_SOA_CONST_ACCESSOR_IMPL TYPE_NAME)

#define _DECLARE_SOA_DATA_MEMBER_IMPL(IS_COLUMN, TYPE, NAME)                                                                        \
  BOOST_PP_IIF(IS_COLUMN,                                                                                                           \
    TYPE * BOOST_PP_CAT(NAME, _);                                                                                                   \
  ,                                                                                                                                 \
    TYPE * BOOST_PP_CAT(NAME, _);                                                                                                   \
  )

#define _DECLARE_SOA_DATA_MEMBER(R, DATA, TYPE_NAME)                                                                                \
  BOOST_PP_EXPAND(_DECLARE_SOA_DATA_MEMBER_IMPL TYPE_NAME)

/*
 * A macro defining a SoA (structure of variable sized arrays variant).
 */
#define declare_SoA_template(CLASS, ...)                                                                                            \
struct CLASS {                                                                                                                      \
                                                                                                                                    \
  /* these could be moved to an external type trait to free up the symbol names */                                                  \
  using self_type = CLASS;                                                                                                          \
                                                                                                                                    \
  /* For CUDA applications, we align to the 128 bytes of the cache lines.                                                           \
   * See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#global-memory-3-0 this is still valid                     \
   * up to compute capability 8.X.                                                                                                  \
   */                                                                                                                               \
  constexpr static size_t defaultAlignment = 128;                                                                                   \
                                                                                                                                    \
  /* dump the SoA internaul structure */                                                                                            \
  SOA_HOST_ONLY                                                                                                                     \
  static void dump(size_t nElements, size_t byteAlignment = defaultAlignment) {                                                     \
    std::cout << #CLASS "(" << nElements << ", " << byteAlignment << "): " << '\n';                                                 \
    std::cout << "  sizeof(" #CLASS "): " << sizeof(CLASS) << '\n';                                                                 \
    std::cout << "  computeDataSize(...): " << computeDataSize(nElements, byteAlignment);                                           \
    size_t offset=0;                                                                                                                \
    _ITERATE_ON_ALL(_DECLARE_SOA_DUMP_INFO, ~, __VA_ARGS__)                                                                         \
    std::cout << std::endl;                                                                                                         \
  }                                                                                                                                 \
  /* Helper function used by caller to externally allocate the storage */                                                           \
  static size_t computeDataSize(size_t nElements, size_t byteAlignment = defaultAlignment) {                                        \
    size_t ret = 0;                                                                                                                 \
    _ITERATE_ON_ALL(_ACCUMULATE_SOA_ELEMENT, ~, __VA_ARGS__)                                                                        \
    return ret;                                                                                                                     \
  }                                                                                                                                 \
                                                                                                                                    \
  size_t nElements() { return nElements_; }                                                                                         \
  size_t byteAlignment() { return byteAlignment_; }                                                                                 \
                                                                                                                                    \
  /* Constructor relying on user provided storage */                                                                                \
  CLASS(std::byte* mem, size_t nElements, size_t byteAlignment = defaultAlignment):                                                 \
      mem_(mem), nElements_(nElements), byteAlignment_(byteAlignment) {                                                             \
    auto curMem = mem_;                                                                                                             \
    _ITERATE_ON_ALL(_ASSIGN_SOA_COLUMN_OR_SCALAR, ~, __VA_ARGS__)                                                                   \
    /* Sanity check: we should have reached the computed size; */                                                                   \
    if(mem_ + computeDataSize(nElements_, byteAlignment_) != curMem)                                                                \
      throw std::out_of_range("In " #CLASS "::" #CLASS ": unexpected end pointer.");                                                \
  }                                                                                                                                 \
                                                                                                                                    \
  /* AoS-like accessor to individual elements */                                                                                    \
  struct const_element {                                                                                                            \
    SOA_HOST_DEVICE                                                                                                                 \
    const_element(CLASS const& soa, int index) :                                                                                    \
      soa_(soa),                                                                                                                    \
      index_(index)                                                                                                                 \
    { }                                                                                                                             \
                                                                                                                                    \
    _ITERATE_ON_ALL(_DECLARE_SOA_CONST_ELEMENT_ACCESSOR, ~, __VA_ARGS__)                                                            \
                                                                                                                                    \
  private:                                                                                                                          \
    CLASS const& soa_;                                                                                                              \
    const int index_;                                                                                                               \
  };                                                                                                                                \
                                                                                                                                    \
  struct element {                                                                                                                  \
    element(size_t index,                                                                                                           \
      /* Turn Boost PP */                                                                                                           \
      _ITERATE_ON_VALUE_TYPE_COMMA(_DECLARE_ELEMENT_VALUE_ARG, index, _VALUE_TYPE_COLUMN, __VA_ARGS__)                              \
    ):                                                                                                                              \
      _ITERATE_ON_VALUE_TYPE_COMMA(_DECLARE_ELEMENT_VALUE_MEMBER_INITIALISATION, index, _VALUE_TYPE_COLUMN, __VA_ARGS__)            \
       {}                                                                                                                           \
    element& operator=(const element& other) {                                                                                      \
      _ITERATE_ON_VALUE_TYPE(_DECLARE_ELEMENT_VALUE_COPY, ~, _VALUE_TYPE_COLUMN, __VA_ARGS__)                                       \
      return *this;                                                                                                                 \
    }                                                                                                                               \
    _ITERATE_ON_VALUE_TYPE(_DECLARE_ELEMENT_VALUE_MEMBER, ~, _VALUE_TYPE_COLUMN, __VA_ARGS__)                                       \
  };                                                                                                                                \
                                                                                                                                    \
  /* AoS-like accessor */                                                                                                           \
  SOA_HOST_DEVICE                                                                                                                   \
  element operator[](size_t index) { return element(index,                                                                          \
    _ITERATE_ON_VALUE_TYPE_COMMA(_DECLARE_ELEMENT_CONSTR_CALL, ~, _VALUE_TYPE_COLUMN, __VA_ARGS__) );                               \
    }                                                                                                                               \
  const element operator[](size_t index) const { return element(index,                                                              \
    _ITERATE_ON_VALUE_TYPE_COMMA(_DECLARE_ELEMENT_CONSTR_CALL, ~, _VALUE_TYPE_COLUMN, __VA_ARGS__) );                               \
  }                                                                                                                                 \
                                                                                                                                    \
  /* accessors */                                                                                                                   \
  _ITERATE_ON_ALL(_DECLARE_SOA_ACCESSOR, ~, __VA_ARGS__)                                                                            \
  _ITERATE_ON_ALL(_DECLARE_SOA_CONST_ACCESSOR, ~, __VA_ARGS__)                                                                      \
                                                                                                                                    \
  /* dump the SoA internal structure */                                                                                             \
  template <typename T> SOA_HOST_ONLY friend void dump();                                                                           \
                                                                                                                                    \
private:                                                                                                                            \
  /* data members */                                                                                                                \
  std::byte* mem_;                                                                                                                  \
  size_t nElements_;                                                                                                                \
  size_t byteAlignment_;                                                                                                            \
  _ITERATE_ON_ALL(_DECLARE_SOA_DATA_MEMBER, ~, __VA_ARGS__)                                                                         \
}

#endif //CUDADataFormats_Common_SoAmacros_h


