/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_FETCH_OP_CUDA_HPP_
#define DESUL_ATOMICS_FETCH_OP_CUDA_HPP_

#ifndef DESUL_CUDA_ARCH_IS_PRE_VOLTA

#define DESUL_HAVE_CUDA_ATOMICS_ASM

#include <desul/atomics/cuda/CUDA_asm.hpp>
#include <type_traits>

namespace desul {
namespace Impl {
// unsigned long long on Volta and later. The inline-asm natives above (red.add / atom.add and the
// min, max, sub forms) are instantiated for uint64_t, which on LP64 platforms is unsigned long, a
// type distinct from unsigned long long, CUDA's own 64-bit atomic type and what user code usually
// holds. A device_atomic_add on an unsigned long long then falls to the generic template, whose
// inner device_atomic_fetch_add call is bound at that template's definition and never sees a native
// overload, so every such add became a compare-and-swap loop (measured 78x slower than red.add on a
// contended counter). These forward to the uint64_t natives where the two types are distinct and
// vanish where they are the same type.
#define DESUL_IMPL_CUDA_DEVICE_ATOMIC_OP_ULL(OP)                                                   \
  template <class MemoryOrder, class MemoryScope, class T = unsigned long long,                    \
            std::enable_if_t<!std::is_same<T, uint64_t>::value && sizeof(T) == sizeof(uint64_t),    \
                             int> = 0>                                                             \
  __device__ void device_atomic_##OP(unsigned long long* ptr, unsigned long long val,              \
                                     MemoryOrder order, MemoryScope scope) {                       \
    device_atomic_##OP(reinterpret_cast<uint64_t*>(ptr), static_cast<uint64_t>(val), order,        \
                       scope);                                                                     \
  }                                                                                                \
  template <class MemoryOrder, class MemoryScope, class T = unsigned long long,                    \
            std::enable_if_t<!std::is_same<T, uint64_t>::value && sizeof(T) == sizeof(uint64_t),    \
                             int> = 0>                                                             \
  __device__ unsigned long long device_atomic_fetch_##OP(                                          \
      unsigned long long* ptr, unsigned long long val, MemoryOrder order, MemoryScope scope) {     \
    return static_cast<unsigned long long>(device_atomic_fetch_##OP(                               \
        reinterpret_cast<uint64_t*>(ptr), static_cast<uint64_t>(val), order, scope));             \
  }
DESUL_IMPL_CUDA_DEVICE_ATOMIC_OP_ULL(add)
DESUL_IMPL_CUDA_DEVICE_ATOMIC_OP_ULL(sub)
DESUL_IMPL_CUDA_DEVICE_ATOMIC_OP_ULL(min)
DESUL_IMPL_CUDA_DEVICE_ATOMIC_OP_ULL(max)
#undef DESUL_IMPL_CUDA_DEVICE_ATOMIC_OP_ULL
}  // namespace Impl
}  // namespace desul


#else

namespace desul {
namespace Impl {

// clang-format off
inline __device__                int device_atomic_fetch_add(               int* ptr,                int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr,  val); }
inline __device__       unsigned int device_atomic_fetch_add(      unsigned int* ptr,       unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr,  val); }
inline __device__ unsigned long long device_atomic_fetch_add(unsigned long long* ptr, unsigned long long val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr,  val); }
inline __device__              float device_atomic_fetch_add(             float* ptr,              float val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr,  val); }
#ifndef DESUL_CUDA_ARCH_IS_PRE_PASCAL
inline __device__             double device_atomic_fetch_add(            double* ptr,             double val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr,  val); }
#endif

inline __device__                int device_atomic_fetch_sub(               int* ptr,                int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicSub(ptr,  val); }
inline __device__       unsigned int device_atomic_fetch_sub(      unsigned int* ptr,       unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicSub(ptr,  val); }
inline __device__ unsigned long long device_atomic_fetch_sub(unsigned long long* ptr, unsigned long long val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr, -val); }
inline __device__              float device_atomic_fetch_sub(             float* ptr,              float val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr, -val); }
#ifndef DESUL_CUDA_ARCH_IS_PRE_PASCAL
inline __device__             double device_atomic_fetch_sub(            double* ptr,             double val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr, -val); }
#endif

inline __device__                int device_atomic_fetch_min(               int* ptr,                int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicMin(ptr,  val); }
inline __device__       unsigned int device_atomic_fetch_min(      unsigned int* ptr,       unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicMin(ptr,  val); }
inline __device__ unsigned long long device_atomic_fetch_min(unsigned long long* ptr, unsigned long long val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicMin(ptr,  val); }

inline __device__                int device_atomic_fetch_max(               int* ptr,                int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicMax(ptr,  val); }
inline __device__       unsigned int device_atomic_fetch_max(      unsigned int* ptr,       unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicMax(ptr,  val); }
inline __device__ unsigned long long device_atomic_fetch_max(unsigned long long* ptr, unsigned long long val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicMax(ptr,  val); }

inline __device__                int device_atomic_fetch_and(               int* ptr,                int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAnd(ptr,  val); }
inline __device__       unsigned int device_atomic_fetch_and(      unsigned int* ptr,       unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAnd(ptr,  val); }
inline __device__ unsigned long long device_atomic_fetch_and(unsigned long long* ptr, unsigned long long val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAnd(ptr,  val); }

inline __device__                int device_atomic_fetch_or (               int* ptr,                int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicOr (ptr,  val); }
inline __device__       unsigned int device_atomic_fetch_or (      unsigned int* ptr,       unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicOr (ptr,  val); }
inline __device__ unsigned long long device_atomic_fetch_or (unsigned long long* ptr, unsigned long long val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicOr (ptr,  val); }

inline __device__                int device_atomic_fetch_xor(               int* ptr,                int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicXor(ptr,  val); }
inline __device__       unsigned int device_atomic_fetch_xor(      unsigned int* ptr,       unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicXor(ptr,  val); }
inline __device__ unsigned long long device_atomic_fetch_xor(unsigned long long* ptr, unsigned long long val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicXor(ptr,  val); }

inline __device__                int device_atomic_fetch_inc(               int* ptr,                         MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr, 1   ); }
inline __device__       unsigned int device_atomic_fetch_inc(      unsigned int* ptr,                         MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr, 1u  ); }
inline __device__ unsigned long long device_atomic_fetch_inc(unsigned long long* ptr,                         MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr, 1ull); }

inline __device__                int device_atomic_fetch_dec(               int* ptr,                         MemoryOrderRelaxed, MemoryScopeDevice) { return atomicSub(ptr,  1  ); }
inline __device__       unsigned int device_atomic_fetch_dec(      unsigned int* ptr,                         MemoryOrderRelaxed, MemoryScopeDevice) { return atomicSub(ptr,  1u ); }
inline __device__ unsigned long long device_atomic_fetch_dec(unsigned long long* ptr,                         MemoryOrderRelaxed, MemoryScopeDevice) { return atomicAdd(ptr, -1ull);}

inline __device__       unsigned int device_atomic_fetch_inc_mod(  unsigned int* ptr,       unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicInc(ptr,  val); }
inline __device__       unsigned int device_atomic_fetch_dec_mod(  unsigned int* ptr,       unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) { return atomicDec(ptr,  val); }
// clang-format on

#define DESUL_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP(FETCH_OP, TYPE)                         \
  template <class MemoryOrder>                                                         \
  __device__ TYPE device_atomic_##FETCH_OP(                                            \
      TYPE* ptr, TYPE val, MemoryOrder, MemoryScopeDevice) {                           \
    __threadfence();                                                                   \
    TYPE return_val =                                                                  \
        device_atomic_##FETCH_OP(ptr, val, MemoryOrderRelaxed(), MemoryScopeDevice()); \
    __threadfence();                                                                   \
    return return_val;                                                                 \
  }                                                                                    \
  template <class MemoryOrder>                                                         \
  __device__ TYPE device_atomic_##FETCH_OP(                                            \
      TYPE* ptr, TYPE val, MemoryOrder, MemoryScopeCore) {                             \
    return device_atomic_##FETCH_OP(ptr, val, MemoryOrder(), MemoryScopeDevice());     \
  }

#define DESUL_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP_INTEGRAL(FETCH_OP) \
  DESUL_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP(FETCH_OP, int)           \
  DESUL_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP(FETCH_OP, unsigned int)  \
  DESUL_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP(FETCH_OP, unsigned long long)

#ifdef DESUL_CUDA_ARCH_IS_PRE_PASCAL

#define DESUL_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP_FLOATING_POINT(FETCH_OP) \
  DESUL_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP(FETCH_OP, float)

#else

#define DESUL_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP_FLOATING_POINT(FETCH_OP) \
  DESUL_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP(FETCH_OP, float)               \
  DESUL_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP(FETCH_OP, double)

#endif

DESUL_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP_INTEGRAL(fetch_min)
DESUL_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP_INTEGRAL(fetch_max)
DESUL_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP_INTEGRAL(fetch_and)
DESUL_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP_INTEGRAL(fetch_or)
DESUL_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP_INTEGRAL(fetch_xor)

DESUL_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP_FLOATING_POINT(fetch_add)
DESUL_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP_INTEGRAL(fetch_add)
DESUL_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP_FLOATING_POINT(fetch_sub)
DESUL_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP_INTEGRAL(fetch_sub)

DESUL_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP_INTEGRAL(fetch_inc)
DESUL_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP_INTEGRAL(fetch_dec)

DESUL_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP(fetch_inc_mod, unsigned int)
DESUL_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP(fetch_dec_mod, unsigned int)

#undef DESUL_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP_FLOATING_POINT
#undef DESUL_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP_INTEGRAL
#undef DESUL_IMPL_CUDA_DEVICE_ATOMIC_FETCH_OP



}  // namespace Impl
}  // namespace desul

#endif

#endif
