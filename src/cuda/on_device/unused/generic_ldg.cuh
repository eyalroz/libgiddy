/*
 * Generic LDG variants
 *
 * Part of Bryan Catanzaro's CUDA generics
 * https://github.com/bryancatanzaro/generics/
 *
 * TODO: This must be reimplemented, similarly to generic_shuffle
 *
 */

#ifndef SRC_CUDA_ON_DEVICE_GENERIC_LDG_CUH_
#define SRC_CUDA_ON_DEVICE_GENERIC_LDG_CUH_

#pragma once
#ifdef __shfl_up
#undef __shfl_up
#undef __shfl_down
#undef __shfl_xor
#endif
#define IMPLEMENTING_LDG_OVERRIDES
#include "cuda/syntax_replacement.h"
#include "cuda/detail/alias.cuh"

namespace detail {

template<typename T,
         typename U=typename working_type<T>::type,
         int r = aliased_size<T, U>::value>
struct load_storage {
    typedef array<U, r> result_type;
    static const int idx = aliased_size<T, U>::value - r;
    __device__ __forceinline__
    static result_type impl(const T* ptr) {
        return result_type(__ldg(((const U*)ptr) + idx),
                           load_storage<T, U, r-1>::impl(ptr));
    }
};

template<typename T, typename U>
struct load_storage<T, U, 1> {
    typedef array<U, 1> result_type;
    static const int idx = aliased_size<T, U>::value - 1;
    __device__ __forceinline__
    static result_type impl(const T* ptr) {
        return result_type(__ldg(((const U*)ptr) + idx));
    }
};

}


#if __CUDA_ARCH__ >= 350
// Device has ldg
template<typename T>
__device__ __forceinline__ T __ldg(const T* ptr) {
    typedef typename detail::working_array<T>::type aliased;
    aliased storage = detail::load_storage<T>::impl(ptr);
    return detail::fuse<T>(storage);
}

#else
//Device does not, fall back.
template<typename T>
__device__ __forceinline__ T __ldg(const T* ptr) {
    return *ptr;
}

#endif




#endif /* SRC_CUDA_ON_DEVICE_GENERIC_LDG_CUH_ */
