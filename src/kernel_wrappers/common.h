
#pragma once
#ifndef KERNEL_WRAPPERS_COMMON_CUH_
#define KERNEL_WRAPPERS_COMMON_CUH_

#include "kernel_wrappers/registered_wrapper.h"
#include "kernels/resolve_launch_configuration.h"
#include "cuda/functors.hpp"         // enough kernels need this to merit always including it
#include "cuda/miscellany.h"
#ifdef __CUDACC__
#include "util/static_block.h"       // for registration in the factory during static init
#endif
#include "util/endianness.h"
#include "util/integer.h"

#include <functional>
#include <utility>

#ifndef __CUDACC__

namespace cuda {

	using util::uint_t;

	template <unsigned N>
	using size_type_by_index_size = cuda::promoted_size<uint_t<N>>;

	template <typename T>
	using size_type_by_index_type = size_type_by_index_size<sizeof(T)>;
}

#endif

/**
 * We sometimes need to reconcile the types util::endianness_t
 * and cuda::endianness_t. We can't just unify them, since the
 * CUDA API stand alone as does the code in util/ ...
 */
inline constexpr cuda::endianness_t translate(util::endianness_t endianness)
{
	return (endianness == util::endianness_t::little_endian) ?
		cuda::endianness_t::little_endian :
		cuda::endianness_t::big_endian;
}

#endif /* KERNEL_WRAPPERS_COMMON_CUH_ */
