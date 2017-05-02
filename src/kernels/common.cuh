
#pragma once
#ifndef KERNELS_COMMON_CUH_
#define KERNELS_COMMON_CUH_

#include "cuda/on_device/imports.cuh"
#include "cuda/on_device/miscellany.cuh"
#include "cuda/on_device/atomics.cuh"
#include "cuda/on_device/primitives/grid.cuh"
#include "cuda/api/device_function.hpp"
#include "cuda/optional_and_any.hpp"
#include "cuda/bit_operations.cuh"
#include "cuda/api/constants.h"
#include "cuda/api/kernel_launch.cuh"
#include "util/math.hpp"
#include "util/integer.h"


namespace cuda {
	using util::uint_t;

	/**
	 * When we pass the length of an array to a kernel or a device-side
	 * function, and we've specified some kind of an index type for that
	 * array, there are two potential pitfalls we must be aware of:
	 *
	 * 1. The input is the maximum possible size with that index,
	 *    i.e. a length of (1 << sizeof(index_type)) - which doesn't
	 *    fit in an index_type value. To fit it, we need a larger type -
	 *    the next smallest power-of-2-sized unsigned integer type
	 *    typically.
	 *
	 * 2. If we us an overly-large size type - there's the risk of the
	 *    kernel running slower because it needs to manipulate larger
	 *    values all the time.
	 *
	 * I've decided to strike the compromise embodied in the following
	 * code. For data with index sizes 1 or 2 we will promote the
	 * overall size type to have 4 bytes; for larger data - the
	 * promotion will do nothing; and we will be assuming the lengths
	 * we are passed are at most (1 << 32) - 1 (or (1 << 64)-1 for
	 * 8-bytes indices). This choice is not entirely arbitrary:
	 * GPUs like uint32_t counters anyway, so the promotion does't
	 * really hurt us; and the chances of wanting to pass 256 values
	 * or 65,536 values to anything are much higher than to want to
	 * pass 4,294,967,295 (1 << 32 - 1) or 1<<64 - 1.
	 *
	 */
	template <unsigned N>
	using size_type_by_index_size = promoted_size<uint_t<N>>;

	template <typename T>
	using size_type_by_index_type = size_type_by_index_size<sizeof(T)>;

}



// Everything below is (almost certainly) only necessary
// for resolving launch configurations (and is for host
// code only)

#include "kernels/launch_config_resolution_params.h"
#include "cuda/syntax_replacement.h"

#endif /* KERNELS_COMMON_CUH_ */
