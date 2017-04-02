/**
 * CUB library wrappers and instantiators
 *
 * This file-pair is necessary so that non-CUDA C++ code can
 * make CUB calls without having to compile with CUDA (and hence actually
 * include the CUB code itself).
 *
 * Now, we could replicate the entirety of CUB and keep just the function
 * declarations, but - we won't do that, since, well, I don't have the time
 * and I don't need all of it right now. In fact, the headers here will just
 * suit my needs rather than allow invocation of CUB functions as-is.
 *
 * @note I have minimized dependence here on any of my other code - just
 * the standard C++ library and CUDA are necessary for the using C++ code
 * to know about (and the implementation relatively frugal also, although
 * not entirely so since the instantiationscan get specific.
 *
 */
#pragma once
#ifndef CUB_WRAPPERS_H_
#define CUB_WRAPPERS_H_

#include <driver_types.h> // for cudaStream_t
#include <cstddef>        // for size_t

#include "util/integer.h"

namespace cub {
namespace device_partition {

template <typename SelectionOp, unsigned IndexSize>
size_t get_scratch_size(util::uint_t<IndexSize> num_items);

template <typename SelectionOp, unsigned IndexSize>
void partition(
	void*                                     device_scratch_area,
	size_t                                    scratch_size,
	typename SelectionOp::result_type const*  data,
	typename SelectionOp::result_type*        partitioned_data,
	util::uint_t<IndexSize>*                  num_selected,
	util::uint_t<IndexSize>                   num_items,
	cudaStream_t                              stream);

} // namespace device_partition

namespace radix_sort {

template <typename Datum> size_t get_scratch_size(size_t num_items);

template <typename Datum>
void sort(
	void*                         device_scratch_area,
	size_t                        scratch_size,
	Datum*          __restrict__  sorted_data,
	Datum const*    __restrict__  input_data,
	size_t                        num_items,
	cudaStream_t                  stream_id,
	bool                          ascending = true);

} // namespace radix_sort
} // namespace cub

#endif /* CUB_WRAPPERS_H_ */
