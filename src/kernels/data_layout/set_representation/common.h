#pragma once
#ifndef SRC_KERNELS_SET_REPRESENTATION_COMMON_H_
#define SRC_KERNELS_SET_REPRESENTATION_COMMON_H_

#include "cuda/bit_vector.cuh"

namespace cuda {
namespace kernels {
namespace set_representation {

using cuda::bit_vector;

enum class sortedness_t : bool {
	Unsorted = false,
	Sorted   = true,
};

} // namespace set_representation
} // namespace kernels
} // namespace cuda

#endif /* SRC_KERNELS_SET_REPRESENTATION_COMMON_H_ */
