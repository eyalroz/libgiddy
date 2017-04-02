#pragma once
#ifndef SRC_KERNELS_ENUMERATED_ELEMENTWISE_TERNARY_OPERATION_CUH_
#define SRC_KERNELS_ENUMERATED_ELEMENTWISE_TERNARY_OPERATION_CUH_


#include "kernels/common.cuh"

namespace cuda {

/**
 * A workaround for a strange NVCC bug; despite serialization_factor_t from "cuda/api/types.h"
 * being visible, I still get a compilation error with the NVCC of CUDA 8.0 RC.
 */
using local_serialization_factor_t = unsigned short;

namespace kernels {
namespace elementwise {
namespace ternary {

using namespace grid_info::linear;


enum : local_serialization_factor_t { DefaultSerializationFactor = 16 };

/*
 * Things to consider:
 * - Quantized version
 * - Use of unsigned for length
 * - Implement a version in which all pointers have the same base and you just get offsets
 * - A tuple or a varargs version of this kernel - instead of unary, binary and ternary?
 */
template<
	unsigned IndexSize, typename TernaryOp, serialization_factor_t SerializationFactor = DefaultSerializationFactor>
__global__ void ternary_operation(
	typename TernaryOp::result_type*                 __restrict__  result,
	const typename TernaryOp::first_argument_type*   __restrict__  first_argument,
	const typename TernaryOp::second_argument_type*  __restrict__  second_argument,
	const typename TernaryOp::third_argument_type*   __restrict__  third_argument,
	uint_t<IndexSize>                                              length)
{
	using index_type = uint_t<IndexSize>;
	auto f = [&](index_type pos) {
		TernaryOp op;
		result[pos] = op(first_argument[pos], second_argument[pos], third_argument[pos]);
	};
	primitives::grid::linear::at_block_stride(length, f, SerializationFactor);
}


template<unsigned IndexSize, typename TernaryOp, serialization_factor_t SerializationFactor = DefaultSerializationFactor>
class launch_config_resolution_params_t final : public cuda::launch_config_resolution_params_t {
public:
	using parent = cuda::launch_config_resolution_params_t;
public:
	launch_config_resolution_params_t(
		device::properties_t            device_properties_,
		size_t                          length_) :
		cuda::launch_config_resolution_params_t(
			device_properties_,
			device_function_t(ternary_operation<IndexSize, TernaryOp, SerializationFactor>)
		)
	{
		grid_construction_resolution            = thread;
		length                                  = length_;
		serialization_option                    = fixed_factor;
		default_serialization_factor            = DefaultSerializationFactor;
	};
};
} // namespace ternary
} // namespace elementwise
} // namespace kernels
} // namespace cuda

#endif /* SRC_KERNELS_ENUMERATED_ELEMENTWISE_TERNARY_OPERATION_CUH_ */
