#pragma once
#ifndef SRC_KERNELS_ENUMERATED_ELEMENTWISE_UNARY_OPERATION_CUH_
#define SRC_KERNELS_ENUMERATED_ELEMENTWISE_UNARY_OPERATION_CUH_

#include "kernels/common.cuh"
#include "kernels/elementwise/unary_operation.cuh"

namespace cuda {
namespace kernels {
namespace enumerated_elementwise {
namespace unary {

using namespace grid_info::linear;

enum { DefaultSerializationFactor = elementwise::unary::DefaultSerializationFactor };


// TODO: Why can't I use the same name? I get instantiation failure for unknown reasons :-(

/*
 * Things to consider:
 * - Quantized version
 * - Use of unsigned for length
 * - Determine number of iterations in advance (and don't check it at each iteration)
 * - Implement a version in which all pointers have the same base and you just get offsets
 * - Maybe drop the serialization factor altogether?
 * - An LHS-in-place and an RHS-in-place version
 */
template<typename UnaryOp, serialization_factor_t SerializationFactor = DefaultSerializationFactor>
__global__ void unary_operation(
	typename UnaryOp::result_type*          __restrict__  result,
	const typename UnaryOp::argument_type*  __restrict__  input,
	typename UnaryOp::enumerator_type                     length)
{
	auto f = [&result, &input](typename UnaryOp::enumerator_type pos) {
		UnaryOp op;
		result[pos] = op(pos, input[pos]);
	};
	primitives::grid::linear::at_block_stride(length, f, SerializationFactor);
}

template<typename UnaryOp, serialization_factor_t SerializationFactor = DefaultSerializationFactor>
class launch_config_resolution_params_t final : public cuda::launch_config_resolution_params_t {
public:
	using parent = cuda::launch_config_resolution_params_t;
public:
	launch_config_resolution_params_t(
		device::properties_t            device_properties_,
		size_t                          length_) :
		cuda::launch_config_resolution_params_t(
			device_properties_,
			device_function_t(unary_operation<UnaryOp, SerializationFactor>)
		)
	{
		grid_construction_resolution            = thread;
		length                                  = length_;
		serialization_option                    = fixed_factor;
		default_serialization_factor            = DefaultSerializationFactor;
	};
};

} // namespace unary
} // namespace enumerated_elementwise
} // namespace kernels
} // namespace cuda

#endif /* SRC_KERNELS_ENUMERATED_ELEMENTWISE_UNARY_OPERATION_CUH_ */
