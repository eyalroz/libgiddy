#pragma once
#ifndef SRC_KERNELS_ELEMENTWISE_UNARY_OPERATION_CUH_
#define SRC_KERNELS_ELEMENTWISE_UNARY_OPERATION_CUH_

#include "kernels/common.cuh"

namespace cuda {
namespace kernels {
namespace elementwise {
namespace unary {

using namespace grid_info::linear;

enum { DefaultSerializationFactor = 16 };


// TODO: Why can't I use the same name? I get instantiation failure for unknown reasons :-(

/*
 * Things to consider:
 * - Quantized version
 * - Use of unsigned for length
 * - Determine number of iterations in advance (and don't check it at each iteration)
 * - Implement a version in which all pointers have the same base and you just get offsets
 * - Maybe drop the serialization factor altogether?
 * - An LHS-in-place and an RHS-in-place version
 * - Supporting a limited-size version
 */
template<unsigned IndexSize, typename UnaryOp, serialization_factor_t SerializationFactor = DefaultSerializationFactor>
__global__ void unary_operation(
	typename UnaryOp::result_type*          __restrict__  output,
	const typename UnaryOp::argument_type*  __restrict__  input,
	size_type_by_index_size<IndexSize>                    length)
{
	auto f = [&output, &input](decltype(length) pos) {
		UnaryOp op;
		output[pos] = op(input[pos]);
	};
	primitives::grid::linear::at_block_stride(length, f, SerializationFactor);
}

template<unsigned IndexSize, typename Op, serialization_factor_t SerializationFactor = DefaultSerializationFactor>
class launch_config_resolution_params_t final : public kernels::launch_config_resolution_params_t {
public:
	using parent = kernels::launch_config_resolution_params_t;
public:
	launch_config_resolution_params_t(
		device::properties_t            device_properties_,
		size_t                          length_) :
		parent(
			device_properties_,
			device_function_t(unary_operation<IndexSize, Op, SerializationFactor>)
		)
	{
		grid_construction_resolution            = thread;
		length                                  = length_;
		serialization_option                    = fixed_factor;
		default_serialization_factor            = DefaultSerializationFactor;
	};
};

} // namespace unary
} // namespace elementwise
} // namespace kernels
} // namespace cuda

#endif /* SRC_KERNELS_ELEMENTWISE_UNARY_OPERATION_CUH_ */
