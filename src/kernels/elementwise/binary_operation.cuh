
#include "kernels/common.cuh"
#include "cuda/functors.hpp"
#include "util/macro.h"

namespace cuda {
namespace kernels {
namespace elementwise {
namespace binary {

using namespace grid_info::linear;

enum : serialization_factor_t { DefaultSerializationFactor = 16 };

// TODO: Why can't I use the same name? I get instantiation failure for unknown reasons :-(

/*
 * Things to consider:
 * - Quantized version
 * - Determine number of iterations in advance (and don't check it at each iteration)
 * - Implement a version in which all pointers have the same base and you just get offsets
 * - Maybe drop the serialization factor altogether?
 * - An LHS-in-place and an RHS-in-place version
 */
template<unsigned IndexSize, typename BinaryOp, serialization_factor_t SerializationFactor = DefaultSerializationFactor>
__global__ void performBinaryOperationAA(
	typename       BinaryOp::result_type*           __restrict__  result,
	const typename BinaryOp::first_argument_type*   __restrict__  lhs,
	const typename BinaryOp::second_argument_type*  __restrict__  rhs,
	uint_t<IndexSize>                                             length)
{
	using index_type = uint_t<IndexSize>;
	// NOTE: Only supporting arrays of at most 4GB elements here. otherwise make it size_t or ptrdiff_t or what-not
	auto f = [&](index_type pos) {
		BinaryOp op;
		result[pos] = op(lhs[pos], rhs[pos]);
	};
	primitives::grid::linear::at_block_stride(length, f, SerializationFactor);
}

template<unsigned IndexSize, typename BinaryOp, serialization_factor_t SerializationFactor = DefaultSerializationFactor>
__global__ void performBinaryOperationSA(
	typename       BinaryOp::result_type*           __restrict__  result,
	const typename BinaryOp::first_argument_type                  lhs,
	const typename BinaryOp::second_argument_type*  __restrict__  rhs,
	uint_t<IndexSize>                                             length)
{
	using index_type = uint_t<IndexSize>;
	auto f = [&](index_type pos) {
		BinaryOp op;
		result[pos] = op(lhs, rhs[pos]);
	};
	primitives::grid::linear::at_block_stride(length, f, SerializationFactor);
}

template<unsigned IndexSize, typename BinaryOp, serialization_factor_t SerializationFactor = DefaultSerializationFactor>
__global__ void performBinaryOperationAS(
	typename       BinaryOp::result_type*           __restrict__  result,
	const typename BinaryOp::first_argument_type*   __restrict__  lhs,
	const typename BinaryOp::second_argument_type                 rhs,
	uint_t<IndexSize>                                             length)
{
	using index_type = uint_t<IndexSize>;
	auto f = [&](index_type pos) {
		BinaryOp op;
		result[pos] = op(lhs[pos], rhs);
	};
	primitives::grid::linear::at_block_stride(length, f, SerializationFactor);
}

template<unsigned IndexSize, typename BinaryOp, serialization_factor_t SerializationFactor = DefaultSerializationFactor>
class launch_config_resolution_params_t final : public cuda::launch_config_resolution_params_t {
public:
	using parent = cuda::launch_config_resolution_params_t;
public:
	launch_config_resolution_params_t(
		device::properties_t            device_properties_,
		size_t                          length_) :
		cuda::launch_config_resolution_params_t(
			device_properties_,
			// Yeah, it's a hack to just use one of the three functions. But -
			// there shouldn't really be three functions, we should just template
			// on the iterator
			device_function_t(performBinaryOperationAA<IndexSize, BinaryOp, SerializationFactor>)
		)
	{
		grid_construction_resolution            = thread;
		length                                  = length_;
		serialization_option                    = fixed_factor;
		default_serialization_factor            = DefaultSerializationFactor;
	};
};


} // namespace binary
} // namespace elementwise
} // namespace kernels
} // namespace cuda
