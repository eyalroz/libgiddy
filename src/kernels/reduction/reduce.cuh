#pragma once
#ifndef SRC_KERNELS_REDUCTION_REDUCE_CUH_
#define SRC_KERNELS_REDUCTION_REDUCE_CUH_

#include "kernels/common.cuh"
#include "cuda/functors.hpp"
#include "cuda/on_device/primitives/warp.cuh"
#include "cuda/on_device/primitives/block.cuh"
#include "cuda/on_device/primitives/grid.cuh"
#include "kernels/reduction/common.cuh"

#include <limits>

namespace cuda {
namespace kernels {
namespace reduction {
namespace reduce {

namespace detail {

namespace thread_to_thread {

/**
 * In this phase of a reduction operation, there is no inter-thread,
 * inter-warp or inter-block communication of any kind, and each thread
 * performs a reduction on the input element it reads (after applying
 * a pretransformation to them).
 *
 * @param result
 * @param data
 * @param length
 * @return
 */
template<unsigned IndexSize, typename ReductionOp, typename InputDatum, typename PretransformOp>
__forceinline__ __device__ typename ReductionOp::result_type reduce(
	const InputDatum*   __restrict__  data,
	uint_t<IndexSize>                 length)
{
	using index_type = uint_t<IndexSize>;
	using result_type = typename ReductionOp::result_type;
	using pretransform_type = typename PretransformOp::result_type;
	static_assert(
		std::is_same<typename ReductionOp::first_argument_type, result_type>::value &&
		std::is_same<typename ReductionOp::second_argument_type, result_type>::value,
		"Only all-same-argument reduction operations are supported at this time.");
		// ... but note these don't have to be the same as InputDatum
	static_assert(std::is_same<
		typename PretransformOp::argument_type, InputDatum>::value,
		"The pretransform op must apply to input of the type specified for the input.");
	using namespace grid_info::linear;

	// single threads reduce independently

	ReductionOp reduction_op;
	result_type thread_result = reduction_op.neutral_value();
	primitives::grid::linear::at_grid_stride(length,
		[&thread_result, data](index_type pos) {
			typename ReductionOp::accumulator accumulation_op;
			auto pretransformed_datum =
				reduction::detail::apply_unary_op<index_type, PretransformOp>(pos, data[pos]);
			accumulation_op(thread_result, pretransformed_datum);
		}
	);
	return thread_result;
}

} // namespace thread_to_thread

} // namespace detail

// TODO: templatize the size/offest as well
// TODO: Specialize for ResutlDatum > InputDatum,
// TODO: Consider overflow/carry for integer sums with ResutlDatum = InputDatum
// TODO: Support type switching at the warp level, block level, and result

template<
	unsigned IndexSize, typename ReductionOp,
	typename InputDatum, typename PretransformOp
	>
__global__ void reduce(
	typename ReductionOp::result_type*  __restrict__  result,
	const InputDatum*                   __restrict__  data,
	uint_t<IndexSize>                                 length)
{
	auto thread_result = detail::thread_to_thread
		::reduce<IndexSize, ReductionOp, InputDatum, PretransformOp>(data, length);
	__syncthreads();
	auto block_result =
		primitives::block::reduce<ReductionOp>(thread_result);
	primitives::block_to_grid
		::accumulation_to_scalar<ReductionOp>(result, block_result);
}

template<
	unsigned IndexSize, typename ReductionOp,
	typename InputDatum, typename PretransformOp
	>
class launch_config_resolution_params_t final : public cuda::launch_config_resolution_params_t {
public:
	using parent = cuda::launch_config_resolution_params_t;
public:
	launch_config_resolution_params_t(
		device::properties_t            device_properties_,
		size_t                          length_,
		optional<shared_memory_size_t>  dynamic_shared_mem_limit = nullopt) :
		cuda::launch_config_resolution_params_t(
			device_properties_,
			device_function_t(reduce<IndexSize, ReductionOp, InputDatum, PretransformOp>),
			dynamic_shared_mem_limit
		)
	{
		auto shared_mem_constraint_on_warps_in_block =
			available_dynamic_shared_memory_per_block /
			(sizeof(typename ReductionOp::result_type) * warp_size);
		auto shared_mem_constraint_on_threads_in_block =
			warp_size * shared_mem_constraint_on_warps_in_block;

		grid_construction_resolution            = block;
		block_resolution_constraints.fixed_threads_per_block
		                                        = std::min(
		                                        	(size_t) device_properties.maxThreadsDim[0],
		                                        	(size_t) shared_mem_constraint_on_threads_in_block);
		length                                  =
			util::div_rounding_up(length_, shared_mem_constraint_on_threads_in_block);
		serialization_option                    = auto_maximized;
		dynamic_shared_memory_requirement.per_block =
			util::round_down(available_dynamic_shared_memory_per_block,
				sizeof(typename ReductionOp::result_type) * warp_size);
		quanta.threads_in_block                 = warp_size;
		keep_gpu_busy_factor                    = 16;
	};
};


} // namespace reduce
} // namespace reduction
} // namespace kernels
} // namespace cuda

#endif /* define SRC_KERNELS_REDUCTION_REDUCE_CUH_ */
