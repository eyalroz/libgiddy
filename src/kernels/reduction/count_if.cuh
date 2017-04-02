#pragma once
#ifndef SRC_KERNELS_REDUCTION_COUNT_IF_CUIH
#define SRC_KERNELS_REDUCTION_COUNT_IF_CUIH


#include "kernels/common.cuh"
#include "reduce.cuh"

namespace cuda {
namespace kernels {
namespace reduction {
namespace count_if {

/*
 *  TODO: This has some code duplication with the reduce() kernel;
 *  in fact, if we have an apply-then-reduce with a unary function, we
 *  could implement this kernel using that one, with our unary function
 *  being a predicate guaranteed to produce zero or one, and the binary
 *  function being plus.
 */

template<unsigned IndexSize, typename UnaryPredicate>
__global__ void count_if(
	uint_t<IndexSize>*                             __restrict__  result,
	const typename UnaryPredicate::argument_type*  __restrict__  data,
	uint_t<IndexSize>                                            length)
{
	using index_type = uint_t<IndexSize>;
	using element_type = typename UnaryPredicate::argument_type;
	using namespace grid_info::linear;
	using reductipn_op = cuda::functors::plus<index_type>;

	// single threads reduce independently

	// TODO: Perhaps act on multiple elements at a time, in each thread, for the case of sizeof(element_type) < 4 ?

	uint_t<IndexSize> thread_result = 0;
	primitives::grid::linear::at_grid_stride(length,
		[&thread_result, data](index_type pos) {
			UnaryPredicate predicate;
			// Cross your fingers and pray that the compiler optimizes this to not have a conditional jmp
			thread_result += predicate(data[pos]) ? 1 : 0;
		}
	);
	__syncthreads();

	uint_t<IndexSize> block_result =
		primitives::block::reduce<reductipn_op, index_type>(thread_result);

	if (thread::is_first_in_block()) {
		typename reductipn_op::accumulator::atomic atomic_accumulation_op;
		atomic_accumulation_op(*result, block_result); // e.g. *result += block_result
	}
}

template<unsigned IndexSize, typename UnaryPredicate>
class launch_config_resolution_params_t final : public cuda::launch_config_resolution_params_t {
public:
	using parent = cuda::launch_config_resolution_params_t;
public:
	launch_config_resolution_params_t(
		device::properties_t            device_properties_,
		size_t                          length_)
		:
		cuda::launch_config_resolution_params_t(
			device_properties_,
			device_function_t(count_if<IndexSize, UnaryPredicate>)
		)
	{
		grid_construction_resolution            = block;
		length                                  = length_;
		serialization_option                    = none;
		dynamic_shared_memory_requirement.per_block
		                                        = IndexSize;
		quanta.threads_in_block                 = warp_size;
//		block_resolution_constraints.max_threads_per_block
//		                                        = 1;
	};
};


} // namespace count_if
} // namespace reduction
} // namespace kernels
} // namespace cuda

#endif /* SRC_KERNELS_REDUCTION_COUNT_IF_CUIH */
