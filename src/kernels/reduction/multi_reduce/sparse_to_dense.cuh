/**
 * Multi-reduce is the task of performing multiple reductions with the same kernel,
 * or sequence of kernels. That is, one has data intended for different 'bins', or
 * groups if you will, and the result of the multi-reduce is the reduction results
 * for all bins (or all bins with any data).
 *
 * This file contains kernels for the SPARSE TO DENSE variant:
 *
 * SPARSE input: consists of pairs of: a value to participate in the reduction,
 * and the bin with which this value is associated. Of course, we prefer a SoA
 * vs AoS layout, so technically the input is two corresponding arrays of equal
 * length - data and indices.
 *
 * DENSE output: This is a single vector of bins, each of which holds
 * the result of a single reduction. The 'reduction index' or 'group index'
 * of each element is implicit - its index in the vector.
 *
 * Another technical caveat is that these kernels do not actually initialize the
 * data, only continue reducing with it, so that it is not their responsibility,
 * for example, to ensure untouched bin reduction results are actually neutral
 * to begin with. For the case of summation, input data will only be added to
 * existing bin results, and it should be a good idea to zero them out in advance.
 *
 * Applying the SPARSE TO DENSE variant is roughly equivalent to running the
 * SQL query:
 *
 *   SELECT SUM(data) FROM some_table
 *   GROUP BY contiguous_numeric_category
 *   ORDER BY contiguous_numeric_category ASC;
 *
 * and you might also hear it referred to as "reduce-by-index", "reduce-by-key"
 * or "group-indexed reduction".
 */


#include "kernels/common.cuh"
#include "common.cuh" // ... common to multi-reduce operations

namespace cuda {

using namespace grid_info::linear;

namespace kernels {
namespace reduction {
namespace multi_reduce {
namespace dynamic_num_reductions {
namespace sparse_to_dense {

enum { DefaultSerializationFactor = 32 };

// A variant which dynamically decides how many histograms per block it can use -
// not even one (and cowardly opts for global atomics), one per block,
// or more - up to one per thread. We assume the more histograms we can use, the
// better, since we reduce the contention for atomics. ... but this assumption
// has not yet been tested to bear out

template <unsigned IndexSize, typename ReductionOp, typename InputDatum, typename ReductionIndex>
__global__ void sparse_to_dense(
	typename ReductionOp::result_type*  __restrict__  target,
	const InputDatum*                   __restrict__  data,
	const ReductionIndex*               __restrict__  indices,
	uint_t<IndexSize>                                 data_length,
	ReductionIndex                                    num_reductions,
	serialization_factor_t                            serialization_factor)
{
	using index_type = uint_t<IndexSize>;
	using result_type = typename ReductionOp::result_type;

	auto single_result_array_size = num_reductions * sizeof(result_type);
	auto available_shared_mem = ptx::special_registers::dynamic_smem_size();
	if (single_result_array_size > available_shared_mem) {
		// Run the variant which uses no shared memory. Although we could
		// try to be smart and use the shared memory we do have - either
		// for the more common values somehow or just for the first values.
		// Also, if we had some global memory scratch space, enough for a
		// full histogram per block, we could just write in there non-atomically,
		// utilize L1/L2 caching, and perform a small dense-to-dense reduction
		// in the end
		return dynamic_num_reductions::detail::thread_to_global::sparse_to_dense<
			IndexSize, ReductionOp, InputDatum, ReductionIndex,
			dynamic_num_reductions::detail::thread_does_reduce_running_sequences_of_same_value,
			dynamic_num_reductions::detail::do_skip_accumulation_of_neutral_values>(
			target, data, indices, data_length, num_reductions, serialization_factor);
	}
	auto num_result_arrays_which_fit_in_shared_mem = available_shared_mem / single_result_array_size;
	// TODO: Space out the arrays so they don't have aligned banks

	auto block_results = shared_memory_proxy<result_type>();

	// TODO: Is it at all worth it to use thread-specific or warp-specific results?
	// Need to test that
	if (num_result_arrays_which_fit_in_shared_mem >= block::num_warps()) {

		auto warp_results = shared_memory_proxy<result_type>() + warp::index_in_block() * num_reductions;

		if (num_result_arrays_which_fit_in_shared_mem >= block::length() + block::num_warps()) {

			primitives::block::fill_n(
				shared_memory_proxy<result_type>(),
				(block::length() + block::num_warps()) * num_reductions, ReductionOp().neutral_value()
			);
			__syncthreads();

			auto thread_results = shared_memory_proxy<result_type>() +
				(block::num_warps() + thread::index_in_block()) * num_reductions;
			dynamic_num_reductions::detail::thread_to_thread::sparse_to_dense<
				IndexSize, ReductionOp, InputDatum, ReductionIndex
			>(thread_results, data, indices, data_length, num_reductions, serialization_factor);
			__syncthreads();

			dynamic_num_reductions::detail::thread_to_warp::dense_to_dense<
				IndexSize, ReductionOp, ReductionIndex
			>(warp_results, thread_results, num_reductions);
		}
		else {
			primitives::block::fill_n(
				shared_memory_proxy<result_type>(),
				block::num_warps() * num_reductions, ReductionOp().neutral_value()
				);
			__syncthreads();
			dynamic_num_reductions::detail::thread_to_warp::sparse_to_dense<
				IndexSize, ReductionOp, InputDatum, ReductionIndex
			>(warp_results, data, indices, data_length, num_reductions, serialization_factor);
		}
		__syncthreads();
//		block_print("Consolidating warp histograms into block histogram");
		dynamic_num_reductions::detail::warp_to_block::dense_to_dense_in_place<
			IndexSize, ReductionOp, ReductionIndex
		>(block_results, warp_results, num_reductions);
	}
	else {
		primitives::block::fill_n(
			shared_memory_proxy<result_type>(), num_reductions, ReductionOp().neutral_value()
		);
		__syncthreads();
		dynamic_num_reductions::detail::thread_to_block::sparse_to_dense<
			IndexSize, ReductionOp, InputDatum, ReductionIndex
		>(block_results, data, indices, data_length, num_reductions, serialization_factor);
	}
	__syncthreads();

	if (grid_info::linear::grid::num_blocks() > 1) {
		dynamic_num_reductions::detail::block_to_global::dense_to_dense<
			IndexSize, ReductionOp, ReductionIndex,
			dynamic_num_reductions::detail::block_to_global::atomic_block_to_global
		>(target, block_results, num_reductions);
	}
	else {
		// TODO: Avoid this code duplication
		dynamic_num_reductions::detail::block_to_global::dense_to_dense<
			IndexSize, ReductionOp, ReductionIndex,
			dynamic_num_reductions::detail::block_to_global::non_atomic_block_to_global
		>(target, block_results, num_reductions);
	}
}


template <unsigned IndexSize, typename ReductionOp, typename InputDatum, typename ReductionIndex>
class launch_config_resolution_params_t final : public cuda::launch_config_resolution_params_t {
public:
	using parent = cuda::launch_config_resolution_params_t;
public:
	launch_config_resolution_params_t(
		device::properties_t            device_properties_,
		size_t                          data_length,
		size_t                          num_reductions,
		optional<shared_memory_size_t>  dynamic_shared_mem_limit = nullopt) :
		cuda::launch_config_resolution_params_t(
			device_properties_,
			device_function_t(sparse_to_dense<IndexSize, ReductionOp, InputDatum, ReductionIndex>),
			dynamic_shared_mem_limit
		)
	{
		using index_type = uint_t<IndexSize>;
		using result_type = typename ReductionOp::result_type;

		auto single_result_array_size = num_reductions * sizeof(result_type);
		auto num_result_arrays_which_fit_in_shared_mem =
			available_dynamic_shared_memory_per_block / single_result_array_size;

		grid_construction_resolution            = thread;
		length                                  = data_length;
		serialization_option                    = runtime_specified_factor; // Make it auto_maximized!
		default_serialization_factor            = DefaultSerializationFactor;
		if (num_result_arrays_which_fit_in_shared_mem > 0) {
			dynamic_shared_memory_requirement.per_block =
				num_result_arrays_which_fit_in_shared_mem * single_result_array_size;
		}
		quanta.threads_in_block                 = warp_size;
	};
};

} // namespace sparse_to_dense
} // namespace dynamic_num_reductions
} // namespace multi_reduce
} // namespace reduction
} // namespace kernels
} // namespace cuda
