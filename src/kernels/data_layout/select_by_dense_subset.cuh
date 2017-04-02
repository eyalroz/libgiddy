#pragma once
#ifndef SRC_KERNELS_ENUMERATED_SELECT_BY_DENSE_SUBSET_CUH
#define SRC_KERNELS_ENUMERATED_SELECT_BY_DENSE_SUBSET_CUH

#include "kernels/data_layout/set_representation/common.h"
#include "kernels/common.cuh"
#include "cuda/bit_vector.cuh"
#include "cuda/on_device/primitives/warp.cuh"
#include "cuda/on_device/primitives/block.cuh"

namespace cuda {
namespace kernels {
namespace select_by_dense_subset {

// Note that select by _sparse_ subset is just the gather operation

using cuda::bit_vector;
using namespace grid_info::linear;

/*
 * Notes:
 *
 * - This kernel is almost copy-pasted from dense_to_unsorted_sparse.
 * Probably one can encapsulate most of its functionality in a templated
 * device-side function and just call it from here.
 * - It really helps if selected is 128-byte aligned
 *
 * TODO: Perhaps have a version of this which doesn't set num_selected?
 * TODO: Perhaps we should templatify per_warp_shared_mem_elements?
 *
 */
template <unsigned IndexSize, unsigned ElementSize>
__global__ void select_by_dense_subset(
	uint_t<ElementSize>*        __restrict__  selected,     // what about alignment??
	uint_t<IndexSize>*                             __restrict__  num_selected,
	const uint_t<ElementSize>*  __restrict__  input_data,
	const typename cuda::bit_vector<uint_t<IndexSize>>::container_type*
		                              __restrict__  raw_dense,
	uint_t<IndexSize>       domain_size,
	cuda::shared_memory_size_t                      per_warp_shared_mem_elements)
{
	using namespace grid_info::linear;
	using index_type = uint_t<IndexSize>;
	using element_type = uint_t<ElementSize>;
	using bit_vector = ::cuda::bit_vector<index_type>;

	// So the idea is for each warp to accumulate a fragment of the overall output in
	// its own region of shared memory, going over the entire input. Whenever this
	// area of shared memory fills up (for a certain warp), it writes all of it to
	// global memory

	// TODO: Perpahs have a warp::get_shared_memory()
	// function, or a warp::shared_memory facade object, into which you feed the
	// size of shared-mem-per-warp and get this warp's shared mem
	element_type* selected_fragment =
		shared_memory_proxy<element_type>() +
		per_warp_shared_mem_elements * warp::index_in_block();

	// This value is maintained by each warp thread independently, but is the same
	// for all warp threads
	index_type selected_fragment_length = 0;

	auto fragment_select_by_fragment_of_the_dense_subset = [&](index_type pos_in_dense_elements) {
		typename bit_vector::container_type dense_container_element =
			raw_dense[pos_in_dense_elements];

		if (bit_vector::is_set(dense_container_element, lane::index())) {
			auto i = bit_vector::index_among_set_bits(dense_container_element, lane::index());
			// Remember we have one grid thread for every single element
			// of the input data column, i.e. for every (non-slack) bit in
			// the bit vector.
			auto input_position = bit_vector::global_index_for(pos_in_dense_elements, lane::index());
			selected_fragment[selected_fragment_length + i] = input_data[input_position];
		}
		selected_fragment_length += count_bits_set(dense_container_element);
		if (selected_fragment_length > per_warp_shared_mem_elements - warp_size) {
			cuda::primitives::warp::collaborative_append_to_global_memory<element_type, index_type>(
				(element_type*) selected, num_selected,
				selected_fragment, selected_fragment_length);
			selected_fragment_length = 0;
		}
	};
	auto raw_dense_length = bit_vector::num_elements_necessary_for(domain_size);
	primitives::warp::at_grid_stride(raw_dense_length,
		fragment_select_by_fragment_of_the_dense_subset);

	// After passing the entire grid we need to 'flush' the shared memory
	// remainder into the global output

	if (selected_fragment_length > 0) {
		cuda::primitives::warp::collaborative_append_to_global_memory<element_type, index_type>(
			selected, num_selected, selected_fragment, selected_fragment_length);
	}
}

template <unsigned IndexSize, unsigned ElementSize>
class launch_config_resolution_params_t final : public cuda::launch_config_resolution_params_t {
public:
	using parent = cuda::launch_config_resolution_params_t;
public:
	launch_config_resolution_params_t(
		device::properties_t            device_properties,
		size_t                          length_,
		optional<shared_memory_size_t>  dynamic_shared_mem_limit = nullopt
		) :
		cuda::launch_config_resolution_params_t(
			device_properties,
			device_function_t(select_by_dense_subset<IndexSize, ElementSize>),
			dynamic_shared_mem_limit
		)
	{
		keep_gpu_busy_factor                    = 50; // TODO: Experiment with this, make it uarch dependent
		grid_construction_resolution            = thread;
		length                                  = length_; // TODO: I'm only 60% sure about this
		quanta.threads_in_block                 = warp_size;
		serialization_option                    = auto_maximized;
				// The actual amount of shared memory used demends on a parameter passed by the calling code:
		// it's ElementSize * per_warp_shared_mem_elements . So we'll just set the maximum here
		dynamic_shared_memory_requirement.per_block =
			available_dynamic_shared_memory_per_block;
	};
};



} //namespace dense_to_sparse
} // namespace kernels
} // namespace cuda


#endif /* SRC_KERNELS_ENUMERATED_SELECT_BY_DENSE_SUBSET_CUH */
