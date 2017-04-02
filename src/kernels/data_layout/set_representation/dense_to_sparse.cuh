#pragma once
#ifndef SRC_KERNELS_SET_REPRESENTATION_DENSE_TO_SPARSE_CUH
#define SRC_KERNELS_SET_REPRESENTATION_DENSE_TO_SPARSE_CUH

#include "common.h"
#include "kernels/common.cuh"
#include "cuda/on_device/primitives/warp.cuh"
#include "cuda/on_device/primitives/block.cuh"

namespace cuda {
namespace kernels {
namespace set_representation {
namespace dense_to_sparse {

using namespace grid_info::linear;

/*
 * TODO:
 * 0. Write earlier or write later? Who knows.
 *
 * 1. Perhaps avoid some atomics by using multiple global
 * atomic counters and rotating among them round-robin, randomly, or
 * just fixing one per warp, using the warp index? If we used 32 of these
 * it could mean as little 1/32 of the atomics pressure... on the other hand,
 * every warp has ~3 KiB of shared memory space, or at least 375 distinct
 * elements, or bout 12 full warps' worth. So how frequently will the warp
 * need to write? ... maybe not so infrequently. On a GTX 780 each SMX allows
 * a full warp to write once per clock cycle. Will we have enough warps on
 * an SMX to keep that pace? ... perhaps with a large grid; since
 * 12 full-warp-read-and-write-to-shared-mem cycles is a _lot_ of clock cycles.
 * Thousands, mayble 10,000 even.
 *
 * Something else we could do is make the shared memory a sort of a circular
 * buffer, with every warp doing shared mem atomics to get another chunk of
 * shared mem to write to, and occasionally (or maybe only some of the warps)
 * taking the time to write some of the shared buffer to main memory.
 *
 * 2. Consider making this a block-stride rather than a grid-stride kernel.
 * Not clear whether that's very useful though.
 *
 * 3. A version of this kernel which accepts the allocated length for sparse,
 * and refrains from filling more than it can fit
 *
 */
template <unsigned IndexSize>
__global__ void dense_to_unsorted_sparse(
	uint_t<IndexSize>*    __restrict__  sparse,
	uint_t<IndexSize>*    __restrict__  num_elements_written,
	const typename cuda::bit_vector<uint_t<IndexSize>>::container_type*
		                        __restrict__  raw_dense,
	uint_t<IndexSize> domain_size)
{
	using index_type = uint_t<IndexSize>;
	using namespace grid_info::linear;
	using bit_vector = ::cuda::bit_vector<index_type>;

	auto num_warps_per_block = block::num_full_warps(); // which is also num_warps() for this kernel
	cuda::shared_memory_size_t per_warp_output_buffer_length =
		(ptx::special_registers::dynamic_smem_size() / IndexSize) / num_warps_per_block;


	// So the idea is for each warp to accumulate a fragment of the overall output in
	// its own region of shared memory, going over the entire input. Whenever some
	// warp's shared mem buffer fills up, that warp writes all of it to global memory
	// (using atomics to determine the destination; the buffering helps reduce the
	// frequency of atomic calls)

	// TODO: Put this next bit in some header, e.g. make a warp::get_shared_memory()
	// function which does this
	index_type* warp_output_buffer =
		shared_memory_proxy<index_type>() +
		per_warp_output_buffer_length * warp::index_in_block();

	// This value is maintained by each warp thread independently, but is the same
	// for all warp threads
	index_type output_fragment_length = 0;

	auto f = [&](index_type pos_in_dense_elements) {
		typename bit_vector::container_type dense_container_element =
			raw_dense[pos_in_dense_elements];

		if (bit_vector::is_set(dense_container_element, lane::index())) {
			auto i = bit_vector::index_among_set_bits(dense_container_element, lane::index());
			// remember we have one grid thread for every single element of the domain,
			// i.e. for every (non-slack) bit in the bit vector
			warp_output_buffer[output_fragment_length + i] =
				bit_vector::global_index_for(pos_in_dense_elements, lane::index());
		}
		output_fragment_length += count_bits_set(dense_container_element);
		if (output_fragment_length > per_warp_output_buffer_length - warp_size) {
			cuda::primitives::warp::collaborative_append_to_global_memory<index_type, index_type>(
				sparse, num_elements_written,
				warp_output_buffer, output_fragment_length);
			output_fragment_length = 0;
		}
	};
	auto raw_dense_length = bit_vector::num_elements_necessary_for(domain_size);
	primitives::warp::at_grid_stride(raw_dense_length, f);

	// After passing the entire grid we need to 'flush' the shared memory
	// remainder into the global output

	if (output_fragment_length > 0) {
		cuda::primitives::warp::collaborative_append_to_global_memory<index_type, index_type>(
			sparse, num_elements_written, warp_output_buffer, output_fragment_length);
	}
}

template <unsigned IndexSize>
class launch_config_resolution_params_t final : public cuda::launch_config_resolution_params_t {
public:
	using parent = cuda::launch_config_resolution_params_t;
public:
	launch_config_resolution_params_t(
		device::properties_t            device_properties_,
		size_t                          input_data_length,
		optional<shared_memory_size_t>  dynamic_shared_mem_limit = nullopt) :
		cuda::launch_config_resolution_params_t(
			device_properties_,
			device_function_t(dense_to_unsorted_sparse<IndexSize>),
			dynamic_shared_mem_limit
		)
	{
		if (dynamic_shared_mem_limit && dynamic_shared_mem_limit < available_dynamic_shared_memory_per_block) {
			util::enforce("Kernel shared memory requirement exceeds specified limit: ",
				available_dynamic_shared_memory_per_block, dynamic_shared_mem_limit.value());
		}

		grid_construction_resolution        = resolution_t::warp;
		length                              = util::div_rounding_up(input_data_length, warp_size);
		serialization_option                = serialization_option_t::auto_maximized;
		keep_gpu_busy_factor                = 50;
			// Really? There's a difference between the last part
			// (each thread has one or two elements in registers,
			// and they interact) and the first part which is essentially
			// per-thread-accumulation. The first needs a high keep-busy
			// factor, the second part needs just 1
		dynamic_shared_memory_requirement.per_block =
			available_dynamic_shared_memory_per_block;
			// We want to make the per-warp shared mem buffers as large as possible,
			// to minimize atomic calls

	};
};


} // namespace dense_to_sparse
} // namespace set_representation
} // namespace kernels
} // namespace cuda

#endif /* SRC_KERNELS_SET_REPRESENTATION_DENSE_TO_SPARSE_CUH */
