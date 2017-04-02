#pragma once
#ifndef SRC_KERNELS_SET_REPRESENTATION_SPARSE_TO_DENSE_CUH
#define SRC_KERNELS_SET_REPRESENTATION_SPARSE_TO_DENSE_CUH

#include "common.h"
#include "kernels/common.cuh"
#include "cuda/on_device/builtins.cuh"
#include "cuda/on_device/primitives/warp.cuh"
#include "cuda/on_device/primitives/block.cuh"

namespace cuda {
namespace kernels {
namespace set_representation {
namespace sparse_to_dense {

using namespace grid_info::linear;

template <unsigned IndexSize>
__device__ void sparse_to_dense_sorted(
	typename bit_vector<uint_t<IndexSize>>::container_type*
		         __restrict__  raw_dense, // maybe pass the actual class instance?
	const uint_t<IndexSize>*  __restrict__  sparse,
	uint_t<IndexSize>     sparse_length)
{
	using namespace grid_info::linear;
	using index_type = uint_t<IndexSize>;
	using bit_vector = typename ::cuda::bit_vector<index_type>;
	using container_type = typename ::cuda::bit_vector<index_type>::container_type;

	/* Generally we'll be working at block stride, since timing is not
	 * uniform across warps and blocks and we don't want some warp to lag
	 * consistently.
	 *
	 * So, for a given warp of consecutive sparse entries, what do we do?
	 * If every sparse input element corresponds to a different container
	 * element, each thread will just need to write a 1-bit-on container
	 * element irrespective of the rest of the grid, and no communication
	 * is necessary. But this will often not be the case, and the thread
	 * needs to make sure it doesn't overwrite other threads' writes to
	 * that element. What's worse is that some of those threads might belong
	 * to other warps! And some of those might be in a different _block_
	 * altogether! Drats.
	 *
	 * The straightforward, but slow, solution would be: Just use atomic
	 * compare-and-swap to bitwise-or the container element, guaranteeing
	 * correctness. Unfortunately that's likely to be quite slow.
	 *
	 * Will it be fast enough if we fall back to atomics...
	 *
	 * - For inter-warp 'collisions'?
	 * - For inter-block 'collisions'?
	 * - Or maybe we need to avoid global atomics altogether?
	 *
	 * To avoid atomics, we'd need to have some (all?) warps read the
	 * following 32 elements to the warp's own 32. With luck,
	 * that would only mean just 128 bytes more in reads per block, since
	 * since most 128 byte cache lines should be fetched once and provided
	 * to 3 consecutive blocks (once as each of previous, current, and
	 * following).
	 *
	 * If we can at all prevent multiple writes to the same container,
	 * we could "agree" that only the first thread with index in a container
	 * is the thread which writes to it, and the other threads merely need
	 * to supply it with the relevant bits.
	 *
	 */

	index_type pos = thread::global_index();

	if (pos >= sparse_length) { return; }

	index_type sparse_element = sparse[pos];
	index_type target_pos = bit_vector::element_index_for(sparse_element);

	// A 'leading lane' is the first lane targeting a specific element in the output,
	// the raw dense buffer; only leading lanes will eventually write anything, and
	// non-leading lanes must communicate their data to the leaders
	bool is_leading_lane;

	if (pos > 0) {
		index_type preceding_sparse_element = sparse[pos - 1];
		// TODO: static same-element method perhaps?
		is_leading_lane = bit_vector::element_index_for(preceding_sparse_element) < target_pos;
	}
	else {
		is_leading_lane = true;
	}

	auto leading_lanes_in_warp = builtins::warp_ballot(is_leading_lane);
	if (leading_lanes_in_warp == 0) {
		// No leading lanes, so no work for us; even if some of the data we've
		// read is relevant - the previous warp will take care of it.
		return;
	}

	// Now we'll do something akin to a shuffle-based reduction, using bitwise or
	// instead of sum, except that this will be done for each leading lane, i.e.
	// for each dense element, separately

//	thread_printf("keep_bits_up_to(leading_lanes_in_warp, lane::index()) = keep_bits_up_to(0x%8X, %2u) = 0x%8X",
//		leading_lanes_in_warp, lane::index(), keep_bits_up_to(leading_lanes_in_warp, lane::index()));
	auto associated_leading_lane =
		find_last_set_up_to(leading_lanes_in_warp, lane::index()) - 1;
		// Note that some lanes have _no_ associated leading lane, since
		// the previous warp will be handling their data. For these lanes,
		// the following values are meaningless, and associated_leading_lane
		// will be warp_size. That's not the most "convenient" value but
		// we'll just work with it.


	int fsa = find_first_set_after(leading_lanes_in_warp, lane::index()) - 1;
	auto no_next_leading_lane_in_this_warp = (fsa == -1);
	auto next_leading_lane =
		no_next_leading_lane_in_this_warp ? warp_size : fsa;
	auto distance_to_next_leading_lane = next_leading_lane - lane::index();
	auto distance_from_leading_lane = lane::index() - associated_leading_lane;


	auto contributors_to_associated_leading_lane =
		(associated_leading_lane == warp_size) ?
		0 : next_leading_lane - associated_leading_lane;
		// So this is where we accounted for those lanes whose data is
		// handled by the previous warp - their (non-existent) leading lane
		// will be considered to have 0 contributors

//	thread_printf("my leader = %2u no next leader = %d next leader = %2u num contributors = %2u",
//		associated_leading_lane, no_next_leading_lane_in_this_warp, next_leading_lane, contributors_to_associated_leading_lane);

	auto accumulated_dense_element = // starting out with just one bit set
		bit_vector::element_bit_mask_for(sparse_element);

	#pragma unroll
	for(int delta = 1; delta < 32; delta <<= 1) {
		if (delta >= contributors_to_associated_leading_lane) break;
		auto accumulated_from_another_lane = __shfl_down(accumulated_dense_element, delta);
		auto got_data_for_my_leading_lane =	delta < distance_to_next_leading_lane;


		if (got_data_for_my_leading_lane) {
			accumulated_dense_element |= accumulated_from_another_lane;
		}
	}

        // We  have now completed a sort of a clipped shuffle-based reduction (but with bitwise-or
        // instead of a sum); and each leading thread has all the bits set in its accumulated
        // dense element... except perhaps for the last leading one. That's because it might have
        // data which the next warp is supposed to handle. Is there a next warp? That is, are
        // there additional active warps?

        // Note: this check is specific to this kernel (or rather, for kernels with
        // one thread per one input element). There may be additional warps in the grid,
        // but they don't do anything
        auto in_last_active_warp = (pos >> log_warp_size) >= (sparse_length >> log_warp_size);

        if (in_last_active_warp) {
                if (is_leading_lane) {
//                      thread_printf("in last active warp, leading lane writing %3d to pos %4d", accumulated_dense_element, target_pos);
                        raw_dense[target_pos] = accumulated_dense_element;
                }
//              else thread_print("in last active warp, but not leading.");
                return;
        }

        // Ok, there _are_ additional active warps, let's peek at the next warp's data (which
        // we've had this warp read as well)

        // TODO: Maybe we shouldn't just peek the whole warp's worth? We could easily compute it
        // using last_sparse_element.
        // Note: If this is the next-to-last warp, the following trinary op
        // ensures two things: That we don't read past the end of the input and that
        // we don't use the result of our "read"
        index_type next_warp_element = (pos + warp_size < sparse_length) ?
                sparse[pos + warp_size] : 0;
                // 0 is in this case an arbitrary value whose target pos is 0, which cannot
                // be the same as the target position for the last leader. Why? If there's
                // just one leader in the warp and it's the first element, then maybe its
                // target position is 0, but then it must cover all 32 elements in the warp,
                // meaning that in the next warp we'll only have elements with higher
                // target position (i.e. with value at least 32).

        auto target_pos_for_last_leader_in_warp =
                primitives::warp::get_from_lane(target_pos, warp::last_lane);

	auto use_next_warp_element =
		(bit_vector::element_index_for(next_warp_element) ==
		target_pos_for_last_leader_in_warp);

	// Now we have a sequence of several lanes, beginning with the first one, but ending
	// _before_ the last leader lane (think about it, it has to be the case), with valid
	// data from the next warp

	auto num_elements_to_use_from_next_warp =
		builtins::population_count(builtins::warp_ballot(use_next_warp_element));

	if (num_elements_to_use_from_next_warp == 0) {
		if (is_leading_lane) {
			raw_dense[target_pos] = accumulated_dense_element;
		}
		return;
	}
	container_type accumulated_from_next_warp;

	// Same reduction as before, but only amongst the peeked elements

	if (use_next_warp_element)  {
		accumulated_from_next_warp = bit_vector::element_bit_mask_for(next_warp_element);
	}
	for(int delta = 1; delta < num_elements_to_use_from_next_warp; delta <<= 1) {
		auto accumulated_from_another_lane = __shfl_down(accumulated_from_next_warp, delta);
		if (lane::index() + delta < num_elements_to_use_from_next_warp) {
			accumulated_from_next_warp |= accumulated_from_another_lane;
		}
	}
	// Only the last leader needs this, but hey - we _are_ instruction-locked after all
	auto accumulated_all_from_next_warp =
		primitives::warp::get_from_lane(accumulated_from_next_warp, warp::first_lane);
	if (is_leading_lane) {
		if (no_next_leading_lane_in_this_warp) {
			accumulated_dense_element |= accumulated_all_from_next_warp;
		}
		raw_dense[target_pos] = accumulated_dense_element;
	}
}

// TODO: Support serialization here? Hmm. Maybe not worth the hassle

template <unsigned IndexSize>
__device__ void sparse_to_dense_unsorted(
	typename bit_vector<uint_t<IndexSize>>::container_type*
		         __restrict__  raw_dense, // maybe pass the actual class instance?
	const uint_t<IndexSize>*  __restrict__  sparse,
	uint_t<IndexSize>     sparse_length)
{
	using index_type = uint_t<IndexSize>;
	using namespace grid_info::linear;
	using bit_vector = typename ::cuda::bit_vector<index_type>;

	/*
	 * Notes:
	 *
	 * - A smart(er) thing to do would be to notice whether or not we're in a sorted _fragment_
	 *   or not, since that should be quite common in many use cases. Of course, it could be argued
	 *   that the caller should distinguish between the cases of expecting them or not. At any rate,
	 *   for now, this is a naive implementation
	 */

	auto pos = thread::global_index();
	if (pos >= sparse_length) { return; }
	auto sparse_element = sparse[pos];
	auto dense_element_index = bit_vector::element_index_for(sparse_element);
	auto bit_index_in_dense_element = bit_vector::intra_element_index_for(sparse_element);
	atomic::set_bit(&raw_dense[dense_element_index], bit_index_in_dense_element);
	return;
}

/**
 *
 * @note The sparse subset representation must be sorted, have no
 * repetitions, and obviously only have values which will map
 * into the allocated raw_dense buffer (that is, less than
 * 32 times the number of elements allocated at {@ref raw_dense}
 *
 * @note this code more-or-less assumes that
 * sizeof(bit_vector::container_type == warp_size == 32)
 *
 * @note this requires a pre-initialization of the dense
 * representation to zero, as the kernel does not write anything
 * to container elements with no bits originating in the sparse
 * elements
 *
 * @note this code can be slightly tweaked to bitwise-or with
 * the target position rather than just write there, meaning it
 * would be an implementation of a union of a dense-represented set
 * with a sparse-represented set (with dense output).
 *
 * @tparam IndexSize
 * @tparam SerializationFactor
 * @param raw_dense
 * @param sparse
 * @param sparse_length
 */
template <sortedness_t Sortedness, unsigned IndexSize>
__global__ void sparse_to_dense(
	typename bit_vector<uint_t<IndexSize>>::container_type*
		         __restrict__  raw_dense, // maybe pass the actual class instance?
	const uint_t<IndexSize>*  __restrict__  sparse,
	uint_t<IndexSize>     sparse_length)
{
	// Ugly, since we can't do partial specialization
	return (Sortedness == sortedness_t::Sorted) ?
		sparse_to_dense_sorted<IndexSize>(raw_dense, sparse, sparse_length) :
		sparse_to_dense_unsorted<IndexSize>(raw_dense, sparse, sparse_length);
}

template <sortedness_t Sortedness, unsigned IndexSize>
class launch_config_resolution_params_t final : public cuda::launch_config_resolution_params_t {
public:
	using parent = cuda::launch_config_resolution_params_t;
public:
	launch_config_resolution_params_t(
		device::properties_t            device_properties_,
		size_t                          sparse_length) :
		cuda::launch_config_resolution_params_t(
			device_properties_,
			device_function_t(sparse_to_dense<Sortedness, IndexSize>)
		)
	{
		grid_construction_resolution            = thread;
		length                                  = sparse_length;
		serialization_option                    = none;
		// quanta.threads_in_block                 = warp_size;
	};
};

} // namespace sparse_to_dense
} // namespace set_representation
} // namespace kernels
} // namespace cuda

#endif /* SRC_KERNELS_SET_REPRESENTATION_SPARSE_TO_DENSE_CUH */
