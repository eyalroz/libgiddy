#pragma once
#ifndef SRC_CUDA_WARP_LEVEL_PRIMITIVES_CUH_
#define SRC_CUDA_WARP_LEVEL_PRIMITIVES_CUH_

#include "cuda/on_device/primitives/common.cuh"

#include "cuda/bit_operations.cuh"
#include "cuda/functors.hpp"
#include "cuda/on_device/generic_shuffle.cuh"
#include "cuda/on_device/builtins.cuh"
#include "cuda/on_device/atomics.cuh"
#include "cuda/on_device/ptx.cuh"

#include <type_traits>

#define __fd__ __device__ __forceinline__

namespace cuda {
namespace primitives {
namespace warp {

using namespace grid_info::linear;

/**
 * Checks whether a predicate holds for an entire warp of threads
 *
 * @param predicate A boolean value (passed as an unsigned int
 * since that's what nVIDIA GPUs actually check with the HW instruction
 * @return true if predicate is non-zero for all threads
 */
__fd__  unsigned all_satisfy(unsigned int predicate)
{
	return ::builtins::all_in_warp_satisfy(predicate);
}


/**
 * Checks whether a predicate holds for none of the threads in a warp
 *
 * @param predicate A boolean value (passed as an unsigned int
 * since that's what nVIDIA GPUs actually check with the HW instruction
 * @return true if predicate is zero for all threads
 */
__fd__  unsigned none_satisfy(unsigned int predicate)
{
	return all_satisfy(!predicate);
}

/**
 * Checks whether a predicate holds for an entire warp of threads
 *
 * @param predicate A boolean value (passed as an unsigned int
 * since that's what nVIDIA GPUs actually check with the HW instruction
 * @return true if predicate is non-zero for all threads
 */
__fd__  unsigned all_lanes_agree_on(unsigned int predicate)
{
	auto ballot_results = ::builtins::warp_ballot(predicate);
	return
		    ballot_results == 0  // none satisfy the predicate
		or ~ballot_results == 0; // all satisfy the predicate);
}


/**
 * Checks whether a predicate holds for at least one of the threads
 * in a warp
 *
 * @param predicate A boolean value (passed as an unsigned int
 * since that's what nVIDIA GPUs actually check with the HW instruction
 * @return true if predicate is non-zero for at least one thread
 */
__fd__  unsigned some_satisfy(unsigned int predicate)
{
	return !none_satisfy(predicate);
}


/**
 * Performs a reduction (e.g. a summation of multiplication) of all elements passed into
 * the function by the threads of a block.
 *
 * @note This ignores overflow! Make sure you use a roomy type. Alternatively, you
 * could implement a two-type version which takes input in one type and works on a
 * bigger one.
 */
template <typename ReductionOp>
__fd__ typename ReductionOp::result_type reduce(typename ReductionOp::first_argument_type value)
{
	static_assert(std::is_same<
		typename ReductionOp::first_argument_type,
		typename ReductionOp::second_argument_type>::value, "A reduction operation "
			"must have the same types for its LHS and RHS");
	static_assert(std::is_same<
		typename ReductionOp::first_argument_type,
		typename ReductionOp::result_type>::value, "The warp shuffle primitive "
			"can only be applied with a reduction op having the same result and "
			"argument types");
	typename ReductionOp::accumulator op_acc;
	// Let's cross our fingers and hope this next variable gets optimized
	// away with return-value optimization (and that the same register is used
	// for everything in the case of
	//
	//   x = reduce<ReductionOp>(x);
	//
	for (int shuffle_mask = half_warp_size; shuffle_mask > 0; shuffle_mask >>= 1)
		op_acc(value, __shfl_xor(value, shuffle_mask));
	return value;
}

template <typename Datum>
__fd__ Datum sum(Datum value)
{
	return reduce<cuda::functors::plus<Datum>>(value);
}

/**
 * Performs a reduction (e.g. a summation of multiplication) of all elements passed into
 * the function by the threads of a block - but with each thread ending up with the reduction
 * result for all threads upto itself.
 *
 * @note What about inclusivity?
 *
 * @todo offer both an inclusive and an exclusive versionn
 */
template <
	typename ReductionOp,
	typename InputDatum,
	inclusivity_t Inclusivity = inclusivity_t::Inclusive>
__fd__ typename ReductionOp::result_type scan(InputDatum value)
{
	using result_type = typename ReductionOp::result_type;
	ReductionOp op;
	typename ReductionOp::accumulator acc_op;

	result_type x;

	if (Inclusivity == inclusivity_t::Exclusive) {
		InputDatum preshuffled = __shfl_up(value, 1);
		// ... and now we can pretend to be doing an inclusive shuffle
		x = lane::is_first() ? op.neutral_value() : preshuffled;
	}
	else { x = value; }

	// x of lane i holds the reduction of values of
	// the lanes i - 2*(offset) ... i - offset , and initially offset = 0
	#pragma unroll
	for (int offset = 1; offset < warp_size; offset <<= 1) {
		result_type shuffled = __shfl_up(x, offset);
		if(lane::index() >= offset) { acc_op(x, shuffled); }
	}
	return x;
}

// TODO: Need to implement a scan-and-reduce warp primitive

template <typename T, inclusivity_t Inclusivity = inclusivity_t::Inclusive, typename Result = T>
__fd__ T prefix_sum(T value)
{
	return scan<cuda::functors::plus<Result>, T, Inclusivity>(value);
}

template <typename T, typename Result = T>
__fd__ T exclusive_prefix_sum(T value)
{
	return prefix_sum<T, inclusivity_t::Exclusive, Result>(value);
}


// Note: This relies on the shuffle generics! Without them it'll
// fail for > 32-bit types.
// Also, let's consider replacing this function with some proxy
// to enable syntax such as
//
//   warp_choose(my_var).from_lane(some_lane_id)
//
template <typename T>
__fd__ T get_from_lane(T value, int source_lane)
{
	return __shfl(value, source_lane);
}

template <typename T>
__fd__ T get_from_first_lane(T value)
{
	return get_from_lane(value, grid_info::linear::warp::first_lane);
}

template <typename T>
__fd__ T get_from_last_lane(T value)
{
	return get_from_lane(value, grid_info::linear::warp::last_lane);
}

template <typename T>
__fd__ void update_from_lane(T& value, int source_lane)
{
	value = __shfl(value, source_lane);
}

template <typename T>
__fd__ void update_from_first_lane(T& value)
{
	update_from_lane(value, grid_info::linear::warp::first_lane);
}

template <typename T>
__fd__ void update_from_last_lane(T& value)
{
	update_from_lane(value, grid_info::linear::warp::last_lane);
}


template <typename Function>
__fd__ typename std::result_of<Function()>::type have_a_single_lane_compute(
	Function f, unsigned designated_computing_lane = grid_info::linear::warp::first_lane)
{
	typename std::result_of<Function()>::type result;
	if (lane::index() == designated_computing_lane) { result = f(); }
	return get_from_lane(result, designated_computing_lane);
}

template <typename Function>
__fd__ typename std::result_of<Function()>::type have_first_lane_compute(Function f)
{
	return have_a_single_lane_compute<Function>(f, grid_info::linear::warp::first_lane);
}

template <typename Function>
__fd__ typename std::result_of<Function()>::type have_last_lane_compute(Function f)
{
	return have_a_single_lane_compute<Function>(f, grid_info::linear::warp::last_lane);
}

/**
 * Count the threads in a warp for which some predicate holds
 *
 * @param predicate the predicate value for each thread (true if non-zero)
 * @return the number of threads in the warp whose {@ref predicate} is true (non-zero)
 */
__fd__ unsigned vote_and_tally(int predicate)
{
	return builtins::population_count(builtins::warp_ballot(predicate));
}

/**
 * Determines which is the first lane within a warp which satisfies some
 * boolean predicate
 *
 * @param[in] predicate an essentially-boolena value representing whether
 * or not the calling thread (lane) satisfies the predicate; typed
 * as an int since that's what CUDA primitives take mostly.
 *
 * @return index of the first lane in the warp for which predicate is non-zero,
 * or warp_size in case no lanes in the warp satisfy the predicate
 */
__fd__ unsigned first_lane_satisfying(int predicate)
{
	return count_trailing_zeros(builtins::warp_ballot(predicate));
}

/**
 * Have all warp threads collaborate in copying
 * data between two memory locations (possibly not in the same memory
 * space), while also converting types.
 *
 * @param target The destination into which to write the converted elements
 * @param source The origin of the data
 * @param length The number of elements available (for reading?] at the
 * source
 */
template <typename T, typename U, typename Size>
__fd__ void cast_and_copy(
	T*        __restrict__  target,
	const U*  __restrict__  source,
	Size                    length)
{
	using namespace grid_info::linear;
	// sometimes this next loop can be unrolled (when length is known
	// at compile time; since the function is inlined)
	#pragma unroll
	for(promoted_size<Size> pos = lane::index(); pos < length; pos += warp_size) {
		target[pos] = source[pos];
	}
}

/**
 * Same as {@ref cast_copy}, except that no casting is done
 */
template <typename T, typename Size>
__fd__ void copy(
	T*        __restrict__  target,
	const T*  __restrict__  source,
	Size                    length)
{
	return cast_and_copy<T, T, Size>(target, source, length);
}


// TODO: Check whether writing this with a forward iterator and std::advance
// yields the same PTX code (in which case we'll prefer that)
template <typename RandomAccessIterator, typename Size, typename T>
inline __device__ void fill_n(RandomAccessIterator start, Size count, const T& value)
{
	T tmp = value;
	for(promoted_size<Size> index = lane::index();
		index < count;
		index += grid_info::linear::warp::size())
	{
		start[index] = tmp;
	}
}

template <typename ForwardIterator, typename T>
inline __device__ void fill(ForwardIterator start, ForwardIterator end, const T& value)
{
    const T tmp = value;
	auto iter = start + lane::index();
    for (; iter < end; iter += grid_info::linear::warp::size())
    {
    	*iter = tmp;
    }
}

/**
 * Use a lookup table to convert numeric indices to a sequence
 * of values of any type
 */
template <typename T, typename I, typename Size, typename U = T>
__fd__ void lookup(
	T*       __restrict__  target,
	const U* __restrict__  lookup_table,
	const I* __restrict__  indices,
	Size                   num_indices)
{
	using namespace grid_info::linear;
	#pragma unroll
	for(promoted_size<Size> pos = lane::index(); pos < num_indices; pos += warp_size) {
		target[pos] = lookup_table[indices[pos]];
	}
}

/**
 * Used by multiple warps, when each warp has a bunch of data it has
 * obtained and all warps' data must be chained into a global-memory
 * vector - with no gaps and no overwriting (but not necessarily in
 * the order of warps, just any order.)
 *
 * @tparam T the type of data elements being copied
 * @tparam Size must fit any index used into the input or output array;
 * for the general case it would be 64-bit, but this is
 * usable also for when you need 32-bit work (e.g. a 32-bit length
 * output variable).
 * @param global_output
 * @param global_output_length
 * @param fragment_to_append
 * @param fragment_length
 */
template <typename T, typename Size = size_t>
__fd__ void collaborative_append_to_global_memory(
	T*     __restrict__  global_output,
	Size*  __restrict__  global_output_length,
	T*     __restrict__  fragment_to_append,
	Size   __restrict__  fragment_length)
{
	using namespace grid_info::linear;
	Size previous_output_size = thread::is_first_in_warp() ?
		atomic::add(global_output_length, fragment_length) : 0;
	Size offset_to_start_writing_at = get_from_lane(
		previous_output_size, grid_info::linear::warp::first_lane);

	// Now the (0-based) positions
	// previous_output_size ... previous_output_size + fragment_length - 1
	// are reserved by this warp; nobody else will write there and we don't need
	// any more atomics

	copy(global_output + offset_to_start_writing_at,
		fragment_to_append, fragment_length);
}

/**
 * @note If you call this for multiple warps and the same destination,
 * you'd better use an atomic accumulator op...
 */

template <typename D, typename S, typename AccumulatingOperation, typename Size>
forceinline __device__ void elementwise_accumulate(
	D*       __restrict__  destination,
	const S* __restrict__  source,
	Size                   length)
{
	AccumulatingOperation op;
	for(promoted_size<Size> pos = lane::index(); pos < length; pos += warp_size) {
		op(destination[pos], source[pos]);
	}
}

/**
 * A variant of elementwise_accumulate for when the length <= warp_size,
 * in which case each thread will get just one source element (possibly
 * a junk or default-constructed element) to work with
 *
 * @note If you call this for multiple warps and the same destination,
 * you'd better use an atomic accumulator op...
 */
template <typename D, typename S, typename AccumulatingOperation, typename Size>
forceinline __device__ void elementwise_accumulate(
	D*       __restrict__  destination,
	const S&               source_element,
	Size                   length)
{
	AccumulatingOperation op;
	if (lane::index() < length) {
		op(destination[lane::index()], source_element);
	}
}

/**
 * A variant of the one-position-per-thread applicator,
 * {@ref primitives::grid::linear::at_grid_stride}: Here each warp works on one
 * input position, advancing by 'grid stride' in the sense of total
 * warps in the grid.
 *
 * @param length The length of the range of positions on which to act
 * @param f The callable for warps to use each position in the sequence
 */
template <typename Function, typename Size = unsigned>
__fd__ void at_grid_stride(Size length, const Function& f)
{
	using namespace ::grid_info;

	auto num_warps_in_grid = linear::grid::num_warps();
	for(// _not_ the global thread index! - one element per warp
		promoted_size<Size> pos = linear::warp::global_index();
		pos < length;
		pos += num_warps_in_grid)
	{
		f(pos);
	}
}

template <typename T>
__fd__ T* get_warp_specific_shared_memory(unsigned elements_allocated_per_warp)
{
	return shared_memory_proxy<T>() +
		elements_allocated_per_warp * grid_info::linear::warp::index_in_block();
}

// A bit ugly, but...
// Note: make sure the first warp thread has not diverged/exited,
// or use the leader selection below
// TODO: Mark this unsafe
#define once_per_warp if (::grid_info::linear::thread::is_first_in_warp())

__fd__ unsigned active_lanes_mask()
{
	return builtins::warp_ballot(1);
		// the result will only have bits set for the lanes which are active;
		// there's no "inference" about what inactive lanes might have passed
		// to the ballot function
}

__fd__ unsigned active_lane_count()
{
	return builtins::population_count(active_lanes_mask());
}

namespace detail {

template <bool PreferFirstLane = true>
__fd__ unsigned select_leader_lane(unsigned active_lanes_mask)
{
	// If clz returns k, that means that the k'th lane (zero-based) is active, and
	// can be chosen as the leader
	//
	// Note: We (safely) assume at least one lane is active, as
	// otherwise clz will return -1
	return PreferFirstLane ?
		builtins::count_leading_zeros(active_lanes_mask) :
		count_trailing_zeros(active_lanes_mask);
}

template <bool PreferFirstLane = true>
__fd__ bool am_leader_lane(unsigned active_lanes_mask)
{
	return select_leader_lane<PreferFirstLane>(active_lanes_mask)
		== grid_info::linear::lane::index();
}

__fd__ bool lane_index_among_active_lanes(unsigned active_lanes_mask)
{
	unsigned preceding_lanes_mask =
		(1 << ptx::special_registers::laneid()) - 1;
	return builtins::population_count(preceding_lanes_mask);
}

template <typename Function, bool PreferFirstLane = true>
__fd__ typename std::result_of<Function()>::type have_a_single_active_lane_compute(
	Function f, unsigned active_lanes_mask)
{
	return have_a_single_lane_compute(f, select_leader_lane<PreferFirstLane>(active_lanes_mask));
}


} // namespace detail

/**
 * This is a mechanism for making exactly one lane act instead of the whole warp,
 * which supports the case of some threads being inactive (e.g. having exited).
 *
 * @tparam PreferFirstLane if true, the first lane will be the acting leader
 * whenever it is active - at the cost of making this functions slightly more
 * expensive (an extra subtraction instruction)
 * @return Iif PreferFirstLane is true, and the first lane is active,
 * then 0; the index of some active lane otherwise.
 */
template <bool PreferFirstLane = true>
__fd__ unsigned select_leader_lane()
{
	return detail::select_leader_lane<PreferFirstLane>(active_lanes_mask());
}

/**
 * This applies the leader lane selection mechanism to obtain a predicate
 * for being the leader lane.
 *
 * @tparam PreferFirstLane if true, the first lane will be the acting leader
 * whenever it is active - at the cost of making this functions slightly more
 * expensive (an extra subtraction instruction)
 * @return 1 for exactly one of the active lanes in each warp, 0 for all others
 */
template <bool PreferFirstLane = true>
__fd__ bool am_leader_lane()
{
	return detail::am_leader_lane<PreferFirstLane>(active_lanes_mask());
}

__fd__ unsigned lane_index_among_active_lanes()
{
	return detail::lane_index_among_active_lanes(active_lanes_mask());
}


template <typename Function, bool PreferFirstLane = true>
__fd__ typename std::result_of<Function()>::type have_a_single_active_lane_compute(Function f)
{
	return detail::have_a_single_active_lane_compute<PreferFirstLane>(f, active_lanes_mask());
}

// TODO: Consider implementing have_first_active_lane_compute and
// have_last_active_lane_compute

/**
 * When every (active) lane in a warp needs to increment a counter,
 * use this function to avoid all of them having to execute atomics;
 * each active lane gets the "previous value" as though they had all
 * incremented in order.
 *
 * @note It's not clear to me how better this is from just using atomics
 * and being done with it
 *
 * @todo extend this to other atomic operations
 */
template <typename T>
__fd__ T active_lanes_increment(T* counter)
{
	auto lanes_mask = active_lanes_mask();
	auto active_lane_count = builtins::population_count(lanes_mask);
	auto perform_all_increments = [counter, active_lane_count]() {
		atomic::add(counter, active_lane_count);
	};
	auto value_before_all_lane_increments =
		have_a_single_lane_compute(perform_all_increments,
			detail::select_leader_lane(lanes_mask));
	// the return value simulates the case of every lane having done its
	// own atomic increment
	return value_before_all_lane_increments +
		detail::lane_index_among_active_lanes(lanes_mask);
}


template <typename T>
struct search_result_t {
	unsigned lane_index { warp_size };
	T value;
	bool is_set() const { return lane_index < warp_size; }
};

/**
 * Have each lane search for its own value of interest within the
 * sorted sequence of single values provided by all the warp lanes.
 *
 * @note The amount of time this function takes is very much
 * data-dependent!
 *
 * @todo Does it matter if the _needles_, as opposed to the
 * _hay straws_, are sorted? I wonder.
 *
 * @param lane_needle the value the current lane wants to search
 * for
 * @param lane_hay_straw the warp_size hay "straws" passed by
 * the lanes make up the entire "haystack" we search in. They
 * _must_ be known to be in order, i < j => straw of lane i <
 * straw of lane j. They are accessed using intra-warp shuffling,
 * solely.
 * @return For lane i, the index of the first lane j with
 * straw-of-j > needle-of-i, along with straw-of-j; if there is no such
 * lane j, warp_size is returned as the lane index and an arbitrary
 * value is returned as the result.
 */
template <typename T, bool AssumeNeedlesAreSorted = false>
__fd__ search_result_t<T> multisearch(const T& lane_needle, const T& lane_hay_straw)
{
	search_result_t<T> result;

	struct {
		unsigned lower, upper; // lower  is inclusive, upper is exclusive
	} bounds;
	if (lane_needle <= lane_hay_straw) {
		bounds.lower = grid_info::linear::warp::first_lane;
		bounds.upper = lane::index();
	}
	else {
		bounds.lower = lane::index() + 1;
		bounds.upper = warp_size;
	}
	enum : unsigned { cutoff_to_linear_search = 6 };
		// is 6 a good choice for a cutoff? should it depend on the microarch?

	while (bounds.upper - bounds.lower >= cutoff_to_linear_search) {
		unsigned mid = (bounds.lower + bounds.upper) / 2;
		auto mid_lane_hay_straw = __shfl(lane_needle, mid);
		if (lane_needle <= mid_lane_hay_straw) { bounds.lower = mid + 1; }
		else { bounds.upper = mid; }
	}
	for(unsigned lane = bounds.lower; lane < bounds.upper; lane++) {
		auto hay_straw = __shfl(lane_needle, lane);
		if (not result.is_set() and hay_straw > lane_needle) {
			result = { lane, hay_straw }; break;
		}
	}
	return result;
}


} // namespace warp
} // namespace primitives
} // namespace cuda

#undef __fd__

#endif /* SRC_CUDA_WARP_LEVEL_PRIMITIVES_CUH_ */
