#pragma once
#ifndef SRC_KERNELS_VFOR_INDEXED_SCATTER_CUH
#define SRC_KERNELS_VFOR_INDEXED_SCATTER_CUH

#include "kernels/common.cuh"
#include "cuda/on_device/primitives/warp.cuh"
#include "cuda/on_device/primitives/block.cuh"
#include "kernels/reduction/scan.cuh"
#include "cuda/on_device/generic_shuffle.cuh"
#include "cuda/functors.hpp"

#include <boost/integer.hpp>

namespace cuda {
namespace kernels {
namespace scatter {
namespace compressed_indices {

using namespace grid_info::linear;

namespace detail {
namespace block {

/**
 * This function encapsulates the dependence of the scattering on the patch
 * run's individual offset size (and hence, offset data type) - so it
 * requires an additional template parameter and multiple instantions.
 */
template <
	unsigned OutputIndexSize, unsigned ElementSize, unsigned InputIndexSize,
	unsigned RunLengthSize, unsigned RunElementSizeInBits
>
__device__ void scatter_single_run(
	uint_t<ElementSize>*                __restrict__  target,
	const uint_t<ElementSize>*          __restrict__  data,
	const unsigned char*  __restrict__  run_offset_bytes,
	uint_t<RunLengthSize>                           run_length,
	uint_t<OutputIndexSize>       scatter_position_baseline_value)
{
	using integral_type = typename boost::uint_t<(int) RunElementSizeInBits>::exact;
	static_assert(std::is_integral<integral_type>::value, "non-integral type");
	auto run_offsets = reinterpret_cast<const integral_type*>(run_offset_bytes);
	primitives::block::at_block_stride(run_length, [&](uint_t<RunLengthSize> pos_within_run) {
		uint_t<OutputIndexSize> scatter_output_position =
			scatter_position_baseline_value + run_offsets[pos_within_run];

//		if (thread::is_last_in_grid()) {
//			thread_printf("scatter_output_position = %u = %u + %u = "
//				"scatter_position_baseline_value + typed_run_elements_start[pos_within_run",
//				scatter_output_position, scatter_position_baseline_value, run_offsets);
//		}
		target[scatter_output_position] = data[pos_within_run];
			// ... this statement may diverge badly in the store part,
			// but at least the load will be nicely coalesced
	});
}

/**
 * This is a sub-operation of @ref scatter , which does not take any
 * anchoring information - for which reason it must process runs sequentially
 * rather than having, say, each warp handle a run. See @ref scatter for
 * a description of the different parameters; they are provided by @ref scatter
 * for a single anchored interval (which is guaranteed to be covered by full
 * runs, and there's no need to start/end in mid-run.
 *
 * @note we do not use this as the overall kernel, since we want to improve
 * parallelism at least by having each block acting independently.
 */
template <unsigned OutputIndexSize, unsigned ElementSize, unsigned InputIndexSize, unsigned RunLengthSize = InputIndexSize>
__device__ void scatter_unanchored(
	uint_t<ElementSize>*            __restrict__  target,
	const uint_t<ElementSize>*      __restrict__  data,
	const uint_t<OutputIndexSize>*  __restrict__  scatter_position_run_baseline_values,
	const unsigned char*                  __restrict__  scatter_position_run_individual_offset_sizes,
	const uint_t<RunLengthSize>*    __restrict__  scatter_position_run_lengths,
	const uint_t<InputIndexSize>*   __restrict__  scatter_position_run_offsets_start_positions,
	const unsigned char*                  __restrict__  scatter_position_offset_bytes,
	uint_t<InputIndexSize>                        num_runs_in_interval
)
{
	uint_t<InputIndexSize> starting_data_pos_for_run = 0;
	for(auto run_index = 0; run_index < num_runs_in_interval; run_index++) {
		auto run_scatter_pos_baseline_value = scatter_position_run_baseline_values[run_index];
		auto run_element_size = scatter_position_run_individual_offset_sizes[run_index];
		auto run_length = scatter_position_run_lengths[run_index];
		const unsigned char* run_element_bytes_start =
			scatter_position_offset_bytes +
			scatter_position_run_offsets_start_positions[run_index] * OutputIndexSize;

		// You could argue "why no size 3, or 5, 6 etc?
		// ... well, the answer is that if the size could be 3, we'll just make it
		// 4, which is not a terrible loss w.r.t. the compressed indices, certainly
		// when you take the data into account - but what's more, it means that it is
		// _not_ 2, i.e. typical destination indices are at a distance of more than
		// 2^16 output elements, i.e. patching is sparse (as for batches of patches
		// close to each other within that segment, another segment could have been
		// added during the compression phase), i.e. the overall impact on the
		// compression ratio is very very low

		// TODO: Use some mechanism for avoiding this code duplication...!

		auto run_data = data + starting_data_pos_for_run;
		switch(run_element_size) {
		case 8:
			detail::block::scatter_single_run<OutputIndexSize, ElementSize, InputIndexSize, RunLengthSize, 8u>(
				target, run_data, run_element_bytes_start, run_length, run_scatter_pos_baseline_value);
			break;
		case 16:
			detail::block::scatter_single_run<OutputIndexSize, ElementSize, InputIndexSize, RunLengthSize, 16u>(
				target, run_data, run_element_bytes_start, run_length, run_scatter_pos_baseline_value);
			break;
		case 32:
			detail::block::scatter_single_run<OutputIndexSize, ElementSize, InputIndexSize, RunLengthSize, 32u>(
				target, run_data, run_element_bytes_start, run_length, run_scatter_pos_baseline_value);
			break;
		case 64:
			detail::block::scatter_single_run<OutputIndexSize, ElementSize, InputIndexSize, RunLengthSize, 64u>(
				target, run_data, run_element_bytes_start, run_length, run_scatter_pos_baseline_value);
			break;
		}
		starting_data_pos_for_run += run_length;
	}
}


} // namespace block

} // namespace detail


/**
 * This is a variant of the Scatter operation for sorted
 * scattering indices, in which the scatter destinations
 * are input in a compressed form: Sequences of related
 * indices, with each sequence having a frame-of-reference
 * value (such that the encoded values are offsets from it),
 * number of bytes per element in the encoding of offsets
 * for that sequence, and the sequence length.
 *
 * This is motivated by the fact that the sorted indices
 * typically share many bytes with their predecessor values;
 * but occasionally this may not be the case, so DELTA
 * compression would be problematic (and we don't want
 * "double patching" - patching of DELTA encoding of the offsets).
 *
 * @note we assume that anchors to multiples of the anchoring
 * period are placed at the beginning of runs. This is a safe
 * assumption, since if it's not the case - the runs can be
 * broken up into multiple, shorter runs with the same FOR value.
 * Is this worth it? Well, it would add at most 1 run for every
 * @p anchoring_period elements of data to scatter, and since
 * @p anchoring_period is typically generous (e.g. 1024, 2048) -
 * it means at most +OutputIndexSize for the FOR value and +InputIndexSize
 * for the length, so at most a factor of 8/1024 for 64-bit Size
 * types. On the other hand, this will also scrap the offsets column,
 * which will _certainly_ cut uint_t<RunLengthSize> * num_anchors .
 * Since the run lengths are small. So even size-wise it's mostly
 * a win.
 *
 * @note this kernel will likely be inefficient for overly small
 * anchoring periods. If that's an issue, alter it so that every
 * block skips several anchors, effectively enlarging the
 * anchoring period - and change the launch config resolver
 * accordingly
 *
 * @note 0-length inputs (i.e. nothing to scatter, no anchors etc.)
 * is not supported
 *
 * @tparam OutputIndexSize size in bytes of indices into the @p target array. It is
 * never smaller than @tparam InputIndexSize (since scattering with overrides, i.e.
 * recurring indices into target, is not supported)
 * @tparam uint_t<ElementSize> the type of elements we'll be writing into @p target
 * @tparam InputIndexSize size in bytes of indices into the @p data array
 * @tparam RunLengthSize Indices into the target parameter come in runs,
 * processed separately; the run lengths may be limited more strictly
 * than indices into @p data , hence a possibly-different size for them.
 *
 * @param[inout] target The array into which input data is written -
 * but not necessarily contiguously, not does it necessarily get
 * completely overwritten (typically, only a fraction of it might be
 * altered)
 * @param[in] data The input to write at different positions into @p target
 * @param[in] scatter_position_run_offsets_start_pos An indication, for each
 * scatter position run, of where its offsets lie within @p scatter_position_bytes,
 * in multiples of OutputIndexSize (which is possible do to the run-level alignment of
 * @p scatter_position_bytes)
 * @param[in] scatter_position_run_lengths the numbers of output position indices;
 * run lengths most be strictly positive (non-0)
 * encoded in each of the runs
 * @param[in] scatter_position_run_individual_offset_sizes the number of bits used to encode the offset
 * of a target position from the run baseline, for every one of the runs. At the moment,
 * the only valid values are be 8, 16, 32 or 64; anything else will fail silently (or not
 * so silently)
 * @param[in] scatter_position_run_baseline_values For each run of output positions, this
 * value is to be added to the offset encoded in @scatter_position_bytes
 * @param[in] scatter_position_bytes the untyped offsets for all output position runs. This
 * array may contain 'gaps' due to alignment requirement changes between consecutive
 * runs: the data for each run starts at an OutputIndexSize-byte-aligned position (and
 * so does the entire array)
 * @param[in] scatter_position_anchors Reundandt data to make it easier for GPU blocks
 * to figure out which patch runs to apply without going through all run lengths: For
 * every @p anchoring_period patches, this array has one value, being the index of
 * the patch run starting with that multiple of the anchoring period as its starting
 * position. It is guaranteed that run lengths are such that a new run begins at
 * every multiple of the anchoring period, so offsets into runs are not necessary alongside
 * the anchors. This array is also used instead of an per-run indication of which
 * patch data values correspond to the run - as for anchored positions it easy (?) to tell.
 * @param[in] anchoring_period the period at which an index into
 * the scattering runs arrays is provided, to avoid having each block needing
 * to accumulate the run lengths to determine where its relevant runs begin
 * @param[in] num_scatter_position_runs the target positions are grouped in contiguous runs,
 * each of which has a baseline value and offsets (of size that's possibly different than
 * other runs); this is the total number of such runs
 * @param[in] data_length the length of @p data and the number of elements
 * to be modified in @p target
 */
template <unsigned OutputIndexSize, unsigned ElementSize, unsigned InputIndexSize, unsigned RunLengthSize = InputIndexSize>
__global__ void scatter(
	uint_t<ElementSize>*            __restrict__  target,
	const uint_t<ElementSize>*      __restrict__  data,
	const uint_t<RunLengthSize>*    __restrict__  scatter_position_run_lengths,
	const unsigned char*                  __restrict__  scatter_position_run_individual_offset_sizes,
	const uint_t<OutputIndexSize>*  __restrict__  scatter_position_run_baseline_values,
	const uint_t<InputIndexSize>*   __restrict__  scatter_position_run_offsets_start_positions,
	const unsigned char*                  __restrict__  scatter_position_offset_bytes,
	const uint_t<InputIndexSize>*   __restrict__  scatter_position_anchors,
	uint_t<InputIndexSize>                        anchoring_period,
	uint_t<InputIndexSize>                        num_output_position_runs,
	uint_t<InputIndexSize>                        data_length)
{
	// Each block is responsible for scattering anchoring_period input elements
	// into the target area

	auto block_anchor_index = block::index();
	auto next_anchor_index  = block::index() + 1;

	auto start_run_index_for_block = scatter_position_anchors[block_anchor_index];
	auto this_block_reaches_end_of_anchors =
		next_anchor_index * anchoring_period >= data_length;

	uint_t<InputIndexSize> starting_data_pos_for_block = block_anchor_index * anchoring_period;
	// TODO: If we assume we have as many blocks as there are anchors,
	// perhaps we could just condition on the block index.
//	uint_t<InputIndexSize> length_to_scatter_in_this_block =
//		this_block_reaches_end_of_anchors ?
//		data_length - starting_data_pos_for_block : anchoring_period;
	uint_t<InputIndexSize> num_runs_to_use_in_this_block =
		this_block_reaches_end_of_anchors ?
			num_output_position_runs - start_run_index_for_block :
			scatter_position_anchors[next_anchor_index] - start_run_index_for_block;

	// block_printf("Applying %2u patch runs starting from run index %2u", num_runs_to_use_in_this_block, start_run_index_for_block);

	// TODO: Should we cache the run start positions, lengths and widths in
	// shared memory? Probably not if we cache their loads are in L1. Can we do that?
	// Perhaps with ldg or opt-in L1 caching for this kernel. If they're not in L1,
	// it's still not entirely clear whether it's worth caching them

	detail::block::scatter_unanchored<OutputIndexSize, ElementSize, InputIndexSize, RunLengthSize>(
		target,
		data + starting_data_pos_for_block,
		scatter_position_run_baseline_values         + start_run_index_for_block,
		scatter_position_run_individual_offset_sizes + start_run_index_for_block,
		scatter_position_run_lengths                 + start_run_index_for_block,
		scatter_position_run_offsets_start_positions + start_run_index_for_block,
		scatter_position_offset_bytes,
		num_runs_to_use_in_this_block);

}

template <unsigned OutputIndexSize, unsigned ElementSize, unsigned InputIndexSize, unsigned RunLengthSize = InputIndexSize>
class launch_config_resolution_params_t final : public cuda::launch_config_resolution_params_t {
public:
	using parent = cuda::launch_config_resolution_params_t;
public:
	launch_config_resolution_params_t(
		device::properties_t            device_properties,
		size_t                          input_data_length,
		size_t                          anchoring_period
		) :
		cuda::launch_config_resolution_params_t(
			device_properties,
			device_function_t(scatter<OutputIndexSize, ElementSize, InputIndexSize, RunLengthSize>), nullopt
		)
	{
		// The i'th block applies the patches whose index (relative to
		// all patches) starts at the i'th anchors (either until the next
		// anchor or the end of the output). If there are less threads in
		// the block than there are patches to apply, the block's thread
		// will iterate over the patches, threads-per-block patches at a
		// "time".

		grid_construction_resolution            = block;
		length                                  = util::div_rounding_up(input_data_length, anchoring_period);
		quanta.threads_in_block                 = warp_size; // threads will check whether they're excessive
		if (device_properties.max_threads_per_block() > anchoring_period) {
			upper_limits.threads_in_block       = anchoring_period;
		}
		serialization_option                    = none;
	};
};


} // namespace compressed_indices
} // namespace scatter
} // namespace kernels
} // namespace cuda

#endif /* SRC_KERNELS_VFOR_INDEXED_SCATTER_CUH */
