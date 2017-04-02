#pragma once
#ifndef SRC_KERNELS_DECOMPRESSION_DELTA_CUH
#define SRC_KERNELS_DECOMPRESSION_DELTA_CUH

/*
 * Decompressing DELTA-compressed data is essentially performing an inclusive
 * prefix sum (scan with binary operator +) starting from some baseline :
 * discrete determinate integration being the opposite of discrete derivation.
 * However, we do not want to be in a situation where the reference value is
 * so far away, and the number of elements to scan so high, as to require a
 * two-phase operation, like in a general-case scan. For this reason the
 * decompression is assisted by baseline values every (run-time) fixed period;
 * that is, we give it the results of something like the first phase of a
 * prefix sum, for free.
 */

#include "kernels/common.cuh"
#include "cuda/functors.hpp"
#include "kernels/reduction/scan.cuh"
#include "util/math.hpp"
#include "cuda/on_device/primitives/grid.cuh"

namespace cuda {
namespace kernels {
namespace decompression {
namespace delta {

using namespace grid_info::linear;

/**
 * The elements in @p compressed_input are differences between the original,
 * uncompressed elements, to be reproduced in @p decompressed. There's also
 * an initial baseline value at @p baseline_values[0] - which is essentially
 * enough to perform the decompression. However, to break the long chains of
 * data dependencies (and to allow for easier chunking, and to avoid the need
 * for two-phased decompression) - the kernel is provided with a baseline
 * value for every consecutive @p baselining_period (i.e. with the additional
 * partial sums of elements 0...baselining_period-1, 0...2*baselining_period-1
 * etc.)
 *
 * Theoretically we could still want to use a two-phased kernel for large
 * baselining periods, but that's not implemented right now.
 *
 * TODO: Template this kernel on the single segment scan parameter, so that
 * the kernel wrapper can benefit from a power-of-two-length block and from
 * nicely-aligning baselining periods
 *
 * @param decompressed the decompression results
 * @param compressed_input the differences between consecutive elements of
 * the uncompressed input
 * @param baseline_values the actual, non-delta, baseline value is provided
 * for different points within the data - within this array, which has a value
 * for each @p baselining_period consecutive elements
 * @param length the total length of the input the kernel is to decompress
 * @param baselining_period the actual, non-delta, baseline value is provided
 * for different points within the data; this is the difference between
 * consecutive such points
 * @parem num_segments_per_block the number of elements to be decompressed by each
 * block (except perhaps for the last one in the grid) - in units of
 * the {@p baselining_period}.
 *
 */
template<unsigned IndexSize, unsigned UncompressedSize, unsigned CompressedSize>
__global__ void decompress(
	uint_t<UncompressedSize>*        __restrict__  decompressed,
	const uint_t<CompressedSize>*    __restrict__  compressed_input,
	const uint_t<UncompressedSize>*  __restrict__  baseline_values,
	uint_t<IndexSize>            length,
	uint_t<IndexSize>            baselining_period)
{
	using compressed_type = uint_t<CompressedSize>;
	using uncompressed_type = uint_t<UncompressedSize>;
	using index_type = uint_t<IndexSize>;
	auto scratch_area = shared_memory_proxy<uncompressed_type>();
	auto periods_per_block = div_rounding_up(block::length(), baselining_period); // could very well be 1 !
	auto length_of_full_scan_segment = baselining_period * periods_per_block;

	for(index_type period_index = block::index();
		period_index * length_of_full_scan_segment < length;
		period_index += grid::num_blocks())
	{
		index_type        segment_start_position = period_index * length_of_full_scan_segment;
		index_type        segment_length         = min(length - segment_start_position, length_of_full_scan_segment);
		uncompressed_type baseline_value         = baseline_values[period_index];

		reduction::scan::detail::scan_segment
			<IndexSize, functors::plus<uncompressed_type>, compressed_type,
			reduction::scan::inclusivity_t::Inclusive,
			reduction::scan::detail::InitialChunkIsntBlockAligned,
			reduction::scan::detail::FinalChunkIsntBlockAligned,
			reduction::scan::detail::BlockLengthIsAPowerOfTwo,
			functors::identity<uncompressed_type>>(
				decompressed,
				scratch_area,
				compressed_input,
				segment_length,
				segment_start_position,
				baseline_value);
	}
}

template<unsigned IndexSize, unsigned UncompressedSize, unsigned CompressedSize>
class launch_config_resolution_params_t final : public cuda::launch_config_resolution_params_t {
public:
	using parent = cuda::launch_config_resolution_params_t;
public:
	launch_config_resolution_params_t(
		device::properties_t            device_properties_,
		size_t                          data_length,
		size_t                          baselining_period,
		optional<shared_memory_size_t>  dynamic_shared_mem_limit = nullopt) :
		cuda::launch_config_resolution_params_t(
			device_properties_,
			device_function_t(decompress<IndexSize, UncompressedSize, CompressedSize>)
		)
	{
		auto num_segments = util::div_rounding_up(data_length, baselining_period);

		grid_construction_resolution            = block;
		serialization_option                    = auto_maximized;
		dynamic_shared_memory_requirement.per_block =
			warp_size * UncompressedSize; // for the "single-segment scan"
		length                                  = num_segments;
		keep_gpu_busy_factor                    = 32; // for now, chosen pretty arbitrarily

		// This is only due to the fact that we have not yet implemented
		// the single-segment-prefix-sum primitive for blocks whose length is
		// not a power of 2; there's nothing inherent to delta decompression or
		// our implementation of it that requires this
		must_have_power_of_2_threads_per_block  = true;

		if (length == 1) {
			block_resolution_constraints.max_threads_per_block
			                                    = std::max<grid_block_dimension_t>(data_length, warp_size);
		}
	};
};

} // namespace delta
} // namespace decompression
} // namespace kernels
} // namespace cuda

#endif /* SRC_KERNELS_DECOMPRESSION_DELTA_CUH */
