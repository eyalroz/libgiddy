#pragma once
#ifndef SRC_KERNELS_DECOMPRESSION_FRAME_OF_REFERENCE_CUH
#define SRC_KERNELS_DECOMPRESSION_FRAME_OF_REFERENCE_CUH

#include "kernels/common.cuh"
#include "kernels/elementwise/unary_operation.cuh"
#include "cuda/functors.hpp"
#include "cuda/stl_array.cuh"
#include "cuda/model_functors.hpp"
#include "cuda/on_device/primitives/block.cuh"

namespace cuda {
namespace kernels {
namespace decompression {
namespace frame_of_reference {

enum { DefaultSerializationFactor = elementwise::unary::DefaultSerializationFactor };

/**
 * This kernel reproduces larger decompressed values using a reference value,
 * obtained using some predictive model (often as simple that just using
 * a single constant frame-of-reference value) for the current batch of input elements,
 * and a per-element offset value - the deviation of the actual data from the
 * prediction/reference.
 *
 * @note The frame-of-reference value is not uniform for the entire data.
 * With this variant of the kernel we allow it to change once every certain
 * number (@p segment_length) of elements. The segment length is uniform over
 * the compressed data in an array / DB column; but it is not a compile-time-fixed
 * value.
 *
 * @note This code is nearly an exact duplicate of the code for @ref model::decompress,
 * with the only difference being the addition of offsets in the compressed input.
 */
template<
	unsigned IndexSize, typename Uncompressed,
	typename Compressed, typename UnaryModelFunction>
__global__ void decompress(
	Uncompressed*       __restrict__    decompressed,
	const Compressed*   __restrict__    compressed_input,
	const typename UnaryModelFunction::coefficients_type*
	                    __restrict__    segment_model_coefficients,
	size_type_by_index_size<IndexSize>  length,
	size_type_by_index_size<IndexSize>  segment_length,
	uint_t<IndexSize>                   num_segments_to_decompress_per_block)
{
	using namespace grid_info::linear;

	// We could theoretically switch between different kernels rather than check this condition.
	if (num_segments_to_decompress_per_block > 1) {
		// Note: This code path could use more testing
		auto starting_segment_index_for_block = num_segments_to_decompress_per_block * block::index();
		auto block_starting_position = starting_segment_index_for_block * segment_length;
		auto position = block_starting_position + thread::index();
		if (position >= length) { return; }
		auto inter_block_segment_index = thread::index() / segment_length;
		auto global_segment_index = starting_segment_index_for_block + inter_block_segment_index;
		// Note: We don't use the overall position in the entire input array as the
		// position to which to apply the frame-of-reference model, only the position
		// relative to the current segment (that way modeling is uniform)
		auto position_within_segment = thread::index() - inter_block_segment_index * segment_length;
		UnaryModelFunction segment_model(segment_model_coefficients[global_segment_index]);
		// Note: The model might not predict a value of type Uncompressed; for example,
		// it may be a fractional value for predicting an integral value, due to, say
		// a best-line-fit over the data
		Uncompressed predicted_value = static_cast<Uncompressed>(segment_model(position_within_segment));
		decompressed[position] = predicted_value + compressed_input[position];
	}
	else {
		// We might need multiple iterations per thread to cover the whole segment
		auto starting_position_for_block = block::index() * segment_length;
		UnaryModelFunction segment_model(segment_model_coefficients[block::index()]);
		auto f = [&](decltype(segment_length) position_within_segment) {
			auto global_position = starting_position_for_block + position_within_segment;
			Uncompressed predicted_value = static_cast<Uncompressed>(segment_model(position_within_segment));
			decompressed[global_position] = predicted_value + compressed_input[global_position];
		};
		primitives::block::at_block_stride(segment_length, f);
	}
}


template<
	unsigned IndexSize, typename Uncompressed,
	typename Compressed, typename UnaryModelFunction>
class launch_config_resolution_params_t final : public kernels::launch_config_resolution_params_t {
public:
	using parent = kernels::launch_config_resolution_params_t;
public:
	launch_config_resolution_params_t(
		device::properties_t            device_properties_,
		size_t                          data_length,
		size_t                          segment_length) :
		parent(
			device_properties_,
			device_function_t(decompress<IndexSize, Uncompressed, Compressed, UnaryModelFunction>)
		)
	{
		if (segment_length == 0) {
			throw std::invalid_argument("The segment length must be a positive number");
		}
		if (segment_length % sizeof(native_word_t) != 0) {
			throw std::invalid_argument("The segment length must be aligned with the GPU's native word size");
		}
		auto num_segments = util::div_rounding_up(data_length, segment_length);
		auto segments_per_block = util::div_rounding_up(device_properties.max_threads_per_block(), segment_length);

		grid_construction_resolution            = block;
		serialization_option                    = none;
		length                                  = util::div_rounding_up(num_segments, segments_per_block);
		if (length == 1) {
			block_resolution_constraints.fixed_threads_per_block
			                                    = data_length;
		}
		else {
			quanta.threads_in_block             = segments_per_block > 1 ? segment_length : 1;
			block_resolution_constraints.max_threads_per_block
			                                    = segments_per_block * segment_length;
		}
	};
};

} // namespace frame_of_reference
} // namespace decompression
} // namespace kernels
} // namespace cuda

#endif /* SRC_KERNELS_DECOMPRESSION_FRAME_OF_REFERENCE_CUH */
