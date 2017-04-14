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
 * With this variant of the kernel we allow it to change once every fixed
 * period (@p modeling period) of elements; one could generalize this further
 * by also encoding a changing length of period.
 *
 * @note This code is nearly an exact duplicate of the codel for model::decompress,
 * with the only difference being the addition of offsets in the compressed input;
 * one can conceive some templatization to reflect that and avoid duplication,
 * but it would be somewhat ugly. Perhaps when we replace columns with more abstract
 * iterators we could unify the code more easily
 *
 * @param decompressed
 * @param compressed_input
 * @param reference_values
 * @param length
 * @param model_function
 * @param framing_period
 */
template<
	unsigned IndexSize, typename Uncompressed,
	typename Compressed, typename UnaryModelFunction>
__global__ void decompress(
	Uncompressed*       __restrict__  decompressed,
	const Compressed*   __restrict__  compressed_input,
	const typename UnaryModelFunction::coefficients_type*
	                    __restrict__  interval_model_coefficients,
	uint_t<IndexSize>                 length,
	uint_t<IndexSize>                 modeling_period,
	uint_t<IndexSize>                 intervals_per_block)
{
	using namespace grid_info::linear;
	using index_type = uint_t<IndexSize>;

	// We could theoretically switch between different kernels rather than check this condition.
	if (intervals_per_block > 1) {
		// Note: This code path could use more testing
		uint_t<IndexSize> starting_interval_index_for_block = intervals_per_block * block::index();
		uint_t<IndexSize> block_starting_position = starting_interval_index_for_block * modeling_period;
		auto position = block_starting_position + thread::index();
		if (position >= length) { return; }
		auto inter_block_interval_index = thread::index() / modeling_period;
		auto global_interval_index = starting_interval_index_for_block + inter_block_interval_index;
		// Note: We don't use the overall position in the entire input array as the
		// position to which to apply the frame-of-reference model, only the position
		// relative to the current segment (that way modeling is uniform)
		auto position_within_interval = thread::index() - inter_block_interval_index * modeling_period;
		UnaryModelFunction period_model(interval_model_coefficients[global_interval_index]);
		// Note: The model might not predict a value of type Uncompressed; for example,
		// it may be a fractional value for predicting an integral value, due to, say
		// a best-line-fit over the data
		Uncompressed predicted_value = static_cast<Uncompressed>(period_model(position_within_interval));
		decompressed[position] = predicted_value + compressed_input[position];
	}
	else {
		// We might need multiple iterations per thread to cover the whole interval
		uint_t<IndexSize> starting_position_for_block = block::index() * modeling_period;
		UnaryModelFunction period_model(interval_model_coefficients[block::index()]);
		auto f = [&](index_type position_within_interval) {
			auto global_position = starting_position_for_block + position_within_interval;
			Uncompressed predicted_value = static_cast<Uncompressed>(period_model(position_within_interval));
			decompressed[global_position] = predicted_value + compressed_input[global_position];
		};
		primitives::block::at_block_stride(modeling_period, f);
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
		size_t                          modeling_period) :
		parent(
			device_properties_,
			device_function_t(decompress<IndexSize, Uncompressed, Compressed, UnaryModelFunction>)
		)
	{
		auto num_periods = util::div_rounding_up(data_length, modeling_period);
		auto periods_per_block = device_properties.max_threads_per_block() / modeling_period;

		grid_construction_resolution            = block;
		serialization_option                    = none;
		length                                  = util::div_rounding_up(num_periods, periods_per_block);
		if (length == 1) {
			block_resolution_constraints.fixed_threads_per_block
			                                    = data_length;
		}
		else {
			quanta.threads_in_block             = periods_per_block > 1 ? modeling_period : 1;
			block_resolution_constraints.max_threads_per_block
			                                    = periods_per_block * modeling_period;
		}
	};
};

} // namespace frame_of_reference
} // namespace decompression
} // namespace kernels
} // namespace cuda

#endif /* SRC_KERNELS_DECOMPRESSION_FRAME_OF_REFERENCE_CUH */
