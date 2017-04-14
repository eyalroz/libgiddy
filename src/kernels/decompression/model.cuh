#pragma once
#ifndef SRC_KERNELS_MODEL_CUH
#define SRC_KERNELS_MODEL_CUH

#include "cuda/on_device/primitives/block.cuh"
#include "kernels/common.cuh"

/*
 * Model decompression is similar to Frame-of-Reference, except that:
 *
 * 1. The data fits the model function (constant, linear, etc.) perfectly,
 *    so no difference/correction needs to be applied.
 * 2. No anchoring is necessary since no data is used
 *
 * Now, you might be asking yourself: "Why would I need this kind of a
 * decompressor? I work with real-world data, not with synthetically
 * generated evaluations of mathematical functions?"
 *
 * The reason is that with patching, this becomes somewhat useful. For
 * example, suppose your input is mostly 0, or some constant value,
 * but occasionally you have some noise, some garbage data point. No
 * need to spend any bytes on storing the constant over a long series
 * of samples - just keep the noise as patch values.
 *
 * Also, sometimes you _do_ need to work with data which is 'perfect',
 * such as columns which are running indices or contiguous keys -
 * which would fit a linear (or affine, if you will) model perfectly.
 *
 */

namespace cuda {
namespace kernels {
namespace decompression {
namespace model {

using namespace grid_info::linear;

template<unsigned IndexSize, typename Uncompressed, typename UnaryModelFunction>
__global__ void decompress(
	Uncompressed* __restrict__   decompressed,
	typename UnaryModelFunction::coefficients_type
	                             model_coefficients,
	uint_t<IndexSize>            length)
{
	using index_type = uint_t<IndexSize>;
	constexpr const serialization_factor_t serialization_factor = warp_size / sizeof(Uncompressed);
	UnaryModelFunction model(model_coefficients);
	auto f = [&decompressed, &length, &model](index_type pos) {
		// Note: The model might not predict a value of type Uncompressed; for example,
		// it may be a fractional value for predicting an integral value, due to, say
		// a best-line-fit over the data
		decompressed[pos] = static_cast<Uncompressed>(model(pos));
	};
	primitives::grid::linear::at_block_stride(length, f, serialization_factor);
}

template<unsigned IndexSize, typename Uncompressed, typename UnaryModelFunction>
__global__ void decompress_with_intervals(
	Uncompressed* __restrict__   decompressed,
	const typename UnaryModelFunction::coefficients_type* __restrict__
	                             interval_model_coefficients,
	uint_t<IndexSize>            modeling_period,
	uint_t<IndexSize>            intervals_per_block,
	uint_t<IndexSize>            length)
{
	using namespace grid_info::linear;
	using index_type = uint_t<IndexSize>;

	index_type starting_interval_index_for_block = intervals_per_block * block::index();
	index_type interval_start_pos = starting_interval_index_for_block * modeling_period;

	if (intervals_per_block > 1) {
		// we only assign multiple periods to a block if one iteration per thread
		// is enough to cover everything

		auto position = interval_start_pos + thread::index();
		if (position >= length) { return; }
		auto inter_block_interval_index = thread::index() / modeling_period;
		auto interval_index = starting_interval_index_for_block + inter_block_interval_index;
		// Note: We don't use the overall position in the entire input array as the
		// position to which to apply the frame-of-reference model, only the position
		// relative to the current segment (that way modeling is uniform)
		auto position_within_interval = thread::index() - inter_block_interval_index * modeling_period;
		UnaryModelFunction period_model(interval_model_coefficients[interval_index]);
		// Note: The model might not predict a value of type Uncompressed; for example,
		// it may be a fractional value for predicting an integral value, due to, say
		// a best-line-fit over the data
		decompressed[position] =static_cast<Uncompressed>(period_model(position_within_interval));
	}
	else {
		// We cover one interval, of length modeling_period
		UnaryModelFunction period_model(interval_model_coefficients[block::index()]);
		auto f = [&](index_type position_within_interval) {
			auto global_position = interval_start_pos + position_within_interval;
			decompressed[global_position] =
				static_cast<Uncompressed>(period_model(position_within_interval));
		};
		primitives::block::at_block_stride(modeling_period, f);
	}
}

template<unsigned IndexSize, typename Uncompressed, typename UnaryModelFunction>
class launch_config_resolution_params_t final : public kernels::launch_config_resolution_params_t {
public:
	using parent = kernels::launch_config_resolution_params_t;
public:
	launch_config_resolution_params_t(
		device::properties_t            device_properties_,
		size_t                          length_) :
		parent(
			device_properties_,
			device_function_t(decompress<IndexSize, Uncompressed, UnaryModelFunction>)
		)
	{
		grid_construction_resolution            = thread;
		length                                  = length_;
		serialization_option                    = fixed_factor;
		default_serialization_factor            = warp_size / sizeof(Uncompressed);
	};
};


} // namespace model
} // namespace decompression
} // namespace kernels
} // namespace cuda

#endif /* SRC_KERNELS_MODEL_CUH */
