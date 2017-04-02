#pragma once
#ifndef SRC_KERNELS_DATA_LAYOUT_SCATTER_CUH
#define SRC_KERNELS_DATA_LAYOUT_SCATTER_CUH

#include "kernels/common.cuh"

namespace cuda {
namespace kernels {
namespace scatter {

static const serialization_factor_t DefaultSerializationFactor { 16 };

/**
 * @note it's entirely possible that the amount of data to scatter will be much smaller than
 * the range into which it's scattered, hence the different types uint_t<OutputIndexSize> and InputSize
 *
 * @tparam uint_t<OutputIndexSize>
 * @tparam uint_t<ElementSize>
 * @tparam InputSize
 * @tparam SerializationFactor
 * @param[inout] target
 * @param[in] data
 * @param[in] indices
 * @param data_length
 */
template <
	unsigned OutputIndexSize,
	unsigned ElementSize,
	unsigned InputIndexSize,
	serialization_factor_t SerializationFactor = DefaultSerializationFactor
>
__global__ void scatter(
	uint_t<ElementSize>*            __restrict__  target,
	const uint_t<ElementSize>*      __restrict__  data,
	const uint_t<OutputIndexSize>*  __restrict__  indices,
	size_t                                              data_length)
{
	using element_type = uint_t<ElementSize>;
	using input_index_type = uint_t<InputIndexSize>;
	using output_index_type = uint_t<OutputIndexSize>;

	auto f = [&](promoted_size<input_index_type> pos) {
		target[indices[pos]] = data[pos];
	};
	primitives::grid::linear::at_block_stride(data_length, f, SerializationFactor);
}

template <
	unsigned OutputIndexSize,
	unsigned ElementSize,
	unsigned InputIndexSize,
	serialization_factor_t SerializationFactor
>
class launch_config_resolution_params_t final : public cuda::launch_config_resolution_params_t {
public:
	using parent = cuda::launch_config_resolution_params_t;
public:
	launch_config_resolution_params_t(
		device::properties_t            device_properties_,
		size_t                          input_data_length_
		) :
		cuda::launch_config_resolution_params_t(
			device_properties_,
			device_function_t(scatter<OutputIndexSize, ElementSize, InputIndexSize, SerializationFactor>),
			nullopt
		)
	{
		grid_construction_resolution            = thread;
		length                                  = input_data_length_; // == the number of indices
		serialization_option                    = fixed_factor;
		default_serialization_factor            = SerializationFactor;
	};
};

} // namespace scatter
} // namespace kernels
} // namespace cuda

#endif /* SRC_KERNELS_DATA_LAYOUT_SCATTER_CUH */
