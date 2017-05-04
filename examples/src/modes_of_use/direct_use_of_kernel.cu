
#include "common.cuh"

#include "kernels/resolve_launch_configuration.h"
#include "cuda/model_functors.hpp"
#include "kernels/decompression/frame_of_reference.cuh"

void decompress_on_device(
	uncompressed_type*              __restrict__  decompressed,
	const compressed_type*          __restrict__  compressed,
	const model_coefficients_type*  __restrict__  segment_model_coefficients,
	size_type                                     length,
	size_type                                     segment_length)
{
	namespace kernel_ns = cuda::kernels::decompression::frame_of_reference;

	auto device_properties = cuda::device::current::get().properties();
	kernel_ns::launch_config_resolution_params_t
		<sizeof(index_type), uncompressed_type, compressed_type, model_type> params(
			device_properties, length, segment_length);
	auto launch_config = cuda::kernels::resolve_launch_configuration(params);

	auto kernel_function =
		cuda::kernels::decompression::frame_of_reference::decompress
				<sizeof(index_type), uncompressed_type, compressed_type, model_type>;
	auto num_blocks = launch_config.grid_dimensions.x;
	auto num_segments = div_rounding_up(length, segment_length);
	auto segments_per_block = util::div_rounding_up(num_segments, num_blocks);

	cuda::launch(kernel_function, launch_config,
		decompressed,
		compressed,
		segment_model_coefficients,
		length,
		segment_length,
		segments_per_block
	);
}
