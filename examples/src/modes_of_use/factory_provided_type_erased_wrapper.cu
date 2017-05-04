
#include "common.cuh"

#include "kernel_wrappers/registered_wrapper.h"

#include <iostream>

void decompress_on_device(
	uncompressed_type*              __restrict__  decompressed,
	const compressed_type*          __restrict__  compressed,
	const model_coefficients_type*  __restrict__  segment_model_coefficients,
	size_type                                     length,
	size_type                                     segment_length)
{
	auto kernel_name =
		"decompression::frame_of_reference::kernel_t<4u, int, short, cuda::functors::unary::parametric_model::constant<4u, int> >";
	auto type_erased_kernel = cuda::registered::kernel_t::produceSubclass(kernel_name);

	auto current_device = cuda::device::current::get();
	auto device_properties = current_device.properties();
	auto kernel_function_attributes = type_erased_kernel->get_device_function().attributes();
		// Yes, CUDA has "device properties" and "function attributes", I didn't invent this :-(

	cuda::registered::kernel_t::arguments_type extra_config_resolution_arguments = {
		{ "length",         (size_t) length         },
		{ "segment_length", (size_t) segment_length },
	};

	auto launch_config = type_erased_kernel->resolve_launch_configuration(
		device_properties,
		kernel_function_attributes,
		extra_config_resolution_arguments );

	cuda::registered::kernel_t::arguments_type extra_kernel_launch_arguments = {
		{ "decompressed",               decompressed                },
		{ "compressed_input",           compressed                  },
		{ "segment_model_coefficients", segment_model_coefficients  },
		{ "length",                     length                      },
		{ "segment_length",             segment_length              },
	};

	type_erased_kernel->enqueue_launch(
		cuda::stream::default_stream_id,
		launch_config,
		extra_kernel_launch_arguments);
}
