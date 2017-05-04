
#include "common.cuh"

#include "kernel_wrappers/decompression/frame_of_reference.cu"


void decompress_on_device(
	uncompressed_type*              __restrict__  decompressed,
	const compressed_type*          __restrict__  compressed,
	const model_coefficients_type*  __restrict__  segment_model_coefficients,
	size_type                                     length,
	size_type                                     segment_length)
{
	namespace kernel_ns = cuda::kernels::decompression::frame_of_reference;

	kernel_ns::kernel_t<sizeof(index_type), uncompressed_type, compressed_type, model_type> kernel;

	auto current_device = cuda::device::current::get();
	auto device_properties = current_device.properties();
	auto kernel_function_attributes = kernel.get_device_function().attributes();
		// Yes, CUDA has "device properties" and "function attributes", I didn't invent this :-(

	auto launch_config = kernel.resolve_launch_configuration(
		device_properties,
		kernel_function_attributes,
		length,
		segment_length);

	// Note: If you want to enqueue the launch on a stream other than the default,
	// use cuda::stream_t's enqueue.kernel_launch method
	//
	cuda::kernel::enqueue_launch(
		kernel,
		launch_config,
		decompressed,
		compressed,
		segment_model_coefficients,
		length,
		segment_length);
}
