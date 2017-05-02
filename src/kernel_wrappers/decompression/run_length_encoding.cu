
#include "kernel_wrappers/common.h"
#ifdef __CUDACC__
#include "kernels/decompression/run_length_encoding.cuh"
#endif

namespace cuda {
namespace kernels {
namespace decompression {
namespace run_length_encoding {

// TODO: This currently ignores the possibility of a sorted variant of the kernel

template<unsigned UncompressedIndexSize, unsigned UncompressedSize, unsigned RunLengthSize, unsigned RunIndexSize = UncompressedIndexSize>
class kernel_t : public cuda::registered::kernel_t {
public:
	REGISTERED_KERNEL_WRAPPER_BOILERPLATE_DEFINITIONS(kernel_t);

	using run_index_type                = uint_t<RunIndexSize>;
	using uncompressed_index_type       = uint_t<UncompressedIndexSize>;
	using uncompressed_type             = uint_t<UncompressedSize>;
	using run_length_type               = uint_t<RunLengthSize>;

	using uncompressed_index_size_type  = size_type_by_index_size<UncompressedIndexSize>;
	using run_index_size_type           = size_type_by_index_size<RunIndexSize>;
	using size_type                     = uncompressed_index_size_type;

	static_assert(util::is_power_of_2(UncompressedIndexSize), "UncompressedIndexSize is not a power of 2");
	static_assert(util::is_power_of_2(RunIndexSize         ), "RunIndexSize is not a power of 2");
	static_assert(util::is_power_of_2(RunLengthSize        ), "RunLengthSize is not a power of 2");


	launch_configuration_t resolve_launch_configuration(
		device::properties_t           device_properties,
		device_function::attributes_t  kernel_function_attributes,
		size_t                         uncompressed_length,
		size_t                         segment_length,
		launch_configuration_limits_t  limits) const
#ifdef __CUDACC__
	{
		launch_config_resolution_params_t<
			UncompressedIndexSize, UncompressedSize, RunLengthSize, RunIndexSize
		> params(
			device_properties,
			uncompressed_length, segment_length);

		return cuda::kernels::resolve_launch_configuration(params, limits);
	}
#else
	;
#endif
};

#ifdef __CUDACC__

template<unsigned UncompressedIndexSize, unsigned UncompressedSize, unsigned RunLengthSize, unsigned RunIndexSize>
launch_configuration_t kernel_t<UncompressedIndexSize, UncompressedSize, RunLengthSize, RunIndexSize>::resolve_launch_configuration(
	device::properties_t           device_properties,
	device_function::attributes_t  kernel_function_attributes,
	arguments_type                 extra_arguments,
	launch_configuration_limits_t  limits) const
{
	auto uncompressed_length  = any_cast<size_t>(extra_arguments.at("length"        ));
	auto segment_length       = any_cast<size_t>(extra_arguments.at("segment_length"));

	return resolve_launch_configuration(
		device_properties, kernel_function_attributes,
		uncompressed_length, segment_length,
		limits);
}

template<unsigned UncompressedIndexSize, unsigned UncompressedSize, unsigned RunLengthSize, unsigned RunIndexSize>
void kernel_t<UncompressedIndexSize, UncompressedSize, RunLengthSize, RunIndexSize>::enqueue_launch(
	stream::id_t                   stream,
	const launch_configuration_t&  launch_config,
	arguments_type                 arguments) const
{

	auto decompressed               = any_cast<uncompressed_type*       >(arguments.at("decompressed"            ));
	auto run_data                   = any_cast<const uncompressed_type* >(arguments.at("run_data"                ));
	auto run_lengths                = any_cast<const run_length_type*   >(arguments.at("run_lengths"             ));
	auto position_anchors           = any_cast<const run_index_type*    >(arguments.at("position_anchors"        ));
	auto intra_run_anchor_offsets   = any_cast<const run_length_type*   >(arguments.at("intra_run_anchor_offsets"));
	auto segment_length             = any_cast<size_type                >(arguments.at("segment_length"          ));
	auto num_segments               = any_cast<size_type                >(arguments.at("num_segments"            ));
	auto num_element_runs           = any_cast<run_index_size_type      >(arguments.at("num_element_runs"        ));
	auto uncompressed_length        = any_cast<size_type                >(arguments.at("length"                  ));

	cuda::kernel::enqueue_launch(
		*this, stream, launch_config,
		decompressed, run_data, run_lengths, position_anchors, intra_run_anchor_offsets,
		segment_length, num_segments, num_element_runs, uncompressed_length
	);
}

template<unsigned UncompressedIndexSize, unsigned UncompressedSize, unsigned RunLengthSize, unsigned RunIndexSize>
const device_function_t kernel_t<UncompressedIndexSize, UncompressedSize, RunLengthSize, RunIndexSize>::get_device_function() const
{
	return reinterpret_cast<const void*>(
		cuda::kernels::decompression::run_length_encoding::decompress<UncompressedIndexSize, UncompressedSize, RunLengthSize, RunIndexSize>);
}


static_block {
	//         UncompressedIndexSize   UncompressedSize  RunLengthSize
	//----------------------------------------------------------------------
	kernel_t < 4,                      1,                1 >::registerInSubclassFactory();
	kernel_t < 4,                      2,                1 >::registerInSubclassFactory();
	kernel_t < 4,                      4,                1 >::registerInSubclassFactory();
	kernel_t < 4,                      8,                1 >::registerInSubclassFactory();

	kernel_t < 4,                      1,                2 >::registerInSubclassFactory();
	kernel_t < 4,                      2,                2 >::registerInSubclassFactory();
	kernel_t < 4,                      4,                2 >::registerInSubclassFactory();
	kernel_t < 4,                      8,                2 >::registerInSubclassFactory();

	kernel_t < 4,                      1,                4 >::registerInSubclassFactory();
	kernel_t < 4,                      2,                4 >::registerInSubclassFactory();
	kernel_t < 4,                      4,                4 >::registerInSubclassFactory();
	kernel_t < 4,                      8,                4 >::registerInSubclassFactory();

	kernel_t < 8,                      4,                1 >::registerInSubclassFactory();
	kernel_t < 8,                      4,                2 >::registerInSubclassFactory();
	kernel_t < 8,                      4,                4 >::registerInSubclassFactory();

	kernel_t < 8,                      1,                8 >::registerInSubclassFactory();
	kernel_t < 8,                      4,                8 >::registerInSubclassFactory();
	kernel_t < 8,                      8,                8 >::registerInSubclassFactory();
}
#endif /* __CUDACC__ */

} // namespace run_length_encoding
} // namespace decompression
} // namespace kernels
} // namespace cuda

