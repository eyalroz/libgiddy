
#include "kernel_wrappers/common.h"
#ifdef __CUDACC__
#include "kernels/decompression/run_position_encoding.cuh"
#endif

namespace cuda {
namespace kernels {
namespace decompression {
namespace run_position_encoding {

// TODO: This currently ignores the possibility of a sorted variant of the kernel

template<unsigned UncompressedIndexSize, unsigned UncompressedSize, unsigned PositionOffsetSize, bool PositionsAreRelative, unsigned RunIndexSize = UncompressedIndexSize>
class kernel_t : public cuda::registered::kernel_t {

public:
	REGISTERED_KERNEL_WRAPPER_BOILERPLATE_DEFINITIONS(kernel_t);

	using run_index_type                = uint_t<RunIndexSize>;
	using uncompressed_index_type       = uint_t<UncompressedIndexSize>;
	using position_offset_type          = uint_t<PositionOffsetSize>;
	using uncompressed_type             = uint_t<UncompressedSize>;

	using index_type                    = uncompressed_index_type;
	using uncompressed_index_size_type  = size_type_by_index_size<UncompressedIndexSize>;
	using run_index_size_type           = size_type_by_index_size<RunIndexSize>;
	using position_offset_size_type     = size_type_by_index_size<PositionOffsetSize>;
	using size_type                     = uncompressed_index_size_type;
	using run_length_type               = position_offset_size_type;
		// They're the same type since the length of a run is the difference between
		// between the end and start position offsets

	static_assert(util::is_power_of_2(UncompressedIndexSize), "UncompressedIndexSize is not a power of 2");
	static_assert(util::is_power_of_2(sizeof(position_offset_type)), "sizeof(PositionOffset) is not a power of 2");
	static_assert(PositionsAreRelative or sizeof(position_offset_type) >= UncompressedIndexSize,
		"If run positions are in absolute values, their type must be able to cover the entire "
		"potential range of data (i.e. their type must be at least as large as the size type");


	launch_configuration_t resolve_launch_configuration(
		device::properties_t           device_properties,
		device_function::attributes_t  kernel_function_attributes,
		size_t                         uncompressed_length,
		size_t                         uncompressed_segment_length,
		launch_configuration_limits_t  limits) const
#ifdef __CUDACC__
	{
		launch_config_resolution_params_t<
			UncompressedIndexSize, UncompressedSize, PositionOffsetSize, PositionsAreRelative, RunIndexSize
		> params(
			device_properties,
			uncompressed_length, uncompressed_segment_length);

		return cuda::kernels::resolve_launch_configuration(params, limits);
	}
#else
	;
#endif
};

#ifdef __CUDACC__

template<unsigned UncompressedIndexSize, unsigned UncompressedSize, unsigned PositionOffsetSize, bool PositionsAreRelative, unsigned RunIndexSize>
launch_configuration_t kernel_t<UncompressedIndexSize, UncompressedSize, PositionOffsetSize, PositionsAreRelative, RunIndexSize>::resolve_launch_configuration(
	device::properties_t           device_properties,
	device_function::attributes_t  kernel_function_attributes,
	arguments_type                 extra_arguments,
	launch_configuration_limits_t  limits) const
{
	auto uncompressed_length         = any_cast<size_t>(extra_arguments.at("length"        ));
	auto uncompressed_segment_length = any_cast<size_t>(extra_arguments.at("segment_length"));


	return resolve_launch_configuration(
		device_properties, kernel_function_attributes,
		uncompressed_length, uncompressed_segment_length,
		limits);
}

template<unsigned UncompressedIndexSize, unsigned UncompressedSize, unsigned PositionOffsetSize, bool PositionsAreRelative, unsigned RunIndexSize>
void kernel_t<UncompressedIndexSize, UncompressedSize, PositionOffsetSize, PositionsAreRelative, RunIndexSize>::enqueue_launch(
	stream::id_t                   stream,
	const launch_configuration_t&  launch_config,
	arguments_type                 arguments) const
{
	// TODO: Would it make sense to pass the average run length per anchored segment?
	// That wouldn't take a lot of space after all

	auto decompressed                = any_cast<uncompressed_type*           >(arguments.at("decompressed"               ));
	auto run_data                    = any_cast<const uncompressed_type*     >(arguments.at("run_data"                   ));
	auto run_start_positions         = any_cast<const position_offset_type*  >(arguments.at("run_start_positions"        ));
	auto position_anchors            = any_cast<const run_index_type*        >(arguments.at("position_anchors"           ));
	auto uncompressed_segment_length = any_cast<position_offset_size_type    >(arguments.at("segment_length"             ));
	auto num_anchors                 = any_cast<uncompressed_index_size_type >(arguments.at("num_anchors"                ));
	auto num_element_runs            = any_cast<run_index_size_type          >(arguments.at("num_element_runs"           ));
	auto uncompressed_length         = any_cast<uncompressed_index_size_type >(arguments.at("length"                     ));

	cuda::kernel::enqueue_launch(
		*this, stream, launch_config,
		decompressed, run_data, run_start_positions, position_anchors,
		uncompressed_segment_length, num_anchors, num_element_runs, uncompressed_length
	);

}

template<unsigned UncompressedIndexSize, unsigned UncompressedSize, unsigned PositionOffsetSize, bool PositionsAreRelative, unsigned RunIndexSize>
const device_function_t kernel_t<UncompressedIndexSize, UncompressedSize, PositionOffsetSize, PositionsAreRelative, RunIndexSize>::get_device_function() const
{
	return reinterpret_cast<const void*>(
		cuda::kernels::decompression::run_position_encoding::decompress<
			UncompressedIndexSize, UncompressedSize, PositionOffsetSize, PositionsAreRelative, RunIndexSize
		>);
}


static_block {
	constexpr bool relative = true;
	constexpr bool absolute = false;

	//         UncompressedIndexSize   Datum   RunPosition  PositionsAreRelative
	//-------------------------------------------------------------------------------

	kernel_t < 4,                      1,      2,           relative >::registerInSubclassFactory();
	kernel_t < 4,                      2,      2,           relative >::registerInSubclassFactory();
	kernel_t < 4,                      4,      2,           relative >::registerInSubclassFactory();
	kernel_t < 4,                      8,      2,           relative >::registerInSubclassFactory();
	kernel_t < 8,                      4,      2,           relative >::registerInSubclassFactory();

	// and now the absolutely-positioned

	kernel_t < 4,                      1,      4,           absolute >::registerInSubclassFactory();
	kernel_t < 4,                      2,      4,           absolute >::registerInSubclassFactory();
	kernel_t < 4,                      4,      4,           absolute >::registerInSubclassFactory();
	kernel_t < 4,                      8,      4,           absolute >::registerInSubclassFactory();
	kernel_t < 8,                      4,      8,           absolute >::registerInSubclassFactory();
}
#endif /* __CUDACC__ */

} // namespace run_position_encoding
} // namespace decompression
} // namespace kernels
} // namespace cuda

