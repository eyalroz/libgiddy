
#include "kernel_wrappers/common.h"
#ifdef __CUDACC__
#include "kernels/decompression/run_position_encoding.cuh"
#endif

namespace cuda {
namespace kernels {
namespace decompression {
namespace run_position_encoding {

// TODO: This currently ignores the possibility of a sorted variant of the kernel

template<unsigned IndexSize, unsigned UncompressedSize, unsigned PositionOffsetSize, bool PositionsAreRelative>
class kernel: public cuda::registered::kernel_t {

public:
	REGISTERED_KERNEL_WRAPPER_BOILERPLATE_DEFINITIONS(kernel);

	using uncompressed_type    = util::uint_t<UncompressedSize>;
	using position_offset_type = util::uint_t<PositionOffsetSize>;

	static_assert(util::is_power_of_2(IndexSize), "IndexSize is not a power of 2");
	static_assert(util::is_power_of_2(sizeof(position_offset_type)), "sizeof(PositionOffset) is not a power of 2");
	static_assert(PositionsAreRelative or sizeof(position_offset_type) >= IndexSize,
		"If run positions are in absolute values, their type must be able to cover the entire "
		"potential range of data (i.e. their type must be at least as large as the size type");
};

#ifdef __CUDACC__

template<unsigned IndexSize, unsigned UncompressedSize, unsigned PositionOffsetSize, bool PositionsAreRelative>
launch_configuration_t kernel<IndexSize, UncompressedSize, PositionOffsetSize, PositionsAreRelative>::resolve_launch_configuration(
	device::properties_t           device_properties,
	device_function::attributes_t  kernel_function_attributes,
	arguments_type                 extra_arguments,
	launch_configuration_limits_t  limits) const
{
	namespace kernel_ns = cuda::kernels::decompression::run_position_encoding;

	auto uncompressed_length       = any_cast<size_t>(extra_arguments.at("uncompressed_length"));
	auto position_anchoring_period = any_cast<size_t>(extra_arguments.at("position_anchoring_period"));
	kernel_ns::launch_config_resolution_params_t<IndexSize, UncompressedSize, PositionOffsetSize, PositionsAreRelative> params(
		device_properties,
		uncompressed_length, position_anchoring_period);

	return cuda::kernels::resolve_launch_configuration(params, limits);
}

template<unsigned IndexSize, unsigned UncompressedSize, unsigned PositionOffsetSize, bool PositionsAreRelative>
void kernel<IndexSize, UncompressedSize, PositionOffsetSize, PositionsAreRelative>::enqueue_launch(
	stream::id_t                   stream,
	const launch_configuration_t&  launch_config,
	arguments_type                 arguments) const
{
	using index_type           = util::uint_t<IndexSize>;
	using uncompressed_type    = util::uint_t<UncompressedSize>;
	using position_offset_type = util::uint_t<PositionOffsetSize>;

	// TODO: Would it make sense to pass the average run length per anchored segment?
	// That wouldn't take a lot of space after all

	auto decompressed               = any_cast<uncompressed_type*          >(arguments.at("decompressed"               ));
	auto run_data                   = any_cast<const uncompressed_type*    >(arguments.at("run_data"                   ));
	auto run_start_positions        = any_cast<const position_offset_type* >(arguments.at("run_start_positions"        ));
	auto position_anchors           = any_cast<const index_type*           >(arguments.at("position_anchors"           ));
	auto position_anchoring_period  = any_cast<position_offset_type        >(arguments.at("position_anchoring_period"  ));
	auto num_anchors                = any_cast<util::uint_t<IndexSize>     >(arguments.at("num_anchors"                ));
	auto num_element_runs           = any_cast<util::uint_t<IndexSize>     >(arguments.at("num_element_runs"           ));
	auto uncompressed_length        = any_cast<util::uint_t<IndexSize>     >(arguments.at("uncompressed_length"        ));

	cuda::kernel::enqueue_launch(
		*this, stream, launch_config,
		decompressed, run_data, run_start_positions, position_anchors,
		position_anchoring_period, num_anchors, num_element_runs, uncompressed_length
	);

}

template<unsigned IndexSize, unsigned UncompressedSize, unsigned PositionOffsetSize, bool PositionsAreRelative>
const device_function_t kernel<IndexSize, UncompressedSize, PositionOffsetSize, PositionsAreRelative>::get_device_function() const
{
	return reinterpret_cast<const void*>(
		cuda::kernels::decompression::run_position_encoding::decompress<
			IndexSize, UncompressedSize, PositionOffsetSize, PositionsAreRelative
		>);
}


static_block {
	constexpr bool relative = true;
	constexpr bool absolute = false;

	//       IndexSize   Datum   RunPosition  PositionsAreRelative
	//-------------------------------------------------------------------------------

	kernel < 4,          1,      2,           relative >::registerInSubclassFactory();
	kernel < 4,          2,      2,           relative >::registerInSubclassFactory();
	kernel < 4,          4,      2,           relative >::registerInSubclassFactory();
	kernel < 4,          8,      2,           relative >::registerInSubclassFactory();

	kernel < 8,          4,      2,           relative >::registerInSubclassFactory();

	// and now the absolutely-positioned

	kernel < 4,          1,      4,           absolute >::registerInSubclassFactory();
	kernel < 4,          2,      4,           absolute >::registerInSubclassFactory();
	kernel < 4,          4,      4,           absolute >::registerInSubclassFactory();
	kernel < 4,          8,      4,           absolute >::registerInSubclassFactory();

	kernel < 8,          4,      8,           absolute >::registerInSubclassFactory();
}
#endif /* __CUDACC__ */

} // namespace run_position_encoding
} // namespace decompression
} // namespace kernels
} // namespace cuda

