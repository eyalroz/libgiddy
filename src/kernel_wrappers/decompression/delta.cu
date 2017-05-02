
#include "kernel_wrappers/common.h"
#ifdef __CUDACC__
#include "kernels/decompression/delta.cuh"
#endif

namespace cuda {
namespace kernels {
namespace decompression {
namespace delta {

// TODO: This currently ignores the possibility of a sorted variant of the kernel

template<unsigned IndexSize, unsigned UncompressedSize, unsigned CompressedSize>
class kernel_t : public cuda::registered::kernel_t {
public:
	REGISTERED_KERNEL_WRAPPER_BOILERPLATE_DEFINITIONS(kernel_t);

	using compressed_type   = uint_t<CompressedSize>;
	using uncompressed_type = uint_t<UncompressedSize>;
	using size_type         = size_type_by_index_size<IndexSize>;

	launch_configuration_t resolve_launch_configuration(
		device::properties_t           device_properties,
		device_function::attributes_t  kernel_function_attributes,
		size_t                         length,
		size_t                         segment_length,
		launch_configuration_limits_t  limits) const
#ifdef __CUDACC__
	{
		launch_config_resolution_params_t<
			IndexSize, UncompressedSize, CompressedSize
		> params(
			device_properties,
			length, segment_length,
			limits.dynamic_shared_memory);

		return cuda::kernels::resolve_launch_configuration(params, limits);
	}
#else
	;
#endif

};

#ifdef __CUDACC__

template<unsigned IndexSize, unsigned UncompressedSize, unsigned CompressedSize>
inline launch_configuration_t kernel_t<IndexSize, UncompressedSize, CompressedSize>::resolve_launch_configuration(
	device::properties_t           device_properties,
	device_function::attributes_t  kernel_function_attributes,
	arguments_type                 extra_arguments,
	launch_configuration_limits_t  limits) const
{
	auto length         = any_cast<size_t>(extra_arguments.at("length"        ));
	auto segment_length = any_cast<size_t>(extra_arguments.at("segment_length"));

	return resolve_launch_configuration(
		device_properties, kernel_function_attributes,
		length, segment_length,
		limits);
}

template<unsigned IndexSize, unsigned UncompressedSize, unsigned CompressedSize>
inline void kernel_t<IndexSize, UncompressedSize, CompressedSize>::enqueue_launch(
		stream::id_t                   stream,
		const launch_configuration_t&  launch_config,
		arguments_type                 arguments) const
{
	using compressed_type   = uint_t<CompressedSize>;
	using uncompressed_type = uint_t<UncompressedSize>;

	auto decompressed      = any_cast<uncompressed_type*       >(arguments.at("decompressed"    ));
	auto compressed_input  = any_cast<const compressed_type*   >(arguments.at("compressed_input"));
	auto anchor_values     = any_cast<const uncompressed_type* >(arguments.at("anchor_values"   ));
	auto length            = any_cast<util::uint_t<IndexSize>  >(arguments.at("length"          ));
	auto segment_length    = any_cast<util::uint_t<IndexSize>  >(arguments.at("segment_length"  ));

	cuda::kernel::enqueue_launch(
		*this, stream, launch_config,
		decompressed, compressed_input, anchor_values, length, segment_length
	);
}

template<unsigned IndexSize, unsigned UncompressedSize, unsigned CompressedSize>
inline const device_function_t kernel_t<IndexSize, UncompressedSize, CompressedSize>::get_device_function() const
{
	return cuda::kernels::decompression::delta::decompress<IndexSize, UncompressedSize, CompressedSize>;
}

static_block {
	//         IndexSize   UncompressedSize   CompressedSize
	// ----------------------------------------------------
	kernel_t < 4,          2,                 1 >::registerInSubclassFactory();
	kernel_t < 4,          4,                 1 >::registerInSubclassFactory();
	kernel_t < 4,          4,                 2 >::registerInSubclassFactory();
	kernel_t < 4,          8,                 1 >::registerInSubclassFactory();
	kernel_t < 4,          8,                 2 >::registerInSubclassFactory();
	kernel_t < 4,          8,                 4 >::registerInSubclassFactory();

	kernel_t < 8,          2,                 1 >::registerInSubclassFactory();
	kernel_t < 8,          4,                 1 >::registerInSubclassFactory();
	kernel_t < 8,          4,                 2 >::registerInSubclassFactory();
	kernel_t < 8,          8,                 1 >::registerInSubclassFactory();
	kernel_t < 8,          8,                 2 >::registerInSubclassFactory();
	kernel_t < 8,          8,                 4 >::registerInSubclassFactory();

}

#endif /* __CUDACC__ */

} // namespace delta
} // namespace decompression
} // namespace kernels
} // namespace cuda
