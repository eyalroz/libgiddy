
#include "kernel_wrappers/common.h"
#include "util/endianness.h"
#ifdef __CUDACC__
#include "kernels/decompression/discard_zero_bytes_variable.cuh"
#endif

namespace cuda {
namespace kernels {
namespace decompression {
namespace discard_zero_bytes {
namespace variable_width {

using util::endianness_t;

template<unsigned IndexSize, /* util::terminal_t EndToPad, */unsigned UncompressedSize, unsigned ElementSizesContainerSize = sizeof(native_word_t)>
class kernel_t : public cuda::registered::kernel_t {
public:
	REGISTERED_KERNEL_WRAPPER_BOILERPLATE_DEFINITIONS(kernel_t);

	using element_size_t     = unsigned; // code duplication with the kernel file!
	using uncompressed_type  = uint_t<UncompressedSize>;
	using element_sizes_container_size_type
		                     = uint_t<ElementSizesContainerSize>;
	using size_type          = size_type_by_index_size<IndexSize>;

	launch_configuration_t resolve_launch_configuration(
		device::properties_t           device_properties,
		device_function::attributes_t  kernel_function_attributes,
		size_t                         length_in_elements,
		size_t                         segment_length,
		launch_configuration_limits_t  limits) const
#ifdef __CUDACC__
	{
		launch_config_resolution_params_t<
			IndexSize, UncompressedSize, ElementSizesContainerSize
		> params(
			device_properties,
			length_in_elements, segment_length);

		return cuda::kernels::resolve_launch_configuration(params, limits);
	}
#else
	;
#endif

};

#ifdef __CUDACC__

template<unsigned IndexSize, /* util::terminal_t EndToPad, */unsigned UncompressedSize, unsigned ElementSizesContainerSize>
launch_configuration_t kernel_t<IndexSize, UncompressedSize, ElementSizesContainerSize>::resolve_launch_configuration(
	device::properties_t            device_properties,
	device_function::attributes_t kernel_function_attributes,
	arguments_type                 extra_arguments,
	launch_configuration_limits_t  limits) const
{
	auto length         = any_cast<size_t>(extra_arguments.at("length"        ));
		// The overall length, in elements

	auto segment_length = any_cast<size_t>(extra_arguments.at("segment_length"));
	// The number of elements in each anchored segment

	return resolve_launch_configuration(
		device_properties, kernel_function_attributes,
		length, segment_length,
		limits);
}

template<unsigned IndexSize, /* util::terminal_t EndToPad, */unsigned UncompressedSize, unsigned ElementSizesContainerSize>
void kernel_t<IndexSize, UncompressedSize, ElementSizesContainerSize>::enqueue_launch(
	stream::id_t                   stream,
	const launch_configuration_t&  launch_config,
	arguments_type                 arguments) const
{
	using index_type = uint_t<IndexSize>;
	using uncompressed_type = uint_t<UncompressedSize>;
	using element_sizes_container_size_type = uint_t<ElementSizesContainerSize>;

	auto decompressed                   = any_cast<uncompressed_type*   >(arguments.at("decompressed"                 ));
	auto compressed_data                = any_cast<const unsigned char* >(arguments.at("compressed_data"              ));
	auto packed_element_sizes           = any_cast<const element_sizes_container_size_type *
	                                                                    >(arguments.at("packed_element_sizes"         ));
	auto position_anchors               = any_cast<const index_type*    >(arguments.at("position_anchors"             ));
	auto segment_length                 = any_cast<index_type           >(arguments.at("segment_length"               ));
	auto length                         = any_cast<index_type           >(arguments.at("length"                       ));
		// The overall length, in elements
	auto min_represented_element_size   = any_cast<element_size_t       >(arguments.at("min_represented_element_size" ));
	auto bits_per_element_size          = any_cast<unsigned             >(arguments.at("bits_per_element_size"        ));
	cuda::kernel::enqueue_launch(
		*this, stream, launch_config,
		decompressed, compressed_data, packed_element_sizes, position_anchors,
		segment_length, length, min_represented_element_size,
		bits_per_element_size
	);
}

template<unsigned IndexSize, /* util::terminal_t EndToPad, */unsigned UncompressedSize, unsigned ElementSizesContainerSize>
const device_function_t kernel_t<IndexSize, UncompressedSize, ElementSizesContainerSize>::get_device_function() const
{
	return reinterpret_cast<const void*>(cuda::kernels::decompression::discard_zero_bytes::variable_width::decompress
		<IndexSize, UncompressedSize, ElementSizesContainerSize>);
}


static_block {
	//         IndexSize  Uncompressed   ElementSizesContainerSize
	//--------------------------------------------------------------------------------------
	kernel_t < 4,         2,             4 >::registerInSubclassFactory();
	kernel_t < 4,         4,             4 >::registerInSubclassFactory();
	kernel_t < 4,         8,             4 >::registerInSubclassFactory();

	kernel_t < 8,         2,             4 >::registerInSubclassFactory();
	kernel_t < 8,         4,             4 >::registerInSubclassFactory();
	kernel_t < 8,         8,             4 >::registerInSubclassFactory();

	kernel_t < 4,         2,             8 >::registerInSubclassFactory();
	kernel_t < 4,         4,             8 >::registerInSubclassFactory();
	kernel_t < 4,         8,             8 >::registerInSubclassFactory();
}

#endif /* __CUDACC__ */

} // namespace variable_width
} // namespace discard_zero_bytes
} // namespace decompression
} // namespace kernels
} // namespace cuda
