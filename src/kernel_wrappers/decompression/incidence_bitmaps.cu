
#include "kernel_wrappers/common.h"
#ifdef __CUDACC__
#include "kernels/decompression/incidence_bitmaps.cuh"
#endif

namespace cuda {
namespace kernels {
namespace decompression {
namespace incidence_bitmaps {


// TODO: This currently ignores the possibility of a sorted variant of the kernel

template<unsigned IndexSize, unsigned UncompressedSize, unsigned BitmapAlignmentInInts, bool UseDictionary = true>
class kernel_t : public cuda::registered::kernel_t {
public:
	using bit_container_t   = standard_bit_container_t;
	using uncompressed_type = uint_t<UncompressedSize>;
	using bitmap_index_type = unsigned char; // code duplication with the .cuh file here
	using size_type         = size_type_by_index_size<IndexSize>;
	using num_bitmaps_type  = size_type_by_index_size<sizeof(bitmap_index_type)>;

	REGISTERED_KERNEL_WRAPPER_BOILERPLATE_DEFINITIONS(kernel_t);

	launch_configuration_t resolve_launch_configuration(
		device::properties_t           device_properties,
		device_function::attributes_t  kernel_function_attributes,
		size_t                         length,
		launch_configuration_limits_t  limits) const
#ifdef __CUDACC__
	{
		launch_config_resolution_params_t<
			IndexSize, UncompressedSize, BitmapAlignmentInInts, UseDictionary
		> params(
			device_properties,
			length);

		return cuda::kernels::resolve_launch_configuration(params, limits);
	}
#else
	;
#endif
};

#ifdef __CUDACC__
template<unsigned IndexSize, unsigned UncompressedSize, unsigned BitmapAlignmentInInts, bool UseDictionary>
launch_configuration_t kernel_t<IndexSize, UncompressedSize, BitmapAlignmentInInts, UseDictionary>::resolve_launch_configuration(
	device::properties_t            device_properties,
	device_function::attributes_t kernel_function_attributes,
	arguments_type                 extra_arguments,
	launch_configuration_limits_t  limits) const
{
	auto length = any_cast<size_t>(extra_arguments.at("length"));

	return resolve_launch_configuration(
		device_properties, kernel_function_attributes,
		length,
		limits);
}

template<unsigned IndexSize, unsigned UncompressedSize, unsigned BitmapAlignmentInInts, bool UseDictionary>
void kernel_t<IndexSize, UncompressedSize, BitmapAlignmentInInts, UseDictionary>::enqueue_launch(
	stream::id_t                   stream,
	const launch_configuration_t&  launch_config,
	arguments_type                 arguments) const
{
	using uncompressed_type = uint_t<UncompressedSize>;
	using bitmap_index_type = unsigned char;

	auto decompressed       = any_cast<uncompressed_type*       >(arguments.at("decompressed"       ));
	auto incidence_bitmaps  = any_cast<const bit_container_t*   >(arguments.at("incidence_bitmaps"  ));
	auto dictionary_entries = any_cast<const uncompressed_type* >(arguments.at("dictionary_entries" ));
	auto bitmap_length      = any_cast<size_type                >(arguments.at("bitmap_length"      ));
	auto num_bitmaps        = any_cast<num_bitmaps_type         >(arguments.at("num_bitmaps"        ));

	cuda::kernel::enqueue_launch(
		*this, stream, launch_config,
		decompressed, incidence_bitmaps, dictionary_entries, bitmap_length, num_bitmaps
	);
}

template<unsigned IndexSize, unsigned UncompressedSize, unsigned BitmapAlignmentInInts, bool UseDictionary>
const device_function_t kernel_t<
	IndexSize, UncompressedSize, BitmapAlignmentInInts, UseDictionary
>::get_device_function() const
{
	return reinterpret_cast<const void*>(cuda::kernels::decompression::incidence_bitmaps
		::decompress<IndexSize, UncompressedSize, BitmapAlignmentInInts, UseDictionary>);
}


static_block {
	//             IndexSize  UncompressedSize  BitmapAlignmentInInts  UseDictionary
	//----------------------------------------------------------------------------------
	kernel_t < 4,  1,         2,                true  >::registerInSubclassFactory();
	kernel_t < 4,  2,         2,                true  >::registerInSubclassFactory();
	kernel_t < 4,  4,         2,                true  >::registerInSubclassFactory();
	kernel_t < 4,  8,         2,                true  >::registerInSubclassFactory();
	kernel_t < 4,  1,         2,                false >::registerInSubclassFactory();
}

#endif /* __CUDACC__ */

} // namespace incidence_bitmaps
} // namespace decompression
} // namespace kernels
} // namespace cuda
