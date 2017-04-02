
#include "kernel_wrappers/common.h"
#ifdef __CUDACC__
#include "kernels/decompression/incidence_bitmaps.cuh"
#endif

namespace cuda {
namespace kernels {
namespace decompression {
namespace incidence_bitmaps {

using bit_container_t = unsigned;
using bitmap_index_t  = unsigned char;

// TODO: This currently ignores the possibility of a sorted variant of the kernel

template<unsigned IndexSize, unsigned UncompressedSize, unsigned BitmapAlignmentInInts, bool UseDictionary = true>
class kernel: public cuda::registered::kernel_t {
public:
	using uncompressed_type = util::uint_t<UncompressedSize>;
	using bitmap_index_type = unsigned char;

	REGISTERED_KERNEL_WRAPPER_BOILERPLATE_DEFINITIONS(kernel);
};

#ifdef __CUDACC__
template<unsigned IndexSize, unsigned UncompressedSize, unsigned BitmapAlignmentInInts, bool UseDictionary>
launch_configuration_t kernel<IndexSize, UncompressedSize, BitmapAlignmentInInts, UseDictionary>::resolve_launch_configuration(
	device::properties_t            device_properties,
	device_function::attributes_t kernel_function_attributes,
	arguments_type                 extra_arguments,
	launch_configuration_limits_t  limits) const
{
	namespace kernel_ns = cuda::kernels::decompression::incidence_bitmaps;

	auto length = any_cast<size_t>(extra_arguments.at("length"));
	kernel_ns::launch_config_resolution_params_t<IndexSize, UncompressedSize, BitmapAlignmentInInts, UseDictionary> params(
		device_properties,
		length);

	return cuda::resolve_launch_configuration(params, limits);
}

template<unsigned IndexSize, unsigned UncompressedSize, unsigned BitmapAlignmentInInts, bool UseDictionary>
void kernel<IndexSize, UncompressedSize, BitmapAlignmentInInts, UseDictionary>::launch(
	stream::id_t                   stream,
	const launch_configuration_t&  launch_config,
	arguments_type                 arguments) const
{
	using uncompressed_type = util::uint_t<UncompressedSize>;
	using bitmap_index_type = unsigned char;

	auto decompressed       = any_cast<uncompressed_type*       >(arguments.at("decompressed"       ));
	auto incidence_bitmaps  = any_cast<const bit_container_t*   >(arguments.at("incidence_bitmaps"  ));
	auto dictionary_entries = any_cast<const uncompressed_type* >(arguments.at("dictionary_entries" ));
	auto bitmap_length      = any_cast<util::uint_t<IndexSize>  >(arguments.at("bitmap_length"      ));
	auto num_bitmaps        = any_cast<bitmap_index_type        >(arguments.at("num_bitmaps"        ));

	cuda::enqueue_launch(
		cuda::kernels::decompression::incidence_bitmaps
		::decompress<IndexSize, UncompressedSize, BitmapAlignmentInInts, UseDictionary>,
		launch_config, stream,
		decompressed, incidence_bitmaps, dictionary_entries, bitmap_length, num_bitmaps
	);
}

template<unsigned IndexSize, unsigned UncompressedSize, unsigned BitmapAlignmentInInts, bool UseDictionary>
const device_function_t kernel<IndexSize, UncompressedSize, BitmapAlignmentInInts, UseDictionary>::get_device_function() const
{
	return reinterpret_cast<const void*>(cuda::kernels::decompression::incidence_bitmaps
		::decompress<IndexSize, UncompressedSize, BitmapAlignmentInInts, UseDictionary>);
}


static_block {
	//                 IndexSize      UncompressedSize  BitmapAlignmentInInts  UseDictionary
	//----------------------------------------------------------------------------------
	kernel < 4,        1,                2,                     true  >::registerInSubclassFactory();
	kernel < 4,        2,                2,                     true  >::registerInSubclassFactory();
	kernel < 4,        4,                2,                     true  >::registerInSubclassFactory();
	kernel < 4,        8,                2,                     true  >::registerInSubclassFactory();
	kernel < 4,        1,                2,                     false >::registerInSubclassFactory();
}

#endif /* __CUDACC__ */

} // namespace incidence_bitmaps
} // namespace decompression
} // namespace kernels
} // namespace cuda
