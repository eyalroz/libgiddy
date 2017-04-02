
#include "kernel_wrappers/common.h"
#ifdef __CUDACC__
#include "kernels/decompression/dictionary.cuh"
#endif

namespace cuda {
namespace kernels {
namespace decompression {
namespace dictionary {

// TODO: This currently ignores the possibility of a sorted variant of the kernel

template<unsigned IndexSize, unsigned UncompressedSize, unsigned DictionaryIndexSize>
class kernel: public cuda::registered::kernel_t {
public:
	REGISTERED_KERNEL_WRAPPER_BOILERPLATE_DEFINITIONS(kernel);

	using dictionary_index_type = util::uint_t<DictionaryIndexSize>;
	using uncompressed_type = util::uint_t<UncompressedSize>;
};

#ifdef __CUDACC__

template<unsigned IndexSize, unsigned UncompressedSize, unsigned DictionaryIndexSize>
launch_configuration_t kernel<IndexSize, UncompressedSize, DictionaryIndexSize>::resolve_launch_configuration(
	device::properties_t           device_properties,
	device_function::attributes_t  kernel_function_attributes,
	arguments_type                 extra_arguments,
	launch_configuration_limits_t  limits) const
{
	namespace kernel_ns = cuda::kernels::decompression::dictionary;

	auto num_dictionary_entries = any_cast<size_t>(extra_arguments.at("num_dictionary_entries"));
	auto data_length            = any_cast<size_t>(extra_arguments.at("data_length"));
	optional<bool> maybe_cache_input_data_in_shared_mem =
		any_cast<bool>(maybe_at(extra_arguments, "cache_input_data_in_shared_mem"));
	auto serialization_factor =
		any_cast<serialization_factor_t>(maybe_at(extra_arguments, "serialization_factor"));

	kernel_ns::launch_config_resolution_params_t<IndexSize, UncompressedSize, DictionaryIndexSize> params(
		device_properties,
		num_dictionary_entries, data_length, maybe_cache_input_data_in_shared_mem,
		limits.dynamic_shared_memory);

	return cuda::resolve_launch_configuration(params, limits, serialization_factor);
}

template<unsigned IndexSize, unsigned UncompressedSize, unsigned DictionaryIndexSize>
void kernel<IndexSize, UncompressedSize, DictionaryIndexSize>::launch(
	stream::id_t                   stream,
	const launch_configuration_t&  launch_config,
	arguments_type                 arguments) const
{
	using dictionary_index_type = util::uint_t<DictionaryIndexSize>;
	using uncompressed_type = util::uint_t<UncompressedSize>;

	auto maybe_cache_data_in_shared_mem =
		any_cast<bool>(maybe_at(arguments, "cache_dictionary_in_shared_mem"));
	auto caching_dictionary_in_shared_mem =
		maybe_cache_data_in_shared_mem.value_or(launch_config.dynamic_shared_memory_size > 0);
	auto maybe_serialization_factor =
		any_cast<serialization_factor_t>(maybe_at(arguments, "serialization_factor"));
	auto serialization_factor = maybe_serialization_factor.value_or(
		cuda::kernels::gather::get_default_serialization_factor(caching_dictionary_in_shared_mem));

	auto decompressed            = any_cast<uncompressed_type*           >(arguments.at("decompressed"           ));
	auto compressed_input        = any_cast<const dictionary_index_type* >(arguments.at("compressed_input"       ));
	auto dictionary_entries      = any_cast<const uncompressed_type*     >(arguments.at("dictionary_entries"     ));
	auto length                  = any_cast<util::uint_t<IndexSize>      >(arguments.at("length"                 ));
	auto num_dictionary_entries  = any_cast<size_t                       >(arguments.at("num_dictionary_entries" ));

	cuda::enqueue_launch(
		cuda::kernels::decompression::dictionary::decompress
		<IndexSize, UncompressedSize, DictionaryIndexSize>,
		launch_config, stream,
		decompressed,
		compressed_input,
		dictionary_entries, // the "input data length" w.r.t. the Gather kernel
		length, // the "number of indices" w.r.t. the Gather kernel
		num_dictionary_entries,
		caching_dictionary_in_shared_mem, serialization_factor
	);
}

template<unsigned IndexSize, unsigned UncompressedSize, unsigned DictionaryIndexSize>
const device_function_t kernel<IndexSize, UncompressedSize, DictionaryIndexSize>::get_device_function() const
{
	return reinterpret_cast<const void*>(cuda::kernels::decompression::dictionary::decompress
		<IndexSize, UncompressedSize, DictionaryIndexSize>);
}

static_block {
	//       IndexSize   UncompressedSize  DictionaryIndexSize
	//------------------------------------------------------------------------
    kernel < 4,          1,                1 >::registerInSubclassFactory();

    kernel < 4,          2,                1 >::registerInSubclassFactory();
    kernel < 4,          2,                2 >::registerInSubclassFactory();

    kernel < 4,          4,                1 >::registerInSubclassFactory();
    kernel < 4,          4,                2 >::registerInSubclassFactory();
    kernel < 4,          4,                4 >::registerInSubclassFactory();

    kernel < 4,          8,                1 >::registerInSubclassFactory();
    kernel < 4,          8,                2 >::registerInSubclassFactory();
    kernel < 4,          8,                4 >::registerInSubclassFactory();
    kernel < 4,          8,                8 >::registerInSubclassFactory();

    kernel < 8,          1,                1 >::registerInSubclassFactory();

    kernel < 8,          2,                1 >::registerInSubclassFactory();
    kernel < 8,          2,                2 >::registerInSubclassFactory();

    kernel < 8,          4,                1 >::registerInSubclassFactory();
    kernel < 8,          4,                2 >::registerInSubclassFactory();
    kernel < 8,          4,                4 >::registerInSubclassFactory();

    kernel < 8,          8,                1 >::registerInSubclassFactory();
    kernel < 8,          8,                2 >::registerInSubclassFactory();
    kernel < 8,          8,                4 >::registerInSubclassFactory();
    kernel < 8,          8,                8 >::registerInSubclassFactory();

}

#endif /* __CUDACC__ */

} // namespace dictionary
} // namespace decompression
} // namespace kernels
} // namespace cuda

