
#include "kernel_wrappers/common.h"
#ifdef __CUDACC__
#include "kernels/data_layout/gather_bits.cuh"
#endif

namespace cuda {
namespace kernels {
namespace gather_bits {

template<unsigned OutputIndexSize, unsigned InputIndexSize>
class kernel: public cuda::registered::kernel_t {
public:
	REGISTERED_KERNEL_WRAPPER_BOILERPLATE_DEFINITIONS(kernel);
};

#ifdef __CUDACC__

template<unsigned OutputIndexSize, unsigned InputIndexSize>
launch_configuration_t kernel<OutputIndexSize, InputIndexSize>::resolve_launch_configuration(
	device::properties_t           device_properties,
	device_function::attributes_t  kernel_function_attributes,
	arguments_type                 extra_arguments,
	launch_configuration_limits_t  limits) const
{
	namespace kernel_ns = cuda::kernels::gather_bits;

	auto num_input_bits        = any_cast<size_t>(extra_arguments.at("num_input_bits"));
	auto num_bit_indices       = any_cast<size_t>(extra_arguments.at("num_bit_indices"));
	auto serialization_factor  =
		any_cast<serialization_factor_t>(maybe_at(extra_arguments, "serialization_factor"));

	kernel_ns::launch_config_resolution_params_t<OutputIndexSize, InputIndexSize> params(
		device_properties,
		num_input_bits, num_bit_indices,
		limits.dynamic_shared_memory);

	return cuda::resolve_launch_configuration(params, limits, serialization_factor);
}

template<unsigned OutputIndexSize, unsigned InputIndexSize>
void kernel<OutputIndexSize, InputIndexSize>::launch(
	stream::id_t                   stream,
	const launch_configuration_t&  launch_config,
	arguments_type                 arguments) const
{
	auto maybe_cache_dictionary_in_shared_mem =
		any_cast<bool>(maybe_at(arguments, "cache_input_data_in_shared_mem"));
	// This kernel only uses (dynamic) shared memory if it caches the input data there; so
	// we check the intention both explicitly and implicitly via the shared memory in the
	// launch config; perhaps it's not such a good idea but I don't want to rock this boat
	// right now
	auto caching_input_data_in_shared_mem =
		maybe_cache_dictionary_in_shared_mem.value_or(launch_config.dynamic_shared_memory_size > 0);
	auto maybe_serialization_factor =
		any_cast<unsigned short>(maybe_at(arguments, "serialization_factor"));

	using input_index_type = util::uint_t<InputIndexSize>;
	using output_index_type = util::uint_t<OutputIndexSize>;
	using bit_container_type = typename cuda::bit_vector<output_index_type>::container_type;

	auto gathered_bits   = any_cast<bit_container_type*       >(arguments.at("gathered_bits"  ));
	auto input_bits      = any_cast<const bit_container_type* >(arguments.at("input_bits"     ));
	auto indices         = any_cast<const input_index_type*   >(arguments.at("indices"        ));
	auto num_input_bits  = any_cast<input_index_type          >(arguments.at("num_input_bits" ));
	auto num_bit_indices = any_cast<output_index_type         >(arguments.at("num_bit_indices"));

	cuda::enqueue_launch(
		cuda::kernels::gather_bits::gather_bits<OutputIndexSize, InputIndexSize>,
		launch_config, stream,
		gathered_bits, input_bits, indices, num_input_bits, num_bit_indices
//		, caching_input_data_in_shared_mem
	);
}

template<unsigned OutputIndexSize, unsigned InputIndexSize>
const device_function_t kernel<OutputIndexSize, InputIndexSize>::get_device_function() const
{
	return reinterpret_cast<const void*>(cuda::kernels::gather_bits::gather_bits<OutputIndexSize, InputIndexSize>);
}

static_block {
	//       OutputIndexSize  InputIndexSize
	//------------------------------------------------------------------
	kernel < 4,               1  >::registerInSubclassFactory();
	kernel < 4,               2  >::registerInSubclassFactory();
	kernel < 4,               4  >::registerInSubclassFactory();
	kernel < 4,               8  >::registerInSubclassFactory();
	kernel < 8,               1  >::registerInSubclassFactory();
	kernel < 8,               2  >::registerInSubclassFactory();
	kernel < 8,               4  >::registerInSubclassFactory();
	kernel < 8,               8  >::registerInSubclassFactory();
}

#endif /* __CUDACC__ */

} // namespace gather_bits
} // namespace kernels
} // namespace cuda


