
#include "kernel_wrappers/common.h"
#ifdef __CUDACC__
#include "kernels/data_layout/gather.cuh"
#endif

namespace cuda {
namespace kernels {
namespace gather {

template <unsigned OutputIndexSize, unsigned ElementSize, unsigned InputIndexSize>
class kernel: public cuda::registered::kernel_t {
public:
	REGISTERED_KERNEL_WRAPPER_BOILERPLATE_DEFINITIONS(kernel);

	using element_type    = util::uint_t<ElementSize>;
	using input_index_type = util::uint_t<InputIndexSize>;
};

#ifdef __CUDACC__

template <unsigned OutputIndexSize, unsigned ElementSize, unsigned InputIndexSize>
launch_configuration_t kernel<OutputIndexSize, ElementSize, InputIndexSize>::resolve_launch_configuration(
	device::properties_t            device_properties,
	device_function::attributes_t   kernel_function_attributes,
	arguments_type                  extra_arguments,
	launch_configuration_limits_t   limits) const
{
	namespace kernel_ns = cuda::kernels::gather;

	auto data_length = any_cast<size_t>(extra_arguments.at("data_length"));
	auto num_indices = any_cast<size_t>(extra_arguments.at("num_indices"));
	optional<bool> maybe_cache_input_data_in_shared_mem =
		any_cast<bool>(maybe_at(extra_arguments, "cache_input_data_in_shared_mem"));
	auto serialization_factor =
		any_cast<serialization_factor_t>(maybe_at(extra_arguments, "serialization_factor"));

	kernel_ns::launch_config_resolution_params_t<OutputIndexSize, ElementSize, InputIndexSize> params(
		device_properties,
		data_length, num_indices, maybe_cache_input_data_in_shared_mem,
		limits.dynamic_shared_memory);

	return cuda::resolve_launch_configuration(params, limits, serialization_factor);
}

template <unsigned OutputIndexSize, unsigned ElementSize, unsigned InputIndexSize>
void kernel<OutputIndexSize, ElementSize, InputIndexSize>::launch(
	stream::id_t                   stream,
	const launch_configuration_t&  launch_config,
	arguments_type                 arguments) const
{
	namespace kernel_ns = cuda::kernels::gather;

	auto reordered_data = any_cast<element_type*           >(arguments.at("reordered_data"));
	auto data           = any_cast<const element_type*     >(arguments.at("data"          ));
	auto indices        = any_cast<const input_index_type* >(arguments.at("indices"       ));
	auto data_length    = any_cast<size_t                  >(arguments.at("data_length"   ));
	auto num_indices    = any_cast<uint_t<OutputIndexSize> >(arguments.at("num_indices"   ));


	// This kernel only uses (dynamic) shared memory if it caches the input data there; so
	// we check the intention both explicitly and implicitly via the shared memory in the
	// launch config; perhaps it's not such a good idea but I don't want to rock this boat
	// right now
	auto caching_input_data_in_shared_mem = (launch_config.dynamic_shared_memory_size > 0);
	auto maybe_serialization_factor =
		any_cast<serialization_factor_t>(maybe_at(arguments, "serialization_factor"));
	auto serialization_factor = maybe_serialization_factor.value_or(
		kernel_ns::get_default_serialization_factor(caching_input_data_in_shared_mem));


	cuda::enqueue_launch(
		kernel_ns::gather<OutputIndexSize, ElementSize, InputIndexSize>,
		launch_config, stream,
		reordered_data, data, indices, data_length, num_indices,
		caching_input_data_in_shared_mem, serialization_factor
	);
}

template <unsigned OutputIndexSize, unsigned ElementSize, unsigned InputIndexSize>
const device_function_t kernel<OutputIndexSize, ElementSize, InputIndexSize>::get_device_function() const
{
	return reinterpret_cast<const void*>(cuda::kernels::gather::gather<OutputIndexSize, ElementSize, InputIndexSize>);
}

static_block {
	//       OutputIndexSize  ElementSize  InputIndexSize
	//------------------------------------------------------------------
	kernel < 4,               1,           1 >::registerInSubclassFactory();
	kernel < 4,               2,           1 >::registerInSubclassFactory();
	kernel < 4,               4,           1 >::registerInSubclassFactory();
	kernel < 4,               8,           1 >::registerInSubclassFactory();

	kernel < 4,               1,           2 >::registerInSubclassFactory();
	kernel < 4,               2,           2 >::registerInSubclassFactory();
	kernel < 4,               4,           2 >::registerInSubclassFactory();
	kernel < 4,               8,           2 >::registerInSubclassFactory();

	kernel < 4,               1,           4 >::registerInSubclassFactory();
	kernel < 4,               2,           4 >::registerInSubclassFactory();
	kernel < 4,               4,           4 >::registerInSubclassFactory();
	kernel < 4,               8,           4 >::registerInSubclassFactory();

	kernel < 4,               4,           8 >::registerInSubclassFactory();

	kernel < 8,               4,           4 >::registerInSubclassFactory();
	kernel < 8,               8,           4 >::registerInSubclassFactory();
	kernel < 8,               4,           8 >::registerInSubclassFactory();
}

#endif /* __CUDACC__ */

} // namespace gather
} // namespace kernels
} // namespace cuda


