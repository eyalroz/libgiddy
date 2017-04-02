
#include "kernel_wrappers/common.h"
#ifdef __CUDACC__
#include "kernels/data_layout/select_by_dense_subset.cuh"
#endif

namespace cuda {
namespace kernels {
namespace select_by_dense_subset {

template<unsigned IndexSize, unsigned ElementSize>
class kernel: public cuda::registered::kernel_t {
public:
	REGISTERED_KERNEL_WRAPPER_BOILERPLATE_DEFINITIONS(kernel);

	using index_type = util::uint_t<IndexSize>;
	using element_type = util::uint_t<ElementSize>;
	using bit_container_type = typename cuda::bit_vector<index_type>::container_type;
};

#ifdef __CUDACC__

template<unsigned IndexSize, unsigned ElementSize>
launch_configuration_t kernel<IndexSize, ElementSize>::resolve_launch_configuration(
	device::properties_t           device_properties,
	device_function::attributes_t  kernel_function_attributes,
	arguments_type                 extra_arguments,
	launch_configuration_limits_t  limits) const
{
	namespace kernel_ns = cuda::kernels::select_by_dense_subset;

	auto length = any_cast<size_t>(extra_arguments.at("length"));
	kernel_ns::launch_config_resolution_params_t<IndexSize, ElementSize> params(
		device_properties,
		length,
		limits.dynamic_shared_memory);
	auto serialization_factor =
		any_cast<serialization_factor_t>(maybe_at(extra_arguments, "serialization_factor"));

	return cuda::resolve_launch_configuration(params, limits, serialization_factor);
}


template<unsigned IndexSize, unsigned ElementSize>
void kernel<IndexSize, ElementSize>::launch(
	stream::id_t                   stream,
	const launch_configuration_t&  launch_config,
	arguments_type                 arguments) const
{
	using index_type = util::uint_t<IndexSize>;
	using element_type = util::uint_t<ElementSize>;

	auto selected     = any_cast<element_type*             >(arguments.at("selected"    ));
	auto num_selected = any_cast<index_type*               >(arguments.at("num_selected"));
	auto input_data   = any_cast<const element_type*       >(arguments.at("input_data"  ));
	auto raw_dense    = any_cast<const bit_container_type* >(arguments.at("raw_dense"   ));
	auto domain_size  = any_cast<index_type                >(arguments.at("domain_size" ));
	auto per_warp_shared_mem_elements =
		                any_cast<shared_memory_size_t>(arguments.at("per_warp_shared_mem_elements"   ));
	cuda::enqueue_launch(
		cuda::kernels::select_by_dense_subset::select_by_dense_subset<IndexSize, ElementSize>,
		launch_config, stream,
		selected, num_selected, input_data, raw_dense, domain_size, per_warp_shared_mem_elements
	);
}

template<unsigned IndexSize, unsigned ElementSize>
const device_function_t kernel<IndexSize, ElementSize>::get_device_function() const
{
	return reinterpret_cast<const void*>(cuda::kernels::select_by_dense_subset::select_by_dense_subset<IndexSize, ElementSize>);
}

static_block {
	//       IndexSize  ElementSize
	//-----------------------------------------------------
	kernel < 4,         1 >::registerInSubclassFactory();
	kernel < 4,         2 >::registerInSubclassFactory();
	kernel < 4,         4 >::registerInSubclassFactory();
	kernel < 8,         1 >::registerInSubclassFactory();
	kernel < 8,         2 >::registerInSubclassFactory();
	kernel < 8,         4 >::registerInSubclassFactory();
}
#endif /* __CUDACC__ */

} //namespace dense_to_sparse
} // namespace kernels
} // namespace cuda

