
#include "kernel_wrappers/common.h"
#ifdef __CUDACC__
#include "kernels/data_layout/set_representation/dense_to_sparse.cuh"
#endif

namespace cuda {
namespace kernels {
namespace set_representation {
namespace dense_to_sparse {

// TODO: This currently ignores the possibility of a sorted variant of the kernel

template<unsigned IndexSize>
class kernel: public cuda::registered::kernel_t {
public:
	REGISTERED_KERNEL_WRAPPER_BOILERPLATE_DEFINITIONS(kernel);

	using index_type = util::uint_t<IndexSize>;
	using bit_container_type = typename cuda::bit_vector<index_type>::container_type;
};

#ifdef __CUDACC__

template<unsigned IndexSize>
launch_configuration_t kernel<IndexSize>::resolve_launch_configuration(
	device::properties_t           device_properties,
	device_function::attributes_t  kernel_function_attributes,
	arguments_type                 extra_arguments,
	launch_configuration_limits_t  limits) const
{
	namespace kernel_ns = cuda::kernels::set_representation::dense_to_sparse;

	auto input_data_length = any_cast<size_t>(extra_arguments.at("input_data_length"));
	kernel_ns::launch_config_resolution_params_t<IndexSize> params(
		device_properties,
		input_data_length,
		limits.dynamic_shared_memory);

	return cuda::resolve_launch_configuration(params, limits);
}

template<unsigned IndexSize>
void kernel<IndexSize>::launch(
	stream::id_t                   stream,
	const launch_configuration_t&  launch_config,
	arguments_type                 arguments) const
{
	auto sparse        = any_cast<index_type*               >(arguments.at("sparse"        ));
	auto sparse_length = any_cast<index_type*               >(arguments.at("sparse_length" ));
	auto raw_dense     = any_cast<const bit_container_type* >(arguments.at("raw_dense"     ));
	auto domain_size   = any_cast<index_type          >(arguments.at("domain_size"   ));

	cuda::enqueue_launch(
		cuda::kernels::set_representation::dense_to_sparse::dense_to_unsorted_sparse<IndexSize>,
		launch_config, stream,
		sparse, sparse_length, raw_dense, domain_size
	);
}

template<unsigned IndexSize>
const device_function_t kernel<IndexSize>::get_device_function() const
{
	return reinterpret_cast<const void*>(
		cuda::kernels::set_representation::dense_to_sparse::dense_to_unsorted_sparse<IndexSize>);
}

static_block {
	kernel < 4 >::registerInSubclassFactory();
	kernel < 8 >::registerInSubclassFactory();
}
#endif /* __CUDACC__ */

} // namespace dense_to_sparse
} // namespace set_representation
} // namespace kernels
} // namespace cuda
