
#include "kernel_wrappers/common.h"
#include "kernels/data_layout/set_representation/common.h" // for the sortedness type
#ifdef __CUDACC__
#include "kernels/data_layout/set_representation/sparse_to_dense.cuh"
#endif

namespace cuda {
namespace kernels {
namespace set_representation {
namespace sparse_to_dense {

// TODO: This currently ignores the possibility of a sorted variant of the kernel

template <sortedness_t Sortedness, unsigned IndexSize>
class kernel: public cuda::registered::kernel_t {
public:
	REGISTERED_KERNEL_WRAPPER_BOILERPLATE_DEFINITIONS(kernel);

	using index_type = util::uint_t<IndexSize>;
	using bit_container_type = typename ::cuda::bit_vector<index_type>::container_type;
};

#ifdef __CUDACC__

template <sortedness_t Sortedness, unsigned IndexSize>
launch_configuration_t kernel<Sortedness, IndexSize>::resolve_launch_configuration(
	device::properties_t            device_properties,
	device_function::attributes_t kernel_function_attributes,
	arguments_type                 extra_arguments,
	launch_configuration_limits_t  limits) const

{
	namespace kernel_ns = cuda::kernels::set_representation::sparse_to_dense;

	auto sparse_length = any_cast<size_t>(extra_arguments.at("sparse_length"));
	kernel_ns::launch_config_resolution_params_t<Sortedness, IndexSize> params(
		device_properties,
		sparse_length);

	return cuda::resolve_launch_configuration(params, limits);
}


template <sortedness_t Sortedness, unsigned IndexSize>
void kernel<Sortedness, IndexSize>::launch(
	stream::id_t                   stream,
	const launch_configuration_t&  launch_config,
	arguments_type                 arguments) const
{
	auto raw_dense     = any_cast<bit_container_type*>(arguments.at("raw_dense"    ));
	auto sparse        = any_cast<const index_type*        >(arguments.at("sparse"       ));
	auto sparse_length = any_cast<index_type         >(arguments.at("sparse_length"));

	cuda::enqueue_launch(
		cuda::kernels::set_representation::sparse_to_dense::sparse_to_dense<Sortedness, IndexSize>,
		launch_config, stream,
		raw_dense, sparse, sparse_length
	);
}

template <sortedness_t Sortedness, unsigned IndexSize>
const device_function_t kernel<Sortedness, IndexSize>::get_device_function() const
{
	return reinterpret_cast<const void*>(cuda::kernels::set_representation::sparse_to_dense::sparse_to_dense<Sortedness, IndexSize>);
}

static_block {
	//       Sortedness              IndexSize
	//-----------------------------------------------------
	kernel < sortedness_t::Sorted,   4 >::registerInSubclassFactory();
	kernel < sortedness_t::Sorted,   8 >::registerInSubclassFactory();
	kernel < sortedness_t::Unsorted, 4 >::registerInSubclassFactory();
	kernel < sortedness_t::Unsorted, 8 >::registerInSubclassFactory();
}
#endif


} // namespace sparse_to_dense
} // namespace set_representation
} // namespace kernels
} // namespace cuda
