
#include "kernel_wrappers/common.h"
#ifdef __CUDACC__
#include "kernels/data_layout/scatter.cuh"
#endif

namespace cuda {
namespace kernels {
namespace scatter {

#ifndef __CUDACC__
enum : serialization_factor_t { DefaultSerializationFactor =  16} ;
#endif

template<unsigned OutputIndexSize, unsigned ElementSize, unsigned InputIndexSize, serialization_factor_t SerializationFactor = DefaultSerializationFactor>
class kernel: public cuda::registered::kernel_t {
public:
	REGISTERED_KERNEL_WRAPPER_BOILERPLATE_DEFINITIONS(kernel);

	using element_type = util::uint_t<ElementSize>;
	using input_index_type = util::uint_t<InputIndexSize>;
	using output_index_type = util::uint_t<OutputIndexSize>;
};

#ifdef __CUDACC__

template<unsigned OutputIndexSize, unsigned ElementSize, unsigned InputIndexSize, serialization_factor_t SerializationFactor>
launch_configuration_t kernel<OutputIndexSize, ElementSize, InputIndexSize, SerializationFactor>::resolve_launch_configuration(
	device::properties_t            device_properties,
	device_function::attributes_t   kernel_function_attributes,
	arguments_type                  extra_arguments,
	launch_configuration_limits_t   limits) const
{
	namespace kernel_ns = cuda::kernels::scatter;

	auto data_length = any_cast<size_t>(extra_arguments.at("data_length"));
	kernel_ns::launch_config_resolution_params_t<OutputIndexSize, ElementSize, InputIndexSize, SerializationFactor> params(
		device_properties,
		data_length);

	return cuda::kernels::resolve_launch_configuration(params, limits);
}


template<unsigned OutputIndexSize, unsigned ElementSize, unsigned InputIndexSize, serialization_factor_t SerializationFactor>
void kernel<OutputIndexSize, ElementSize, InputIndexSize, SerializationFactor>::enqueue_launch(
	stream::id_t                   stream,
	const launch_configuration_t&  launch_config,
	arguments_type                 arguments) const
{
	auto target       = any_cast<element_type*            >(arguments.at("target"        ));
	auto data         = any_cast<const element_type*      >(arguments.at("data"          ));
	auto indices      = any_cast<const output_index_type* >(arguments.at("indices"       ));
	auto data_length  = any_cast<size_t                   >(arguments.at("data_length"   ));


	cuda::kernel::enqueue_launch(
		*this, stream, launch_config,
		target, data, indices, data_length
	);
}

template<unsigned OutputIndexSize, unsigned ElementSize, unsigned InputIndexSize, serialization_factor_t SerializationFactor>
const device_function_t kernel<OutputIndexSize, ElementSize, InputIndexSize, SerializationFactor>::get_device_function() const
{
	return reinterpret_cast<const void*>(cuda::kernels::scatter::scatter<OutputIndexSize, ElementSize, InputIndexSize, SerializationFactor>);
}


static_block {
	//       OutputIndexSize   ElementSize  InputIndexSize
	//------------------------------------------------------------------------------
	kernel < 4,                1,           1        >::registerInSubclassFactory();
	kernel < 4,                1,           2        >::registerInSubclassFactory();
	kernel < 4,                1,           4        >::registerInSubclassFactory();
	kernel < 4,                1,           8        >::registerInSubclassFactory();
	kernel < 4,                2,           1        >::registerInSubclassFactory();
	kernel < 4,                2,           2        >::registerInSubclassFactory();
	kernel < 4,                2,           4        >::registerInSubclassFactory();
	kernel < 4,                2,           8        >::registerInSubclassFactory();
	kernel < 4,                4,           1        >::registerInSubclassFactory();
	kernel < 4,                4,           2        >::registerInSubclassFactory();
	kernel < 4,                4,           4        >::registerInSubclassFactory();
	kernel < 4,                4,           8        >::registerInSubclassFactory();
	kernel < 4,                8,           1        >::registerInSubclassFactory();
	kernel < 4,                8,           2        >::registerInSubclassFactory();
	kernel < 4,                8,           4        >::registerInSubclassFactory();
	kernel < 4,                8,           8        >::registerInSubclassFactory();

	kernel < 8,                1,           1        >::registerInSubclassFactory();
	kernel < 8,                1,           2        >::registerInSubclassFactory();
	kernel < 8,                1,           4        >::registerInSubclassFactory();
	kernel < 8,                1,           8        >::registerInSubclassFactory();
	kernel < 8,                2,           1        >::registerInSubclassFactory();
	kernel < 8,                2,           2        >::registerInSubclassFactory();
	kernel < 8,                2,           4        >::registerInSubclassFactory();
	kernel < 8,                2,           8        >::registerInSubclassFactory();
	kernel < 8,                4,           1        >::registerInSubclassFactory();
	kernel < 8,                4,           2        >::registerInSubclassFactory();
	kernel < 8,                4,           4        >::registerInSubclassFactory();
	kernel < 8,                4,           8        >::registerInSubclassFactory();
	kernel < 8,                8,           1        >::registerInSubclassFactory();
	kernel < 8,                8,           2        >::registerInSubclassFactory();
	kernel < 8,                8,           4        >::registerInSubclassFactory();
	kernel < 8,                8,           8        >::registerInSubclassFactory();
}


#endif /* __CUDACC__ */

} // namespace scatter
} // namespace kernels
} // namespace cuda
