
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
class kernel_t : public cuda::registered::kernel_t {
public:
	REGISTERED_KERNEL_WRAPPER_BOILERPLATE_DEFINITIONS(kernel_t);

	using element_type = uint_t<ElementSize>;
	using input_index_type = uint_t<InputIndexSize>;
	using input_size_type   = size_type_by_index_size<InputIndexSize>;
	using output_index_type = uint_t<OutputIndexSize>;
	using output_size_type  = size_type_by_index_size<OutputIndexSize>;

	launch_configuration_t resolve_launch_configuration(
		device::properties_t           device_properties,
		device_function::attributes_t  kernel_function_attributes,
		size_t                         data_length,
		launch_configuration_limits_t  limits) const
#ifdef __CUDACC__
	{
		launch_config_resolution_params_t<
			OutputIndexSize, ElementSize, InputIndexSize, SerializationFactor
		> params(
			device_properties,
			data_length);

		return cuda::kernels::resolve_launch_configuration(params, limits, SerializationFactor);
	}
#else
	;
#endif
};

#ifdef __CUDACC__

template<unsigned OutputIndexSize, unsigned ElementSize, unsigned InputIndexSize, serialization_factor_t SerializationFactor>
launch_configuration_t kernel_t<OutputIndexSize, ElementSize, InputIndexSize, SerializationFactor>::resolve_launch_configuration(
	device::properties_t            device_properties,
	device_function::attributes_t   kernel_function_attributes,
	arguments_type                  extra_arguments,
	launch_configuration_limits_t   limits) const
{
	auto data_length = any_cast<size_t>(extra_arguments.at("data_length"));

	return resolve_launch_configuration(
		device_properties, kernel_function_attributes,
		data_length,
		limits);
}


template<unsigned OutputIndexSize, unsigned ElementSize, unsigned InputIndexSize, serialization_factor_t SerializationFactor>
void kernel_t<OutputIndexSize, ElementSize, InputIndexSize, SerializationFactor>::enqueue_launch(
	stream::id_t                   stream,
	const launch_configuration_t&  launch_config,
	arguments_type                 arguments) const
{
	auto target       = any_cast<element_type*            >(arguments.at("target"        ));
	auto data         = any_cast<const element_type*      >(arguments.at("data"          ));
	auto indices      = any_cast<const output_index_type* >(arguments.at("indices"       ));
	auto data_length  = any_cast<input_size_type          >(arguments.at("data_length"   ));


	cuda::kernel::enqueue_launch(
		*this, stream, launch_config,
		target, data, indices, data_length
	);
}

template<unsigned OutputIndexSize, unsigned ElementSize, unsigned InputIndexSize, serialization_factor_t SerializationFactor>
const device_function_t kernel_t<OutputIndexSize, ElementSize, InputIndexSize, SerializationFactor>::get_device_function() const
{
	return reinterpret_cast<const void*>(cuda::kernels::scatter::scatter<OutputIndexSize, ElementSize, InputIndexSize, SerializationFactor>);
}


static_block {
	//         OutputIndexSize   ElementSize  InputIndexSize
	//------------------------------------------------------------------------------
	kernel_t < 4,                1,           1        >::registerInSubclassFactory();
	kernel_t < 4,                1,           2        >::registerInSubclassFactory();
	kernel_t < 4,                1,           4        >::registerInSubclassFactory();
	kernel_t < 4,                1,           8        >::registerInSubclassFactory();
	kernel_t < 4,                2,           1        >::registerInSubclassFactory();
	kernel_t < 4,                2,           2        >::registerInSubclassFactory();
	kernel_t < 4,                2,           4        >::registerInSubclassFactory();
	kernel_t < 4,                2,           8        >::registerInSubclassFactory();
	kernel_t < 4,                4,           1        >::registerInSubclassFactory();
	kernel_t < 4,                4,           2        >::registerInSubclassFactory();
	kernel_t < 4,                4,           4        >::registerInSubclassFactory();
	kernel_t < 4,                4,           8        >::registerInSubclassFactory();
	kernel_t < 4,                8,           1        >::registerInSubclassFactory();
	kernel_t < 4,                8,           2        >::registerInSubclassFactory();
	kernel_t < 4,                8,           4        >::registerInSubclassFactory();
	kernel_t < 4,                8,           8        >::registerInSubclassFactory();

	kernel_t < 8,                1,           1        >::registerInSubclassFactory();
	kernel_t < 8,                1,           2        >::registerInSubclassFactory();
	kernel_t < 8,                1,           4        >::registerInSubclassFactory();
	kernel_t < 8,                1,           8        >::registerInSubclassFactory();
	kernel_t < 8,                2,           1        >::registerInSubclassFactory();
	kernel_t < 8,                2,           2        >::registerInSubclassFactory();
	kernel_t < 8,                2,           4        >::registerInSubclassFactory();
	kernel_t < 8,                2,           8        >::registerInSubclassFactory();
	kernel_t < 8,                4,           1        >::registerInSubclassFactory();
	kernel_t < 8,                4,           2        >::registerInSubclassFactory();
	kernel_t < 8,                4,           4        >::registerInSubclassFactory();
	kernel_t < 8,                4,           8        >::registerInSubclassFactory();
	kernel_t < 8,                8,           1        >::registerInSubclassFactory();
	kernel_t < 8,                8,           2        >::registerInSubclassFactory();
	kernel_t < 8,                8,           4        >::registerInSubclassFactory();
	kernel_t < 8,                8,           8        >::registerInSubclassFactory();
}


#endif /* __CUDACC__ */

} // namespace scatter
} // namespace kernels
} // namespace cuda
