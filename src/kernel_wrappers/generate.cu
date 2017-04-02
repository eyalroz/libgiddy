
#include "common.h"
#ifdef __CUDACC__
#include "kernels/generate.cuh"
#endif

#include <ratio>

namespace cuda {
namespace kernels {
namespace generate {

#ifndef __CUDACC__
enum : serialization_factor_t { DefaultSerializationFactor = 16 };
#endif

template<unsigned IndexSize, typename UnaryFunction, serialization_factor_t SerializationFactor = DefaultSerializationFactor>
class kernel: public cuda::registered::kernel_t {
public:
	REGISTERED_KERNEL_WRAPPER_BOILERPLATE_DEFINITIONS(kernel);

	using Datum             = typename UnaryFunction::result_type;
};

#ifdef __CUDACC__

template<unsigned IndexSize, typename UnaryFunction, serialization_factor_t SerializationFactor>
launch_configuration_t kernel<IndexSize, UnaryFunction, SerializationFactor>::resolve_launch_configuration(
	device::properties_t             device_properties,
	device_function::attributes_t  kernel_function_attributes,
	arguments_type                   extra_arguments,
	launch_configuration_limits_t    limits) const
{
	namespace kernel_ns = cuda::kernels::generate;

	auto length = any_cast<size_t>(extra_arguments.at("length"));
	kernel_ns::launch_config_resolution_params_t<IndexSize, UnaryFunction, SerializationFactor> params(
		device_properties,
		length);

	return cuda::resolve_launch_configuration(params, limits);
}

template<unsigned IndexSize, typename UnaryFunction, serialization_factor_t SerializationFactor>
void kernel<IndexSize, UnaryFunction, SerializationFactor>::launch(
	stream::id_t                   stream,
	const launch_configuration_t&  launch_config,
	arguments_type                 arguments) const
{
	using index_type = uint_t<IndexSize>;
	cuda::enqueue_launch(
		cuda::kernels::generate::generate<IndexSize, UnaryFunction, SerializationFactor>,
		launch_config, stream,
		any_cast<Datum*      >(arguments.at("result")),
		any_cast<index_type  >(arguments.at("length"))
	);
}

template<unsigned IndexSize, typename UnaryFunction, serialization_factor_t SerializationFactor>
const device_function_t kernel<IndexSize, UnaryFunction, SerializationFactor>::get_device_function() const
{
	return reinterpret_cast<const void*>(cuda::kernels::generate::generate<IndexSize, UnaryFunction, SerializationFactor>);
}

static_block {
	using half = functors::constant_by_ratio<float, std::ratio<1,2>>;
	using third = functors::constant_by_ratio<float, std::ratio<1,3>>;

	//       IndexSize  UnaryFunction
	//-----------------------------------------------------------------------------------------
	kernel < 4,         functors::identity<int8_t>                               >::registerInSubclassFactory();
	kernel < 4,         functors::identity<uint8_t>                              >::registerInSubclassFactory();
	kernel < 4,         functors::identity<int32_t>                              >::registerInSubclassFactory();
	kernel < 4,         functors::constant_by_ratio<float, std::ratio<1, 3> >    >::registerInSubclassFactory();
	kernel < 4,         functors::affine_transform_nullaries<float, third, half> >::registerInSubclassFactory();
	kernel < 4,         functors::fixed_modulus<uint32_t, 7>                     >::registerInSubclassFactory();
	kernel < 4,         functors::constant<uint8_t, 0>                           >::registerInSubclassFactory();
	kernel < 4,         functors::constant<int32_t, 1>                           >::registerInSubclassFactory();
	kernel < 4,         functors::affine_transform<int32_t, 0, 0>                >::registerInSubclassFactory();
	kernel < 4,         functors::affine_transform<int32_t, 0, 123>              >::registerInSubclassFactory();
	kernel < 8,         functors::identity<int32_t>                              >::registerInSubclassFactory();
	kernel < 8,         functors::constant_by_ratio<float, std::ratio<1, 3>>     >::registerInSubclassFactory();
	kernel < 8,         functors::affine_transform_nullaries<float, third, half> >::registerInSubclassFactory();
	kernel < 8,         functors::fixed_modulus<uint32_t, 7>                     >::registerInSubclassFactory();
	kernel < 8,         functors::identity<int8_t>                               >::registerInSubclassFactory();
	kernel < 8,         functors::identity<uint8_t>                              >::registerInSubclassFactory();
	kernel < 8,         functors::constant<uint8_t, 0>                           >::registerInSubclassFactory();
	kernel < 8,         functors::constant<int32_t, 1>                           >::registerInSubclassFactory();
	kernel < 8,         functors::affine_transform<int32_t, 0, 0>                >::registerInSubclassFactory();
	kernel < 8,         functors::affine_transform<int32_t, 0, 123>              >::registerInSubclassFactory();
}
#endif /* __CUDACC__ */

} // namespace generate
} // namespace kernels
} // namespace cuda
