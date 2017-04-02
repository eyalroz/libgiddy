
#include "kernel_wrappers/common.h"
#ifdef __CUDACC__
#include "kernels/elementwise/unary_operation.cuh"
#endif

namespace cuda {
namespace kernels {
namespace elementwise {
namespace unary {

#ifndef __CUDACC__
enum : serialization_factor_t { DefaultSerializationFactor = 16 };
#endif

template <unsigned IndexSize, typename Op, serialization_factor_t SerializationFactor = DefaultSerializationFactor>
class kernel: public cuda::registered::kernel_t {
public:
	REGISTERED_KERNEL_WRAPPER_BOILERPLATE_DEFINITIONS(kernel);

	using output_type = typename Op::result_type;
	using input_type  = typename Op::argument_type;

};

#ifdef __CUDACC__

template<unsigned IndexSize, typename Op, serialization_factor_t SerializationFactor>
launch_configuration_t kernel<IndexSize, Op, SerializationFactor>::resolve_launch_configuration(
	device::properties_t             device_properties,
	device_function::attributes_t  kernel_function_attributes,
	arguments_type                   extra_arguments,
	launch_configuration_limits_t    limits) const
{
	namespace kernel_ns = cuda::kernels::elementwise::unary;

	auto length = any_cast<size_t>(extra_arguments.at("length"));
	kernel_ns::launch_config_resolution_params_t<IndexSize, Op, SerializationFactor> params(
		device_properties,
		length);

	return cuda::resolve_launch_configuration(params, limits);
}


template <unsigned IndexSize, typename Op, serialization_factor_t SerializationFactor>
void kernel<IndexSize, Op, SerializationFactor>::launch(
	stream::id_t                   stream,
	const launch_configuration_t&  launch_config,
	arguments_type                 arguments) const
{
	using index_type = uint_t<IndexSize>;

	cuda::enqueue_launch(
			cuda::kernels::elementwise::unary::unary_operation<IndexSize, Op, SerializationFactor>,
			launch_config, stream,
			any_cast<output_type*      >(arguments.at("output"   )),
			any_cast<const input_type* >(arguments.at("input"    )),
			any_cast<index_type        >(arguments.at("length"   ))
	);
}

template <unsigned IndexSize, typename Op, serialization_factor_t SerializationFactor>
const device_function_t kernel<IndexSize, Op, SerializationFactor>::get_device_function() const
{
	return reinterpret_cast<const void*>(cuda::kernels::elementwise::unary::unary_operation<IndexSize, Op, SerializationFactor>);
}

template<unsigned IndexSize, typename T>
using copy_kernel = kernel <IndexSize, cuda::functors::identity<T>>;

static_block {
	namespace functors = cuda::functors;
	kernel < 4,         functors::cast<float, int32_t>       >::registerInSubclassFactory();
	kernel < 4,         functors::identity<uint8_t>          >::registerInSubclassFactory();
	kernel < 4,         functors::absolute<float>            >::registerInSubclassFactory();
	kernel < 4,         functors::absolute<int32_t>          >::registerInSubclassFactory();
	kernel < 4,         functors::absolute<int8_t>           >::registerInSubclassFactory();
	kernel < 4,         functors::population_count<uint32_t> >::registerInSubclassFactory();
	kernel < 8,         functors::identity<uint8_t>          >::registerInSubclassFactory();
	kernel < 8,         functors::negate<int32_t>            >::registerInSubclassFactory();
	kernel < 8,         functors::population_count<uint32_t> >::registerInSubclassFactory();

	copy_kernel < 4, uint8_t  >::registerInSubclassFactory();
	copy_kernel < 4, uint16_t >::registerInSubclassFactory();
	copy_kernel < 4, uint32_t >::registerInSubclassFactory();
	copy_kernel < 4, uint64_t >::registerInSubclassFactory();
	copy_kernel < 4, float    >::registerInSubclassFactory();
	copy_kernel < 4, double   >::registerInSubclassFactory();
	copy_kernel < 8, uint8_t  >::registerInSubclassFactory();
	copy_kernel < 8, uint16_t >::registerInSubclassFactory();
	copy_kernel < 8, uint32_t >::registerInSubclassFactory();
	copy_kernel < 8, uint64_t >::registerInSubclassFactory();
	copy_kernel < 8, float    >::registerInSubclassFactory();
	copy_kernel < 8, double   >::registerInSubclassFactory();



}
#endif /* __CUDACC__ */

} // namespace unary
} // namespace elementwise
} // namespace kernels
} // namespace cuda
