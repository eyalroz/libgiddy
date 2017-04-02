
#include "kernel_wrappers/common.h"
#ifdef __CUDACC__
#include "kernels/enumerated_elementwise/unary_operation.cuh"
#endif

namespace cuda {
namespace kernels {
namespace enumerated_elementwise {
namespace unary {

#ifndef __CUDACC__
enum : serialization_factor_t { DefaultSerializationFactor = 16 };
#endif

template <typename Op, serialization_factor_t SerializationFactor = DefaultSerializationFactor>
class kernel: public cuda::registered::kernel_t {
public:
	REGISTERED_KERNEL_WRAPPER_BOILERPLATE_DEFINITIONS(kernel);

	using input_type   = typename Op::argument_type;
	using index_type   = typename Op::enumerator_type;
	using output_type  = typename Op::result_type;

};

#ifdef __CUDACC__

template <typename Op, serialization_factor_t SerializationFactor>
launch_configuration_t kernel<Op, SerializationFactor>::resolve_launch_configuration(
	device::properties_t           device_properties,
	device_function::attributes_t  kernel_function_attributes,
	arguments_type                 extra_arguments,
	launch_configuration_limits_t  limits) const
{
	namespace kernel_ns = cuda::kernels::enumerated_elementwise::unary;

	auto length = any_cast<size_t>(extra_arguments.at("length"));
	kernel_ns::launch_config_resolution_params_t<Op, SerializationFactor> params(
		device_properties,
		length);

	return cuda::resolve_launch_configuration(params, limits);
}

template <typename Op, serialization_factor_t SerializationFactor>
void kernel<Op, SerializationFactor>::launch(
	stream::id_t                   stream,
	const launch_configuration_t&  launch_config,
	arguments_type                 arguments) const
{
	auto output = any_cast<output_type*      >(arguments.at("output"   ));
	auto input  = any_cast<const input_type* >(arguments.at("input"    ));
	auto length = any_cast<index_type        >(arguments.at("length"   ));

	cuda::enqueue_launch(
			cuda::kernels::enumerated_elementwise::unary::unary_operation<Op, SerializationFactor>,
			launch_config, stream,
			output, input, length
	);
}

template <typename Op, serialization_factor_t SerializationFactor>
const device_function_t kernel<Op, SerializationFactor>::get_device_function() const
{
	return reinterpret_cast<const void*>(cuda::kernels::enumerated_elementwise::unary::unary_operation<Op, SerializationFactor>);
}


static_block {
	using namespace cuda;
	using cuda::functors::enumerated::as_enumerated_unary;
	namespace functors = cuda::functors;

	//       Op
	//                                                   LHS        RHS       Result
	//------------------------------------------------------------------------------------
	kernel < as_enumerated_unary < functors::plus      < uint32_t, int32_t, int32_t > > >::registerInSubclassFactory();
	kernel < as_enumerated_unary < functors::multiplies< uint32_t, float,   float   > > >::registerInSubclassFactory();
	kernel < as_enumerated_unary < functors::plus      < uint32_t, int32_t, int32_t > > >::registerInSubclassFactory();
	kernel < as_enumerated_unary < functors::multiplies< uint64_t, float,   float   > > >::registerInSubclassFactory();
}


#endif /* __CUDACC__ */

} // namespace unary
} // namespace enumerated_elementwise
} // namespace kernels
} // namespace cuda
