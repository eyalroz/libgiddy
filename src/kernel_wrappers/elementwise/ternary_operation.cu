
#include "kernel_wrappers/common.h"
#ifdef __CUDACC__
#include "kernels/elementwise/ternary_operation.cuh"
#endif

namespace cuda {
namespace kernels {
namespace elementwise {
namespace ternary {

#ifndef __CUDACC__
enum : serialization_factor_t { DefaultSerializationFactor = 16 };
#endif

template <unsigned IndexSize, typename Op, serialization_factor_t SerializationFactor = DefaultSerializationFactor>
class kernel: public cuda::registered::kernel_t {
public:
	REGISTERED_KERNEL_WRAPPER_BOILERPLATE_DEFINITIONS(kernel);

	using result_type          = typename Op::result_type;
	using first_argument_type  = typename Op::first_argument_type;
	using second_argument_type = typename Op::second_argument_type;
	using third_argument_type  = typename Op::third_argument_type;
};

#ifdef __CUDACC__

template<unsigned IndexSize, typename Op, serialization_factor_t SerializationFactor>
launch_configuration_t kernel<IndexSize, Op, SerializationFactor>::resolve_launch_configuration(
	device::properties_t             device_properties,
	device_function::attributes_t  kernel_function_attributes,
	arguments_type                   extra_arguments,
	launch_configuration_limits_t    limits) const
{
	namespace kernel_ns = cuda::kernels::elementwise::ternary;

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
	cuda::enqueue_launch(
			cuda::kernels::elementwise::ternary::ternary_operation<IndexSize, Op, SerializationFactor>,
			launch_config, stream,
			any_cast<result_type*               >(arguments.at("result"          )),
			any_cast<const first_argument_type* >(arguments.at("first_argument"  )),
			any_cast<const second_argument_type*>(arguments.at("second_argument" )),
			any_cast<const third_argument_type* >(arguments.at("third_argument"  )),
			any_cast<util::uint_t<IndexSize>    >(arguments.at("length"          ))
	);
}

template <unsigned IndexSize, typename Op, serialization_factor_t SerializationFactor>
const device_function_t kernel<IndexSize, Op, SerializationFactor>::get_device_function() const
{
	return reinterpret_cast<const void*>(cuda::kernels::elementwise::ternary::ternary_operation<IndexSize, Op, SerializationFactor>);
}

static_block {
	namespace functors = cuda::functors;

	//       IndexSize  TernaryOperation
	//---------------------------------------------------------------------
	kernel < 4,         functors::clip<int>                       >::registerInSubclassFactory();
	kernel < 4,         functors::clip<float>                     >::registerInSubclassFactory();
	kernel < 4,         functors::between_or_equal_ternary<float> >::registerInSubclassFactory();
	kernel < 4,         functors::if_then_else<float>             >::registerInSubclassFactory();
	kernel < 8,         functors::clip<float>                     >::registerInSubclassFactory();
}
#endif /* __CUDACC__ */

} // namespace ternary
} // namespace elementwise
} // namespace kernels
} // namespace cuda
