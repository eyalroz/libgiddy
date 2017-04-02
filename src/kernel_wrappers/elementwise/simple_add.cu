
#include "kernel_wrappers/common.h"
#ifdef __CUDACC__
#include "kernels/elementwise/simple_add.cuh"
#endif

namespace cuda {
namespace kernels {
namespace elementwise {
namespace simple_add {

template <typename Datum>
class kernel: public cuda::registered::kernel_t {
public:
	REGISTERED_KERNEL_WRAPPER_BOILERPLATE_DEFINITIONS(kernel);

};

#ifdef __CUDACC__

template <typename Datum>
launch_configuration_t kernel<Datum>::resolve_launch_configuration(
	device::properties_t             device_properties,
	device_function::attributes_t  kernel_function_attributes,
	arguments_type                   extra_arguments,
	launch_configuration_limits_t    limits) const
{
	namespace kernel_ns = cuda::kernels::elementwise::simple_add;

	auto length = any_cast<size_t>(extra_arguments.at("length"));
	kernel_ns::launch_config_resolution_params_t<Datum> params(
		device_properties,
		length);

	return cuda::resolve_launch_configuration(params, limits);
}

template <typename Datum>
void kernel<Datum>::launch(
	stream::id_t                   stream,
	const launch_configuration_t&  launch_config,
	arguments_type                 arguments) const
{
	cuda::enqueue_launch(
			cuda::kernels::elementwise::simple_add::simple_add<Datum>,
			launch_config, stream,
			any_cast<Datum*        >(arguments.at("result"   )),
			any_cast<const Datum*  >(arguments.at("left_hand_side"      )),
			any_cast<const Datum*  >(arguments.at("right_hand_side"      )),
			any_cast<size_t>        (arguments.at("length"   ))
	);
}

template <typename Datum>
const device_function_t kernel<Datum>::get_device_function() const
{
	return reinterpret_cast<const void*>(cuda::kernels::elementwise::simple_add::simple_add<Datum>);
}


#ifdef DEBUG
static_block {
	kernel < int8_t    >::registerInSubclassFactory();
	kernel < int16_t   >::registerInSubclassFactory();
	kernel < int32_t   >::registerInSubclassFactory();
	kernel < int64_t   >::registerInSubclassFactory();
	kernel < uint8_t   >::registerInSubclassFactory();
	kernel < uint16_t  >::registerInSubclassFactory();
	kernel < uint32_t  >::registerInSubclassFactory();
	kernel < uint64_t  >::registerInSubclassFactory();
	kernel < double    >::registerInSubclassFactory();
	kernel < float     >::registerInSubclassFactory();
}
#endif
#endif /* __CUDACC__ */

} // namespace simple_add
} // namespace elementwise
} // namespace kernels
} // namespace cuda
