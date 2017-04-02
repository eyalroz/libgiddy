
#include "kernel_wrappers/common.h"
#ifdef __CUDACC__
#include "kernels/reduction/count_if.cuh"
#endif

namespace cuda {
namespace kernels {
namespace reduction {
namespace count_if {

template<unsigned IndexSize, typename UnaryPredicate>
class kernel: public cuda::registered::kernel_t {
public:
	REGISTERED_KERNEL_WRAPPER_BOILERPLATE_DEFINITIONS(kernel);

	using predicate_argument_type = typename UnaryPredicate::argument_type;
};

#ifdef __CUDACC__

template<unsigned IndexSize, typename UnaryPredicate>
launch_configuration_t kernel<IndexSize, UnaryPredicate>::resolve_launch_configuration(
	device::properties_t           device_properties,
	device_function::attributes_t  kernel_function_attributes,
	arguments_type                 extra_arguments,
	launch_configuration_limits_t  limits) const
{
	namespace kernel_ns = cuda::kernels::reduction::count_if;

	auto length = any_cast<size_t>(extra_arguments.at("length"));
	kernel_ns::launch_config_resolution_params_t<IndexSize, UnaryPredicate> params(
		device_properties,
		length);

	return cuda::resolve_launch_configuration(params, limits);
}

template<unsigned IndexSize, typename UnaryPredicate>
void kernel<IndexSize, UnaryPredicate>::launch(
	stream::id_t                    stream,
	const launch_configuration_t&   launch_config,
	arguments_type                  arguments) const
{
	using index_type = uint_t<IndexSize>;

	auto result = any_cast<index_type*                    >(arguments.at("result" ));
	auto data   = any_cast<const predicate_argument_type* >(arguments.at("data"   ));
	auto length = any_cast<util::uint_t<IndexSize>        >(arguments.at("length" ));

	cuda::enqueue_launch(
		cuda::kernels::reduction::count_if::count_if<IndexSize, UnaryPredicate>,
		launch_config, stream,
		result, data, length
	);
}

template<unsigned IndexSize, typename UnaryPredicate>
const device_function_t kernel<IndexSize, UnaryPredicate>::get_device_function() const
{
	return reinterpret_cast<const void*>(cuda::kernels::reduction::count_if::count_if<IndexSize, UnaryPredicate>);
}


static_block {

	// 32-bit count results

	//       IndexSize UnaryPredicate
	//--------------------------------------------------------------
	kernel < 4,        functors::non_zero<uint32_t>            >::registerInSubclassFactory();
	kernel < 4,        functors::non_zero<uint64_t>            >::registerInSubclassFactory();
	kernel < 4,        functors::about_zero<float,  bool, 20>  >::registerInSubclassFactory();
	kernel < 4,        functors::about_zero<double, bool, 20>  >::registerInSubclassFactory();
	kernel < 4,        functors::is_zero<int32_t>              >::registerInSubclassFactory();

	// 64-bit count results

	//       IndexSize UnaryPredicate
	//--------------------------------------------------------------
	kernel < 8,        functors::non_zero<uint32_t>           >::registerInSubclassFactory();
	kernel < 8,        functors::non_zero<uint64_t>           >::registerInSubclassFactory();
	kernel < 8,        functors::about_zero<float,  bool, 20> >::registerInSubclassFactory();
	kernel < 8,        functors::about_zero<double, bool, 20> >::registerInSubclassFactory();
	kernel < 8,        functors::is_zero<int32_t>             >::registerInSubclassFactory();
}
#endif /* __CUDACC__ */

} // namespace count_if
} // namespace reduction
} // namespace kernels
} // namespace cuda

