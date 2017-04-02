
#include "kernel_wrappers/common.h"
#ifdef __CUDACC__
#include "kernels/reduction/multi_reduce/histogram.cuh"
#endif

namespace cuda {
namespace kernels {
namespace reduction {
namespace histogram {


#ifndef __CUDACC__
enum : serialization_factor_t { DefaultSerializationFactor = 32 };
#endif

template <unsigned IndexSize, unsigned BinIndexSize>
class kernel: public cuda::registered::kernel_t {
public:
	REGISTERED_KERNEL_WRAPPER_BOILERPLATE_DEFINITIONS(kernel);
};

#ifdef __CUDACC__


template <unsigned IndexSize, unsigned BinIndexSize>
launch_configuration_t kernel<IndexSize, BinIndexSize>::resolve_launch_configuration(
	device::properties_t           device_properties,
	device_function::attributes_t  kernel_function_attributes,
	arguments_type                 extra_arguments,
	launch_configuration_limits_t  limits) const
{
	namespace kernel_ns = cuda::kernels::reduction::histogram;

	auto data_length      = any_cast<size_t >(extra_arguments.at("data_length"     ));
	auto histogram_length = any_cast<size_t >(extra_arguments.at("histogram_length"));

	kernel_ns::launch_config_resolution_params_t<IndexSize, BinIndexSize> params(
		device_properties,
		data_length, histogram_length,
		limits.dynamic_shared_memory);

	auto serialization_factor = any_cast<serialization_factor_t>(maybe_at(extra_arguments, "serialization_factor"));

	return cuda::resolve_launch_configuration(params, limits, serialization_factor);
}


template <unsigned IndexSize, unsigned BinIndexSize>
void kernel<IndexSize, BinIndexSize>::launch(
	stream::id_t                   stream,
	const launch_configuration_t&  launch_config,
	arguments_type                 arguments) const
{
	using index_type      = uint_t<IndexSize>;
	using bin_index_type = uint_t<BinIndexSize>;

	auto histogram        = any_cast<index_type*           >(arguments.at("histogram"        ));
	auto bin_indices      = any_cast<const bin_index_type* >(arguments.at("bin_indices"      ));
	auto data_length      = any_cast<index_type            >(arguments.at("data_length"      ));
	auto histogram_length = any_cast<bin_index_type        >(arguments.at("histogram_length" ));

	auto maybe_serialization_factor =
		any_cast<serialization_factor_t>(maybe_at(arguments, "serialization_factor"));
	auto serialization_factor = maybe_serialization_factor.value_or(DefaultSerializationFactor);
	cuda::enqueue_launch(
		cuda::kernels::reduction::histogram::histogram<IndexSize, BinIndexSize>,
		launch_config, stream,
		histogram, bin_indices, data_length, histogram_length, serialization_factor
	);
}

template <unsigned IndexSize, unsigned BinIndexSize>
const device_function_t kernel<IndexSize, BinIndexSize>::get_device_function() const
{
	return reinterpret_cast<const void*>(cuda::kernels::reduction::histogram::histogram<IndexSize, BinIndexSize>);
}

static_block {
	//       IndexSize  BinIndexSize
	//----------------------------------------------
	kernel < 4,         1 >::registerInSubclassFactory();
	kernel < 4,         2 >::registerInSubclassFactory();
	kernel < 4,         4 >::registerInSubclassFactory();
	kernel < 4,         8 >::registerInSubclassFactory();
	kernel < 8,         4 >::registerInSubclassFactory();
	kernel < 8,         8 >::registerInSubclassFactory();
}
#endif /* __CUDACC__ */

} // namespace histogram
} // namespace reduction
} // namespace kernels
} // namespace cuda

