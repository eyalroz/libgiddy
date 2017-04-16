
#include "kernel_wrappers/common.h"
#include "cuda/model_functors.hpp"
#ifdef __CUDACC__
#include "kernels/decompression/model.cuh"
#endif

namespace cuda {
namespace kernels {
namespace decompression {
namespace model {

template<unsigned IndexSize, typename Uncompressed, typename UnaryModelFunction>
class kernel_t : public cuda::registered::kernel_t {
public:
	using model_coefficients = typename UnaryModelFunction::coefficients_type;
	REGISTERED_KERNEL_WRAPPER_BOILERPLATE_DEFINITIONS(kernel_t);

	launch_configuration_t resolve_launch_configuration(
		device::properties_t           device_properties,
		device_function::attributes_t  kernel_function_attributes,
		size_t                         length,
		launch_configuration_limits_t  limits) const
#ifdef __CUDACC__
	{
		launch_config_resolution_params_t<
			IndexSize, Uncompressed, UnaryModelFunction
		> params(
			device_properties,
			length);

		return cuda::kernels::resolve_launch_configuration(params, limits);
	}
#else
	;
#endif
};

#ifdef __CUDACC__

template<unsigned IndexSize, typename Uncompressed, typename UnaryModelFunction>
launch_configuration_t kernel_t<IndexSize, Uncompressed, UnaryModelFunction>::resolve_launch_configuration(
	device::properties_t           device_properties,
	device_function::attributes_t  kernel_function_attributes,
	arguments_type                 extra_arguments,
	launch_configuration_limits_t  limits) const
{
	auto length = any_cast<size_t>(extra_arguments.at("length"));

	return resolve_launch_configuration(
		device_properties, kernel_function_attributes,
		length,
		limits);
}


template<unsigned IndexSize, typename Uncompressed, typename UnaryModelFunction>
void kernel_t<IndexSize, Uncompressed, UnaryModelFunction>::enqueue_launch(
	stream::id_t                   stream,
	const launch_configuration_t&  launch_config,
	arguments_type                 arguments) const
{
	using index_type = uint_t<IndexSize>;
	using model_coefficients_type =
		typename UnaryModelFunction::coefficients_type;
	auto length          = any_cast<index_type>(arguments.at("length"));

	auto decompressed       = any_cast<Uncompressed*           >(arguments.at("decompressed"       ));
	auto model_coefficients = any_cast<model_coefficients_type >(arguments.at("model_coefficients" ));

	cuda::kernel::enqueue_launch(
		*this, stream, launch_config,
		decompressed, model_coefficients, length);
}

template<unsigned IndexSize, typename Uncompressed, typename UnaryModelFunction>
const device_function_t kernel_t<IndexSize, Uncompressed, UnaryModelFunction>::get_device_function() const
{
	return reinterpret_cast<const void*>(cuda::kernels::decompression::model::decompress
		<IndexSize, Uncompressed, UnaryModelFunction>);
}


static_block {
	namespace functors = cuda::functors;
	namespace unary_models = ::cuda::functors::unary::parametric_model;

	//         IndexSize  Uncompressed   UnaryModelFunction
	//-------------------------------------------------------------------------------
	kernel_t < 4,         int16_t,       unary_models::linear  < 4, int16_t >  >::registerInSubclassFactory();
	kernel_t < 4,         int32_t,       unary_models::linear  < 4, int32_t >  >::registerInSubclassFactory();
	kernel_t < 8,         int32_t,       unary_models::linear  < 8, int32_t >  >::registerInSubclassFactory();
	kernel_t < 4,         int16_t,       unary_models::constant< 4, int16_t >  >::registerInSubclassFactory();
	kernel_t < 4,         int32_t,       unary_models::constant< 4, int32_t >  >::registerInSubclassFactory();
}

#endif /* __CUDACC__ */

} // namespace model
} // namespace decompression
} // namespace kernels
} // namespace cuda
