
#include "kernel_wrappers/common.h"
#ifdef __CUDACC__
#include "kernels/reduction/reduce.cuh"
#endif

namespace cuda {
namespace kernels {
namespace reduction {
namespace reduce {

template <
	unsigned IndexSize,
	typename ReductionOp,
	typename InputDatum,
	typename PretransformOp = functors::identity<InputDatum>
>
class kernel: public cuda::registered::kernel_t {
public:
	REGISTERED_KERNEL_WRAPPER_BOILERPLATE_DEFINITIONS(kernel);

	using result_type        = typename ReductionOp::result_type;
};

#ifdef __CUDACC__


template <unsigned IndexSize, typename ReductionOp, typename InputDatum, typename PretransformOp>
launch_configuration_t kernel<IndexSize, ReductionOp, InputDatum, PretransformOp>
::resolve_launch_configuration(
	device::properties_t           device_properties,
	device_function::attributes_t  kernel_function_attributes,
	arguments_type                 extra_arguments,
	launch_configuration_limits_t  limits) const
{
	namespace kernel_ns = cuda::kernels::reduction::reduce;

	auto length = any_cast<size_t>(extra_arguments.at("length"));
	kernel_ns::launch_config_resolution_params_t<IndexSize, ReductionOp, InputDatum, PretransformOp> params(
		device_properties,
		length,
		limits.dynamic_shared_memory);

	return cuda::resolve_launch_configuration(params, limits);
}

template <unsigned IndexSize, typename ReductionOp, typename InputDatum, typename PretransformOp>
void kernel<IndexSize, ReductionOp, InputDatum, PretransformOp>::launch(
	stream::id_t                     stream,
	const launch_configuration_t&   launch_config,
	arguments_type                  arguments) const
{
	using index_type = uint_t<IndexSize>;

	auto result = any_cast<result_type*      >(arguments.at("result" ));
	auto data   = any_cast<const InputDatum* >(arguments.at("data"   ));
	auto length = any_cast<index_type        >(arguments.at("length" ));

	cuda::enqueue_launch(
		cuda::kernels::reduction::reduce::reduce
		<IndexSize, ReductionOp, InputDatum, PretransformOp>,
		launch_config, stream,
		result, data, length
	);
}

template <
	unsigned IndexSize, typename ReductionOp, typename InputDatum, typename PretransformOp>
const device_function_t kernel<IndexSize, ReductionOp, InputDatum, PretransformOp>::get_device_function() const
{
	return reinterpret_cast<const void*>(cuda::kernels::reduction::reduce::reduce
		<IndexSize, ReductionOp, InputDatum, PretransformOp>);
}

template <unsigned CountSize, typename BitContainer = unsigned int>
using count_bits_set_kernel = kernel<CountSize, cuda::functors::plus<uint_t<CountSize>>, BitContainer,
	cuda::functors::population_count<uint_t<CountSize>, BitContainer>>;
template <unsigned IndexSize, typename BitContainer = unsigned>
using dense_minimum_kernel = kernel<IndexSize, cuda::functors::minimum<uint_t<IndexSize>>, BitContainer,
	functors::global_index_of_first_set_bit<IndexSize, BitContainer>>;

template <unsigned IndexSize, typename BitContainer = unsigned>
using dense_maximum_kernel = kernel<IndexSize, cuda::functors::maximum<uint_t<IndexSize>>, BitContainer,
	functors::global_index_of_last_set_bit<IndexSize, BitContainer>>;

static_block {
	namespace functors = cuda::functors;

	// no pretransform

	//       IndexSize   ReductionOp                InputDatum
	//-----------------------------------------------------------------------
	kernel < 4,         functors::plus<int32_t>,   int32_t     >::registerInSubclassFactory();
	kernel < 4,         functors::plus<int64_t>,   int32_t     >::registerInSubclassFactory();
	kernel < 8,         functors::plus<int64_t>,   int32_t     >::registerInSubclassFactory();
	kernel < 4,         functors::plus<float>,     float       >::registerInSubclassFactory();
	kernel < 4,         functors::plus<double>,    float       >::registerInSubclassFactory();
	kernel < 4,         functors::plus<double>,    double      >::registerInSubclassFactory();

	// unenumerated pretransform

	//       IndexSize  ReductionOp                InputDatum  PretransformOp
	//------------------------------------------------------------------------
	kernel < 4,         functors::plus<uint32_t>,  uint32_t,   functors::is_zero<uint32_t> >::registerInSubclassFactory();


	count_bits_set_kernel <4>::registerInSubclassFactory();
	count_bits_set_kernel <8>::registerInSubclassFactory();

	// enumerated pretransform

	//       IndexSize  ReductionOp                InputDatum  PretransformOp
	//------------------------------------------------------------------------
	kernel < 4,         functors::plus<uint32_t>,  uint32_t,   functors::enumerated::as_enumerated_unary<functors::plus<uint32_t> > >::registerInSubclassFactory();

	dense_minimum_kernel < 4 >::registerInSubclassFactory();
	dense_minimum_kernel < 8 >::registerInSubclassFactory();
	dense_maximum_kernel < 4 >::registerInSubclassFactory();
	dense_maximum_kernel < 8 >::registerInSubclassFactory();
}
#endif /* __CUDACC__ */

} // namespace reduce
} // namespace reduction
} // namespace kernels
} // namespace cuda
