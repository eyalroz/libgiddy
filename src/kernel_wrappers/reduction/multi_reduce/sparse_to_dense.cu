
#include "kernel_wrappers/common.h"
#ifdef __CUDACC__
#include "kernels/reduction/multi_reduce/sparse_to_dense.cuh"
#endif

namespace cuda {
namespace kernels {
namespace reduction {
namespace multi_reduce {
namespace dynamic_num_reductions {
namespace sparse_to_dense {

#ifndef __CUDACC__
enum : serialization_factor_t { DefaultSerializationFactor = 32 };
#endif

template <unsigned IndexSize, typename ReductionOp, typename InputDatum,
		  typename ReductionIndex>
class kernel: public cuda::registered::kernel_t {
public:
	REGISTERED_KERNEL_WRAPPER_BOILERPLATE_DEFINITIONS(kernel);

	using result_type = typename ReductionOp::result_type;
};

#ifdef __CUDACC__

template <unsigned IndexSize, typename ReductionOp, typename InputDatum, typename ReductionIndex>
launch_configuration_t kernel<IndexSize, ReductionOp, InputDatum, ReductionIndex>::resolve_launch_configuration(
	device::properties_t             device_properties,
	device_function::attributes_t  kernel_function_attributes,
	arguments_type                   extra_arguments,
	launch_configuration_limits_t    limits) const
{
	namespace kernel_ns =
		::cuda::kernels::reduction::multi_reduce::dynamic_num_reductions::sparse_to_dense
		;

	auto length = any_cast<size_t>(extra_arguments.at("length"));
	auto num_distinct_indices = any_cast<size_t>(extra_arguments.at("num_distinct_indices"));
	auto serialization_factor =
		any_cast<serialization_factor_t>(maybe_at(extra_arguments, "serialization_factor"));
	kernel_ns::launch_config_resolution_params_t<
		IndexSize, ReductionOp, InputDatum, ReductionIndex> params(
		device_properties,
		length, num_distinct_indices,
		limits.dynamic_shared_memory);

	return cuda::resolve_launch_configuration(params, limits, serialization_factor);
}

template <unsigned IndexSize, typename ReductionOp, typename InputDatum, typename ReductionIndex>
void kernel<IndexSize, ReductionOp, InputDatum, ReductionIndex>::launch(
	stream::id_t                     stream,
	const launch_configuration_t&   launch_config,
	arguments_type                  arguments) const
{
	using index_type = uint_t<IndexSize>;

	auto target               = any_cast<result_type*          >(arguments.at("target"               ));
	auto data                 = any_cast<const InputDatum*     >(arguments.at("data"                 ));
	auto indices              = any_cast<const ReductionIndex* >(arguments.at("indices"              ));
	auto length               = any_cast<index_type            >(arguments.at("length"               ));
	auto num_distinct_indices = any_cast<ReductionIndex        >(arguments.at("num_distinct_indices" ));


	auto maybe_serialization_factor =
		any_cast<serialization_factor_t>(maybe_at(arguments, "serialization_factor"));
	auto serialization_factor = maybe_serialization_factor.value_or(DefaultSerializationFactor);

	cuda::enqueue_launch(
		sparse_to_dense<IndexSize, ReductionOp, InputDatum, ReductionIndex>,
		launch_config, stream,
		target, data, indices, length, num_distinct_indices,serialization_factor
	);
}

template <unsigned IndexSize, typename ReductionOp, typename InputDatum, typename ReductionIndex>
const device_function_t kernel<IndexSize, ReductionOp, InputDatum, ReductionIndex>::get_device_function() const
{
	return reinterpret_cast<const void *>(
		sparse_to_dense<IndexSize, ReductionOp, InputDatum, ReductionIndex>);
}

static_block {
	namespace functors = cuda::functors;

	//       IndexSize  ReductionOp                   InputDatum   ReductionIndex
	//-------------------------------------------------------------------------------------
	kernel < 4,         functors::plus<int>,          int32_t,     uint32_t >::registerInSubclassFactory();
	kernel < 4,         functors::plus<uint32_t>,     uint32_t,    uint32_t >::registerInSubclassFactory();
	kernel < 4,         functors::maximum<int>,       int32_t,     uint32_t >::registerInSubclassFactory();
	kernel < 4,         functors::maximum<uint32_t>,  uint32_t,    uint32_t >::registerInSubclassFactory();
	kernel < 8,         functors::plus<uint32_t>,     uint32_t,    uint32_t >::registerInSubclassFactory();
	kernel < 8,         functors::plus<uint32_t>,     uint32_t,    uint64_t >::registerInSubclassFactory();
}
#endif


} // namespace sparse_to_dense
} // namespace dynamic_num_reductions
} // namespace multi_reduce
} // namespace reduction
} // namespace kernels
} // namespace cuda

