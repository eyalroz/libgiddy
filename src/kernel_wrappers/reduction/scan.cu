/**
 * Note this file contains three kernel wrappers: Scanning is done using either
 * one kernel with no intermediate reductions, or two kernels, one reducing
 * segments of the input and the other finalizing the scan using those reduction
 * results. So 1 + (1 + 1) = 3 kernels overall:
 *
 *   - scan_single_segment
 *   - reduce_segments
 *   - scan_using_segment_reductions
 *
 */

#include "kernel_wrappers/common.h"
#ifdef __CUDACC__
#include "kernels/reduction/scan.cuh"
#endif

namespace cuda {
namespace kernels {
namespace reduction {
namespace scan {

#ifdef __CUDACC__
using inclusivity_t = primitives::inclusivity_t;
#else
enum inclusivity_t : bool {
	Exclusive = false,
	Inclusive = true
};
#endif


using cuda::functors::enumerated::functor_is_enumerated_t;

namespace reduce_segments {

template<unsigned IndexSize, typename ReductionOp, typename InputDatum,
	typename PretransformOp = functors::identity<InputDatum> >
class kernel: public cuda::registered::kernel_t {
public:
	REGISTERED_KERNEL_WRAPPER_BOILERPLATE_DEFINITIONS(kernel);

	using index_type    = util::uint_t<IndexSize>;
	using result_type   = typename ReductionOp::result_type;

	// TODO: This needs generalization. Many other kernels may
	// theoretically have output lengths depending on the
	// launch configuration
	index_type resolve_segment_length(
		index_type                     total_length,
		const launch_configuration_t&  launch_config) const
	{
		auto num_segments = launch_config.grid_dimensions.x;
		return util::round_up_to_power_of_2(
			util::div_rounding_up(total_length, num_segments), warp_size);
	}
};

#ifdef __CUDACC__



// Note the same configuration is used both for the reduce phase and the scan
// phase; for now the code is duplicated (but, well, actually the duplication is
// across numerous kernels)
template<unsigned IndexSize, typename ReductionOp, typename InputDatum, typename PretransformOp>
launch_configuration_t kernel<IndexSize, ReductionOp, InputDatum, PretransformOp>
::resolve_launch_configuration(
	device::properties_t            device_properties,
	device_function::attributes_t kernel_function_attributes,
	arguments_type                  extra_arguments,
	launch_configuration_limits_t   limits) const
{
	namespace kernel_ns = cuda::kernels::reduction::scan::reduce_segments;

	auto length = any_cast<size_t>(extra_arguments.at("length"));
	kernel_ns::launch_config_resolution_params_t<IndexSize, ReductionOp, InputDatum, PretransformOp> params(
		device_properties,
		length);
	// Note that if the segment_length is not specified, this will still work, and caller
	// can retrieve the effective segment length from the launch_config

	return cuda::resolve_launch_configuration(params, limits);
}

template<unsigned IndexSize, typename ReductionOp, typename InputDatum, typename PretransformOp>
void kernel<IndexSize, ReductionOp, InputDatum, PretransformOp>::launch(
	stream::id_t                   stream,
	const launch_configuration_t&  launch_config,
	arguments_type                 arguments) const
{
	using index_type = uint_t<IndexSize>;

	auto segment_reductions = any_cast<result_type*      >(arguments.at("segment_reductions" ));
	auto data               = any_cast<const InputDatum* >(arguments.at("data"               ));
	auto length             = any_cast<index_type        >(arguments.at("length"             ));
	auto segment_length = resolve_segment_length(length, launch_config);

	cuda::enqueue_launch(
		cuda::kernels::reduction::scan::reduce_segments::reduce_segments
		<IndexSize, ReductionOp, InputDatum, PretransformOp>,
		launch_config, stream,
		segment_reductions, data, length, segment_length
	);
}

template <unsigned CountSize, typename BitContainer = unsigned int>
using prefix_count_set_bits_kernel = kernel<CountSize, cuda::functors::plus<uint_t<CountSize>>, BitContainer,
		cuda::functors::population_count<uint_t<CountSize>, BitContainer>>;

template<unsigned IndexSize, typename ReductionOp, typename InputDatum, typename PretransformOp>
const device_function_t kernel<IndexSize, ReductionOp, InputDatum, PretransformOp>::get_device_function() const
{
	return reinterpret_cast<const void*>(cuda::kernels::reduction::scan::reduce_segments::reduce_segments
		<IndexSize, ReductionOp, InputDatum, PretransformOp>);
}


static_block {
	//       IndexSize           ReductionOp                      InputDatum
	//	PretransformOp
	//-------------------------------------------------------------------------
	kernel < 4,                  functors::plus<unsigned>,        unsigned    >::registerInSubclassFactory();
	kernel < 4,                  functors::plus<int>,             int         >::registerInSubclassFactory();
	kernel < 4,                  functors::plus<int64_t>,         int         >::registerInSubclassFactory();
	kernel < 8,                  functors::plus<int64_t>,         int         >::registerInSubclassFactory();
	kernel < 4,                  functors::plus<float>,           float       >::registerInSubclassFactory();
	kernel < 4,                  functors::plus<double>,          float       >::registerInSubclassFactory();
	kernel < 4,                  functors::plus<double>,          double      >::registerInSubclassFactory();

	prefix_count_set_bits_kernel < 4 >::registerInSubclassFactory();
	prefix_count_set_bits_kernel < 8 >::registerInSubclassFactory();

	kernel < 4,                  functors::plus<unsigned>,        unsigned,
		functors::enumerated::as_enumerated_unary<functors::plus<unsigned> >>::registerInSubclassFactory();
}

#endif

} // namespace reduce_segments

namespace scan_using_segment_reductions {

template<
	unsigned IndexSize, typename ReductionOp,
	typename InputDatum,
	bool Inclusivity = inclusivity_t::Inclusive,
	typename PretransformOp = functors::identity<InputDatum>
	>
class kernel: public cuda::registered::kernel_t {
public:
	REGISTERED_KERNEL_WRAPPER_BOILERPLATE_DEFINITIONS(kernel);

	using result_type        = typename ReductionOp::result_type;

};

#ifdef __CUDACC__


// Note the same configuration is used both for the reduce phase and the scan
// phase; for now the code is duplicated (but, well, actually the duplication is
// across numerous kernels)
template<
	unsigned IndexSize, typename ReductionOp, typename InputDatum, bool Inclusivity,
	typename PretransformOp>
launch_configuration_t kernel<IndexSize, ReductionOp, InputDatum, Inclusivity, PretransformOp>
::resolve_launch_configuration(
	device::properties_t             device_properties,
	device_function::attributes_t  kernel_function_attributes,
	arguments_type                   extra_arguments,
	launch_configuration_limits_t    limits) const
{
	namespace kernel_ns = cuda::kernels::reduction::scan::scan_using_segment_reductions;

	auto length = any_cast<size_t>(extra_arguments.at("length"));
	// Here we require the segment_length to be specified - since we cannot otherwise
	// determine how many segment reduction values we've been provided with and where
	// to apply them
	auto segment_length = any_cast<size_t>(extra_arguments.at("segment_length"));
	kernel_ns::launch_config_resolution_params_t<IndexSize, ReductionOp, InputDatum, Inclusivity, PretransformOp> params(
		device_properties,
		length, segment_length,
		limits.dynamic_shared_memory);
	return cuda::resolve_launch_configuration(params, limits);
}

template<
	unsigned IndexSize, typename ReductionOp, typename InputDatum,
	bool Inclusivity, typename PretransformOp>
void kernel<IndexSize, ReductionOp, InputDatum, Inclusivity, PretransformOp>::launch(
	stream::id_t                    stream,
	const launch_configuration_t&   launch_config,
	arguments_type                  arguments) const
{
	using index_type = uint_t<IndexSize>;

	auto result             = any_cast<result_type*       >(arguments.at("result"             ));
	auto segment_reductions = any_cast<const result_type* >(arguments.at("segment_reductions" ));
	auto data               = any_cast<const InputDatum*  >(arguments.at("data"               ));
	auto length             = any_cast<index_type         >(arguments.at("length"             ));
	auto segment_length     = any_cast<index_type         >(arguments.at("segment_length"     ));

	cuda::enqueue_launch(
		cuda::kernels::reduction::scan::scan_using_segment_reductions::scan_using_segment_reductions
		<IndexSize, ReductionOp, InputDatum, Inclusivity, PretransformOp>,
		launch_config, stream,
		result, segment_reductions, data, length, segment_length
	);
}

template<
	unsigned IndexSize, typename ReductionOp, typename InputDatum,
	bool Inclusivity, typename PretransformOp>
const device_function_t kernel<IndexSize, ReductionOp, InputDatum, Inclusivity, PretransformOp>::get_device_function() const
{
	return reinterpret_cast<const void*>(cuda::kernels::reduction::scan::scan_using_segment_reductions::scan_using_segment_reductions
		<IndexSize, ReductionOp, InputDatum, Inclusivity, PretransformOp>);
}

template <
	unsigned CountSize, bool Inclusivity = inclusivity_t::Inclusive,
	typename BitContainer = unsigned int>
using prefix_count_set_bits_kernel = kernel<CountSize, cuda::functors::plus<uint_t<CountSize>>, BitContainer,
		Inclusivity, cuda::functors::population_count<uint_t<CountSize>, BitContainer>>;


static_block {
	constexpr const auto Exclusive = inclusivity_t::Exclusive;
	constexpr const auto Inclusive = inclusivity_t::Inclusive;

	//       IndexSize           ReductionOp                      InputDatum  Inclusivity
	//	PretransformOp
	//---------------------------------------------------------------------------------------------------------------------------------------
	kernel < 4,                  functors::plus<uint32_t>,        uint32_t,   Inclusive  >::registerInSubclassFactory();
	kernel < 4,                  functors::plus<int32_t>,             int32_t,        Inclusive  >::registerInSubclassFactory();
	kernel < 4,                  functors::plus<int64_t>,   int32_t,        Inclusive  >::registerInSubclassFactory();
	kernel < 8,        functors::plus<int64_t>,   int32_t,        Inclusive  >::registerInSubclassFactory();
	kernel < 4,                  functors::plus<float>,           float,      Inclusive  >::registerInSubclassFactory();
	kernel < 4,                  functors::plus<double>,          float,      Inclusive  >::registerInSubclassFactory();
	kernel < 4,                  functors::plus<double>,          double,     Inclusive  >::registerInSubclassFactory();
	kernel < 4,                  functors::plus<uint32_t>,        uint32_t,   Exclusive  >::registerInSubclassFactory();
	kernel < 4,                  functors::plus<int32_t>,             int32_t,        Exclusive  >::registerInSubclassFactory();
	kernel < 4,                  functors::plus<int64_t>,   int32_t,        Exclusive  >::registerInSubclassFactory();
	kernel < 8,        functors::plus<int64_t>,   int32_t,        Exclusive  >::registerInSubclassFactory();
	kernel < 4,                  functors::plus<float>,           float,      Exclusive  >::registerInSubclassFactory();
	kernel < 4,                  functors::plus<double>,          float,      Exclusive  >::registerInSubclassFactory();
	kernel < 4,                  functors::plus<double>,          double,     Exclusive  >::registerInSubclassFactory();
	kernel < 4,                  functors::plus<uint32_t>,        uint32_t,   Inclusive,
		functors::enumerated::as_enumerated_unary<functors::plus<uint32_t> >>::registerInSubclassFactory();
	kernel < 4,                  functors::plus<uint32_t>,        uint32_t,   Exclusive,
		functors::enumerated::as_enumerated_unary<functors::plus<uint32_t> >>::registerInSubclassFactory();

	prefix_count_set_bits_kernel < 4            >::registerInSubclassFactory();
	prefix_count_set_bits_kernel < 8            >::registerInSubclassFactory();
	prefix_count_set_bits_kernel < 4, Exclusive >::registerInSubclassFactory();
	prefix_count_set_bits_kernel < 8, Exclusive >::registerInSubclassFactory();


}

#endif /* CUDACC */

} // namespace scan_using_segment_reductions

namespace scan_single_segment {

template<
	unsigned IndexSize, typename ReductionOp,
	typename InputDatum,
	bool Inclusivity = inclusivity_t::Inclusive,
	typename PretransformOp = functors::identity<InputDatum>
	>
class kernel: public cuda::registered::kernel_t {
public:
	REGISTERED_KERNEL_WRAPPER_BOILERPLATE_DEFINITIONS(kernel);

	using result_type        = typename ReductionOp::result_type;
};

#ifdef __CUDACC__

template<
	unsigned IndexSize, typename ReductionOp, typename InputDatum,
	bool Inclusivity, typename PretransformOp>
// Again, the same launch configuration as the above
launch_configuration_t kernel<IndexSize, ReductionOp, InputDatum, Inclusivity, PretransformOp>
::resolve_launch_configuration(
	device::properties_t            device_properties,
	device_function::attributes_t kernel_function_attributes,
	arguments_type                 extra_arguments,
	launch_configuration_limits_t  limits) const
/*{
	return cuda::kernels::reduction::scan::resolve_launch_configuration(
		device_properties, kernel_function_attributes,
		any_cast<size_t>(extra_arguments.at("length")),
		IndexSize,
		limits.block_size);
}*/
{
	namespace kernel_ns = cuda::kernels::reduction::scan::scan_single_segment;

	auto length = any_cast<size_t>(extra_arguments.at("length"));
	kernel_ns::launch_config_resolution_params_t<IndexSize, ReductionOp, InputDatum, Inclusivity, PretransformOp> params(
		device_properties,
		length,
		limits.dynamic_shared_memory);

	return cuda::resolve_launch_configuration(params, limits);
}


template<
	unsigned IndexSize, typename ReductionOp, typename InputDatum,
	bool Inclusivity, typename PretransformOp>
void kernel<IndexSize, ReductionOp, InputDatum, Inclusivity, PretransformOp>::launch(
	stream::id_t                   stream,
	const launch_configuration_t&  launch_config,
	arguments_type                 arguments) const
{
	using index_type    = util::uint_t<IndexSize>;

	auto result = any_cast<result_type*       >(arguments.at("result" ));
	auto data   = any_cast<const InputDatum*  >(arguments.at("data"   ));
	auto length = any_cast<index_type         >(arguments.at("length" ));

	cuda::enqueue_launch(
		cuda::kernels::reduction::scan::scan_single_segment::scan_single_segment
		<IndexSize, ReductionOp, InputDatum, Inclusivity, PretransformOp>,
		launch_config, stream,
		result, data, length
	);
}

template<unsigned IndexSize, typename ReductionOp, typename InputDatum,
	bool Inclusivity, typename PretransformOp>
const device_function_t kernel<IndexSize, ReductionOp, InputDatum, Inclusivity, PretransformOp>::get_device_function() const
{
	return reinterpret_cast<const void*>(cuda::kernels::reduction::scan::scan_single_segment::scan_single_segment
		<IndexSize, ReductionOp, InputDatum, Inclusivity, PretransformOp>);
}

template <
	unsigned CountSize,
	bool Inclusivity = inclusivity_t::Inclusive,
	typename BitContainer = unsigned int>
using prefix_count_set_bits_kernel = kernel<CountSize, cuda::functors::plus<uint_t<CountSize>>, BitContainer,
		Inclusivity, cuda::functors::population_count<uint_t<CountSize>, BitContainer>>;

static_block {
	namespace functors = cuda::functors;
	constexpr const auto Exclusive = inclusivity_t::Exclusive;

	// No pretransform, inclusive

	//       IndexSize  ReductionOp                InputDatum   Inclusivity  PretransformOp
	//-----------------------------------------------------------------------------------------------------
	kernel < 4,         functors::plus<uint32_t>,  uint32_t     >::registerInSubclassFactory();
	kernel < 4,         functors::plus<int32_t>,   int32_t      >::registerInSubclassFactory();
	kernel < 4,         functors::plus<int64_t>,   int32_t      >::registerInSubclassFactory();
	kernel < 8,         functors::plus<int64_t>,   int32_t      >::registerInSubclassFactory();
	kernel < 4,         functors::plus<float>,     float        >::registerInSubclassFactory();
	kernel < 4,         functors::plus<double>,    float        >::registerInSubclassFactory();
	kernel < 4,         functors::plus<double>,    double       >::registerInSubclassFactory();

	// No pretransform, exclusive

	kernel < 4,         functors::plus<int32_t>,   int32_t,     Exclusive>::registerInSubclassFactory();
	kernel < 4,         functors::plus<double>,    double,      Exclusive>::registerInSubclassFactory();

	// Unenumerated pretransform, inclusive & exclusive

	constexpr auto Inclusive = cuda::primitives::inclusivity_t::Inclusive;

	prefix_count_set_bits_kernel < 4                   >::registerInSubclassFactory();
	prefix_count_set_bits_kernel < 8                   >::registerInSubclassFactory();
	prefix_count_set_bits_kernel < 4,        Exclusive >::registerInSubclassFactory();
	prefix_count_set_bits_kernel < 8,        Exclusive >::registerInSubclassFactory();

	// Enumerated pretransform

	kernel < 4,        functors::plus<uint32_t>, uint32_t, Inclusive,
		functors::enumerated::as_enumerated_unary<functors::plus<uint32_t> >>::registerInSubclassFactory();
	kernel < 4,        functors::plus<uint32_t>, uint32_t, Exclusive,
		functors::enumerated::as_enumerated_unary<functors::plus<uint32_t> >>::registerInSubclassFactory();
}

#endif

} // namespace scan_single_segment


} // namespace scan
} // namespace reduction
} // namespace kernels
} // namespace cuda
