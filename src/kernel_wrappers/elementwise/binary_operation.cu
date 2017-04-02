
#include "kernel_wrappers/common.h"
#ifdef __CUDACC__
#include "kernels/elementwise/binary_operation.cuh"
#endif

namespace cuda {
namespace kernels {
namespace elementwise {
namespace binary {

#ifndef __CUDACC__
enum : serialization_factor_t { DefaultSerializationFactor = 16 };
#endif

template<unsigned IndexSize, typename BinaryOp, bool LHSScalarity, bool RHSScalarity,
	serialization_factor_t SerializationFactor = DefaultSerializationFactor>
class kernel: public cuda::registered::kernel_t {
public:
	REGISTERED_KERNEL_WRAPPER_BOILERPLATE_DEFINITIONS(kernel);

	using result_type          = typename BinaryOp::result_type;
	using first_argument_type  = typename BinaryOp::first_argument_type;
	using second_argument_type = typename BinaryOp::second_argument_type;


};

#ifdef __CUDACC__

template<unsigned IndexSize, typename BinaryOp, bool LHSScalarity, bool RHSScalarity, serialization_factor_t SerializationFactor>
launch_configuration_t kernel<IndexSize, BinaryOp, LHSScalarity, RHSScalarity, SerializationFactor>::resolve_launch_configuration(
	device::properties_t             device_properties,
	device_function::attributes_t  kernel_function_attributes,
	arguments_type                   extra_arguments,
	launch_configuration_limits_t    limits) const
{
	namespace kernel_ns = cuda::kernels::elementwise::binary;

	auto length = any_cast<size_t>(extra_arguments.at("length"));
	kernel_ns::launch_config_resolution_params_t<IndexSize, BinaryOp, SerializationFactor> params(
		device_properties,
		length);

	return cuda::resolve_launch_configuration(params, limits);
}


template<unsigned IndexSize, typename BinaryOp, bool LHSScalarity, bool RHSScalarity, serialization_factor_t SerializationFactor>
void kernel<IndexSize, BinaryOp, LHSScalarity, RHSScalarity, SerializationFactor>::launch(
	stream::id_t                   stream,
	const launch_configuration_t&  launch_config,
	arguments_type                 arguments) const
{
	if (!LHSScalarity && !RHSScalarity) {
		auto result          = any_cast<result_type*                >(arguments.at("result"   ));
		auto left_hand_side  = any_cast<const first_argument_type*  >(arguments.at("left_hand_side"      ));
		auto right_hand_side = any_cast<const second_argument_type* >(arguments.at("right_hand_side"      ));
		auto length          = any_cast<util::uint_t<IndexSize>     >(arguments.at("length"   ));
		cuda::enqueue_launch(
			cuda::kernels::elementwise::binary::performBinaryOperationAA<IndexSize, BinaryOp, SerializationFactor>,
			launch_config, stream,
			result, left_hand_side, right_hand_side, length
		);
	}
	else if (LHSScalarity && !RHSScalarity) {
		cuda::enqueue_launch(
			cuda::kernels::elementwise::binary::performBinaryOperationSA<IndexSize, BinaryOp, SerializationFactor>,
			launch_config, stream,
			any_cast<result_type*                >(arguments.at("result"   )),
			any_cast<first_argument_type         >(arguments.at("left_hand_side"      )),
			any_cast<const second_argument_type* >(arguments.at("right_hand_side"      )),
			any_cast<util::uint_t<IndexSize>     >(arguments.at("length"   ))
		);
	}
	else if (!LHSScalarity && RHSScalarity) {
		cuda::enqueue_launch(
			cuda::kernels::elementwise::binary::performBinaryOperationAS<IndexSize, BinaryOp, SerializationFactor>,
			launch_config, stream,
			any_cast<result_type*                >(arguments.at("result"   )),
			any_cast<const first_argument_type*  >(arguments.at("left_hand_side"      )),
			any_cast<second_argument_type        >(arguments.at("right_hand_side"      )),
			any_cast<util::uint_t<IndexSize>     >(arguments.at("length"   ))
		);
	}
	else {
		throw std::out_of_range("Can't have scalar-scalar binary ops");
	}
}

template<unsigned IndexSize, typename BinaryOp, bool LHSScalarity, bool RHSScalarity, serialization_factor_t SerializationFactor>
const device_function_t kernel<IndexSize, BinaryOp, LHSScalarity, RHSScalarity, SerializationFactor>::get_device_function() const
{
	return reinterpret_cast<const void*>(cuda::kernels::elementwise::binary::performBinaryOperationAS<IndexSize, BinaryOp, SerializationFactor>);
}


static_block {
	const auto Scalar = true;
	const auto Array  = false;

	namespace functors = ::cuda::functors;

	// Arithmetic

	//                                                      Scalarity
	//       IndexSize   BinaryOp                          LHS     RHS
	//--------------------------------------------------------------------------------
	kernel < 4,          functors::plus<int8_t>,           Array,  Array  >::registerInSubclassFactory();
	kernel < 4,          functors::plus<int16_t>,          Array,  Array  >::registerInSubclassFactory();
	kernel < 4,          functors::plus<int32_t>,          Array,  Array  >::registerInSubclassFactory();
	kernel < 4,          functors::plus<int64_t>,          Array,  Array  >::registerInSubclassFactory();
	kernel < 4,          functors::minus<int32_t>,         Array,  Array  >::registerInSubclassFactory();
	kernel < 4,          functors::multiplies<int32_t>,    Array,  Array  >::registerInSubclassFactory();
	kernel < 4,          functors::plus<int32_t>,          Array,  Scalar >::registerInSubclassFactory();
	kernel < 4,          functors::plus<int32_t>,          Scalar, Array  >::registerInSubclassFactory();
	kernel < 4,          functors::plus<float>,            Array,  Array  >::registerInSubclassFactory();
	kernel < 4,          functors::plus<float>,            Scalar, Array  >::registerInSubclassFactory();
	kernel < 4,          functors::plus<float>,            Array,  Scalar >::registerInSubclassFactory();
	kernel < 4,          functors::minus<float>,           Array,  Array  >::registerInSubclassFactory();
	kernel < 4,          functors::minus<float>,           Scalar, Array  >::registerInSubclassFactory();
	kernel < 4,          functors::minus<float>,           Array,  Scalar >::registerInSubclassFactory();

	kernel < 8,         functors::plus<float>,             Array,  Array  >::registerInSubclassFactory();
	kernel < 8,         functors::plus<float>,             Scalar, Array  >::registerInSubclassFactory();
	kernel < 8,         functors::plus<float>,             Array,  Scalar >::registerInSubclassFactory();

	// Comparisons

	kernel < 8,         functors::greater_equal<float>,   Array,  Array  >::registerInSubclassFactory();
	kernel < 8,         functors::less_equal<int32_t>,    Array,  Array  >::registerInSubclassFactory();
	kernel < 8,         functors::equals<int32_t>,        Array,  Array  >::registerInSubclassFactory();
	kernel < 8,         functors::equals<float>,          Array,  Array  >::registerInSubclassFactory();
	kernel < 8,         functors::equals<double>,         Array,  Array  >::registerInSubclassFactory();
	kernel < 8,         functors::equals<int16_t>,        Array,  Array  >::registerInSubclassFactory();
	kernel < 8,         functors::equals<uint64_t>,       Array,  Array  >::registerInSubclassFactory();

}
#endif

/*
// Equality comparisons are important enough for the test harness for us to want to explicitly instantiate them
// for a bunch of types

template <typename Op> using performBinaryOperationAA_t = decltype(performBinaryOperationAA<Op>);
template <typename T> using comparisonAA_t = performBinaryOperationAA_t<::cuda::functors::equals<T>>;

#define INSTANTIATE_COMPARISONAA_KERNEL(_type) \
template __global__ comparisonAA_t< _type > performBinaryOperationAA<::cuda::functors::equals< _type >>;

MAP(INSTANTIATE_COMPARISONAA_KERNEL, char, unsigned char, short, unsigned short, int, unsigned int, long, unsigned long, long long, unsigned long long, float, double);
*/

} // namespace binary
} // namespace elementwise
} // namespace kernels
} // namespace cuda
