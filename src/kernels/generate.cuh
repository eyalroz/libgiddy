#pragma once
#ifndef SRC_KERNELS_GENERATE_CUH
#define SRC_KERNELS_GENERATE_CUH

#include "common.cuh"

/*
 * TODO: 'Generate' is actually the nullary-function equivalent of
 * enumerated_elementwise::unary_operation; we should probably try some variadic-template
 * magic to unify the two
 *
 */

namespace cuda {
namespace kernels {
namespace generate {

// TODO: is it worth it to have a serialization factor here at all? Probably not;
// on the other hand, if we were to generalize this to an k-ary + position
// function kernel, i.e.
//
//   result[i] = f(i, x_{i,1}, ... x_{i,k})
//
// then maybe it would make more sense. Maybe.
//
static constexpr serialization_factor_t DefaultSerializationFactor { 16 };

using namespace grid_info::linear;

template<unsigned IndexSize, typename UnaryFunction, serialization_factor_t SerializationFactor = DefaultSerializationFactor>
__global__ void generate(
	typename UnaryFunction::result_type*   __restrict__  result,
	uint_t<IndexSize>                                    length)
{
	using index_type = uint_t<IndexSize>;
/*
	// Length-aware generation - not doing this right now;
	// Make it another kernel
	auto f = [&result, &length](index_type pos) {
		result[pos] = UnaryFunction()(pos, length);
	};
*/
	auto f = [&result, &length](index_type pos) {
		result[pos] = UnaryFunction()(pos);
	};
	primitives::grid::linear::at_block_stride(length, f, SerializationFactor);
}

template<unsigned IndexSize, typename UnaryFunction, serialization_factor_t SerializationFactor = DefaultSerializationFactor>
class launch_config_resolution_params_t final : public cuda::launch_config_resolution_params_t {
public:
	using parent = cuda::launch_config_resolution_params_t;
public:
	launch_config_resolution_params_t(
		device::properties_t            device_properties_,
		size_t                          length_) :
		cuda::launch_config_resolution_params_t(
			device_properties_,
			device_function_t(generate<IndexSize, UnaryFunction, SerializationFactor>)
		)
	{
		grid_construction_resolution            = thread;
		length                                  = length_;
		serialization_option                    = fixed_factor;
		default_serialization_factor            = DefaultSerializationFactor;
	};
};


} // namespace generate
} // namespace kernels
} // namespace cuda

#endif /* SRC_KERNELS_GENERATE_CUH */
