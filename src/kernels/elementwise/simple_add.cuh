
#include "kernels/common.cuh"

namespace cuda {
namespace kernels {
namespace elementwise {
namespace simple_add {


template <typename Datum>
__global__ void simple_add(
	Datum*        __restrict__  result,
	const Datum*  __restrict__  lhs,
	const Datum*  __restrict__  rhs,
	unsigned                    length)
{
	unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < length) {
		result[i] = lhs[i] + rhs[i];
	}
}

template <typename Datum>
class launch_config_resolution_params_t final : public cuda::launch_config_resolution_params_t {
public:
	using parent = cuda::launch_config_resolution_params_t;
public:
	launch_config_resolution_params_t(
		device::properties_t            device_properties_,
		size_t                          length_) :
		cuda::launch_config_resolution_params_t(
			device_properties_,
			device_function_t(simple_add<Datum>)
		)
	{
		grid_construction_resolution            = thread;
		length                                  = length_;
		serialization_option                    = none;
	};
};


} // namespace simple_add
} // namespace elementwise
} // namespace kernels
} // namespace cuda
