
#ifndef CUDA_ON_DEVICE_PRIMITIVES_COMMON_CUH_
#define CUDA_ON_DEVICE_PRIMITIVES_COMMON_CUH_

#include "cuda/api/types.h"
#include "cuda/on_device/miscellany.cuh"
#include "cuda/on_device/math.cuh"
#ifdef DEBUG
#include "cuda/on_device/printing.cuh"
#endif

#include <type_traits>

namespace cuda {
namespace primitives {

enum inclusivity_t : bool {
	Exclusive = false,
	Inclusive = true
};

namespace detail {
template <typename T>
struct elements_per_full_warp_write {
    enum { value = 4 / gcd_constexpr<unsigned>(4,sizeof(T)) };
};
} // namespace detail

} // namespace primitives
} // namespace cuda

#endif /* CUDA_ON_DEVICE_PRIMITIVES_COMMON_CUH_ */
