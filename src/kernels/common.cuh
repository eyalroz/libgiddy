
#pragma once
#ifndef KERNELS_COMMON_CUH_
#define KERNELS_COMMON_CUH_

#include "cuda/on_device/imports.cuh"
#include "cuda/on_device/miscellany.cuh"
#include "cuda/on_device/atomics.cuh"
#include "cuda/on_device/primitives/grid.cuh"
#include "cuda/api/device_function.hpp"
#include "cuda/optional_and_any.hpp"
#include "cuda/bit_operations.cuh"
#include "cuda/api/constants.h"
#include "cuda/api/kernel_launch.cuh"
#include "util/math.hpp"
#include "util/integer.h"

namespace cuda {
	using util::uint_t;
}


// Everything below is (almost certainly) only necessary
// for resolving launch configurations (and is for host
// code only)

#include "kernels/launch_config_resolution_params.h"
#include "cuda/syntax_replacement.h"

#endif /* KERNELS_COMMON_CUH_ */
