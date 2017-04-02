
#pragma once
#ifndef KERNEL_WRAPPERS_COMMON_CUH_
#define KERNEL_WRAPPERS_COMMON_CUH_

#include "kernel_wrappers/registered_wrapper.h"
#include "cuda/linear_grid.h"
#include "cuda/functors.hpp"         // enough kernels need this to merit always including it
#ifdef __CUDACC__
#include "util/static_block.h"       // for registration in the factory during static init
#endif

#include <functional>
#include <utility>

#endif /* KERNEL_WRAPPERS_COMMON_CUH_ */
