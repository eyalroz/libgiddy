/**
 * import some definitions from the cuda namespace into the default
 * namespace
 */
#pragma once
#ifndef CUDA_ON_DEVICE_IMPORTS_CUH_
#define CUDA_ON_DEVICE_IMPORTS_CUH_

using cuda::warp_size;
using cuda::half_warp_size;
using cuda::squared_warp_size;
using cuda::log_warp_size;
using cuda::lane_index_mask;


#endif /* CUDA_ON_DEVICE_IMPORTS_CUH_ */
