#pragma once
#ifndef SRC_KERNELS_REDUCTION_MULTI_REDUCE_DENSE_TO_DENSE_CUH
#define SRC_KERNELS_REDUCTION_MULTI_REDUCE_DENSE_TO_DENSE_CUH

/**
 * Multi-reduce is the task of performing multiple reductions with the same kernel,
 * or sequence of kernels. That is, one has data intended for different 'bins', or
 * groups if you will, and the result of the multi-reduce is the reduction results
 * for all bins (or all bins with any data).
 *
 * This file contains kernels for the DENSE TO DENSE variant:
 *
 * DENSE input: These are multiple vectors of values, all of the same length;
 * if we fix some index i into these vectors, the inputs are exactly those
 * of a regular reduction. The (implicit) index of each element in each vector
 * is its 'group index' or 'reduction index'.
 *
 * DENSE output: This is a single vector of bins, each of which holds
 * the result of a single reduction. The 'reduction index' or 'group index'
 * of each element is implicit - its index in the vector.
 *
 * Another technical caveat is that these kernels do not actually initialize the
 * data, only continue reducing with it, so that it is not their responsibility,
 * for example, to ensure untouched bin reduction results are actualy neutral.
 * For the case of summation, input data will only be added to existing bin
 * results, and it should be a good idea to zero them out in advance.
 */

#include "kernels/common.cuh"
#include "common.cuh"

namespace cuda {
namespace kernels {
namespace reduction {

} // namespace reduction
} // namespace kernels
} // namespace cuda

#endif /* SRC_KERNELS_REDUCTION_MULTI_REDUCE_DENSE_TO_DENSE_CUH */
