#ifndef CUDA_MISCELLANY_H_
#define CUDA_MISCELLANY_H_

#include "cuda/api/types.h"

#include <type_traits>

namespace cuda {

/**
 * Use this type when walking index variables collaboratively among
 * multiple threads along some array. The reason you can't just use
 * the original size type is twofold:
 *
 * 1. Occasionally you might have
 *
 *      pos += blockDim.x * blockDim.y * blockDim.z;
 *
 *    e.g. when you work at block stride on a linear input. Well,
 *    if the type of pos is not large enough (e.g. char) - you would
 *    get into infinite loops.
 *
 * 2. The native integer type on a GPU is 32-bit - at least on CUDA;
 *    so there's not much sense of keeping an index variable in some
 *    smaller type. At best, the compiler will switch it to a 32-bit
 *    value; at worst, you'll waste time putting it into and out of
 *    32-bit variables
 */
template <typename Size>
using promoted_size = typename std::common_type<Size, native_word_t>::type;

} // namespace cuda

#endif /* CUDA_MISCELLANY_H_ */
