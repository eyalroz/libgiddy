#pragma once
#ifndef SRC_CUDA_ON_DEVICE_PTX_CUH_
#define SRC_CUDA_ON_DEVICE_PTX_CUH_

#include <cstdint>     // for uintXX_t types
#include <type_traits> // for std::is_unsigned

#ifndef STRINGIFY
#define STRINGIFY(_q) #_q
#endif

namespace ptx {

/*
 * First, a pointer-size-related definition. Always use this as (part of) the
 * constraint string for pointer arguments to PTX asm instructions
 * (see http://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#constraints)
 * it is intended to support compilation both in 64-bit and 32-bit modes.
 */

#if defined(_WIN64) || defined(__LP64__)
# define PTR_CONSTRAINT "l"
#else
# define PTR_CONSTRAINT "r"
#endif

/**
 * See @link http://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers
 */
namespace special_registers {


#define DEFINE_SPECIAL_REGISTER_GETTER(special_register_name, value_type, width_in_bits) \
__forceinline__ __device__ value_type special_register_name () \
{ \
	value_type ret;  \
	if (std::is_unsigned<value_type>::value) { \
		asm ("mov.u" STRINGIFY(width_in_bits) " %0, %" STRINGIFY(special_register_name) ";" : "=r"(ret)); \
	} \
	else { \
		asm ("mov.s" STRINGIFY(width_in_bits) " %0, %" STRINGIFY(special_register_name) ";" : "=r"(ret)); \
	} \
	return ret; \
} \

DEFINE_SPECIAL_REGISTER_GETTER( laneid,             uint32_t,          32);
DEFINE_SPECIAL_REGISTER_GETTER( warpid,             uint32_t,          32);
DEFINE_SPECIAL_REGISTER_GETTER( gridid,             uint64_t,          64);
DEFINE_SPECIAL_REGISTER_GETTER( smid,               uint32_t,          32);
DEFINE_SPECIAL_REGISTER_GETTER( nsmid,              uint32_t,          32);
DEFINE_SPECIAL_REGISTER_GETTER( clock,              uint32_t,          32);
DEFINE_SPECIAL_REGISTER_GETTER( clock64,            uint64_t,          64);
DEFINE_SPECIAL_REGISTER_GETTER( lanemask_lt,        uint32_t,          32);
DEFINE_SPECIAL_REGISTER_GETTER( lanemask_le,        uint32_t,          32);
DEFINE_SPECIAL_REGISTER_GETTER( lanemask_eq,        uint32_t,          32);
DEFINE_SPECIAL_REGISTER_GETTER( lanemask_ge,        uint32_t,          32);
DEFINE_SPECIAL_REGISTER_GETTER( lanemask_gt,        uint32_t,          32);
DEFINE_SPECIAL_REGISTER_GETTER( dynamic_smem_size,  uint32_t,          32);
DEFINE_SPECIAL_REGISTER_GETTER( total_smem_size,    uint32_t,          32);

#undef DEFINE_SPECIAL_REGISTER_GETTER

/*
 * Not defining getters for:
 *
 * %tid                      - available as threadIdx
 * %ntid                     - available as blockDim
 * %warpid                   - not interesting
 * %nwarpid                  - not interesting
 * %ctaid                    - available is blockId
 * %nctaid                   - available as gridDim
 * %pm0, ..., %pm7           - not interesting, for now (performance monitoring)
 * %pm0_64, ..., %pm7_64     - not interesting, for now (performance monitoring)
 * %envreg0, ..., %envreg31  - not interesting, for now (performance monitoring)
 */


} // namespace special_registers


template <typename T>
__forceinline__ __device__ T ldg(const T* ptr)
{
#if __CUDA_ARCH__ >= 320
	return __ldg(ptr);
#else
	return *ptr; // maybe we should ld.cg or ld.cs here?
#endif
}

/**
 * See @link http://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-isspacep
 */
#define DEFINE_IS_IN_MEMORY_SPACE(_which_space) \
__forceinline__ __device__ int is_in_ ## _which_space ## _memory (const void *ptr) \
{ \
	int result; \
	asm ("{" \
		".reg .pred p;\n\t" \
		"isspacep." STRINGIFY(_which_space) " p, %1;\n\t" \
		"selp.b32 %0, 1, 0, p;\n\t" \
		"}" \
		: "=r"(result) : PTR_CONSTRAINT(ptr)); \
	return result; \
}

DEFINE_IS_IN_MEMORY_SPACE(const)
DEFINE_IS_IN_MEMORY_SPACE(global)
DEFINE_IS_IN_MEMORY_SPACE(local)
DEFINE_IS_IN_MEMORY_SPACE(shared)

#undef DEFINE_IS_IN_MEMORY_SPACE

#undef PTR_CONSTRAINT

/**
 * @brief Find the last non-sign bit in a signed or an unsigned integer value
 *
 * @note See @link http://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-bfind
 *
 * @param val the value in which to find non-sign bits
 * @return the bit index (counting from least significant bit being 0) of the first
 * bit which is 0 if @p val is positive, or of the first bit which is 1 if @p val is negative. If @p val has only
 * sign bits (i.e. if it's 0 or if its type is signed and its bits are all 1) - the value 0xFFFFFFFF (-1) is returned
 */
#define DEFINE_BFIND(value_type, width_in_bits) \
__forceinline__ __device__ uint32_t bfind(value_type val) \
{ \
	value_type ret;  \
	if (std::is_unsigned<value_type>::value) { \
		asm ("bfind.u" STRINGIFY(width_in_bits) " %0, %1;" : "=r"(ret): "r"(val)); \
	} \
	else { \
		asm ("bfind.s" STRINGIFY(width_in_bits) " %0, %1;" : "=r"(ret): "r"(val)); \
	} \
	return ret; \
}

DEFINE_BFIND(int32_t,  32);
DEFINE_BFIND(int64_t,  64);
DEFINE_BFIND(uint32_t, 32);
DEFINE_BFIND(uint64_t, 64);

} // namespace ptx



#endif /* SRC_CUDA_ON_DEVICE_PTX_CUH_ */
