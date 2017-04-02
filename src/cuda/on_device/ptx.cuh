#pragma once
#ifndef SRC_CUDA_ON_DEVICE_PTX_CUH_
#define SRC_CUDA_ON_DEVICE_PTX_CUH_

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

namespace special_registers {


#define DEFINE_SPECIAL_REGISTER_GETTER(special_register_name, value_type, width_in_bits) \
__forceinline__ __device__ value_type special_register_name () \
{ \
	value_type ret;  \
	if (std::is_unsigned<value_type>::value) { \
		asm volatile ("mov.u" STRINGIFY(width_in_bits) " %0, %" STRINGIFY(special_register_name) ";" : "=r"(ret)); \
	} \
	else { \
		asm volatile ("mov.s" STRINGIFY(width_in_bits) " %0, %" STRINGIFY(special_register_name) ";" : "=r"(ret)); \
	} \
	return ret; \
} \

DEFINE_SPECIAL_REGISTER_GETTER( laneid,             unsigned,           32);
DEFINE_SPECIAL_REGISTER_GETTER( warpid,             unsigned,           32);
DEFINE_SPECIAL_REGISTER_GETTER( gridid,             unsigned long long, 64);
DEFINE_SPECIAL_REGISTER_GETTER( smid,               unsigned,           32);
DEFINE_SPECIAL_REGISTER_GETTER( nsmid,              unsigned,           32);
DEFINE_SPECIAL_REGISTER_GETTER( clock,              unsigned,           32);
DEFINE_SPECIAL_REGISTER_GETTER( clock64,            unsigned long long, 64);
DEFINE_SPECIAL_REGISTER_GETTER( lanemask_lt,        unsigned,           32);
DEFINE_SPECIAL_REGISTER_GETTER( lanemask_le,        unsigned,           32);
DEFINE_SPECIAL_REGISTER_GETTER( lanemask_eq,        unsigned,           32);
DEFINE_SPECIAL_REGISTER_GETTER( lanemask_ge,        unsigned,           32);
DEFINE_SPECIAL_REGISTER_GETTER( lanemask_gt,        unsigned,           32);
DEFINE_SPECIAL_REGISTER_GETTER( dynamic_smem_size,  unsigned,           32);
DEFINE_SPECIAL_REGISTER_GETTER( total_smem_size,    unsigned,           32);

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
template <typename T>
__forceinline__ __device__ T load_global_with_non_coherent_cache(const T* ptr) { return ldg(ptr); }

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

} // namespace ptx



#endif /* SRC_CUDA_ON_DEVICE_PTX_CUH_ */
