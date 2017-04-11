/*
 * These are templated wrappers for actual primitives. No function
 * here performs any calculation; non-builtin operations belong in
 * other files.
 */
#ifndef SRC_CUDA_ON_DEVICE_BUILTINS_CUH_
#define SRC_CUDA_ON_DEVICE_BUILTINS_CUH_

#include <device_functions.h>
#include <cub/util_ptx.cuh>
#include "ptx.cuh"

#define __df__ __device__ __forceinline__

namespace builtins {

/*
 * Based on wrappers for built-in instructions from <device_functions.h>
 */

/**
 * When multiplying two n-bit numbers, the result may take up to 2n bits.
 * without upcasting, the value of x * y is the lower n bits of the result;
 * this lets you get the upper bits, without performing a 2n-by-2n multiplication
 */
template <typename T>
__df__  T multiplication_high_bits(std::enable_if<std::is_integral<T>::value, T> x, T y);
__df__  int                multiplication_high_bits( int x,                int y)                { return __mulhi(x, y);    }
__df__  unsigned           multiplication_high_bits( unsigned x,           unsigned y)           { return __umulhi(x, y);   }
__df__  long long          multiplication_high_bits( long long x,          long long y)          { return __mul64hi(x, y);  }
__df__  unsigned long long multiplication_high_bits( unsigned long long x, unsigned long long y) { return __umul64hi(x, y); }

/**
 * Division which becomes faster and less precise than regular "/",
 * when --use-fast-math is specified; otherwise it's the same as regular "/".
 */
template <typename T>
__df__  T divide(std::enable_if<std::is_floating_point<T>::value, T> dividend, T divisor);
__df__  float  divide(float dividend, float divisor)   { return fdividef(dividend, divisor); }
__df__  double divide(double dividend, double divisor) { return fdividef(dividend, divisor); }


template <typename T> __df__ int population_count(T x);
__df__ int population_count(unsigned int x)       { return __popc(x); }
__df__ int population_count(unsigned long long x) { return __popcll(x); }

template <typename T, typename S> __df__ int sum_with_absolute_difference(T x, T y, S z);
__df__ int      sum_with_absolute_difference(int x, int y, int      z) { return __sad (x, y, z); }
__df__ unsigned sum_with_absolute_difference(int x, int y, unsigned z) { return __usad(x, y, z); }

template <typename T> __df__ int absolute_value(T x);
__df__ int             absolute_value(int x)     { return abs(x);   }
__df__ long        absolute_value(long x)        { return labs(x);  }
__df__ long long   absolute_value(long long x)   { return llabs(x); }
__df__ float           absolute_value(float x)   { return fabsf(x); }
__df__ long long   absolute_value(double x)      { return fabs(x);  }

template <typename T> __df__ T bit_reverse(T x);
__df__ int                bit_reverse(int x)                { return __brev(x);   }
__df__ unsigned           bit_reverse(unsigned x)           { return __brev(x);   }
__df__ long long          bit_reverse(long long x)          { return __brevll(x); }
__df__ unsigned long long bit_reverse(unsigned long long x) { return __brevll(x); }

__df__ unsigned warp_ballot     (int cond) { return __ballot(cond); }
__df__ int all_in_warp_satisfy  (int cond) { return __all(cond);    }
__df__ int any_in_warp_satisfies(int cond) { return __any(cond);    }

// 1-based; returns 0 if no bits are set
template <typename T> __df__ int find_first_set(T x);
__df__ int find_first_set(int x)                { return __ffs(x);   }
__df__ int find_first_set(unsigned int x)       { return __ffs(x);   }
__df__ int find_first_set(long long x)          { return __ffsll(x); }
__df__ int find_first_set(unsigned long long x) { return __ffsll(x); }

template <typename T> __df__ int count_leading_zeros(T x);
__df__ int count_leading_zeros(int x)                { return __clz(x);   }
__df__ int count_leading_zeros(unsigned int x)       { return __clz(x);   }
__df__ int count_leading_zeros(long long x)          { return __clzll(x); }
__df__ int count_leading_zeros(unsigned long long x) { return __clzll(x); }

template <typename T> __df__ T minimum(T x, T y);
__df__ int                 minimum(int x, int y)                               { return min(x,y);    }
__df__ unsigned int        minimum(unsigned int x, unsigned int y)             { return umin(x,y);   }
__df__ long long           minimum(long x, long y)                             { return llmin(x,y);  }
__df__ unsigned long long  minimum(unsigned long x, unsigned long y)           { return ullmin(x,y); }
__df__ long long           minimum(long long x, long long y)                   { return llmin(x,y);  }
__df__ unsigned long long  minimum(unsigned long long x, unsigned long long y) { return ullmin(x,y); }
__df__ float               minimum(float x, float y)                           { return fminf(x,y);  }
__df__ double              minimum(double x, double y)                         { return fmin(x,y);   }

template <typename T> __df__ T maximum(T x, T y);
__df__ int                 maximum(int x, int y)                               { return max(x,y);    }
__df__ unsigned int        maximum(unsigned int x, unsigned int y)             { return umax(x,y);   }
__df__ long long           maximum(long x, long y)                             { return llmax(x,y);  }
__df__ unsigned long long  maximum(unsigned long x, unsigned long y)           { return ullmax(x,y); }
__df__ long long           maximum(long long x, long long y)                   { return llmax(x,y);  }
__df__ unsigned long long  maximum(unsigned long long x, unsigned long long y) { return ullmax(x,y); }
__df__ float               maximum(float x, float y)                           { return fmaxf(x,y);  }
__df__ double              maximum(double x, double y)                         { return fmax(x,y);   }

/*
 * Based on wrappers for built-in instructions from <cub/util_ptx.cuh>
 */

template <typename T> __df__ T shift_right_then_add(T x, unsigned int shift, T addend);
__df__ unsigned shift_right_then_add(unsigned x, unsigned int shift, unsigned addend) { return cub::SHR_ADD(x, shift, addend); }

template <typename T> __df__ T shift_left_then_add(T x, unsigned int shift, T addend);
__df__ unsigned shift_left_then_add(unsigned x, unsigned int shift, unsigned addend) { return cub::SHL_ADD(x, shift, addend); }

/**
 * Extracts the bits with 0-based indices start_pos...start_pos+length-1, counting
 * from least to most significant, from a bit field field. Has sign extension semantics
 * for signed inputs which are bit tricky, see in the PTX ISA guide:
 *
 * http://docs.nvidia.com/cuda/parallel-thread-execution/index.html
 *
 * TODO: CUB 1.5.2's BFE wrapper seems kind of fishy. Why does Duane Merill not use PTX for extraction from 64-bit fields?
 * For now only adopting his implementation for the 32-bit case
 */
template <typename T> __df__ T extract_from_bit_field(T bit_field, unsigned int start_pos, unsigned int num_bits);

__df__ unsigned extract_from_bit_field(unsigned bit_field, unsigned int start_pos, unsigned int num_bits) { return cub::BFE(bit_field, start_pos, num_bits); }

template <typename T> __df__ T insert_into_bit_field(T bit_field, T bits_to_insert, unsigned int start_pos, unsigned int num_bits);
__df__ unsigned insert_into_bit_field(unsigned bit_field, unsigned bits_to_insert, unsigned int start_pos, unsigned int num_bits)
{
	unsigned ret;
	cub::BFI(ret, bit_field, bits_to_insert, start_pos, num_bits);
	return ret;
}

template <typename T> __df__ T add_three(T x, T y, T z);
__df__ unsigned add_three(unsigned x, unsigned y, unsigned z) { return cub::IADD3(x, y, z); }

template <typename T> __df__ T select_bytes(T x, T y, unsigned byte_selector);
__df__ unsigned select_bytes(unsigned x, unsigned y, unsigned byte_selector) { return cub::PRMT(x, y, byte_selector); }


/*
 * Special register getter wrappers
 */

__df__ unsigned           lane_index()                     { return ptx::special_registers::laneid();            }
__df__ unsigned           symmetric_multiprocessor_index() { return ptx::special_registers::smid();              }
__df__ unsigned long long grid_index()                     { return ptx::special_registers::gridid();            }
__df__ unsigned int       dynamic_shared_memory_size()     { return ptx::special_registers::dynamic_smem_size(); }
__df__ unsigned int       total_shared_memory_size()       { return ptx::special_registers::total_smem_size();   }

namespace lane_masks {
__df__ unsigned int       preceding()                      { return ptx::special_registers::lanemask_lt();       }
__df__ unsigned int       preceding_and_self()             { return ptx::special_registers::lanemask_le();       }
__df__ unsigned int       self()                           { return ptx::special_registers::lanemask_eq();       }
__df__ unsigned int       succeeding_and_self()            { return ptx::special_registers::lanemask_ge();       }
__df__ unsigned int       succeeding()                     { return ptx::special_registers::lanemask_gt();       }
} // namespace lane_masks

} // namespace builtins

#endif /* SRC_CUDA_ON_DEVICE_BUILTINS_CUH_ */
