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
template <typename T> __df__  T multiplication_high_bits(T x, T y);
template <> __df__  int                multiplication_high_bits<int               >( int x,                int y)                { return __mulhi(x, y);    }
template <> __df__  unsigned           multiplication_high_bits<unsigned          >( unsigned x,           unsigned y)           { return __umulhi(x, y);   }
template <> __df__  long long          multiplication_high_bits<long long         >( long long x,          long long y)          { return __mul64hi(x, y);  }
template <> __df__  unsigned long long multiplication_high_bits<unsigned long long>( unsigned long long x, unsigned long long y) { return __umul64hi(x, y); }

/**
 * Division which becomes faster and less precise than regular "/",
 * when --use-fast-math is specified; otherwise it's the same as regular "/".
 */
template <typename T> __df__  T divide(T dividend, T divisor);
template <> __df__  float  divide<float >(float dividend, float divisor)   { return fdividef(dividend, divisor); }
template <> __df__  double divide<double>(double dividend, double divisor) { return fdividef(dividend, divisor); }


template <typename T> __df__ int population_count(T x);
template <> __df__ int population_count<unsigned          >(unsigned int x)       { return __popc(x); }
template <> __df__ int population_count<unsigned long long>(unsigned long long x) { return __popcll(x); }

template <typename T, typename S> __df__ S sum_with_absolute_difference(T x, T y, S z);
template <> __df__ int      sum_with_absolute_difference<int     >(int x, int y, int      z)           { return __sad (x, y, z); }
template <> __df__ unsigned sum_with_absolute_difference<unsigned>(unsigned x, unsigned y, unsigned z) { return __usad(x, y, z); }

template <typename T> __df__ T absolute_value(T x);
template <> __df__ int         absolute_value<int                >(int x)         { return abs(x);   }
template <> __df__ long        absolute_value<long               >(long x)        { return labs(x);  }
template <> __df__ long long   absolute_value<long long          >(long long x)   { return llabs(x); }
template <> __df__ float       absolute_value<float              >(float x)       { return fabsf(x); }
template <> __df__ double      absolute_value<double             >(double x)      { return fabs(x);  }

template <typename T> __df__ T bit_reverse(T x);
template <> __df__ int                bit_reverse<int               >(int x)                { return __brev(x);   }
template <> __df__ unsigned           bit_reverse<unsigned          >(unsigned x)           { return __brev(x);   }
template <> __df__ long long          bit_reverse<long long         >(long long x)          { return __brevll(x); }
template <> __df__ unsigned long long bit_reverse<unsigned long long>(unsigned long long x) { return __brevll(x); }

__df__ unsigned warp_ballot     (int cond) { return __ballot(cond); }
__df__ int all_in_warp_satisfy  (int cond) { return __all(cond);    }
__df__ int any_in_warp_satisfies(int cond) { return __any(cond);    }

template <typename T> __df__ uint32_t find_last_non_sign_bit(T x);
template <>  __df__ uint32_t find_last_non_sign_bit<int               >(int x)                  { return ptx::bfind(x);            }
template <>  __df__ uint32_t find_last_non_sign_bit<unsigned          >(unsigned x)             { return ptx::bfind(x);            }
template <>  __df__ uint32_t find_last_non_sign_bit<long              >(long x)                 { return ptx::bfind((int64_t) x);  }
template <>  __df__ uint32_t find_last_non_sign_bit<unsigned long     >(unsigned long x)        { return ptx::bfind((uint64_t) x); }
template <>  __df__ uint32_t find_last_non_sign_bit<long long         >(long long x)            { return ptx::bfind((int64_t) x);  }
template <>  __df__ uint32_t find_last_non_sign_bit<unsigned long long>(unsigned long long  x)  { return ptx::bfind((uint64_t) x); }

template <typename T>
__forceinline__ __device__ T load_global_with_non_coherent_cache(const T* ptr)  { return ptx::ldg(ptr); }

template <typename T> __df__ int count_leading_zeros(T x);
template <> __df__ int count_leading_zeros<int               >(int x)                { return __clz(x);   }
template <> __df__ int count_leading_zeros<unsigned          >(unsigned int x)       { return __clz(x);   }
template <> __df__ int count_leading_zeros<long long         >(long long x)          { return __clzll(x); }
template <> __df__ int count_leading_zeros<unsigned long long>(unsigned long long x) { return __clzll(x); }

template <typename T> __df__ T minimum(T x, T y);
template <> __df__ int                 minimum<int               >(int x, int y)                               { return min(x,y);    }
template <> __df__ unsigned int        minimum<unsigned          >(unsigned int x, unsigned int y)             { return umin(x,y);   }
template <> __df__ long                minimum<long              >(long x, long y)                             { return llmin(x,y);  }
template <> __df__ unsigned long       minimum<unsigned long     >(unsigned long x, unsigned long y)           { return ullmin(x,y); }
template <> __df__ long long           minimum< long long        >(long long x, long long y)                   { return llmin(x,y);  }
template <> __df__ unsigned long long  minimum<unsigned long long>(unsigned long long x, unsigned long long y) { return ullmin(x,y); }
template <> __df__ float               minimum<float             >(float x, float y)                           { return fminf(x,y);  }
template <> __df__ double              minimum<double            >(double x, double y)                         { return fmin(x,y);   }

template <typename T> __df__ T maximum(T x, T y);
template <> __df__ int                 maximum<int               >(int x, int y)                               { return max(x,y);    }
template <> __df__ unsigned int        maximum<unsigned          >(unsigned int x, unsigned int y)             { return umax(x,y);   }
template <> __df__ long                maximum<long              >(long x, long y)                             { return llmax(x,y);  }
template <> __df__ unsigned long       maximum<unsigned long     >(unsigned long x, unsigned long y)           { return ullmax(x,y); }
template <> __df__ long long           maximum< long long        >(long long x, long long y)                   { return llmax(x,y);  }
template <> __df__ unsigned long long  maximum<unsigned long long>(unsigned long long x, unsigned long long y) { return ullmax(x,y); }
template <> __df__ float               maximum<float             >(float x, float y)                           { return fmaxf(x,y);  }
template <> __df__ double              maximum<double            >(double x, double y)                         { return fmax(x,y);   }

/*
 * Based on wrappers for built-in instructions from <cub/util_ptx.cuh>
 */

template <typename T> __df__ T shift_right_then_add(T x, unsigned int shift, T addend);
template <> __df__ unsigned shift_right_then_add(unsigned x, unsigned int shift, unsigned addend) { return cub::SHR_ADD(x, shift, addend); }

template <typename T> __df__ T shift_left_then_add(T x, unsigned int shift, T addend);
template <> __df__ unsigned shift_left_then_add(unsigned x, unsigned int shift, unsigned addend) { return cub::SHL_ADD(x, shift, addend); }

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

template <> __df__ unsigned extract_from_bit_field(unsigned bit_field, unsigned int start_pos, unsigned int num_bits) { return cub::BFE(bit_field, start_pos, num_bits); }

template <typename T> __df__ T insert_into_bit_field(T bit_field, T bits_to_insert, unsigned int start_pos, unsigned int num_bits);
template <> __df__ unsigned insert_into_bit_field(unsigned bit_field, unsigned bits_to_insert, unsigned int start_pos, unsigned int num_bits)
{
	unsigned ret;
	cub::BFI(ret, bit_field, bits_to_insert, start_pos, num_bits);
	return ret;
}

template <typename T> __df__ T add_three(T x, T y, T z);
template <> __df__ unsigned add_three(unsigned x, unsigned y, unsigned z) { return cub::IADD3(x, y, z); }

template <typename T> __df__ T select_bytes(T x, T y, unsigned byte_selector);
template <> __df__ unsigned select_bytes(unsigned x, unsigned y, unsigned byte_selector) { return cub::PRMT(x, y, byte_selector); }


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
