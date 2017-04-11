#pragma once
#ifndef CUDA_BIT_OPERATIONS_CUH_
#define CUDA_BIT_OPERATIONS_CUH_

#include <type_traits>

#ifdef __CUDA_ARCH__
#include "cuda/on_device/builtins.cuh"
#else
#include "cuda/faux_builtins.hpp"
#endif

#ifndef CHAR_BIT
#define CHAR_BIT 8
#endif

// TODO: Address potential redundancies between code here and host code

namespace cuda {

template <typename T>
struct size_in_bits { enum { value = sizeof(T) * CHAR_BIT }; };

#ifdef __CUDACC__
#define __fhd__  __forceinline__ __host__ __device__
#define __fd__   __forceinline__ __device__
#else
#define __fhd__ inline
#define __fd__  inline
#endif

/*
 * Bring up builtins into this namespace
 *
 * The code here is intended to be used mostly on the device; and there we obviously want to use
 * nVIDIA's SM's built-in instructions. On the host, however, they can't be used - yet we want
 * the code to read the same. The answer? Using a different namespace alias when compiling host-side
 * and device-side code; the host-side code, in faux_builtins, is obviously often slower.
 */

#ifdef __CUDA_ARCH__
using namespace builtins;
#else
namespace builtins = faux_builtins;
#endif


/*
 * Functions which are primitives on many other platforms, but not on CUDA devices
 */

template <typename T> constexpr __fhd__ T rotate_left(T x, unsigned num_positions)
{
	return x << num_positions | x >> (size_in_bits<T>::value - num_positions);
}
template <typename T> constexpr __fhd__ T rotate_right(T x, unsigned num_positions)
{
	return x >> num_positions | x << (size_in_bits<T>::value - num_positions);
}

/**
 * @brief counts the number initial zeros when considering the binary representation
 * of a number from least to most significant digit
 * @param x the number whose representation is to be counted
 * @return the number of initial zero bits before the first 1; if x is 0, -1 is returned
 */
template <typename T> __fhd__ int count_trailing_zeros(T x) { return builtins::find_first_set(x) - 1; }

/**
 * Get the 1-based bit index of the last bit in a number which is set (to 1)
 *
 * @tparam[in] StrictSemantics find_last_set is supposed to return 0 if no
 * bits are set in the argument. However, CUDA does not currently have such a
 * primitive, and we must use other ones, with the transformation not maintaining
 * these semantics. We thus have the choice of either wasting a few more operations
 * on correcting the result, or be more frugal but have slightly different semantics.
 * @param x the value whose bits are to be examined
 * @return the 1-based index of the last set bit of {@ref x}; if no bit is set, we return 0
 * for StrictSemantics true or 33 for StrictSemantics false.
 */
template <typename T, bool StrictSemantics = true> __fhd__ int find_last_set(T x)
{
	if (StrictSemantics) {
		auto reverse_ffs = find_first_set(reverse_bits(x));
		return (!!reverse_ffs) * (sizeof(T) * CHAR_BIT + 1 - reverse_ffs) ;
	}
	else {
		return (sizeof(T) * CHAR_BIT + 1) - find_first_set(reverse_bits(x)) ;

	}
}

// Some aliases for builtins
template <typename T> __fhd__ int count_one_bits(T x)  { return builtins::population_count(x); }
template <typename T> __fhd__ int count_bits_set(T x)  { return builtins::population_count(x); }
template <typename T> __fhd__ int reverse_bits(T x)    { return builtins::bit_reverse(x);      }


/*
 * Odds and ends
 */

/**
 * Keep only a single bit (be it 0 or 1) from a value,
 * zeroing all the rest.
 *
 * @param x The original value
 * @param bit_index_to_keep this bit will be the same in the output as in x; 0-based
 * @return an all-zero value in all bits except at {@ref bit_index_to_keep}, and the
 * same value as x at that bit
 */
template <typename T>
__fhd__ T keep_single_bit(T x, unsigned bit_index_to_keep)
{
	return x & 1 << bit_index_to_keep;
}

// TODO: Make sure this behaves as expected when no bits
// are set, i.e. overflows and gives us 0
template <typename T>
__fhd__ T keep_only_lowest_set_bit(T x)
{
	return ((T)1) << count_trailing_zeros(x);
}

// TODO: Make sure this behaves as expected when no bits
// are set, i.e. overflows and gives us 0
template <typename T>
__fhd__ T keep_only_highest_set_bit(T x)
{
	return ((T)1) << count_leading_zeros(x);
}

/**
 * Discard (zero-out) all bits of x starting
 * at the bit with index {@ref bit_index}, counting
 * from the LSB to the MSB
 *
 * @param x A value (or rather, a sequence of bits)
 * @param bit_index the index of the first bit to discard; 0-based
 * @return The bits of x of indices 0...bit_index-1 (0-based)
 */
template <typename T>
constexpr __fhd__ unsigned int keep_bits_before(T x, unsigned bit_index)
{
	return x & ((1 << bit_index) - 1);
}

/**
 * Discard (zero-out) all bits of x after
 * the bit with index {@ref bit_index}, counting
 * from the LSB to the MSB
 *
 * @param x A value (or rather, a sequence of bits)
 * @param bit_index the index of the last bit not to discard; 0-based
 * @return The bits of x of indices 0...bit_index (0-based)
 */
template <typename T>constexpr
__fhd__ unsigned int keep_bits_up_to(T x, unsigned bit_index)
{
	return keep_bits_before(x, bit_index + 1);
}

/**
 * Discard (zero-out) all bits of x with (0-based) index
 * less than {@ref bit_index}, counting
 * from the LSB to the MSB
 *
 * @param x A value (or rather, a sequence of bits)
 * @param bit_index the index of the first bit to discard; 0-based
 * @return All bits of x starting at {@ref bit_index} (0-based),
 * in their original positions, i.e. not shifted
 */
template <typename T> constexpr
__fhd__ unsigned int keep_bits_starting_from(T x, unsigned bit_index)
{
	return x & ~((1 << bit_index) - 1);
}

/**
 * Discard (zero-out) all bits of x with (0-based) index
 * at most {@ref bit_index}, counting
 * from the LSB to the MSB
 *
 * @param x A value (or rather, a sequence of bits)
 * @param bit_index the index of the last bit to discard; 0-based
 * @return All bits of x starting at {@ref bit_index}-1 (0-based),
 * in their original positions, i.e. not shifted
 */
template <typename T> constexpr
__fhd__ unsigned int keep_bits_after(T x, unsigned bit_index)
{
	return keep_bits_starting_from(x, bit_index + 1);
}

/**
 * Finds the lowest bit set in x with index no lower than {@ref bit_index}
 *
 * @note same semantics as find_first_set - 1-based with 0 for no
 * relevant bit set.
 *
 * @param x A type interpreted as a seuqnece of bits, LSB to MSB
 * @param bit_index an index in the sequence at which to start looking for 1's
 * @return the minimum index which is at least {@ref bit_index} such that {@ref x} has 1
 * at that bit; 0 if no such bit exists
 */
template <typename T>
__fhd__ unsigned int find_first_set_starting_from(T x, unsigned bit_index)
{
	return find_first_set(keep_bits_starting_from(x, bit_index));
}

template <typename T>
__fhd__ unsigned int find_first_set_after(T x, unsigned bit_index)
{
	return find_first_set_starting_from(x, bit_index + 1);
}


template <typename T, bool StrictSemantics = true>
__fhd__ unsigned int find_last_set_up_to(T x, unsigned bit_index)
{
	return find_last_set<T, StrictSemantics>(keep_bits_up_to(x, bit_index));
}

// Notes: this won't work with bit_index = 0
template <typename T, bool StrictSemantics = true>
__fhd__ unsigned int find_last_set_before_alt(T x, unsigned bit_index)
{
	return find_last_set_up_to(x, bit_index - 1);
}

template <unsigned PowerOf2, typename T>
__fhd__ T constexpr round_up_to_multiple_of_power_of_2(T x)
{
	/*
	 * This relies on:
	 * 1. PowerOf2 actually being a power of 2
	 * 2. comparisons resulting in either 0 or 1. Apparently this holds in CUDA.
	 */
	return x + (x & (PowerOf2 - 1) == 0) * PowerOf2;
}

/**
 * Round down a number to the closest multiple of a modulus - when the modulus is
 * known power of 2.
 *
 * @note Useful when y is _not_ known to be a power of 2 _at_compile_time_, by the
 * compiler - but you the coder know it will indeed always be a power of 2.
 *
 * @tparam T unsigned integral type
 * @tparam S unsigned integral type
 * @param x any value
 * @param modulus a power of 2 (that is, 1, 2, 4, 8 etc. - but not 0)
 * @return the value of x - x % modulus
 */template <typename T, typename S>
__fd__ constexpr
typename std::common_type<T,S>::type round_down_to_multiple_of_power_of_2(const T& x, const S& modulus)
{
	using result_type = typename std::common_type<T,S>::type;
	return ( (result_type) x) & ~( ((result_type) modulus) - 1 );
}

 /**
  * Computes the modular value of a number when the modulus is known to be a power of 2.
  *
  * @note Useful when y is _not_ known to be a power of 2 _at_compile_time_, by the
  * compiler - but you the coder know it will indeed always be a power of 2.
  *
  * @tparam T unsigned integral type
  * @tparam S unsigned integral type
  * @param x any value
  * @param modulus 2 a power of 2 (that is, 1, 2, 4, 8 etc. - but not 0)
  * @return the value of x % modulus
  */
 template <typename T, typename S>
 __fd__ constexpr
 typename std::common_type<T,S>::type modulo_power_of_2(const T x, const S& modulus) {
 	using result_type = typename std::common_type<T,S>::type;
 	return ( (result_type) x) & ( ((result_type) modulus) - 1);
 }


/**
 * @note careful, this may overflow!
 */
template <typename T, typename S>
__fd__ constexpr
typename std::common_type<T,S>::type
round_up_to_multiple_of_power_of_2(const T& x, const S& power_of_2) {
	using result_type = typename std::common_type<T,S>::type;
	return round_down_to_multiple_of_power_of_2 (
		(result_type) x + (result_type) power_of_2 - 1,
		(result_type) power_of_2);
}

template <typename T> __fhd__ T round_up_to_next_power_of_2(T x)
{
	// Can we reduce the number of operations on this thing?
	return keep_only_highest_set_bit(x - 1) << 1;
}

template <typename T> constexpr __fhd__ T all_one_bits()
{
	return ~((T)0);
}

template <typename T> constexpr __fhd__ T one_bits(unsigned k)
{
	return (((T)1) << k) - 1;
}

// Results undefined for x < 1...
template <typename T> __fhd__ int ilog_2(T x)
{
	return highest_bit_set(x) - 1;
}

template <typename T> constexpr __fhd__ T lowest_k_bits(T x, unsigned k)
{
	return x & ((1 << k) - 1);
}

template <typename T> constexpr __fhd__  T highest_k_bits(T x, unsigned k)
{
// more verbose C++14 version of this:
//	auto num_low_bits = size_in_bits<T>::value - k;
//	return x >> num_low_bits ;
//
// ... which is the same as:
	return x >> (size_in_bits<T>::value - k);
}

template <typename T> T constexpr __fhd__  bit_subsequence(T x, unsigned starting_bit, unsigned num_bits)
{
	return lowest_k_bits(x >> starting_bit, num_bits);
}


template <typename T> constexpr __fhd__  T clear_lower_k_bits(T x, unsigned k)
{
	return x & ~((1 << k) - 1);
}

template <typename T> constexpr __fhd__  T clear_higher_k_bits(T x, unsigned k)
{
// more verbose C++14 version of this:
//	auto num_bits_to_clear = size_in_bits<T>::value - k;
//	return lowest_k_bits(num_bits_to_clear);
	return lowest_k_bits(size_in_bits<T>::value - k);
}

template <typename T> constexpr __fhd__  T clear_bit(T x, unsigned bit_index_to_clear)
{
	return (x & ~(1 << bit_index_to_clear));
}

#undef __fhd__
#undef __fd__

} // namespace cuda



#endif /* CUDA_BIT_OPERATIONS_CUH_ */
