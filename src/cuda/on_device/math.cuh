#pragma once
#ifndef SRC_CUDA_ON_DEVICE_MATH_CUH_
#define SRC_CUDA_ON_DEVICE_MATH_CUH_

// Though this is an include file, it should not be visible outside
// the op store's implementation
#include "cuda/api/constants.h"
#include "cuda/syntax_replacement.h"
#include "cuda/api/types.h"
#include "cuda/on_device/builtins.cuh"
	// for absolute_value(), sum_of_absolute_differences(), minimum(), maximum() etc...

#define __fd__ __forceinline__ __device__

template <typename T>
__fd__ constexpr bool is_power_of_2(T val) { return (val & (val-1)) == 0; }
	// Yes, this works: Only if val had exactly one 1 bit will subtracting 1 switch
	// all of its 1 bits.


template <typename T>
constexpr inline T& modular_inc(T& x, T modulus) { return (x+1) % modulus; }

template <typename T>
constexpr inline T& modular_dec(T& x, T modulus) { return (x-1) % modulus; }

namespace detail {

template <typename T>
constexpr T ipow(T base, unsigned exponent, T coefficient) {
	return exponent == 0 ? coefficient :
		ipow(base * base, exponent >> 1, (exponent & 0x1) ? coefficient * base : coefficient);
}

} // namespace detail

template <typename T>
constexpr T ipow(T base, unsigned exponent)
{
	return detail::ipow(base, exponent, 1);
}

template <typename T, typename S>
__fd__ constexpr T div_rounding_up(const T& dividend, const S& divisor)
{
	return (dividend + divisor - 1) / divisor;
}

template <typename T, typename S>
__fd__ constexpr T div_by_power_of_2(const T& dividend, const S& divisor)
{
	// TODO: Can we do better than this?
#ifndef __CUDA_ARCH__
#ifdef SRC_UTIL_BUILTINS_HPP_
	return dividend >> util::builtins::count_trailing_zeros(divisor);
#else
	return dividend >> (builtins::find_first_set(divisor) - 1);
#endif
#else
	return dividend >> (builtins::find_first_set(divisor) - 1);
#endif
}

template <typename T>
constexpr __fd__ int log2_constexpr(T val) { return val ? 1 + log2_constexpr(val >> 1) : -1; }

template <typename T, typename S, S Divisor>
__fd__ constexpr T div_by_fixed_power_of_2(const T& dividend)
{
	return dividend >> log2_constexpr(Divisor);
}


template <typename T, typename S>
__fd__ constexpr T div_by_power_of_2_rounding_up(const T& dividend, const S& divisor)
{
	return div_by_power_of_2(dividend, divisor) + ((dividend & (divisor-1)) > 0);
}

template <typename T, typename S, S Divisor>
__fd__ constexpr T div_by_fixed_power_of_2_rounding_up(const T& dividend)
{
	return div_by_fixed_power_of_2<T,S,Divisor>(dividend) + ((dividend & (Divisor-1)) > 0);
}

template <typename T>
__fd__ constexpr T num_warp_sizes_to_cover(const T& x)
{
	return div_by_fixed_power_of_2<T,unsigned short, warp_size>(x) + ((x & (warp_size-1)) > 0);
}

template <typename T, typename S>
__fd__ constexpr T round_down(const T& x, const S& y)
{
	return x - x%y;
}

template <typename T>
__fd__ constexpr T round_down_to_warp_size(typename std::enable_if<std::is_unsigned<T>::value, T>::type const & x)
{
	return x & warp_size - 1;
}

template <typename T, typename S>
__fd__ constexpr T round_up(const T& x, const S& y)
{
	return round_down(x+y-1, y);
}

template <typename T, typename S>
__fd__ constexpr typename std::common_type<T,S>::type
round_down_to_power_of_2(const T& x, const S& power_of_2)
{
	using result_type = typename std::common_type<T,S>::type;
	return ((result_type) x) & ~(((result_type) power_of_2) - 1);
}

/**
 * @note careful, this may overflow!
 */
template <typename T, typename S>
__fd__ constexpr typename std::common_type<T,S>::type
round_up_to_power_of_2(const T& x, const S& power_of_2) {
	using result_type = typename std::common_type<T,S>::type;
	return round_down_to_power_of_2 ((result_type) x + (result_type) power_of_2 - 1, (result_type) power_of_2);
}

/**
 * @note careful, this may overflow!
 */
template <typename T>
__fd__ constexpr T round_up_to_full_warps(const T& x) {
	return round_up_to_power_of_2(x, warp_size);
}

template <typename T, typename Lower = T, typename Upper = T>
constexpr inline bool between_or_equal(const T& x, const Lower& l, const Upper& u) { return (l <= x) && (x <= u); }

template <typename T, typename Lower = T, typename Upper = T>
constexpr inline bool strictly_between(const T& x, const Lower& l, const Upper& u) { return (l < x) && (x < u); }

// TODO: I don't like this implementation much

template <typename T>
__fd__ T gcd(T u, T v)
{
	while (v != 0) {
		T r = u % v;
		u = v;
		v = r;
	}
	return u;
}

template <typename T>
__fd__ T lcm(T u, T v)
{
	return (u / gcd(u,v)) * v;
}

template <typename T>
__device__ constexpr T gcd_constexpr(T u, T v)
{
	return (v == 0) ? u : gcd_constexpr(v, u % v);
}

template <typename T>
__device__ constexpr T lcm_constexpr(T u, T v)
{
	return (u / gcd_constexpr(u,v)) * v;
}


namespace detail {
template <typename T>
__device__ constexpr T sqrt_constexpr_helper(T x, T low, T high)
{
	// this ugly macro cant be replaced by a lambda
	// or the use of temporary variable, as in C++11, a constexpr
	// function must have a single statement
#define sqrt_constexpr_HELPER_MID ((low + high + 1) / 2)
	return low == high ?
		low :
		((x / sqrt_constexpr_HELPER_MID < sqrt_constexpr_HELPER_MID) ?
			sqrt_constexpr_helper(x, low, sqrt_constexpr_HELPER_MID - 1) :
			sqrt_constexpr_helper(x, sqrt_constexpr_HELPER_MID, high));
#undef sqrt_constexpr_HELPER_MID
}

} // namespace detail

template <typename T>
__device__ constexpr T sqrt_constexpr(T& x)
{
  return detail::sqrt_constexpr_helper(x, 0, x / 2 + 1);
}

template <typename T>
__fd__ unsigned ilog2(std::enable_if<std::is_unsigned<T>::value, T> x) {
  return (CHAR_BIT * sizeof(T) - 1) - builtins::count_leading_zeros(x);
}

#undef __fd__

#endif /* SRC_CUDA_ON_DEVICE_MATH_CUH_ */
