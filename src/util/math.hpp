#pragma once
#ifndef SRC_UTIL_MATH_HPP_
#define SRC_UTIL_MATH_HPP_

#include <cmath>
#include <stdexcept>
#include <type_traits> // for std::enable_if
#include <utility> // for std::move
#include "util/builtins.hpp"

// TODO: Insert everything else in this file into the util namespace
namespace util {

template <typename T, typename LB, typename UB>
constexpr typename std::common_type<T, UB, LB>::type
clip(const T& x, const LB& lower_bound, const UB& upper_bound)
{
	using common_type = typename std::common_type<T, UB, LB>::type;
	return (common_type) x < (common_type) lower_bound ?
		lower_bound : ( (common_type) x > (common_type) upper_bound ? upper_bound : x );
}

template <typename T, typename LB, typename UB>
constexpr typename std::common_type<T, UB, LB>::type
clamp(const T& x, const UB& lower_bound, const UB& upper_bound)
{
	return clip(x, lower_bound, upper_bound);
}

// TODO: Maybe place the next two in separate files
template <typename T, typename U = T>
constexpr T& update_min(T& minimum_to_update, const U& new_value)
{
	return (minimum_to_update > new_value) ? minimum_to_update = new_value : minimum_to_update;
}

template <typename T, typename U = T>
constexpr T& update_max(T& maximum_to_update, const U& new_value)
{
	return (maximum_to_update < new_value) ? maximum_to_update = new_value : maximum_to_update;
}

/**
* Divides the left-hand-side by the right-hand-side, rounding up
* to an integral multiple of the right-hand-side, e.g. (9,5) -> 2 , (10,5) -> 2, (11,5) -> 3.
*
* @param dividend the number to divide
* @param divisor the number of by which to divide
* @return The least integer multiple of {@link divisor} which is greater-or-equal to
* the non-integral division dividend/divisor.
*
* @note sensitive to overflow, i.e. if dividend > std::numeric_limits<S>::max() - divisor,
* the result will be incorrect
*/
template <typename S, typename T>
constexpr inline S div_rounding_up(const S& dividend, const T& divisor) {
	return (dividend + divisor - 1) / divisor;
/*
    std::div_t div_result = std::div(dividend, divisor);
    return div_result.quot + !(!div_result.rem);
*/
}

template <typename T>
constexpr int log2_of_power_of_2(T p)
{
	using unsigned_type = typename std::make_unsigned<T>::type;
	return builtins::count_trailing_zeros(static_cast<unsigned_type>(p));
//	return builtins::count_trailing_zeros(p);
}

template <typename T, typename S>
constexpr T div_by_power_of_2(const T& dividend, const S& divisor)
{
	return dividend >> log2_of_power_of_2(divisor);
}

template <typename T, typename S>
constexpr T div_by_power_of_2_rounding_up(const T& dividend, const S& divisor)
{
/*  // Only safe for C++14 and later
	using unsigned_type = typename std::make_unsigned<S>::type;
	unsigned_type unsigned_divisor = divisor;
	constexpr auto log_2_of_divisor = util::builtins::count_trailing_zeros(unsigned_divisor);
	constexpr auto mask = unsigned_divisor - 1;
	auto correction_for_rounding_up = ((dividend & mask) + mask) >> log_2_of_divisor;

	return (dividend >> log_2_of_divisor) + correction_for_rounding_up;
	*/
	return (dividend >> log2_of_power_of_2(divisor)) +
		(((dividend & (divisor - 1)) + (divisor - 1)) >> log2_of_power_of_2(divisor));
}

/**
* A more 'semantically clear' version for x % y == 0
*/
template <typename S, typename T>
constexpr inline int divides(const S& divisor, const T& dividend) {
	return dividend % divisor == 0;
}

/**
* A more 'semantically clear' version for x % y == 0
*/
template <typename S, typename T>
constexpr inline int is_divisible_by(const T& dividend, const S& divisor) {
	return divides(divisor, dividend);
}

/**
 * @brief Calculates floor(log_2(x)), i.e. the integral part of the
 * base-2 logarithm of x
 */
template <typename T>
inline unsigned ilog2(
	typename std::enable_if<std::is_unsigned<T>::value, T>::type x) {
	return (CHAR_BIT * sizeof(T) - 1) - util::builtins::count_leading_zeros(x);
}

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


/**
* Round up the first parameter to the closest multiple of the second.
*/
template <typename S, typename T>
constexpr inline S round_up(const S& number_to_round, const T& modulus) {
	return (number_to_round + modulus - 1) - ((number_to_round + modulus - 1) % modulus);
}
/**
* Round up the first parameter to the closest multiple of the second.
*/
template <typename S, typename T>
constexpr inline S round_down(const S& number_to_round, const T& modulus) {
	return number_to_round - (number_to_round % modulus);
}

// TODO: Use value params, not references
template <typename T, typename S>
inline constexpr typename std::common_type<T,S>::type
round_down_to_power_of_2(const T& x, const S& power_of_2)
{
	using result_type = typename std::common_type<T,S>::type;
	return ((result_type) x) & ~(((result_type) power_of_2) - 1);
}

template <typename T>
inline T round_down_to_power_of_2(
	typename std::enable_if<std::is_unsigned<T>::value, T>::type const& x)
{
	return 1 << log2_of_power_of_2<T>(x);
}

/**
 * @note careful, this may overflow!
 */
template <typename T, typename S>
inline constexpr typename std::common_type<T,S>::type
round_up_to_power_of_2(const T& x, const S& power_of_2) {
	// Maybe this works instead:
	// return ( x + power_of_2 - 1 ) & - power_of_2;
	using result_type = typename std::common_type<T,S>::type;
	return round_down_to_power_of_2 ((result_type) x + (result_type) power_of_2 - 1, (result_type) power_of_2);
}

/**
 * Rounds x up to the next power of 2 - provided x is
 * greater than 1
 *
 * @note may overflow
 */
template <typename T>
inline T round_up_to_power_of_2(const T& x_greater_than_one)
{
	return ((T)1) << (ilog2<T>(x_greater_than_one - 1) + 1);
}

template <typename T>
constexpr inline T round_up_to_power_of_2_constexpr(const T& x_greater_than_one)
{
	return ((T)1) << ceil_log2_constexpr(x_greater_than_one);
}


// Some constexpr math - C++11 only

namespace detail {
template <typename T>
constexpr T sqrt_constexpr_helper(T x, T low, T high)
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
constexpr T sqrt_constexpr(T& x)
{
  return detail::sqrt_constexpr_helper(x, 0, x / 2 + 1);
}


template <typename T, unsigned Base>
constexpr int log_constexpr(T val) { return val ? 1 + log_constexpr<T, Base>(val / Base) : -1; }

template <typename T, unsigned Base>
constexpr int floor_log_constexpr(T val) { return log_constexpr<T, Base>(val); }

template <typename T, unsigned Base>
constexpr int ceil_log_constexpr(T val) { return val ? 1 + log_constexpr<T, Base>(val - 1) : -1; }

template <typename T>
constexpr int log10_constexpr(T val) { return log_constexpr<T, 10>(val); }

template <typename T>
constexpr int log2_constexpr(T val) { return val ? 1 + log2_constexpr(val >> 1) : -1; }

template <typename T>
int floor_log2(T val) { return ilog2<T>(val); }

template <typename T>
int ceil_log2(T val) { return val ? 1 + ilog2<T>(val - 1) : -1; }

template <typename T>
constexpr int ceil_log2_constexpr(T val) { return ceil_log_constexpr<T, 2>(val); }

template <typename T>
inline constexpr bool is_power_of_2(T val) { return (val & (val-1)) == 0; } // Yes, this works

/**
 * I don't want to drag of all <algorithm> just for max.
 */
template <typename T>
static inline const T& max(const T& a, const T& b) { return (a > b) ? a : b; }

// TODO:
// - These names are too general
// - There are "better" rule-of-thumb distance functions for floating point
//   values, e.g. using ULPs (units at last position). Find some and choose
//   one, this is too hackish

template <typename T>
static inline T get_distance(
	const typename std::enable_if<std::is_floating_point<T>::value, T>::type& lhs,
	const T& rhs, bool use_relative_distance = false, T non_zero_threshold = 0)
{
	using std::fabs;
	return use_relative_distance ?
		fabs(lhs - rhs) / max(max(fabs(rhs), fabs(rhs)), non_zero_threshold) :
		fabs(lhs - rhs);
}

template <typename T>
static inline T get_distance(
	const typename std::enable_if<!std::is_floating_point<T>::value, T>::type& lhs,
	const T& rhs, bool use_relative_distance, T non_zero_threshold)
{
	return std::fabs(lhs - rhs);
}

template <typename T>
constexpr inline T& modular_inc(T& x, T modulus) { return (x+1) % modulus; }

template <typename T>
constexpr inline T& modular_dec(T& x, T modulus) { return (x-1) % modulus; }

template <typename T, typename Lower = T, typename Upper = T>
constexpr inline bool between_or_equal(const T& x, const Lower& l, const Upper& u) { return (l <= x) && (x <= u); }

template <typename T, typename Lower = T, typename Upper = T>
constexpr inline bool strictly_between(const T& x, const Lower& l, const Upper& u) { return (l < x) && (x < u); }

template <typename T>
inline T gcd(T u, T v)
{
	while (v != 0) {
		T r = u % v;
		u = v;
		v = r;
	}
	return u;
}

template <typename T>
inline T lcm(T u, T v)
{
	return (u / gcd(u,v)) * v;
}

template <typename T>
constexpr T gcd_constexpr(T u, T v)
{
	return (v == 0) ? u : gcd(v, u % v);
}

template <typename T>
constexpr T lcm_constexpr(T u, T v)
{
	return (u / gcd_constexpr(u,v)) * v;
}

/**
 * @brief Calculates the binomial coefficient indexed by n and k
 *
 * This implementation was adapted from:
 * http://codereview.stackexchange.com/q/58236/64513
 * it is based on the algorithm described here: http://blog.plover.com/math/choose.html
 * and the relevant follow-up: http://blog.plover.com/math/choose-2.html
 *
 * @param n The number of elements
 * @param k The number of elements in each subset
 * @return The number of k-element subsets in a n-element set
 *
 * @throw std::overflow_error   If the computation caused an overflow
 */
template <typename T>
T binomial_coefficient(T n, T k)
{
	if (k > n) return 0;
	if (n - k < k) k = n - k;
	T r = 1;
	for (T d = 1; d <= k; d++) {
		T mult = n;
		bool divided = true;
		if (mult % d == 0) { mult /= d; }
		else if (r % d == 0) { r /= d; }
		else {divided = false; }
		const T r_mult = r * mult;
		if ((r_mult / mult) != r) { throw std::overflow_error("Overflow"); }
		r = r_mult;
		if (!divided) r /= d;
		n--;
	}
	return r;
}

namespace detail {

	template<typename T>
	struct identity: public std::unary_function<T, T> {
		T operator()(const T& x) const { return x; }
	};

} // namespace detail

/**
 * An implementation Kahan accumulation, based on:
 * http://en.wikipedia.org/wiki/Kahan_summation_algorithm
 *
 * It's intended to reduce the floating-point error when summing
 * up many elements - at the cost of increased time complexity;
 * it's not the absolute minimum error you can get, though
 * (e.g. by sorting the data before summation)
 *
 * @return the sum of all elements in the range
 */
template <
	typename ForwardIterator,
	typename Sum = typename std::iterator_traits<ForwardIterator>::value_type,
	typename UnaryOp = detail::identity<typename std::iterator_traits<ForwardIterator>::value_type>
>
Sum kahan_summation(ForwardIterator begin, ForwardIterator end, UnaryOp op = UnaryOp())
{
    Sum sum = 0;
    Sum compensation = 0;

    for (auto it = begin; it != end; ++it) {
    	auto compensated_datum = op(*it) - compensation;
    	Sum updated_sum = sum + compensated_datum;
        compensation = (updated_sum - sum) - compensated_datum;
        	// with infinite precision, (temp - sum) would be exactly equal
        	// to compensated_datum, and this would become 0. However, only
        	// the more-significant part of compensated_datum is 'preserved' in
        	// the transition from sum to temp (which we will soon complete);
        	// subtracting compensated_datum keeps the less-significant part,
        	// negated, in compensation.
        	// At the next interation we shall negate it again and add this
        	// remainder to compensated_datum
        sum = updated_sum;
    }
    return sum;
}

} /* namespace util */

#endif /* SRC_UTIL_MATH_HPP_ */
