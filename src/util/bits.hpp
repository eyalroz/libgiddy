#pragma once
#ifndef SRC_UTIL_BITS_HPP_
#define SRC_UTIL_BITS_HPP_

#include <cstddef> // for size_t
#include <type_traits>

namespace util {

using std::size_t;

enum { bits_per_byte = 8, bits_per_char = 8, log_bits_per_char = 3 };
	// these values are guaranteed to be valid as per the C++ standard,
	// we don't have to include climits for them


template <typename T>
constexpr inline T num_bits_to_num_bytes(T n_bits) {
	return (n_bits  + bits_per_char - 1) >> log_bits_per_char;
}

/**
 * The number bits in the representation of a value of type T
 */
template <typename T>
struct size_in_bits { enum : size_t { value = sizeof(T) << log_bits_per_char }; };

// TODO: Perhaps a version of type_size_to_fit ?

template <typename T> inline T constexpr lowest_k_bits(T x, unsigned k)
{
	return x & ((1 << k) - 1);
}

template <typename T> inline T constexpr highest_k_bits(T x, unsigned k)
{
	return x >> (size_in_bits<T>::value - k);
}

template <typename T> inline T constexpr bit_subsequence(T x, unsigned starting_bit, unsigned num_bits)
{
	return lowest_k_bits(x >> starting_bit, num_bits);
}


template <typename T> inline T constexpr clear_lower_k_bits(T x, unsigned k)
{
	return x & ~((1 << k) - 1);
}

template <typename T> inline T constexpr clear_higher_k_bits(T x, unsigned k)
{
	return lowest_k_bits(size_in_bits<T>::value - k);
}

template <typename T> inline bool constexpr has_bit_set(const T& x, unsigned bit_index)
{
		    return x & (1 << bit_index);
}

template <typename T> inline T constexpr set_bit(const T& x, unsigned bit_index)
{
		    return x | (1 << bit_index);
}

template <typename T> inline T constexpr clear_bit(T x, unsigned bit_index)
{
		    return x & ~(1 << bit_index);
}

template <typename T>
constexpr inline typename std::enable_if<std::is_integral<T>::value, T>::type all_one_bits()
{
	return ~((T)0);
}

} // namespace util


#endif /* SRC_UTIL_BITS_HPP_ */
