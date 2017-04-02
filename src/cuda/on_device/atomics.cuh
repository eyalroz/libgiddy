#pragma once
#ifndef CUDA_ON_DEVICE_ATOMICS_CUH_
#define CUDA_ON_DEVICE_ATOMICS_CUH_

/* Utilities for on-device code, e.g. getting global thread index etc. */

#ifndef forceinline
#define forceinline __forceinline__
#endif

#include <cuda_runtime.h>
#include "cuda/api/constants.h"
#include <device_atomic_functions.h>
#include <functional>
#include "cuda/syntax_replacement.h"

#ifdef DEBUG
#include "cuda/on_device/printing.cuh"
#endif

/*
 ************************************************************************
 * Atomics wrappers
 ************************************************************************
 */

namespace atomic {

template <typename T>  forceinline __device__ T compare_and_swap(
	typename std::enable_if<
		sizeof(T) == sizeof(int) or sizeof(T) == sizeof(long long int), T
	>::type	* __restrict__ address,
	const T& compare,
	const T& val)
{
	// This switch is necessary due to atomicCAS being defined
	// only for a very small selection of types - int and unsigned long long
	switch(sizeof(T)) {
	case sizeof(int):
			return (T) (atomicCAS(
				reinterpret_cast<int*      >(address),
				reinterpret_cast<const int&>(compare),
				reinterpret_cast<const int&>(val)
			));
	case sizeof(long long int):
		return (T) (atomicCAS(
				reinterpret_cast<unsigned long long*      >(address),
				reinterpret_cast<const unsigned long long&>(compare),
				reinterpret_cast<const unsigned long long&>(val)
			));
	default: return T(); // should not be able to get here
	}
}

/*
// The following variants are necessary because it seems atomicCAS is only
// defined for int and for unsigned long; we should actually generalize this
// to support any type of the appropriate size

template <>  forceinline __device__ uint64_t compare_and_swap<unsigned long>(
	unsigned long* __restrict__ address, const unsigned long& compare, const unsigned long& val)
{
	return atomicCAS(
		reinterpret_cast<unsigned long long*>(address),
		reinterpret_cast<const unsigned long long&>(compare),
		reinterpret_cast<const unsigned long long&>(val));
}
*/

template <typename T, typename BinaryFunction>
forceinline __device__ T apply_atomically(BinaryFunction f, T* __restrict__ address, const T& val)
{
	auto actual_previous_value = *address;
	T expected_previous_value;
	do {
		expected_previous_value = actual_previous_value;
		T prospective_new_value = f(expected_previous_value, val);
		actual_previous_value = atomic::compare_and_swap(address, expected_previous_value,
			prospective_new_value);
	} while (actual_previous_value != expected_previous_value);
	return actual_previous_value;
}

template <typename T, typename UnaryFunction>
forceinline __device__ T apply_atomically(UnaryFunction f, T* __restrict__ address)
{
	auto actual_previous_value = *address;
	T expected_previous_value;
	do {
		expected_previous_value = actual_previous_value;
		T prospective_new_value = f(expected_previous_value);
		actual_previous_value = atomic::compare_and_swap(address, expected_previous_value,
			prospective_new_value);
	} while (actual_previous_value != expected_previous_value);
	return actual_previous_value;
}

namespace detail {

/**
 * Use CUDA intrinsics where possible and relevant to reinterpret the bits
 * of values of different types
 *
 * @param x[in]  the value to reinterpret. No references please!
 * @return the reinterpreted value
 */
template <typename ToInterpret, typename Interpreted>
__device__ forceinline Interpreted reinterpret(
	typename std::enable_if<
		!std::is_same<
			typename std::decay<ToInterpret>::type, // I actually just don't want references here
			typename std::decay<Interpreted>::type>::value && // I actually just don't want references here
		sizeof(ToInterpret) == sizeof(Interpreted), ToInterpret>::type x)
{
	return x;
}

template<> forceinline __device__ double reinterpret<long long int, double>(long long int x) { return __longlong_as_double(x); }
template<> forceinline __device__ long long int reinterpret<double, long long int>(double x) { return __double_as_longlong(x); }

template<> forceinline __device__ double reinterpret<unsigned long long int, double>(unsigned long long int x) { return __longlong_as_double(x); }
template<> forceinline __device__ unsigned long long int reinterpret<double, unsigned long long int>(double x) { return __double_as_longlong(x); }

template<> forceinline __device__ float reinterpret<int, float>(int x) { return __int_as_float(x); }
template<> forceinline __device__ int reinterpret<float, int>(float x) { return __float_as_int(x); }

// The default (which should be 32-or-less-bit types
template <typename T, typename = void>
struct add_impl {
	__device__ forceinline T operator()(T*  __restrict__ address, const T& val) const
	{
		return atomicAdd(address, val);
	}
};

template <typename T>
struct add_impl<T,
	typename std::enable_if<
		!std::is_same<T, unsigned long long int>::value &&
		sizeof(T) == sizeof(unsigned long long int)
	>::type> {
	using surrogate_t = unsigned long long int;

	__device__ forceinline T operator()(T*  __restrict__ address, const T& val) const
	{
		auto address_ = reinterpret_cast<surrogate_t*>(address);

		// TODO: Use apply_atomically

		surrogate_t previous_ = *address_;
		surrogate_t expected_previous_;
		do {
			expected_previous_ = previous_;
			T updated_value = reinterpret<surrogate_t, T>(previous_) + val;
			previous_ = atomicCAS(address_, expected_previous_,
				reinterpret<T, surrogate_t>(updated_value));
		} while (expected_previous_ != previous_);
		T rv = reinterpret<surrogate_t, T>(previous_);
		return rv;
	}
};

} // namespace detail

template <typename T>
forceinline __device__ T add(T* __restrict__ address, const T& val)
{
	return detail::add_impl<T>()(address, val);
}


// TODO: Consider making apply_atomically take functors,
// including the functors header and having using statements here
// instead of actual definitions
template <typename T>
forceinline __device__ T bitwise_or (T* __restrict__ address, const T& val)
{
	auto f = [](const T& x, const T& y) { return x | y; };
	return apply_atomically(f, address, val);
}

template <typename T>
forceinline __device__ T bitwise_and (T* __restrict__ address, const T& val)
{
	auto f = [](const T& x, const T& y) { return x & y; };
	return apply_atomically(f, address, val);
}

template <typename T>
forceinline __device__ T bitwise_xor (T* __restrict__ address, const T& val)
{
	auto f = [](const T& x, const T& y) { return x ^ y; };
	return apply_atomically(f, address, val);
}

template <typename T>
forceinline __device__ T bitwise_not (T* __restrict__ address)
{
	auto f = [](const T& x) { return ~x; };
	return apply_atomically(f, address);
}

template <typename T>
forceinline __device__ T set_bit (T* __restrict__ address, const unsigned bit_index)
{
	auto f = [](const T& x, const unsigned y) { return x | 1 << y; };
	return apply_atomically(f, address, bit_index);
}
template <typename T>
forceinline __device__ T unset_bit (T* __restrict__ address, const unsigned bit_index)
{
	auto f = [](const T& x, const unsigned y) { return x & ~(1 << y); };
	return apply_atomically(f, address, bit_index);
}


// This next #if-#endif block is intended to supply us with 64-bits of some
// atomic primitives only available with compute capability 3.2 or higher
// (which is not currently a requirement. CC 3.0 sort of is.)
#if __CUDA_ARCH__ < 320

template <typename T>
forceinline __device__ T atomicMin (T* __restrict__ address, const T& val)
{
	auto f = [](const T& x, const T& y) { return x < y ? x : y; };
	return apply_atomically(f, address, val);
}

template <typename T>
forceinline __device__ T atomicMax (T* __restrict__ address, const T& val)
{
	auto f = [](const T& x, const T& y) { return x > y ? x : y; };
	return apply_atomically(f, address, val);
}

template <typename T>
forceinline __device__ T atomicAnd (T* __restrict__ address, const T& val)
{
	auto f = [](const T& x, const T& y) { return x && y; };
	return apply_atomically(f, address, val);
}

template <typename T>
forceinline __device__ T atomicOr (T* __restrict__ address, const T& val)
{
	auto f = [](const T& x, const T& y) { return x || y; };
	return apply_atomically(f, address, val);
}

template <typename T>
forceinline __device__ T atomicXor (T* __restrict__ address, const T& val)
{
	auto f = [](const T& x, const T& y) { return (x && !y) || (y && !x); };
	return apply_atomically(f, address, val);
}

#else

template <typename T>
forceinline __device__ T atomicMin(
	T* __restrict__ address, const T& val)
{
	return ::atomicMin(address, val);
}

template <typename T>
forceinline __device__ T atomicMax(
	T* __restrict__ address, const T& val)
{
	return ::atomicMax(address, val);
}

template<> forceinline __device__ unsigned long atomicMin<unsigned long>(
	unsigned long* __restrict__ address, const unsigned long& val)
{
	return ::atomicMin(
		reinterpret_cast<unsigned long long*>(address),
		reinterpret_cast<const unsigned long long&>(val)
	);
}

template<> forceinline __device__ unsigned long atomicMax<unsigned long>(
	unsigned long* __restrict__ address, const unsigned long& val)
{
	return ::atomicMax(
		reinterpret_cast<unsigned long long*>(address),
		reinterpret_cast<const unsigned long long&>(val)
	);
}

#endif /* __CUDA_ARCH__ >= 320 */

// TODO:
// - Consider using non-const references rather than pointers - but make sure you get the same PTX!
// - Apply the same 64-bit special-casing we have for add to all of the other atomic ops

template <typename T>  forceinline __device__ T subtract   (T* __restrict__ address, const T& val)  { return atomicSub(address, val);  }
template <typename T>  forceinline __device__ T exchange   (T* __restrict__ address, const T& val)  { return atomicExch(address, val); }
template <typename T>  forceinline __device__ T min        (T* __restrict__ address, const T& val)  { return atomicMin(address, val);  }
template <typename T>  forceinline __device__ T max        (T* __restrict__ address, const T& val)  { return atomicMax(address, val);  }
template <typename T>  forceinline __device__ T logical_and(T* __restrict__ address, const T& val)  { return atomicAnd(address, val);  }
template <typename T>  forceinline __device__ T logical_or (T* __restrict__ address, const T& val)  { return atomicOr(address, val);   }
template <typename T>  forceinline __device__ T logical_xor(T* __restrict__ address, const T& val)  { return atomicXor(address, val);  }

template <typename T>  forceinline __device__ T increment  (
	T* __restrict__ address,
	const T& maximum_existing_value_to_increment = 0)
{
	return atomicInc(address, maximum_existing_value_to_increment);
}
template <typename T>  forceinline __device__ T decrement  (
	T* __restrict__ address,
	const T& maximum_existing_value_to_increment = 0)
{
	return atomicDec(address, maximum_existing_value_to_increment);
}



} // namespace atomic

#endif /* CUDA_ON_DEVICE_ATOMICS_CUH_ */
