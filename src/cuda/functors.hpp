#pragma once
#ifndef CUDA_FUNCTORS_H_
#define CUDA_FUNCTORS_H_

/*
 * Based on the C++ standard. Earlier revisions of this file claimed to
 * be based on GCC 4.9.3's<functional>, but no code was actually copied
 * and the API is part of the C++ standard.
 */

// TODO: C++14 added std-forward-based variants of the standard functors,
// addressing various issues mentioned in Stephen T. Lavavej's presentation:
// https://channel9.msdn.com/Events/GoingNative/2013/Don-t-Help-the-Compiler
// we should probably do something similar when CUDA starts supporting C++14

#include "cuda/bit_operations.cuh"

#ifdef __CUDA_ARCH__
#include "cuda/on_device/atomics.cuh"
#include "cuda/on_device/builtins.cuh"
#include "cuda/on_device/miscellany.cuh"
#include "cuda/on_device/math.cuh"
#else
#include "cuda/faux_builtins.hpp"
#endif
#include "util/integer.h"

// Should we push this down into on_device/limits.cuh? Like we have with the builtins?
#ifdef __CUDA_ARCH__
#include "cuda/on_device/limits.cuh"
template <typename T>
using numeric_limits = cuda::numeric_limits<T>;
#include <device_functions.h>
#else
#include <limits>
#include <cmath>
template <typename T>
using numeric_limits = std::numeric_limits<T>;

#endif /* __CUDA_ARCH__ */

#include <utility>
#include <functional>

/**
 * See: http://stackoverflow.com/a/43521790/1593077 ;
 * this cannot be done in C++17 or older without resorting to macros
 */
#define RETURNS(...) noexcept(noexcept(__VA_ARGS__)) -> decltype(__VA_ARGS__) { return __VA_ARGS__; }

#define OVERLOAD_SET_TYPE(...) \
  struct { \
    template<class...Args> \
    __device__ __host__ auto operator()(Args&&...args)const \
    RETURNS( __VA_ARGS__( std::forward<Args>(args)... ) ) \
  }


// Yes this _is_missing from std!
// TODO: Perhaps include this from elsewhere?
namespace std {
template<typename Result>
struct nullary_function {
    /// @c result_type is the return type
    typedef Result result_type;
};

template<typename Arg1, typename Arg2, typename Arg3, typename Result>
struct ternary_function {
    /// @c first_argument_type is the type of the first argument
    typedef Arg1     first_argument_type;

    /// @c second_argument_type is the type of the second argument
    typedef Arg2     second_argument_type;

    /// @c third_argument_type is the type of the third argument
    typedef Arg3     third_argument_type;

    /// @c result_type is the return type
    typedef Result   result_type;
};
} // namespace std


namespace cuda {
namespace functors {

#ifdef __CUDACC__
#define __fhd__ __forceinline__ __host__ __device__
#define __fd__  __forceinline__ __device__
#else
#define __fhd__ inline
#define __fd__  inline
#endif


// Arithmetic operations

template<typename LHS, typename RHS = LHS, typename Result = LHS>
struct plus: public std::binary_function<LHS, RHS, Result> {
	__fhd__ Result operator()(const LHS& x, const RHS& y) const { return x + y; }
	struct accumulator {
		__fhd__ Result operator()(
			typename std::enable_if<std::is_same<LHS, RHS>::value, Result>::type& x, const RHS& y) const { return x += y; }
		struct atomic {
#ifdef __CUDA_ARCH__
			__fd__ Result operator()(
				typename std::enable_if<std::is_same<LHS,RHS>::value, Result>::type& x,
				const RHS& y) const { return ::atomic::add(&x,y); }
#endif /* __CUDA_ARCH__ */
		};
		__fhd__ static Result neutral_value() { return 0; };
	};
	__fhd__ static Result neutral_value() { return 0; };
};

template<typename LHS, typename RHS = LHS, typename Result = LHS>
struct minus: public std::binary_function<LHS, RHS, Result> {
	__fhd__ Result operator()(const LHS& x, const RHS& y) const { return x - y; }
	struct accumulator {
		__fhd__ Result operator()(
			typename std::enable_if<std::is_same<LHS, RHS>::value, Result>::type& x, const RHS& y) const { return x -= y; }
		struct atomic {
#ifdef __CUDA_ARCH__
			__fd__ Result operator()(
				typename std::enable_if<std::is_same<LHS,RHS>::value, Result>::type& x,
				const RHS& y) const { return ::atomic::subtract(&x,y); }
#endif /* __CUDA_ARCH__ */
			__fhd__ static Result neutral_value() { return 0; };
		};
	};
	__fhd__ static Result neutral_value() { return 0; };
};

template<typename LHS, typename RHS = LHS, typename Result = LHS>
struct multiplies: public std::binary_function<LHS, RHS, Result> {
	__fhd__ Result operator()(const LHS& x, const RHS& y) const { return x * y; }
	struct accumulator {
		__fhd__ Result operator()(
			typename std::enable_if<std::is_same<LHS, RHS>::value, Result>::type& x, const RHS& y) const { return x *= y; }
		struct atomic {
#ifdef __CUDA_ARCH__
			// TODO: implement this using atomicCAS
#endif /* __CUDA_ARCH__ */
		};
		__fhd__ static Result neutral_value() { return 1; };
	};
	__fhd__ static Result neutral_value() { return 1; };
};

template<typename LHS, typename RHS = LHS, typename Result = LHS>
struct divides: public std::binary_function<LHS, RHS, Result> {
	__fhd__ Result operator()(const LHS& x, const RHS& y) const { return x / y; }
	struct accumulator {
		__fhd__ Result operator()(
			typename std::enable_if<std::is_same<LHS, RHS>::value, Result>::type& x, const RHS& y) const { return x /= y; }
		struct atomic {
#ifdef __CUDA_ARCH__
			// TODO: implement this using atomicCAS
#endif /* __CUDA_ARCH__ */
		};
		__fhd__ static Result neutral_value() { return 1; };
	};
	__fhd__ static Result neutral_value() { return 1; };
};

template<typename LHS, typename RHS = LHS, typename Result = LHS>
struct modulus: public std::binary_function<LHS, RHS, Result> {
	__fhd__ Result operator()(const LHS& x, const RHS& y) const { return x % y; }
	struct accumulator {
		__fhd__ Result operator()(
			typename std::enable_if<std::is_same<LHS, RHS>::value, Result>::type& x, const RHS& y) const { return x %= y; }
		struct atomic {
#ifdef __CUDA_ARCH__
			// TODO: implement this using atomicCAS
#endif /* __CUDA_ARCH__ */
		};
	};
};

template<typename T>
struct minimum: public std::binary_function<T, T, T> {
	__fhd__ T operator()(const T& x, const T& y) const { return x < y ? x : y; }
	struct accumulator {
		__fhd__ T operator()(T& x, const T& y) const { if (x < y) x = y; return x; }
		struct atomic {
#ifdef __CUDA_ARCH__
			__fd__ T operator()(T& x, const T& y) const { return ::atomic::min(&x,y); }
#endif /* __CUDA_ARCH__ */
		};
		__fhd__ static T neutral_value() { return numeric_limits<T>::max(); };
	};
	__fhd__ static T neutral_value() { return numeric_limits<T>::max(); };
};

template<typename T>
struct maximum: public std::binary_function<T, T, T> {
	__fhd__ T operator()(const T& x, const T& y) const { return x > y ? x : y; }
	struct accumulator {
		__fhd__ T operator()(T& x, const T& y) const { if (x > y) x = y; return x; }
#ifdef __CUDA_ARCH__
		struct atomic {
			__fd__ T operator()(T& x, const T& y) const { return ::atomic::max(&x,y); }
		};
#endif /* __CUDA_ARCH__ */
		__fhd__ static T neutral_value() { return numeric_limits<T>::min(); }
	};
	__fhd__ static T neutral_value() { return numeric_limits<T>::min(); }
};


template<typename T, typename R = std::conditional<std::is_integral<T>::value, unsigned long long int, double>>
struct next: public std::unary_function<T, R> {
	__fhd__ R operator()(const T& x) const { return x + 1; }
};

template<typename T, typename R = std::conditional<std::is_integral<T>::value, unsigned long long int, double>>
struct previous: public std::unary_function<T, R> {
	__fhd__ R operator()(const T& x) const { return x - 1; }
};

template<typename Result, typename Argument>
struct sign: public std::unary_function<Argument, Result> {
	__fhd__ Result operator()(const Argument& x) const { return (x > 0);  }
};

template<typename T>
struct negate: public std::unary_function<T, T> {
	__fhd__ T operator()(const T& x) const { return -x; }
};

template<typename T>
using algebraic_negation = negate<T>;

template<typename T>
struct inverse: public std::unary_function<T, T> {
	__fhd__ T operator()(const T& x) const { return ((T) 1) / x; }
};

template<typename T>
struct preincrement: public std::unary_function<T, T> {
	__fhd__ T operator()(T& x) const { return ++x; }
};

template<typename T>
struct postincrement: public std::unary_function<T, T> {
	__fhd__ T operator()(T& x) const { return x++; }
};

template<typename T>
using increment = preincrement<T>;

template<typename T>
struct predecrement: public std::unary_function<T, T> {
	__fhd__ T operator()(T& x) const { return --x; }
};

template<typename T>
struct postdecrement: public std::unary_function<T, T> {
	__fhd__ T operator()(T& x) const { return x--; }
};

template<typename T>
using decrement = predecrement<T>;

template<typename T>
struct absolute: public std::unary_function<T, T> {
	__fhd__ T operator()(const T& x) const { return (x >= 0) ? x : (-x); }
};

template<>
struct absolute<int>: public std::unary_function<int, int> {
	__fhd__ int operator()(const int& x) const { return abs(x); }
};

template<>
struct absolute<long int>: public std::unary_function<long int, long int> {
	__fhd__ long int operator()(const long int& x) const { return labs(x); }
};

template<>
struct absolute<long long int>: public std::unary_function<long long int, long long int> {
	__fhd__ long long int operator()(const long long int& x) const { return llabs(x); }
};

template<>
struct absolute<float>: public std::unary_function<float, float> {
	__fhd__ float operator()(const float& x) const { return fabsf(x); }
};

template<>
struct absolute<double>: public std::unary_function<double, double> {
	__fhd__ double operator()(const double& x) const { return fabs(x); }
};

template<typename CastFrom, typename CastTo>
struct cast: public std::unary_function<CastFrom, CastTo> {
	__fhd__ CastTo operator()(const CastFrom& x) const { return static_cast<CastTo>(x); }
};

template<typename T, T Value>
struct constant:
	public std::nullary_function<T>,
	public std::unary_function<T, T>,
	public std::binary_function<T, T, T>
{
	using result_type = T;
	using argument_type = T;
	__fhd__ T operator()() const { return Value; }
	__fhd__ T operator()(const T& x) const { return Value; }
	__fhd__ T operator()(const T& x, const T& y) const { return Value; }
};

template<typename T>
struct zero:
	public std::nullary_function<T>,
	public std::unary_function<T, T>,
	public std::binary_function<T, T, T>
{
	using result_type = T;
	using argument_type = T;
	__fhd__ T operator()() const { return 0; }
	__fhd__ T operator()(const T& x) const { return 0; }
	__fhd__ T operator()(const T& x, const T& y) const { return 0; }
};

template<typename T, typename U, T First, U Second>
struct constant_pair:
	public std::nullary_function<std::pair<T,U>>
{
	__fhd__ T operator()() const { return std::make_pair(First, Second); }
};

template<typename T, typename Ratio>
struct constant_by_ratio:
	public std::nullary_function<T>,
	public std::unary_function<T, T>,
	public std::binary_function<T, T, T>
{
	using result_type = T;
	using argument_type = T;
	__fhd__ T operator()() const { return static_cast<T>(Ratio::num) / Ratio::den; }
	__fhd__ T operator()(const T& x) const { return (*this)(); }
	__fhd__ T operator()(const T& x, const T& y) const { return (*this)(); }
};

/*
// These work! You may use them if you like
#define CONSTANT_UNARY_FUNCTOR_ID(suffix) EXPAND_THEN_CONCATENATE( constant_unary_, suffix )

#define DEFINE_CONSTANT_UNARY_FUNCTOR_IMPL(_value,_id) \
template<typename T> \
struct _id: public std::unary_function<T, T> { \
	__fhd__ T operator()(const T& x) const { return _value; } \
};

#define DEFINE_CONSTANT_UNARY_FUNCTOR(_value, _id_suffix) \
	DEFINE_CONSTANT_UNARY_FUNCTOR_IMPL(_value, CONSTANT_UNARY_FUNCTOR_ID(_id_suffix) )

#define CONSTANT_NULLARY_FUNCTOR_ID(suffix) EXPAND_THEN_CONCATENATE( constant_, suffix )

#define DEFINE_CONSTANT_NULLARY_FUNCTOR_IMPL(_value,_id) \
template<typename T> \
struct _id: public std::nullary_function<T> { \
	__fhd__ T operator()() const { return _value; } \
};

#define DEFINE_CONSTANT_NULLARY_FUNCTOR(_value, _id_suffix) \
	DEFINE_CONSTANT_NULLARY_FUNCTOR_IMPL(_value, CONSTANT_NULLARY_FUNCTOR_ID(_id_suffix) )
*/
template<typename T>
struct identity: public std::unary_function<T, T> {
	__fhd__ T operator()(const T& x) const { return x; }
};


template<typename T, T A, T B>
struct affine_transform: public std::unary_function<T, T> {
	__fhd__ T operator()(const T& x) const { return A*x + B; }
};

// This is a workaround for the inability to use floats as template parameters
template<typename T, typename NullaryFunctionForA, typename NullaryFunctionForB>
struct affine_transform_nullaries: public std::unary_function<T, T> {
	__fhd__ T operator()(const T& x) const
	{
		auto a_f = NullaryFunctionForA();
		auto b_f = NullaryFunctionForB();
		return a_f() * x + b_f();
	}
};

template<typename T, T A>
using linear_transform = affine_transform<T,A,0>;

template<typename T, typename NullaryFunctionForA>
using linear_transform_nullaries = affine_transform_nullaries<T,NullaryFunctionForA,constant<T, 0>>;

/*
// Not using this one, since the above implementation is more general,
// and would apply to types with no multiplication or no unit element
template<typename T, T A>
using identity = affine<T,A,0>;

// Ditto for this:
template<typename T, T A>
using constant = affine<T,0,B>;
*/

template<typename BinaryFunction, typename RHSNullaryFunction>
class curry_right_hand_side:
	public std::unary_function<typename BinaryFunction::first_argument_type, typename BinaryFunction::result_type> {
public:
	using first_argument_type   = typename BinaryFunction::first_argument_type;
	using second_argument_type  = typename BinaryFunction::second_argument_type;
	using result_type           = typename BinaryFunction::second_argument_type;

	__fhd__ result_type operator()(
		const first_argument_type& x) const
	{
		BinaryFunction f;
		RHSNullaryFunction c;
		return f(x, c());
	}
};


template<typename BinaryFunction, typename LHSNullaryFunction>
class curry_left_hand_side:
	public std::unary_function<typename BinaryFunction::second_argument_type, typename BinaryFunction::result_type> {
public:
	using first_argument_type   = typename BinaryFunction::first_argument_type;
	using second_argument_type  = typename BinaryFunction::second_argument_type;
	using result_type           = typename BinaryFunction::second_argument_type;
	__fhd__ result_type operator()(
		const second_argument_type& x) const
	{
		BinaryFunction f;
		LHSNullaryFunction c;
		return f(c(), x);
	}
};

template<typename FirstFunctor, typename SecondFunctor>
class compose_unaries:
	public std::unary_function<
		typename FirstFunctor::argument_type,
		typename SecondFunctor::result_type>
{
public:
	using argument_type   = typename FirstFunctor::argument_type;
	using result_type     = typename SecondFunctor::result_type;
	__fhd__ result_type operator()(const argument_type& x) const
	{
		FirstFunctor ff;
		SecondFunctor sf;
		return sf(ff(x));
	}
};

template<typename FirstFunctor, typename SecondFunctor>
class compose_unary_on_binary:
	public std::binary_function<
		typename FirstFunctor::first_argument_type,
		typename FirstFunctor::second_argument_type,
		typename SecondFunctor::result_type>
{
public:
	using first_argument_type   = typename FirstFunctor::first_argument_type;
	using second_argument_type   = typename FirstFunctor::second_argument_type;
	using result_type     = typename SecondFunctor::result_type;
	__fhd__ result_type operator()(
		const first_argument_type& x, const second_argument_type& y) const
	{
		FirstFunctor ff;
		SecondFunctor sf;
		return sf(ff(x, y));
	}
};


template<typename T, typename S, S Factor, typename R>
struct scale: public curry_left_hand_side<multiplies<S, T, R>, constant<S, Factor>> {};

template<typename T>
using twice = scale<T, int, 2, T>;

template<typename T, T Modulus>
struct fixed_modulus: public curry_right_hand_side<modulus<T>, constant<T, Modulus>> {};

template<typename Argument, typename Result>
struct strictly_between: public std::binary_function<Argument, std::pair<Argument,Argument>, Result> {
	__fhd__ Result operator()(const Argument& x, const std::pair<Argument, Argument>& y) const
	{
#ifdef __CUDA_ARCH__
		return strictly_between(y.first, x, y.last);
#else
		return (y.first < x) && (x < y.last);
#endif
	}
};

template<typename Argument, typename Result>
struct between_or_equal: public std::binary_function<Argument, std::pair<Argument,Argument>, Result> {
	__fhd__ Result operator()(const Argument& x, const std::pair<Argument, Argument>& y) const
	{
#ifdef __CUDA_ARCH__
		return between_or_equal(y.first, x, y.last);
#else
		return (y.first <= x) && (x <= y.last);
#endif
	}
};

template<typename Argument, typename Result, Argument LowerBound, Argument UpperBound>
struct strictly_between_unary: public curry_right_hand_side<strictly_between<Argument, Result>, constant_pair<Argument, Argument, LowerBound, UpperBound>> {
	using parent = curry_right_hand_side<strictly_between<Argument, Result>, constant_pair<Argument, Argument, LowerBound, UpperBound>>;
	using parent::parent;
	// are ww inheriting the get method?
};

template<typename Argument, typename Result, Argument LowerBound, Argument UpperBound>
struct between_or_equal_unary: public curry_right_hand_side<between_or_equal<Argument, Result>, constant_pair<Argument, Argument, LowerBound, UpperBound>> {
	using parent = curry_right_hand_side<between_or_equal<Argument, Result>, constant_pair<Argument, Argument, LowerBound, UpperBound>>;
	using parent::parent;
	// are ww inheriting the get method?
};

template<typename BinaryFunctor, typename T, T Value>
struct accumulate_from_rhs: public std::unary_function<T, T> {
	__fhd__ T operator()(const T& x) const { return BinaryFunctor()(Value, x); }
};


template<typename T, typename R = bool>
struct equals: public std::binary_function<T, T, R> {
	__fhd__ R operator()(const T& x, const T& y) const { return x == y; }
};

template<typename T, typename R = bool>
struct not_equals: public std::binary_function<T, T, R> {
	__fhd__ R operator()(const T& x, const T& y) const { return x != y; }
};

template<typename T, typename R = bool>
struct greater: public std::binary_function<T, T, R> {
	__fhd__ R operator()(const T& x, const T& y) const { return x > y; }
};

template<typename T, typename R = bool>
struct less: public std::binary_function<T, T, R> {
	__fhd__ R operator()(const T& x, const T& y) const { return x < y; }
};

template<typename T, typename R = bool>
struct greater_equal: public std::binary_function<T, T, R> {
	__fhd__ R operator()(const T& x, const T& y) const { return x >= y; }
};

template<typename T, typename R = bool>
struct less_equal: public std::binary_function<T, T, R> {
	__fhd__ R operator()(const T& x, const T& y) const { return x <= y; }
};

template<typename T>
struct is_zero: public curry_right_hand_side<equals<T>, zero<T>> { };

// Perhaps just use identity?
template<typename T, typename R = bool>
struct non_zero: public std::unary_function<T, R> {
	__fhd__ R operator()(const T& x) const { return x != 0; }
};

template<typename T, typename R, unsigned BinaryPrecisionDigits> struct about_zero :
	public std::unary_function<typename std::enable_if<std::is_floating_point<T>::value>::type, R>
{
	using result_type = T;
	using argument_type = T;
	__fhd__ T operator()(const T& x) const
	{
		double epsilon = 1. / (1ull << BinaryPrecisionDigits);
		return -epsilon < x && x < epsilon;
	}
};


template<typename T>
struct is_non_negative: public curry_right_hand_side<greater_equal<T>, zero<T>> { };

template<typename T>
struct is_positive: public curry_right_hand_side<greater<T>, zero<T>> { };

template<typename T, typename R = bool>
struct even: public std::unary_function<T, R> {
	__fhd__ R operator()(const T& x) const { return x % 2 == 0; }
};

template<typename T, typename R = bool>
struct odd: public std::unary_function<T, R> {
	__fhd__ R operator()(const T& x) const { return x % 2 != 0; }
};

// logical operations

// TODO: Perhaps make the accumulator variants also use the R types?

template<typename T, typename R = bool>
struct logical_and: public std::binary_function<T, T, R> {
	__fhd__ R operator()(const T& x, const T& y) const { return x && y; }
	struct accumulator {
		__fhd__ T operator()(T& x, const T& y) const { return x = x && y; }
		struct atomic {
#ifdef __CUDA_ARCH__
			__fd__ T operator()(T& x, const T& y) const { return ::atomic::logical_and(&x,y); }
#endif /* __CUDA_ARCH__ */
		};
	};
	__fhd__ static T neutral_value() { return static_cast<T>(true); };
};

template<typename T, typename R = bool>
struct logical_or: public std::binary_function<T, T, R> {
	__fhd__ R operator()(const T& x, const T& y) const { return x || y; }
	struct accumulator {
		__fhd__ T operator()(T& x, const T& y) const { return x = x || y;}
		struct atomic {
#ifdef __CUDA_ARCH__
			__fd__ T operator()(T& x, const T& y) const { return ::atomic::logical_or(&x,y); }
#endif /* __CUDA_ARCH__ */
		};
	};
	__fhd__ static T neutral_value() { return static_cast<T>(false); };
};

template<typename T, typename R = bool>
struct logical_not: public std::unary_function<T, R> {
	__fhd__ R operator()(const T& x) const { return !x; }
};
template<typename T>
using logical_negation = logical_not<T>;

template<typename T, typename R = bool>
struct logical_xor: public std::binary_function<T, T, R> {
	__fhd__ R operator()(const T& x, const T& y) const { return (!x != !y); }
	struct accumulator {
		__fhd__ T operator()(T& x, const T& y) const { return x = (!x != !y); }
		struct atomic {
#ifdef __CUDA_ARCH__
			__fd__ T operator()(T& x, const T& y) const { return ::atomic::logical_xor(&x,y); }
#endif /* __CUDA_ARCH__ */
		};
	};
};

// Bitwise operations

template<typename T>
struct bit_and: public std::binary_function<T, T, T> {
	__fhd__ T operator()(const T& x, const T& y) const { return x & y; }
};

template<typename T>
struct bit_or: public std::binary_function<T, T, T> {
	__fhd__ T operator()(const T& x, const T& y) const { return x | y; }
};

template<typename T>
struct bit_xor: public std::binary_function<T, T, T> {
	__fhd__ T operator()(const T& x, const T& y) const { return x ^ y; }
};

template<typename T>
struct bit_not: public std::unary_function<T, T> {
	__fhd__ T operator()(const T& x) const { return ~x; }
};

template<typename T>
using bitwise_negation = bit_not<T>;

template<typename T, typename S = unsigned>
struct shift_left: public std::binary_function<T, S, T> {
	__fhd__ T operator()(const T& x, const S& n_bits) const { return x << n_bits; }
};

template<typename T, typename S = unsigned>
struct shift_right: public std::binary_function<T, S, T> {
	__fhd__ T operator()(const T& x, const S& n_bits) const { return x >> n_bits; }
};


template<typename T, typename S = unsigned>
struct find_first_set: public std::unary_function<T, S> {
	__fhd__ S operator()(const T& x) const {
		return find_first_set(x);
	}
};

template<typename T, typename S = unsigned>
struct find_last_set: public std::unary_function<T, S> {
	__fd__ S operator()(const T& x) const { return cuda::find_last_set(x); }
};

template<typename T>
struct reverse_bits: public std::unary_function<T, T> {
	__fhd__ T operator()(const T& x) const {
		return builtins::bit_reverse(x);
	}
};

template<typename T>
struct keep_only_lowest_bit: public std::unary_function<T, T> {
	__fd__ T operator()(const T& x) const { return cuda::keep_only_lowest_set_bit(x); }
};

template<typename T>
struct keep_only_highest_bit: public std::unary_function<T, T> {
	__fd__ T operator()(const T& x) const { return cuda::keep_only_highest_set_bit(x); }
};

template<typename Result, typename BitContainer = standard_bit_container_t>
struct population_count: public std::unary_function<BitContainer, Result> {
	__fhd__ Result operator()(const BitContainer& x) const {
		return builtins::population_count(x);
	}
};


// Note: Want to negate a functor? Just use compose<my_functor, negate>


// Some ternary ops

template<typename T>
struct clip: public std::ternary_function<T, T, T, T> {
	__fhd__ T operator()(const T& x, const T& range_min, const T& range_max) const { return x < range_min ? range_min : ( x > range_max ? range_max : x ); }
};

template<typename T>
struct strictly_between_ternary: public std::ternary_function<T, T, T, bool> {
	__fhd__ T operator()(const T& range_min, const T& x, const T& range_max) const { return (x <= range_max) && (x >= range_min); }
};

template<typename T>
struct between_or_equal_ternary: public std::ternary_function<T, T, T, bool> {
	__fhd__ T operator()(const T& range_min, const T& x, const T& range_max) const { return (x <= range_max) && (x >= range_min); }
};

template<typename T>
struct if_then_else: public std::ternary_function<bool, T, T, T> {
	__fhd__ T operator()(const bool& x, const T& y, const T& z) const { return x ? y : z; }
};

template<typename T>
struct fused_multiply_add: public std::ternary_function<T, T, T, T> {
	__fhd__ T operator()(const T& x, const T& y, const T& z) const { return x * y + z; }
};

template<typename T>
using fma = fused_multiply_add<T>;


namespace enumerated {

/*
 * We refer to functions (resp: functors) which, in addition to their "regular"
 * arguments, also take an initial index - and are applied to a sequence of
 * indexed elements - as enumerated functions (resp: enumerated functors).
 *
 * Here are some definitions related to those.
 */


/**
 * Indicates whether the pre-reduction transform operation only
 * takes the data element, or also the index of the element in the
 * input array (so essentially, we're using the same kernel and
 * kerenl test adapters for two kinds of kernels)
 *
 * TODO: This can be avoided if we add an is_enumerated trait to
 * cuda::functors::enumerated and use it below
 */
enum class functor_is_enumerated_t : bool {
	Unenumerated = false, //!< Regular
	Enumerated   = true//!< Enumerated
};

template<unsigned IndexSize, typename Arg, typename Result>
struct unary_function
{
	using enumerator_type       = util::uint_t<IndexSize>;
	using argument_type         = Arg;
	using result_type           = Result;
};

template<unsigned IndexSize, typename FirstArgument, typename SecondArgument, typename Result>
struct binary_function
{
	using enumerator_type       = util::uint_t<IndexSize>;
	using first_argument_type   = FirstArgument;
	using second_argument_type  = SecondArgument;
	using result_type           = Result;
};

template<typename BinaryOp>
struct as_enumerated_unary :
	public std::enable_if<std::is_unsigned<typename BinaryOp::first_argument_type>::value, BinaryOp>::type,
	public unary_function<sizeof(typename BinaryOp::first_argument_type), typename BinaryOp::second_argument_type,
	typename BinaryOp::result_type>
{
	// Resolving ambiguity
	using result_type = typename BinaryOp::result_type;
};

} // namespace enumerated

// 1-based! 0 means nothing is set
// TODO: Document me!
template<unsigned IndexSize, typename BitContainer = standard_bit_container_t>
struct global_index_of_last_set_bit :
	public std::enable_if<
		std::is_unsigned<BitContainer>::value,
		cuda::functors::enumerated::unary_function<IndexSize, BitContainer, util::uint_t<IndexSize> >
	>::type
{
	using index_type = util::uint_t<IndexSize>;

	__fhd__ index_type operator()(
		const index_type& pos, const BitContainer& element_bits) const
	{
		constexpr auto num_bits_per_element = sizeof(BitContainer) * CHAR_BIT;
		auto first_bit_set_in_reverse =
			cuda::find_first_set(builtins::bit_reverse(element_bits));
		return first_bit_set_in_reverse ?
			num_bits_per_element * pos +
			(sizeof(BitContainer) * CHAR_BIT - first_bit_set_in_reverse + 1) : 0;
	}
};

// This could use some refinement, to be a composition of simpler
// and more general constructs; but for now - we need it.
// Note: Result is 1-based, all-1-bits result means nothing is set
// TODO: Document me!
template<unsigned IndexSize, typename BitContainer = standard_bit_container_t>
struct global_index_of_first_set_bit :
	public std::enable_if<
		std::is_unsigned<BitContainer>::value,
		cuda::functors::enumerated::unary_function<IndexSize, BitContainer, util::uint_t<IndexSize> >
	>::type
{
	using index_type = util::uint_t<IndexSize>;

	__fhd__ index_type operator()(
		const index_type& pos, const BitContainer& element_bits) const
	{
		constexpr auto num_bits_per_element = sizeof(BitContainer) * CHAR_BIT;
		auto first_set_in_element = cuda::find_first_set(element_bits);
		return first_set_in_element ?
			num_bits_per_element * pos + first_set_in_element :
			cuda::all_one_bits<index_type>();
	}
};

} // namespace functors
} // namespace cuda

#undef __fhd__
#undef __fd__

#endif /* CUDA_FUNCTORS_H_ */
