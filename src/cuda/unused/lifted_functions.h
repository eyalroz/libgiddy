#pragma once
#ifndef CUDA_FUNCTORS_H_
#define CUDA_FUNCTORS_H_

/**
 * This header contains only forward declarations - for use by code not compiled with nvcc,
 * which needs to make calls to nvcc code referring to the functors below (for example,
 * get a kernel to run and apply 'plus' instead of 'multiplies' as an operator). The actual
 * definition cannot appear here, since it requires nvcc's __device__ and __host__ indicators.
 */

// Yes this _is_missing from std!
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

// Arithmetic
template<typename LHS, typename RHS = LHS, typename Result = LHS> struct plus;
template<typename LHS, typename RHS = LHS, typename Result = LHS> struct minus;
template<typename LHS, typename RHS = LHS, typename Result = LHS> struct multiply;
template<typename LHS, typename RHS = LHS, typename Result = LHS> struct divide;
template<typename LHS, typename RHS = LHS, typename Result = LHS> struct modulus;


template<typename T> struct negate;
template<typename T> struct absolute; // abs() / labs() / fabs() / etc.

template<typename T> struct minimum;
template<typename T> struct maximum;

// Comparison
template<typename T, typename R = bool> struct equals;
template<typename T, typename R = bool> struct not_equals;
template<typename T, typename R = bool> struct greater;
template<typename T, typename R = bool> struct less;
template<typename T, typename R = bool> struct greater_equal;
template<typename T, typename R = bool> struct less_equal;
template<typename T, typename R = bool> struct is_zero;
template<typename T, typename R = bool> struct is_non_negative;

// Logical operations
template<typename T, typename R = bool> struct logical_and;
template<typename T, typename R = bool> struct logical_or;
template<typename T, typename R = bool> struct logical_not;
template<typename T, typename R = bool> struct logical_xor;

// Bitwise operations
template<typename T> struct bit_and;
template<typename T> struct bit_or;
template<typename T> struct bit_xor;
template<typename T, typename S> struct shift_left;
template<typename T, typename S> struct shift_right;
template<typename T, typename S> struct find_first_set; // 1-based; 0 means none is set
template<typename T, typename S> struct find_last_set;
template<typename T> struct reverse_bits;
template<typename T> struct keep_only_highest_bit;
template<typename T> struct keep_only_lowest_bit;
template<typename T, typename S> struct population_count;

template<typename T> struct bit_not;

// Fix either the left-hand-side or right-hand-side argument
// of a binary function to a constant value, yielding a
// unary function
//
// TODO: Perhaps 

template<typename BinaryFunction, typename LHSNullaryFunction> class curry_left_hand_side;
template<typename BinaryFunction, typename RHSNullaryFunction> class curry_right_hand_side;

// Functor Negation

template<typename Predicate> class unary_negate;
template<typename Predicate> class binary_negate;

// Unary operations
template<typename CastFrom, typename CastTo> struct cast;
template<typename T, T Value> struct constant;


// Fixes one of the arguments of binary function on
// construction, so when it's called it's actually a unary
// function. See: https://en.wikipedia.org/wiki/Currying
// Useful for applying the same binary function to multiple
// values on one side and a constant value on the other
template<typename BinaryFunction> class curry_right_hand_side;
template<typename BinaryFunction> class curry_left_hand_side;

template<typename T> struct negate;
template<typename T> using algebraic_negation = negate<T>;
template<typename T> struct inverse;
template<typename T> struct preincrement;
template<typename T> struct postincrement;
template<typename T> struct predecrement;
template<typename T> struct postdecrement;

//template<typename Argument, typename Result, Argument LowerBound, Argument UpperBound> struct strictly_between;
template<typename T, typename R = bool> struct strictly_between;
template<typename T, typename R = bool> struct between_or_equal;


// Should these be constexpr's?
template<typename Predicate>
inline unary_negate<Predicate> not1(const Predicate& pred)
{
    return unary_negate<Predicate>(pred);
}
template<typename Predicate>
inline binary_negate<Predicate> not2(const Predicate& pred)
{
    return binary_negate<Predicate>(pred);
}

// Some ternary ops; perhaps distribute them above?

template<typename T> struct clip;
// template<typename R, typename T> struct is_between_ternary;
template<typename P, typename T> struct if_then_else;

} // namespace cuda

#endif /* CUDA_FUNCTORS_H_ */
