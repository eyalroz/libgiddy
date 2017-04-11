/**
 * Several decompression-related kernels use parametric models; since
 * these are templated over, we need these models available as functors -
 * hence this file. They are actually perfectly ordinary unary functors,
 * of the same kind you would find in cuda/functors.hpp - except for
 * one characterizing feature, which is that they have data members:
 * a cuda::array's of model coefficients.
 *
 */
#pragma once
#ifndef SRC_KERNELS_DECOMPRESSION_MODEL_FUNCTORS_HPP
#define SRC_KERNELS_DECOMPRESSION_MODEL_FUNCTORS_HPP

#include "cuda/functors.hpp"
#include "cuda/stl_array.cuh"

#include <functional>

#ifdef __CUDA_ARCH__
#define __fhd__ __forceinline__ __host__ __device__
#else
#define __fhd__ inline
#endif

namespace cuda {
namespace functors {

template <typename T, unsigned NumCoefficients>
struct unary_model_function_t : public std::unary_function<T, T>{
	using parent = std::unary_function<T, T>;
	using result_type = typename parent::result_type;
	using coefficient_type = T; // I could have different types, but, well...
	using coefficients_type = cuda::array<coefficient_type, NumCoefficients>;

	enum : unsigned { num_coefficients = NumCoefficients };

	__fhd__ unary_model_function_t(
		const coefficients_type& coefficients_) : coefficients(coefficients_) { }
	__fhd__ unary_model_function_t(
		const unary_model_function_t& other) : coefficients(other.coefficients) { }

	const coefficients_type coefficients;
};

// TODO: Check the unrolling during compilation
template <typename T, unsigned NumCoefficients>
struct polynomial_model : unary_model_function_t<T, NumCoefficients>{
	using parent = unary_model_function_t<T, NumCoefficients>;
	using parent::parent;
	using parent::coefficients;
	using parent::coefficients_type;

	__forceinline__ __host__ __device__ T operator()(const T& x) {
		T result = 1;
		T power_of_x = 1;
#ifdef __CUDA_ARCH__
		#pragma unroll
#endif
		for(const auto& coefficient : coefficients) {
			power_of_x *= x;
			result  += coefficient * power_of_x;
		}
		return result;
	}
};

template <typename T>
struct linear_model : unary_model_function_t<T, 2>{
	using parent = unary_model_function_t<T, 2>;
	using parent::parent;
	using parent::coefficients;
	using parent::coefficients_type;

	__forceinline__ __host__ __device__ T operator()(const T& x) {
		return coefficients[0] + coefficients[1] * x;
	}
};

// TODO: Check that when this next one is used,
// we don't get a gratuituous load/register copy/etc.
// due to the supposed passing of X and the calculation,
// and that the coefficient is simply assigned out
// to wherever the result is going.
template <typename T>
struct constant_model : unary_model_function_t<T, 1>{
	using parent = unary_model_function_t<T, 1>;
	using parent::parent;
	using parent::coefficients;
	using coefficients_type = parent::coefficients_type;

	__forceinline__ __host__ __device__ T operator()(const T& x) {
		return coefficients[0];
	}
};

} // namespace functors
} // namespace cuda

#undef __fhd__

#endif /* SRC_KERNELS_DECOMPRESSION_MODEL_FUNCTORS_HPP */
