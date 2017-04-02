#pragma once
#ifndef CUDA_MODEL_FUNCTORS_HPP_
#define CUDA_MODEL_FUNCTORS_HPP_

#include "cuda/functors.hpp"
#include "cuda/stl_array.cuh"

#include <functional>

#define __fhd__ __forceinline__ __host__ __device__

namespace cuda {
namespace functors {
namespace unary {
namespace parametric_model {

/**
 * These are templates for sequence functions:
 * Taking an index i and producing the i'th element in a modeled
 * (finite) data series. The specific models here - inheriting
 * @ref model::function_t - are the simple ones you could
 * easily imagine (and yet - already useful).
 */

template <unsigned IndexSize, typename T, unsigned ModelDimension>
struct function_t : public std::unary_function<util::uint_t<IndexSize>, T>{
	enum { index_size = IndexSize };
	using index_type = util::uint_t<IndexSize>;
	using size_type = index_type;
	using model_coefficient_type = T;
	using parent = std::unary_function<index_type, T>;
	using result_type = typename parent::result_type;
	using model_coefficients_type = cuda::array<model_coefficient_type, ModelDimension>;

	using model_dimensions_size_type = decltype(ModelDimension);
	enum : model_dimensions_size_type { model_dimension = ModelDimension };

	__fhd__ function_t(const model_coefficients_type& model_coefficients_) : model_coefficients(model_coefficients_) { }
	__fhd__ function_t(const function_t& other) : model_coefficients(other.model_coefficients) { }

	const model_coefficients_type model_coefficients;
};

// TODO: Check the unrolling during compilation
template <unsigned IndexSize, typename T, unsigned NumModelCoefficients>
struct polynomial : public function_t<IndexSize, T, NumModelCoefficients>{
	using parent = function_t<IndexSize, T, NumModelCoefficients>;
	using parent::parent;
	using parent::model_coefficients;
	using index_type = util::uint_t<IndexSize>;

	__fhd__  T operator()(const index_type& x) const
	{
		T result = 1;
		T power_of_x = 1;
#ifdef __CUDA_ARCH__
		#pragma unroll
#endif
		for(const auto& coefficient : model_coefficients) {
			power_of_x *= x;
			result += coefficient * power_of_x;
		}
		return result;
	}

	__fhd__ polynomial& operator=(
		const polynomial& other) {
		model_coefficients = other.model_coefficients;
		return *this;
	}


};

template <unsigned IndexSize, typename T>
struct linear : public polynomial<IndexSize, T, 2>
{
	using parent = polynomial<IndexSize, T, 2>;
	using parent::parent;
	using parent::model_coefficients;
	using index_type = util::uint_t<IndexSize>;

	__fhd__  T operator()(const index_type& x) const
	{
		return model_coefficients[0] + model_coefficients[1] * x;
	}
};

// TODO: Check that when this next one is used,
// we don't get a gratuituous load/register copy/etc.
// due to the supposed passing of X and the calculation,
// and that the coefficient is simply assigned out
// to wherever the result is going.
template <unsigned IndexSize, typename T>
struct constant : public polynomial<IndexSize, T, 1>
{
	using parent = polynomial<IndexSize, T, 1>;
	using parent::parent;
	using parent::model_coefficients;
	using index_type = util::uint_t<IndexSize>;

	__fhd__  T operator()(const index_type& x) const
	{
		return model_coefficients[0];
	}
};

template <unsigned IndexSize, typename T>
struct zero : public function_t<IndexSize, T, 0>
{
	using parent = function_t<IndexSize, T, 0>;
	using parent::parent;
	using parent::model_coefficients;
	using index_type = util::uint_t<IndexSize>;

	__fhd__  T operator()(const index_type& x) { return 0; }
};

} // namespace model
} // namespace unary
} // namespace functors
} // namespace cuda

#endif /* CUDA_MODEL_FUNCTORS_HPP_ */
