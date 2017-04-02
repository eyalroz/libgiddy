#pragma once
#ifndef CUDA_STL_ARRAY_CUH_
#define CUDA_STL_ARRAY_CUH_

#include "cuda/syntax_replacement.h"

// No exceptions here
// #include <stdexcept>

// #include <bits/stl_algobase.h>
// #include <bits/range_access.h>

namespace cuda {

template<typename T, size_t NumElements>
struct array_traits
{
	typedef T type[NumElements];

	__forceinline__ __device__ __host__
	static constexpr T& reference(const type& t, size_t n) noexcept { return const_cast<T&>(t[n]); }
	__forceinline__ __device__ __host__
	static constexpr T* pointer(const type& t) noexcept             { return const_cast<T*>(t); }
};

template<typename T>
struct array_traits<T, 0>
{
	struct type { };

	__forceinline__ __device__ __host__
	static constexpr T& reference(const type&, size_t) noexcept     { return *static_cast<T*>(nullptr); }
	__forceinline__ __device__ __host__
	static constexpr T* pointer(const type&) noexcept               { return nullptr; }
};

  /**
   *  @brief A standard container for storing a fixed size sequence of elements.
   *
   *  @ingroup sequences
   *
   * Based on std::array
   *
   *  Sets support random access iterators.
   *
   *  @tparam T              Type of element. Required to be a complete type.
   *  @tparam NumElemements  Number of elements.
  */
template<typename T, size_t NumElements>
struct array
{
	typedef T                                       value_type;
	typedef value_type*                             pointer;
	typedef const value_type*                       const_pointer;
	typedef value_type&                             reference;
	typedef const value_type&                       const_reference;
	typedef value_type*                             iterator;
	typedef const value_type*                       const_iterator;
	typedef size_t                                  size_type;
	typedef std::ptrdiff_t                          difference_type;
	typedef std::reverse_iterator<iterator>         reverse_iterator;
	typedef std::reverse_iterator<const_iterator>   const_reverse_iterator;

	// Support for zero-sized arrays mandatory.
	typedef array_traits<T, NumElements>            array_traits_type;

	typename array_traits_type::type   elements;

	// No explicit construct/copy/destroy for aggregate type.

	// DR 776.
	__forceinline__ __device__ __host__
	void fill(const value_type& u)
	{
		//std::fill_n(begin(), size(), __u);
		for(size_type i = 0; i < NumElements; i++) { elements[i] = u; }
	}

	// Does the noexcept matter here?
	__forceinline__ __device__ __host__
	void swap(array& other) noexcept(noexcept(swap(std::declval<T&>(), std::declval<T&>())))
	{
		// std::swap_ranges(begin(), end(), other.begin());
		for(size_type i = 0; i < NumElements; i++)
		{
			auto x = elements[i];
			auto y = other.elements[i];
			elements[i] = y;
			other.elements[i] = x;
		}
	}

	// Iterators.

	__forceinline__ __device__ __host__
	iterator begin() noexcept { return iterator(data()); }

	__forceinline__ __device__ __host__
	const_iterator begin() const noexcept { return const_iterator(data()); }

	__forceinline__ __device__ __host__
	iterator end() noexcept { return iterator(data() + NumElements); }

	__forceinline__ __device__ __host__
	const_iterator end() const noexcept { return const_iterator(data() + NumElements); }

	__forceinline__ __device__ __host__
	reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }

	__forceinline__ __device__ __host__
	const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }

	__forceinline__ __device__ __host__
	reverse_iterator rend() noexcept { return reverse_iterator(begin()); }

	__forceinline__ __device__ __host__
	const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }

	__forceinline__ __device__ __host__
	const_iterator cbegin() const noexcept
	{ return const_iterator(data()); }

	__forceinline__ __device__ __host__
	const_iterator cend() const noexcept { return const_iterator(data() + NumElements); }

	__forceinline__ __device__ __host__
	const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(end()); }

	__forceinline__ __device__ __host__
	const_reverse_iterator crend() const noexcept { return const_reverse_iterator(begin()); }

	// Capacity.
	__forceinline__ __device__ __host__
	constexpr size_type size() const noexcept { return NumElements; }

	__forceinline__ __device__ __host__
	constexpr size_type max_size() const noexcept { return NumElements; }

	__forceinline__ __device__ __host__
	constexpr bool empty() const noexcept { return size() == 0; }

	// Element access.
	__forceinline__ __device__ __host__
	reference operator[](size_type n) noexcept { return array_traits_type::reference(elements, n); }

	__forceinline__ __device__ __host__
	constexpr const_reference operator[](size_type n) const noexcept { return array_traits_type::reference(elements, n); }

	// Note: no bounds checking.
	__forceinline__ __device__ __host__
	reference at(size_type n) { return array_traits_type::reference(elements, n); }

	__forceinline__ __device__ __host__
	constexpr const_reference at(size_type n) const
	{
		// No bounds checking
		return array_traits_type::reference(elements, n);
	}

	__forceinline__ __device__ __host__
	reference front() noexcept
	{ return *begin(); }

	__forceinline__ __device__ __host__
	constexpr const_reference front() const noexcept
	{ return array_traits_type::reference(elements, 0); }

	__forceinline__ __device__ __host__
	reference back() noexcept
	{ return NumElements ? *(end() - 1) : *end(); }

	__forceinline__ __device__ __host__
	constexpr const_reference back() const noexcept
	{
		return NumElements ?
			array_traits_type::reference(elements, NumElements - 1) :
			array_traits_type::reference(elements, 0);
	}

	__forceinline__ __device__ __host__
	pointer data() noexcept { return array_traits_type::pointer(elements); }

	__forceinline__ __device__ __host__
	const_pointer data() const noexcept { return array_traits_type::pointer(elements); }
};

// Array comparisons.
template<typename T, size_t NumElements>
inline bool operator==(const array<T, NumElements>& one, const array<T, NumElements>& two)
{
	return std::equal(one.begin(), one.end(), two.begin());
}

template<typename T, size_t NumElements>
inline bool
operator!=(const array<T, NumElements>& one, const array<T, NumElements>& two)
{ return !(one == two); }

template<typename T, size_t NumElements>
/* __forceinline__ __device__ */ __host__
inline bool
operator<(const array<T, NumElements>& a, const array<T, NumElements>& b)
{
	return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
}

template<typename T, size_t NumElements>
/* __forceinline__ __device__ */ __host__
inline bool operator>(const array<T, NumElements>& one, const array<T, NumElements>& two)
{
	return two < one;
}

template<typename T, size_t NumElements>
/* __forceinline__ __device__ */ __host__
inline bool operator<=(const array<T, NumElements>& one, const array<T, NumElements>& two)
{
	return !(one > two);
}

template<typename T, size_t NumElements>
/* __forceinline__ __device__ */ __host__
inline bool operator>=(const array<T, NumElements>& one, const array<T, NumElements>& two)
{
	return !(one < two);
}

// Specialized algorithms.
template<typename T, size_t NumElements>
__forceinline__ __device__ __host__
void swap(array<T, NumElements>& one, array<T, NumElements>& two)
noexcept(noexcept(one.swap(two)))
{
	one.swap(two);
}

template<size_t Integer, typename T, size_t NumElements>
__forceinline__ __device__ __host__
constexpr T& get(array<T, NumElements>& arr) noexcept
{
	static_assert(Integer < NumElements, "index is out of bounds");
	return array_traits<T, NumElements>::reference(arr.elements, Integer);
}

template<size_t Integer, typename T, size_t NumElements>
__forceinline__ __device__ __host__
constexpr T&& get(array<T, NumElements>&& arr) noexcept
{
	static_assert(Integer < NumElements, "index is out of bounds");
	return std::move(cuda::get<Integer>(arr));
}

template<size_t Integer, typename T, size_t NumElements>
__forceinline__ __device__ __host__
constexpr const T& get(const array<T, NumElements>& arr) noexcept
{
	static_assert(Integer < NumElements, "index is out of bounds");
	return array_traits<T, NumElements>::reference(arr.elements, Integer);
}

} // namespace cuda

// TODO: Do we really need this when we have no CUDA'ized std::tuple?
namespace cuda {

// Tuple interface to class template array.

/// tuple_size
template<typename T> class tuple_size;

/// Partial specialization for cuda::array
template<typename T, size_t NumElements>
struct tuple_size<cuda::array<T, NumElements>> : public std::integral_constant<size_t, NumElements> { };

/// tuple_element
template<size_t Integer, typename T> class tuple_element;

/// Partial specialization for cuda::array
template<size_t Integer, typename T, size_t NumElements>
struct tuple_element<Integer, cuda::array<T, NumElements>>
{
	static_assert(Integer < NumElements, "index is out of bounds");
	typedef T type;
};

} // namespace cuda

#endif /* CUDA_STL_ARRAY_CUH_ */
