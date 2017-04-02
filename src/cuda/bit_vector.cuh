#ifndef CUDA_BIT_VECTOR_CUH_
#define CUDA_BIT_VECTOR_CUH_

#include "cuda/api/constants.h"
#include "cuda/syntax_replacement.h"
#include "util/miscellany.hpp"
#include "util/math.hpp"
#include "cuda/bit_operations.cuh"

#include "gsl/gsl-lite.h"

#include <climits> // for CHAR_BIT

#define __hd__ __host__ __device__
#define __fhd__ __forceinline__ __host__ __device__

namespace cuda {

namespace detail {
template <typename T>
__hd__ static constexpr int log2_constexpr(T val)
{
	return val ? 1 + log2_constexpr<T>(val >> 1) : -1;
}

} // namespace detail

/**
 * This class is a no-storage-owned facade over an array of elements
 * (typically GPUs' native 32-bit-size elements) which are interpreted
 * as vectors of bits - hence, a long vector of bits.
 *
 * @note There are often penalties for using char's rather than 32-bit
 * int's or unsigned ints in CUDA, hence the default choice of container
 * type. See, e.g. : http://stackoverflow.com/q/26993351/1593077
 *
 * @todo Is it ok to use host types here? What happens when host
 * types and device types differ? (e.g. size_t is 32-bit on some  host,
 * or rather int/unsigned int is 64-bit?)
 *
 */
template <typename Size = size_t, typename BitContainer = unsigned int>
class bit_vector {

// ---------------------------------------------------------------------------
public: // types
	using container_type    = BitContainer;
	using element_type      = BitContainer;
	using value_type        = bool; // ... but we won't be using value_type below
	using size_type         = Size;
	using size_of_size_type = unsigned char;
	// using difference_type = Difference;

// ---------------------------------------------------------------------------
public: // constants

	enum : unsigned {
		bits_per_element     = sizeof(container_type) * CHAR_BIT,
		log_bits_per_element = detail::log2_constexpr<unsigned>(bits_per_element),
	};

// ---------------------------------------------------------------------------
public: // ctors & dtor (all non-owning)

	__hd__ bit_vector() : data(nullptr), length(0) { }; // use of this ctor is discouraged
	__hd__ bit_vector(bit_vector& other) = default;
	__hd__ bit_vector(container_type* __restrict__ buffer, size_type length_in_bits) : data(buffer), length(length_in_bits) { };
	__hd__ bit_vector(const gsl::span<container_type>& bit_containers)
		: bit_vector(bit_containers.data(), bit_containers.length() * bits_per_element) { };
	__hd__ bit_vector(const gsl::span<container_type>& bit_containers, size_type length_in_bits)
		: bit_vector(bit_containers.data(), length_in_bits) { };
	__hd__ ~bit_vector() = default; // We don't own the storage, so nothing to free

// ---------------------------------------------------------------------------
public: // operators

	__hd__ bit_vector& operator=(const bit_vector& other) = default;
	__hd__ bool operator[](size_type index) const
	{
		return is_set(element_for(index), intra_element_index_for(index));
	}

// ---------------------------------------------------------------------------
public: // getters

	// I'm not using size() or length(); I wonder if that's a good idea
	__hd__ size_type length_in_bits() const { return length; }

// ---------------------------------------------------------------------------
public: // non-getter const methods

	__hd__ bool empty() const { return length == 0; }

	__hd__ size_of_size_type slack_length() { return num_slack_bits_for(length);	}
	__hd__ size_of_size_type num_slack_bits() const { return num_slack_bits_for(length);	}

	// TODO: Perhaps rename to "length in containers" or "length_in_bit_containers", or "num_containers"??
	__hd__ size_type length_in_elements() const
	{
#ifndef __CUDA_ARCH__
		using util::div_rounding_up;
#endif
		return div_rounding_up(length, bits_per_element); // TODO: Does this need optimization?
	}

	__hd__ size_type size_in_bytes() const
	{
		return sizeof(container_type) * length_in_elements();
	}

	__hd__ const container_type& element_for(size_type index) const
	{
		return data[element_index_for(index)];
	}

	// TODO: Consider writing a proxy, wrapping a container element
	// reference, with a [] const method, as well as set, unset
	// and flip methods,  taking a bit index
	//
	__hd__ static bool is_set(
		const container_type& container, unsigned index_within_element)
	{
		return is_set_uncast(container, index_within_element);
	}

	__hd__ const container_type& last_container() const
	{
		return element_for(length - 1);
	}

// ---------------------------------------------------------------------------
public: // mutators

	__host__ void unset_all() { std::memset(data,    0, size_in_bytes()); }
	__host__ void set_all()   { std::memset(data, 0xFF, size_in_bytes()); }


	__hd__ container_type&       element_for(size_type index)
	{
		return data[element_index_for(index)];
	}

	__hd__ container_type& last_container()
	{
		return element_for(length-1);
	}

	__hd__ void clear_slack() {
		container_type& the_last_container = last_container();
		the_last_container = mask_out_slack_bits(the_last_container, length);
	}


// ---------------------------------------------------------------------------
public: // single bit mutators

	__hd__ container_type& set(size_type index)
	{
		return set(element_for(index), intra_element_index_for(index));
	}
	__hd__ container_type& unset(size_type index)
	{
		return unset(element_for(index), intra_element_index_for(index));
	}
	__hd__ container_type& flip(size_type index)
	{
		return flip(element_for(index), intra_element_index_for(index));
	}
	// bit_value should be 0 or 1 - a boolean really; but I don't want to risk
	// any sort of casting happening here; Is it safe to make it a bool?
	__hd__ container_type& set_to(size_type index, container_type bit_value)
	{
//		container_type& el = element_for(index);
//		auto ii = intra_element_index_for(index);
//		return set_to(el, ii , bit_value);
		return set_to(element_for(index), intra_element_index_for(index), bit_value);
	}


// ---------------------------------------------------------------------------
public: // statics

	__hd__ static size_type element_index_for(size_type index)
	{
		// TODO: Maybe just use size_in_bits<bit_container_type>() here?
#ifdef __CUDA__ARCH__
		// No errors
		return index >> log_bits_per_element;
#else
		return index >>
			// This replacement, while ugly and defeating the purpose
			// of the second enum constant, prevents a warning by nvcc:
			//
			//   warning: right shift count >= width of type
			//
			// which has no merit AFAICT.
			//
			detail::log2_constexpr<size_type>(bits_per_element);
			// log_bits_per_element;
#endif
	}

	__hd__ static unsigned intra_element_index_for(size_type index)
	{
		return modulo_power_of_2(index, bits_per_element);
	}

	__hd__ static container_type element_bit_mask_for(size_type index)
	{
		return 1 << intra_element_index_for(index);
	}

	// TODO: Confusing name; perhaps element_bit_mask_for_intra ?
	__hd__ static container_type intra_element_bit_mask_for(unsigned index_within_element)
	{
		return 1 << index_within_element;
	}

	__hd__ static constexpr container_type element_for_single_bit(size_type index_within_element)
	{
		return 1 << index_within_element;
	}

	__hd__ static size_type global_index_for(
		size_type container_element_index, unsigned index_within_element)
	{
		return (container_element_index << log_bits_per_element) + index_within_element;
	}

	__hd__ static container_type is_set_uncast(
		const container_type& container, unsigned index_within_element)
	{
		return container & intra_element_bit_mask_for(index_within_element);
	}


	/**
	 * Returns the 0-based index of the specified bit
	 * (which is assumed to be set to 1 in the container element)
	 * among all bits set to 1 in that element, e.g. if bits 2, 5, 17, 30
	 * are set and all other bits are unset, index_among_set_bits(17) will
	 * return 2 while index_among_set_bits(5) will return 1.
	 *
	 * @param container a datum interpreted as a sequence of
	 * {@ref bits_per_element} bits
	 * @param index_within_element the index of the set bits among _all_ bits
	 * (as opposed to among the set bits)
	 * @return the index k such that exactly k bits with indices lower
	 * than {@ref index_within_element} are set within {@ref container}
	 */
	__hd__ static unsigned index_among_set_bits(
		container_type& container, unsigned index_within_element)
	{
		return count_bits_set(container & ((1 << index_within_element) - 1));
	}
	__hd__ static container_type& set(
		container_type& container, unsigned index_within_element)
	{
		return container |= intra_element_bit_mask_for(index_within_element);
	}
	__hd__ static container_type& unset(
		container_type& container, unsigned index_within_element)
	{
		return container &= ~intra_element_bit_mask_for(index_within_element);
	}
	__hd__ static container_type& flip(
		container_type& container, unsigned index_within_element)
	{
		return container ^= intra_element_bit_mask_for(index_within_element);
	}

	__hd__ static container_type& set_to(
		container_type& container,
		unsigned index_within_element,
		container_type bit_value)
	{
		unset(container, index_within_element);
		container |= bit_value << index_within_element;
		return container;
	}


	__hd__ static constexpr size_type num_elements_necessary_for(size_type num_bits)
	{
		return (num_bits + bits_per_element - 1) >> log_bits_per_element;
			// == div_rounding_up(num_bits, bits_per_element)
	}

	__hd__ static size_of_size_type num_slack_bits_for(size_type vector_length_in_bits)
	{
		return bits_per_element - modulo_power_of_2(vector_length_in_bits, bits_per_element);
	}

	__hd__ static container_type mask_out_slack_bits(container_type last_container, size_type total_length)
	{
		auto num_bits_used_in_last_elements = modulo_power_of_2(total_length, bits_per_element);
		return cuda::lowest_k_bits(last_container, num_bits_used_in_last_elements);
	}

// ---------------------------------------------------------------------------
protected: // data members
	container_type* __restrict__ data;   // Non-owning!
	size_type                    length; // in bits
};

template <typename Size = size_t, typename BitContainer = unsigned int>
using const_bit_vector = bit_vector<Size, const BitContainer>;

} // namespace cuda

#undef __hd__
#undef __fhd__

#endif /* CUDA_BIT_VECTOR_CUH_ */
