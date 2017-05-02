#ifndef UTIL_INTEGER_H_
#define UTIL_INTEGER_H_

#include "util/bits.hpp"

// TODO: Consider dropping all of the boost stuff
#include <boost/integer.hpp>
#include <ostream>
#include <istream>
#include <cstring> // for memcpy and memset
#include <type_traits>

namespace util {

template <unsigned N, bool Signed>
struct integer_traits_t {
	static_assert(N <= sizeof(unsigned long long), "larger sizes not supported, for now");

public: // types and constants
	enum { num_bytes = N, num_bits = N * 8 };
	enum : bool { signedness = Signed };
	using byte = unsigned char;
	using value_type = byte[N];
	using fast_builtin_type =
		typename std::conditional<Signed,
			typename boost::int_t<num_bits>::fast,
			typename boost::uint_t<num_bits>::fast
		>::type;
	using least_builtin_type =
	typename std::conditional<Signed,
		typename boost::int_t<num_bits>::least,
		typename boost::uint_t<num_bits>::least
	>::type;

};

namespace detail {

enum : bool { is_signed = true, is_unsigned = false, isnt_signed = false };

/**
 * (UNTESTED!) A hopefully-fast integer-like class with arbitrary size
 *
 *
 * @note Heavily dependent on compiler optimizations...
 * @note For now, assumes little-endianness
 * @note For now, limited to small sizes
 *
 */
template <unsigned N, bool Signed>
class int_t final
{
	static_assert(N <= sizeof(unsigned long long), "larger sizes not supported, for now");

public:
	using traits = integer_traits_t<N, Signed>;
	using fast_builtin_type = typename traits::fast_builtin_type;
	using least_builtin_type = typename traits::least_builtin_type;
	using byte = unsigned char; // That should really happen elsewhere...

protected: // data members
	typename traits::value_type value; // Note it is _not_ necessarily aligned

public: // constructors
	int_t() noexcept = default;
	int_t(const int_t& x) noexcept = default;
	int_t(int_t&& x) noexcept = default;

protected: // building blocks for converting ctors, assignments and conversion operators

	static constexpr size_t min(size_t x, size_t y) { return x < y ? x : y; }

	// TODO: This doesn't have sign extension!
	template <typename I>
	int_t& assign(I x) noexcept
	{
		auto x_bytes = (const byte*) &x;

		for (auto j = 0; j < min(sizeof(I), N); j++) {
			value[j] = x_bytes[j];
		}
		for (auto j = min(sizeof(I), N); j < N; j++) {
			value[j] = 0;
		}
		return *this;
	}

	// TODO: This doesn't have sign extension!
	template <typename I>
	I as_integer() const noexcept
	{
		I result;

		if (sizeof(I) > N) { result = 0; }

		auto result_bytes = (byte*) &result;
		for (auto j = 0; j < min(sizeof(I), N); j++) {
			result_bytes[j] = value[j];
		}
		return result;
	}

public: // converting constructors
	int_t(char                x) noexcept { assign<char               >(x); }
	int_t(signed char         x) noexcept { assign<signed char        >(x); }
	int_t(unsigned char       x) noexcept { assign<unsigned char      >(x); }
	int_t(short               x) noexcept { assign<short              >(x); }
	int_t(unsigned short      x) noexcept { assign<unsigned short     >(x); }
	int_t(int                 x) noexcept { assign<int                >(x); }
	int_t(unsigned            x) noexcept { assign<unsigned           >(x); }
	int_t(long                x) noexcept { assign<long               >(x); }
	int_t(unsigned long       x) noexcept { assign<unsigned long      >(x); }
	int_t(long long           x) noexcept { assign<long long          >(x); }
	int_t(unsigned long long  x) noexcept { assign<unsigned long long >(x); }
	~int_t() = default;

public: // operators
	int_t& operator = (const int_t& other) noexcept = default;
	int_t& operator = (int_t&& other) noexcept = default;

	int_t& operator = (char                x) noexcept { return assign<char               >(x); }
	int_t& operator = (signed char         x) noexcept { return assign<signed char        >(x); }
	int_t& operator = (unsigned char       x) noexcept { return assign<unsigned char      >(x); }
	int_t& operator = (short               x) noexcept { return assign<short              >(x); }
	int_t& operator = (unsigned short      x) noexcept { return assign<unsigned short     >(x); }
	int_t& operator = (int                 x) noexcept { return assign<int                >(x); }
	int_t& operator = (unsigned            x) noexcept { return assign<unsigned           >(x); }
	int_t& operator = (long                x) noexcept { return assign<long               >(x); }
	int_t& operator = (unsigned long       x) noexcept { return assign<unsigned long      >(x); }
	int_t& operator = (long long           x) noexcept { return assign<long long          >(x); }
	int_t& operator = (unsigned long long  x) noexcept { return assign<unsigned long long >(x); }


	int_t& operator += (const fast_builtin_type& other) noexcept { return *this = as_fast_builtin() + other; }
	int_t& operator -= (const fast_builtin_type& other) noexcept { return *this = as_fast_builtin() - other; }
	int_t& operator *= (const fast_builtin_type& other) noexcept { return *this = as_fast_builtin() * other; }
	int_t& operator /= (const fast_builtin_type& other)          { return *this = as_fast_builtin() / other; }
	int_t& operator += (const int_t& other) noexcept { return operator+=(other.as_fast_builtin()); }
	int_t& operator -= (const int_t& other) noexcept { return operator-=(other.as_fast_builtin()); }
	int_t& operator *= (const int_t& other) noexcept { return operator*=(other.as_fast_builtin()); }
	int_t& operator /= (const int_t& other)          { return operator/=(other.as_fast_builtin()); }

	bool operator == (const int_t& other) noexcept { return value == other.value; }
	bool operator != (const int_t& other) noexcept { return value != other.value; }

protected: // conversion operators
	operator fast_builtin_type() const noexcept { return as_integer<fast_builtin_type>(); }

protected: // non-mutator methods
	fast_builtin_type as_fast_builtin() const noexcept  { return as_integer<fast_builtin_type>();  }
	fast_builtin_type as_least_builtin() const noexcept { return as_integer<least_builtin_type>(); }
};

// Additional operators which can make do with public members

template <unsigned N, bool Signed> bool operator >  (const int_t<N, Signed>&x, const int_t<N, Signed>& y) noexcept { return x.as_fast_builtin() >  y.as_fast_builtin(); }
template <unsigned N, bool Signed> bool operator <  (const int_t<N, Signed>&x, const int_t<N, Signed>& y) noexcept { return x.as_fast_builtin() <  y.as_fast_builtin(); }
template <unsigned N, bool Signed> bool operator >= (const int_t<N, Signed>&x, const int_t<N, Signed>& y) noexcept { return x.as_fast_builtin() >= y.as_fast_builtin(); }
template <unsigned N, bool Signed> bool operator <= (const int_t<N, Signed>&x, const int_t<N, Signed>& y) noexcept { return x.as_fast_builtin() <= y.as_fast_builtin(); }

template <unsigned N, bool Signed> int_t<N, Signed>& operator ++ (int_t<N, Signed>& i) noexcept { return (i += 1); }
template <unsigned N, bool Signed> int_t<N, Signed>& operator -- (int_t<N, Signed>& i) noexcept { return (i -= 1); }
template <unsigned N, bool Signed>
int_t<N, Signed> operator++ (int_t<N, Signed>& i, int) noexcept
{
	int_t<N, Signed> result = i;
	i += 1;
	return result;
}
template <unsigned N, bool Signed>
int_t<N, Signed> operator-- (int_t<N, Signed>& i, int) noexcept
{
	int_t<N, Signed> result = i;
	i -= 1;
	return result;
}

template <unsigned N, bool Signed>
std::ostream& operator<<(std::ostream& os, int_t<N, Signed> i) { return os << i.as_least_builtin(); }
template <unsigned N, bool Signed>
std::istream& operator>>(std::istream& is, int_t<N, Signed> i)
{
	typename int_t<N, Signed>::fast_builtin_type fast_builtin;
	is >> fast_builtin;
	i = fast_builtin;
	return is;
}

} // namespace detail

template <unsigned N, bool Signed = detail::is_signed>
struct int_t;

template <unsigned N>
using uint_t = typename int_t<N, detail::is_unsigned>::type;


template <unsigned N, bool Signed>
struct int_t {
	using type = detail::int_t<N, Signed>;
};

template<> struct int_t<1, detail::is_signed   > { using type = int8_t;   };
template<> struct int_t<2, detail::is_signed   > { using type = int16_t;  };
template<> struct int_t<4, detail::is_signed   > { using type = int32_t;  };
template<> struct int_t<8, detail::is_signed   > { using type = int64_t;  };
template<> struct int_t<1, detail::is_unsigned > { using type = uint8_t;  };
template<> struct int_t<2, detail::is_unsigned > { using type = uint16_t; };
template<> struct int_t<4, detail::is_unsigned > { using type = uint32_t; };
template<> struct int_t<8, detail::is_unsigned > { using type = uint64_t; };

namespace detail {
namespace {
	template <typename T>
	constexpr int log2_constexpr(T val) { return val ? 1 + log2_constexpr(val >> 1) : -1; }

	template <typename T>
	constexpr int ceil_log2_constexpr(T val) { return val ? 1 + log2_constexpr<T>(val - 1) : -1; }

	template <typename T>
	constexpr inline T round_up_to_power_of_2_constexpr(const T& x_greater_than_one)
	{
		return ((T)1) << ceil_log2_constexpr(x_greater_than_one);
	}

} // namespace (anonymous)
} // namespace detail

/**
 * @brief A type trait for working with sizes of subsets of a type
 *
 * Consider some type, say an unsigned type, T. Its domain of possible
 * values is 0...sizeof(T)*CHAR_BIT - 1 . Now, if we take a subset of
 * this domain, what can its size be? 0...sizeof(T)*CHAR_BIT ; problem
 * is, this larger domain can't be represented by T! That's what this
 * type trait is about; it provides a type for representing this
 * slightly-larger domain, as well as its size.
 *
 * The trait is only defined when there exists a "natural" unsigned
 * integral type which can represent the larger domain, e.g. it
 * isn't defined for integer types of size 8 and higher; for those,
 * see @ref capped_domain_size instead.
 *
 */
template <typename T>
struct domain_size {
	using type = typename std::enable_if<
		sizeof(T) < 8,
		uint_t<detail::round_up_to_power_of_2_constexpr(sizeof(T) + 1)>
	>::type;

	enum : size_t { value = ((size_t)1) << (
		sizeof(typename std::enable_if<
			(sizeof(T) < sizeof(size_t)) and std::is_integral<T>::value, T
		>::type) * bits_per_char) };
};

/**
 * Same as @ref domain_size<T> but pretending the domain size
 * of 8-byte-sized types fits inside a {@code uint64_t} (otherwise a bunch
 * of code won't compile)
 */
template <typename T>
struct capped_domain_size {
	static_assert(std::is_integral<T>::value, "Not supported for non-integral types");
	using type = uint_t<sizeof(T) >= 8 ? 8 :
		detail::round_up_to_power_of_2_constexpr(sizeof(T) + 1)>;
	enum : bool { cap_reached = (sizeof(T) == 8) };
	enum : size_t { value =
		sizeof(T) < sizeof(size_t) ?
			( ((size_t) 1) << sizeof(T) * bits_per_byte) - 1 :
			std::numeric_limits<size_t>::max()
	};
};

} // namespace util

// Decide when/how to enable the following
#if 0


#ifdef __CUDACC__
#define __hd__  __host__ __device__
#else
#define __hd__
#endif /* __CUDACC__ */

namespace std {
template<unsigned N, bool Signed>
struct numeric_limits<util::detail::int_t<N, Signed>>
{
protected:
	using int_t = util::int_t<N, Signed>;
	using num_bits = util::int_t<N>::num_bits;

	// Note:
	// log_10(2)   ~= 0.301029995664
	// 643L / 2136 ~= 0.301029962547
	//
	// so we approximate the former with the latter here; it's
	// good enough for N's which are not extremely high
	//
	constexpr size_t approximate_log10_of_2(size_t x) { return x * 643L / 2136; }

public:

	static constexpr bool is_specialized = true;

	__hd__ static constexpr int_t min() noexcept { return 0; }

	__hd__ static constexpr int_t max() noexcept
	{
		return 1 << (num_bits - 1) + ((1 << (num_bits - 1)) - 1);
	}

	__hd__ static constexpr int_t
	lowest() noexcept { return min(); }

	static constexpr int digits = num_bits;
	static constexpr int digits10 = approximate_log10_of_2(num_bits) + 1;
	static constexpr int max_digits10 = 0;
	static constexpr bool is_signed = false;
	static constexpr bool is_integer = true;
	static constexpr bool is_exact = true;
	static constexpr int radix = 2;

	__hd__ static constexpr int_t
	epsilon() noexcept { return 0; }

	__hd__ static constexpr int_t
	round_error() noexcept { return 0; }

	static constexpr int min_exponent = 0;
	static constexpr int min_exponent10 = 0;
	static constexpr int max_exponent = 0;
	static constexpr int max_exponent10 = 0;

	static constexpr bool has_infinity = false;
	static constexpr bool has_quiet_NaN = false;
	static constexpr bool has_signaling_NaN = false;
	static constexpr float_denorm_style has_denorm = denorm_absent;
	static constexpr bool has_denorm_loss = false;

	__hd__ static constexpr
	int_t infinity() noexcept { return int_t(); }

	__hd__ static constexpr int_t
	quiet_NaN() noexcept { return int_t(); }

	__hd__ static constexpr int_t
	signaling_NaN() noexcept { return int_t(); }

	__hd__ static constexpr int_t
	denorm_min() noexcept { return static_cast<int_t>(0); }

	static constexpr bool is_iec559 = false;
	static constexpr bool is_bounded = true;
	static constexpr bool is_modulo = true;

	static constexpr bool traps = true; // TODO: I have no idea what I need to set this to
	static constexpr bool tinyness_before = false;
	static constexpr float_round_style round_style = round_toward_zero;
  };
}
#endif

#endif /* UTIL_INTEGER_H_ */
