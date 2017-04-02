#pragma once
#ifndef MISCELLANY_H_
#define MISCELLANY_H_

// This prevents spurious Boost warning messages, see:
// http://stackoverflow.com/q/1814548/1593077
#ifndef BOOST_SYSTEM_NO_DEPRECATED
#define BOOST_SYSTEM_NO_DEPRECATED 1
#endif

#include "util/math.hpp" // for kinda_equal and ilog2
#include "util/index_sequence.hpp"

#include <cstring> // for std::memset()
#include <string>
#include <tuple>
#include <type_traits>
#include <algorithm>
#include <numeric> // for std::accumulate()
#include <initializer_list> // used by is_in

#include <cstddef>
#include <climits> // for CHAR_BIT
#include <limits> // for addition_will_overflow

// Use LIKELY() and UNLIKELY() to provide the compiler
// with hints regarding what you expect a conditional
// jump to usually do
#if defined(__GNUC__) && __GNUC__ >= 4
#ifndef UNLIKELY
#define LIKELY(x)   (__builtin_expect((x), 1))
#define UNLIKELY(x) (__builtin_expect((x), 0))
#endif /* UNLIKELY */
#else /* defined(__GNUC__) && __GNUC__ >= 4 */
#ifndef UNLIKELY
#define LIKELY(x)   (x)
#define UNLIKELY(x) (x)
#endif /* UNLIKELY */
#endif /* defined(__GNUC__) && __GNUC__ >= 4 */

// TODO: Insert everything else in this file into the util namespace
namespace util {

using std::size_t;

enum { bits_per_byte = CHAR_BIT, bits_per_char = CHAR_BIT, log_bits_per_char = 3 };

/**
 * Call the same function (without possibility of different instantiation)
 * for each one of the elements of a parameter pack
 *
 * @param f The function to call
 * @param args the arguments on which to call {@ref f}
 */
template <typename F, typename... Args>
void for_each_argument(F f, Args&&... args) {
	[](...){}((f(std::forward<Args>(args)), 0)...);
}

inline void* memzero(void *p, size_t n) { return std::memset(p, 0, n); }

} // namespace util

// useful for overloading as a replacement for partial function template specialization
template <typename T> struct tag {};



// TODO: Same thing for transform!

template <typename Signature, Signature& func>
struct functor;

template<typename R, typename ... Args, R (&func)(Args...)>
struct functor<R(Args...), func> {
	template<typename ... Ts>
	R operator()(Ts&&... xs) const	{
		return func(std::forward<Ts>(xs)...);
	}
};

// Based on this StackOverflow post:
// http://stackoverflow.com/a/11414631/1593077
//
template <template<typename> class Trait, typename Head, typename ...Tail>
struct check_all {
  enum { value = Trait<Head>::value && check_all<Trait, Tail...>::value };
};

template <template<typename> class Trait, typename Head>
struct check_all<Trait, Head> {
  enum { value = Trait<Head>::value };
};

// TODO: implement transform_if and container-assignment/construction transform

// Purely a convenience, please ignore
#define dp(x) std::cerr << x << '\n' << std::flush;

/*
 * Manual unrolling of a loop; the loop body can be represented
 * by a lambda, taking an iterator.
 *
 * Usage:
 *
 * Instead of
 *
 *   for(auto it = v.begin(); it < v.end(); it++) {
 *       // do stuff with it
 *   }
 *
 * we have
 *
 *  auto& f = [&](decltype(v.begin()) it) {
 *      // do stuff with it
 *  };
 *
 *   for(auto it = v.begin(); it < v.end(); it++ ) {
 *       unroller<N>.unroll(it, [&](decltype(it)) {
 *            // do stuff with it
 *       });
 *   }
 *
 * of course, this does not handle the case of v's length not
 * being a multiple of N, that's your problem.
 *
 */
template <size_t N> struct unroller {
	template <typename ForwardIterator, typename Function>
	void unroll(const Function& f, ForwardIterator& it) {
		f(it); ++it;
		unroll<N-1, ForwardIterator, Function>(f, it);
	}
};
template <> struct unroller<0> {
	template <typename ForwardIterator, typename Function>
	void unroll(const Function& f, ForwardIterator& it) { }
};

template <> struct unroller<1> {
	template <typename ForwardIterator, typename Function>
	void unroll(const Function& f, ForwardIterator& it) { f(it); }
};

namespace util {

// TODO: Cast everything to the common type, don't use the common-among-two each time
template <class T, class U >
constexpr typename std::common_type<T,U>::type
min(const T & a, const U & b)
{
	using R = typename std::common_type<T,U>::type;
	return (R) a < (R) b ? a : b;
}

template <class T, class U, class ... R >
constexpr typename std::common_type<T, U, R...>::type
min(const T & a, const U & b, const R &... c)
{
	return min(min(a, b), c...);
}

template <typename Iterator>
Iterator safe_advance(Iterator it, Iterator end, size_t delta)
{
	size_t advanced_distance = 0;
	while(it != end && advanced_distance++ < delta) { it++; }
	if (advanced_distance < delta) {
		throw std::runtime_error(
			"Could only advance iterator by " + std::string(advanced_distance) +
			" before reaching the sentinel.");
	}
	return it;
}

template <typename T>
static bool kinda_equal(
	const typename std::enable_if<std::is_floating_point<T>::value, T>::type& lhs,
	const T& rhs,
	bool floating_point_absolute_differences = false,
	double precision_threshold = 1e-06)
{
	auto distance =  ::util::get_distance(
		lhs, rhs, !floating_point_absolute_differences, (T) precision_threshold);
	return distance < precision_threshold;
}

template <typename T>
static bool kinda_equal(
	const typename std::enable_if<!std::is_floating_point<T>::value, T>::type& lhs,
	const T& rhs,
	bool floating_point_absolute_differences = false,
	double precision_threshold = 1e-06)
{
	return lhs == rhs;
}

template <typename T>
constexpr inline T num_bits_to_num_bytes(T n_bits) {
	return (n_bits  + bits_per_char - 1) >> log_bits_per_char;
}

/**
 * The number bits in the representation of a value of type T
 */
template <typename T>
struct size_in_bits { enum : size_t { value = sizeof(T) << log_bits_per_char }; };

/**
 * The number of distinct values of type T
 */
template <typename T>
struct domain_size {
	enum : size_t { value = ((size_t)1) << (
		sizeof(typename std::enable_if<
			sizeof(T) < sizeof(size_t) and std::is_integral<T>::value, T
		>::type) * bits_per_char) };
};

/**
 * The number of distinct values of type T
 */
template <typename T>
struct capped_domain_size {
	enum : size_t { value = ((size_t)1) << (
		sizeof(typename std::enable_if<
			sizeof(T) < sizeof(size_t) and std::is_integral<T>::value, T
		>::type) * bits_per_char) };
};

/**
 * Same as @ref domain_length<T> but allowing for a supposedly
 * lower-by-1 domain size for size_t-sized types, which cannot hold
 * the actual value.
 */
template<>
struct capped_domain_size<long> {
	enum : size_t { value = std::numeric_limits<
		typename std::enable_if<
			sizeof(long) == sizeof(size_t), long
		>::type>::max() };
};
template<>
struct capped_domain_size<unsigned long> {
	enum : size_t { value = std::numeric_limits<
		typename std::enable_if<
			sizeof(unsigned long) == sizeof(size_t), unsigned long
		>::type>::max() };
};
template<>
struct capped_domain_size<long long> {
	enum : size_t { value = std::numeric_limits<
		typename std::enable_if<
			sizeof(long long) == sizeof(size_t), long long
		>::type>::max() };
};
template<>
struct capped_domain_size<unsigned long long> {
	enum : size_t { value = std::numeric_limits<
		typename std::enable_if<
			sizeof(unsigned long long) == sizeof(size_t), unsigned long long
		>::type>::max() };
};

template <typename T>
unsigned char type_size_to_fit(typename std::enable_if<std::is_unsigned<T>::value, T>::type x)
{
//	return util::round_down_to_power_of_2(util::ceil_log2(x) / CHAR_BIT);
	if (                   x <= std::numeric_limits<uint8_t >::max()) { return 1; }
	if (sizeof(T) >= 2 and x <= std::numeric_limits<uint16_t>::max()) { return 2; }
	if (sizeof(T) >= 4 and x <= std::numeric_limits<uint32_t>::max()) { return 4; }
	if (sizeof(T) >= 8 and x <= std::numeric_limits<uint64_t>::max()) { return 8; }
	// Kind of a lame way to handle other cases:
	throw std::logic_error("Unable to determine size to fit an unsigned value");
}

template <typename T>
unsigned char type_size_to_fit(
	typename std::enable_if<std::is_integral<T>::value and not std::is_unsigned<T>::value, T>::type x)
{
	if (                   x >= std::numeric_limits<int8_t >::min() and x <= std::numeric_limits<int8_t >::max()) { return 1; }
	if (sizeof(T) >= 2 and x >= std::numeric_limits<int16_t>::min() and x <= std::numeric_limits<int16_t>::max()) { return 2; }
	if (sizeof(T) >= 4 and x >= std::numeric_limits<int32_t>::min() and x <= std::numeric_limits<int32_t>::max()) { return 4; }
	if (sizeof(T) >= 8 and x >= std::numeric_limits<int64_t>::min() and x <= std::numeric_limits<int64_t>::max()) { return 8; }
	// Kind of a lame way to handle other cases:
	throw std::logic_error("Unable to determine size to fit a signed value");
}

template <typename T>
unsigned char num_bytes_to_fit(typename std::enable_if<std::is_unsigned<T>::value, const T>::type x) {
	auto highest_bit_index = util::floor_log2(x);
	return num_bits_to_num_bytes(highest_bit_index + 1);
}

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

template<typename InputIterator, typename Separator>
inline std::ostream& print_separated(
	std::ostream&     os,
	const Separator&  separator,
	InputIterator     start,
	InputIterator     end)
{
	if (start != end)   { os << *(start++); }
	while (start < end) { os << separator << *(start++); }
	return os;
}

namespace interleave {

// Parameter pack interleaving
//
// Due to: http://stackoverflow.com/a/37983898/1593077

template<class...> struct pack {};

template<class T = pack<>, class...>
struct concat { using type = T; };

template<class... T1, class... T2, class... Ts>
struct concat<pack<T1...>, pack<T2...>, Ts...>
	: concat<pack<T1..., T2...>, Ts...> {};

template<class... Ts>
using concat_t = typename concat<Ts...>::type;

template<class... Us>
struct interleave {
	 template<class... Vs>
	 using with = concat_t<pack<Us, Vs>...>;
};

} // namespace interleave

namespace detail {

template <typename T>
constexpr T add_decimal_digits(T value_so_far, unsigned digit)
{
	return ((std::numeric_limits<T>::max() - (T) digit) / (T) 10) < value_so_far ?
		value_so_far : add_decimal_digits<T>(value_so_far * 10 + digit, digit);
}

} // namespace detail

/**
 * Produce the highest number fitting into type T which is a sequence of
 * decimal digits identical to @p fill_digit .
 */
template <typename T>
constexpr T fill_decimal_digits(unsigned fill_digit) {
	return detail::add_decimal_digits<T>(fill_digit, fill_digit);
}

/**
 * This is a replacement for std::align() in case you don't have it for some reason -
 * since it is indeed missing from GCC 4.9.x, see:
 * https://gcc.gnu.org/bugzilla/show_bug.cgi?id=57350
 *
 * @note will fail if pn is close to its maximum value
 *
 * @param alignment[in]  the desired alignment
 * @param size[in]       the size of the storage to be aligned
 * @param ptr[inout]     pointer to contiguous storage of at least space bytes
 * @param space[inout]   the size of the buffer in which to operate
 */
inline void *align( std::size_t alignment, std::size_t size, void *&ptr, std::size_t &space ) {
      auto pn = reinterpret_cast< std::uintptr_t >( ptr );
      auto aligned = ( pn + alignment - 1 ) & - alignment;
      auto new_space = space - ( aligned - pn );
      if ( new_space < size ) return nullptr;
      space = new_space;
      return ptr = reinterpret_cast< void * >( aligned );
}

template <typename T>
inline bool is_aligned(const T* ptr, size_t alignment)
{
    auto pn = reinterpret_cast< std::uintptr_t >( ptr );
    auto aligned = ( pn + alignment - 1 ) & - alignment;
    return pn == aligned;
}

// TODO: Other operations? Different types for x and y?
/**
 * @brief Checks whether one can add up two numbers without experiencing
 * an integer overflow (which is undefined behavior!)
 *
 * @return true if the additional will overflow, false if it's safe to do
 */
template <typename N>
inline constexpr bool addition_will_overflow(N x, N y)
{
	return std::is_unsigned<N>::value ?
		(x > std::numeric_limits<N>::max() - y) :
		(y > 0 && x > std::numeric_limits<N>::max() - y) ||
	    (y < 0 && x < std::numeric_limits<N>::min() - y);
}

/**
 * @brief The size of a type necessary to represent a number
 * @param x the value to be represented
 * @return the minimum power-of-2 size (in bytes) of a type which can
 * represent @p x (i.e. we ignore the possibility of using 3-byte
 * integers etc.)
 */
inline unsigned char min_size_to_represent(unsigned long long x) {
	unsigned char n = num_bits_to_num_bytes(util::ilog2<size_t>(x) + 1);
	if (n > 4) { return 8; }
	if (n > 2) { return 4; }
	return n;
}

template<typename T, class ... Args>
T in(T x, Args ... args)
{
    for (auto& arg : {args...}) {
        if (x == arg) { return true; }
    }
    return false;
}

/**
 * Calculates the mean of a fixed number of elements
 *
 * @note There are all sorts of implicit requirements on the element type;
 * this will likely fail for non-numeric types
 *
 * @tparam[in] T the type of elements to be considered
 * @param[in] args the elements for which the mean is to be calculated
 * @return the mean value, or a default-constructed T, which for numeric
 * types is 0
 */
template <typename T, class... Args, std::size_t N = sizeof...(Args)>
std::common_type<T,double> mean(Args... args) {
  std::array<T, N> arr = {args...};
  if (N > 0) return std::accumulate(std::begin(arr), std::end(arr), T{}) / static_cast<double>(N);
  return T{};
}

template <typename Comparator>
struct reverse_comparator
{
public:
    explicit reverse_comparator(Comparator comparator) : comparator_(comparator) {}
    template <typename T1, typename T2>
    bool operator() (const T1& lhs, const T2& rhs) const { return reverse_comparator(rhs, lhs); }
protected:
    Comparator comparator_;
};

/**
 * Shorthand for the type of the N'th element in a template parameter pack
 */
template <size_t N, typename... Ts>
using nth_type_in_pack = typename std::tuple_element<N, std::tuple<Ts...> >::type;

namespace detail
{
    template <typename T, std::size_t...Is>
    constexpr std::array<T, sizeof...(Is)> make_uniform_array(const T& val, std::index_sequence<Is...>)
    {
        return {(static_cast<void>(Is), val)...};
    }
} // namespace detail

/**
 * Returns an std::array constexpr-initialized to have all elements
 * set to the same value. C++11-compatible. Copied from:
 * http://stackoverflow.com/a/23905500/1593077
 *
 * @param val the value to set all array elements to
 * @return the constructed array
 */
template <typename T, std::size_t N>
constexpr std::array<T, N> make_uniform_array(const T& val)
{
    return detail::make_uniform_array(val, std::make_index_sequence<N>());
}

namespace detail
{
	template <typename F, typename... Args, size_t... Indices>
	void for_each_argument_indexed(F f, Args&&... args, std::index_sequence<Indices...>) {
		[](...){}((f(std::forward<Args>(args), Indices, sizeof...(args)), 0)...);
	}
} // namespace detail

/**
 * Call the same function (without possibility of different instantiation)
 * for each one of the elements of a parameter pack - with the index
 * of that element in the parameter pack and the pack's total size
 *
 * @param f The function to call
 * @param args the arguments on which to call {@ref f}
 */
template <typename F, typename... Args>
void for_each_argument_indexed(F f, Args&&... args) {
	detail::for_each_argument_indexed(f, args..., std::make_index_sequence<sizeof...(Args)>());
}

} // namespace util


#endif /* MISCELLANY_H_ */
