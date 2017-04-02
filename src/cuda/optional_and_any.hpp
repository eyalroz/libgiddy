// CUDA only supoorts C++11, while optional and any are C++14 experimental
// and C++17 proper features. For now, CUDA code can't use them - while host
// code does use them already. Instead, CUDA code will use boost's any and optional
// classes, by including this file. As soon as CUDA supports C++14 - we should
// drop this file and just go with the standard.

#ifndef CUDA_USE_BOOST_OPTIONAL_AND_ANY_HPP_
#define CUDA_USE_BOOST_OPTIONAL_AND_ANY_HPP_

#include <boost/any.hpp>
#include <boost/optional.hpp>
#include <boost/none_t.hpp>

namespace cuda {

const auto nullopt = boost::none;

template <typename T>
using optional = boost::optional<T>;

using any = boost::any;

/*template<typename T>
inline T any_cast(any& operand)
{
    return boost::any_cast<T>(operand);
}*/

template<typename T>
inline T any_cast(const any& operand)
{
    return boost::any_cast<T>(operand);
}

template<typename ValueType>
inline optional<ValueType> any_cast(const optional<any>& operand)
{
    return operand ?
    	optional<ValueType>(boost::any_cast<ValueType>(operand.value())) :
    	optional<ValueType>(nullopt);
}


/*
template<typename T>
inline const T any_cast(const any& operand)
{
    return boost::any_cast<T>(operand);
}
*/

/*

template<typename T>
inline const T any_cast(const any&& operand)
{
    return boost::any_cast(std::forward(operand));
}
*/


// Note:: util/stl_algorithms.hpp has something similar using the standard library optional
template <typename C, typename E>
inline optional<typename C::mapped_type> maybe_at(const C& container, const E& element) {
	auto find_result = container.find(element);
	return (find_result == container.end()) ?
		nullopt : optional<typename C::mapped_type>(find_result->second);
}


template <typename T, const char* MissingValueIndicator>
inline std::string to_string(const boost::none_t& n)
{
	return MissingValueIndicator;
}

template <typename T>
inline std::string to_string(const boost::none_t& n)
{
	return to_string<T,"(unset)">(n);
}


template <typename T, const char* MissingValueIndicator>
inline std::string to_string(const optional<T>& x)
{
	if (!x) { return MissingValueIndicator; }
	return std::to_string(x.value());
}

template <typename T>
inline std::string to_string(const optional<T>& x)
{
	if (!x) { return "(unset)"; }
	return std::to_string(x.value());
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const optional<T>& x)
{
	if (!x) { return os << std::string("(unset)"); }
	return os << x.value();
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const boost::none_t& n)
{
	return os << std::string("(unset)");
}

template<class T, class ...Args>
T& emplace_or(optional<T>& opt, Args&&...args) {
	if (!opt) { opt.emplace(std::forward<Args>(args)...); }
	return *opt;
}

template<class T>
inline optional<T> value_if(bool condition, T&& value) {
	if (!condition) return {};
	return value;
}

} // namespace cuda

// This next bit is useful for allowing
// non-NVCC-compiled code to use cuda::optional with host-side optional's

#ifndef __CUDACC__
#if __cplusplus == 201402L
#include <experimental/optional>
namespace cuda {
template <typename T>
using host_code_optional = std::experimental::optional<T>;
} // namespace cuda
#else
#if __cplusplus < 201402L
#include "util/optional.hpp"
namespace cuda {
template <typename T>
using host_code_optional = std::experimental::optional<T>;
} // namespace cuda
#else
#if __cplusplus > 201402L
// should be C++17 or later
#include <optional>
namespace cuda {
template <typename T>
using host_code_optional = std::optional<T>;
} // namespace cuda
#else
#error "something strange is going on with the __cplusplus version"
#endif /* __cplusplus > 201402L */
#endif /* __cplusplus == 201402L */
#endif /* __cplusplus < 201402L */

namespace cuda {

// Kind of a lame name, feel free to suggest an alternative
template<class T>
inline optional<T> adapt(const host_code_optional<T>& o)
{
	return o ? o.value() : optional<T>();
}

} // namespace cuda

#endif /* __CUDACC__ */


#endif /* CUDA_USE_BOOST_OPTIONAL_AND_ANY_HPP_ */
