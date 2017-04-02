/*
 * Poor man's reflection utilities!
 *
 * C++ itself doesn't have reflection; and integrating a full-fledged reflection library is a lot of hassle.
 * Also, I'm not the greatest of C++ gurus and I need to get some stuff actually up and running, quick and
 * dirty, with reflection. So - this is the result. It's ugly and inconsistent, I know.
 *
 * TODO: Place everything here within util
 */

#ifndef UTIL_POOR_MANS_REFLECTION_H_
#define UTIL_POOR_MANS_REFLECTION_H_

#include "util/macro.h"
#include "util/cstring.hpp"

#include <iostream>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <cstdint>
#include <cstddef>
#include <typeinfo>
#include <cstring>
#include <type_traits>

namespace util {

/**
 * A variant of sizeof which takes the string name of a type, rather
 * than the type itself or a variable. It does not support all types,
 * but support for all fundamental types (without type construction using
 * arrays, variant, optionals, structs, unions, classes, pointers).
 *
 * @note could probably be implemented as a constexpr function with C++14
 * relatively easily.
 *
 * @param data_type_name the type's name, e.g. "char" or "unsigned long int"
 * @return the size in bytes of a variable of the specified type
 * @throws util::runtime_error if the type cannot be resolved
 */
std::size_t size_of(const std::string& data_type_name);

} // namespace util

#define ALL_FUNDAMENTAL_NONVOID_TYPES \
	bool, \
	wchar_t, char16_t, char32_t, \
	char, signed char, unsigned char, \
	short, short int, signed short, unsigned short, signed short int, unsigned short int, \
	long, long int, signed long, unsigned long, signed long int, unsigned long int, \
	long long, long long int, signed long long, unsigned long long, signed long long int, unsigned long long int, \
	int, signed, unsigned, signed int, unsigned int, \
	float, double, long double

#define ALL_INTEGRAL_TYPES_NO_DUPES \
	char, signed char, unsigned char, \
	signed short int, unsigned short int, \
	signed long int, unsigned long int, \
	signed long long int, unsigned long long int, \
	signed int, unsigned int

#define ALL_NUMERIC_TYPES_NO_DUPES \
	ALL_INTEGRAL_TYPES_NO_DUPES, \
	float, double, long double

#define ALL_FUNDAMENTAL_NONVOID_TYPES_NO_DUPES \
	bool, ALL_NUMERIC_TYPES_NO_DUPES

#define ALL_FUNDAMENTAL_TYPES void, ALL_FUNDAMENTAL_NONVOID_TYPES

#define POOR_MANS_REFLECTION_RESOLVER_FUNCTIONS_ENTRY(_identifier, _type) \
		::util::cstring::constexpr_equals(type_name, #_type) ? _identifier<_type> : \

// -----------------------


#define POOR_MANS_REFLECTION_RESOLVER(_getter_name, _function_template_name) \
decltype(&_function_template_name<int>) _getter_name(const char* type_name) \
{ \
	return  \
MAP_BINARY(POOR_MANS_REFLECTION_RESOLVER_FUNCTIONS_ENTRY, _function_template_name, ALL_FUNDAMENTAL_NONVOID_TYPES ) \
	_function_template_name<int>; \
} // tough cookies, if we don't have your type - this will fail at runtime; that's the price of constexpr'ness

#define POOR_MANS_CONSTEXPR_REFLECTION_RESOLVER(_getter_name, _function_template_name) \
constexpr POOR_MANS_REFLECTION_RESOLVER(_getter_name, _function_template_name)

// -----------------------

#define POOR_MANS_REFLECTION_RESOLVER_TRAIT_VALUE_ENTRY(_trait_name, _type) \
		::util::cstring::constexpr_equals(type_name, #_type) ? _trait_name<_type >::value : \


#define POOR_MANS_REFLECTION_TRAIT_RESOLVER(_getter_name, _trait_name) \
constexpr bool _getter_name(const char* type_name) \
{ \
	return  \
MAP_BINARY(POOR_MANS_REFLECTION_RESOLVER_TRAIT_VALUE_ENTRY, _trait_name, ALL_FUNDAMENTAL_NONVOID_TYPES ) \
	false; \
} // tough cookies, if we don't have your type - this will fail at runtime; that's the price of constexpr'ness

namespace util {

#define MAKE_REFLECTED_STD_TYPE_TRAIT(_trait_name) \
	POOR_MANS_REFLECTION_TRAIT_RESOLVER(_trait_name, std::_trait_name);

MAKE_REFLECTED_STD_TYPE_TRAIT(is_floating_point);
MAKE_REFLECTED_STD_TYPE_TRAIT(is_integral);
MAKE_REFLECTED_STD_TYPE_TRAIT(is_unsigned);
MAKE_REFLECTED_STD_TYPE_TRAIT(is_arithmetic);
MAKE_REFLECTED_STD_TYPE_TRAIT(is_abstract);
MAKE_REFLECTED_STD_TYPE_TRAIT(is_array);
MAKE_REFLECTED_STD_TYPE_TRAIT(is_pod);
MAKE_REFLECTED_STD_TYPE_TRAIT(is_enum);
MAKE_REFLECTED_STD_TYPE_TRAIT(is_const);
MAKE_REFLECTED_STD_TYPE_TRAIT(is_class);
MAKE_REFLECTED_STD_TYPE_TRAIT(is_empty);

#undef MAKE_REFLECTED_STD_TYPE_TRAIT
} // namespace util

// -----------------------


#define INSTANTIATE_TEMPLATED_CLASS(_identifier, ...) \
namespace detail { \
decltype(&_identifier<__VA_ARGS__>) UNIQUE_IDENTIFIER(instantiate)() \
{ \
	return &_identifier<__VA_ARGS__>; \
} \
} // namespace detail

#define INSTANTIATE_WITH_SPECIFIC_TEMPLATE_ARGUMENTS_INNER(_function, _type_of_function, ...) \
template _type_of_function<__VA_ARGS__> _function<__VA_ARGS__>

#define INSTANTIATE_WITH_SPECIFIC_TEMPLATE_ARGUMENTS(_function, ...) \
template<typename T> using _##_function##_t = decltype(_function<T>); \
INSTANTIATE_WITH_SPECIFIC_TEMPLATE_ARGUMENTS_INNER(_function, _##_function##_t, __VA_ARGS__)


#define INSTANTIATE_MEMBER_OF_TEMPLATED_CLASS(_class_name, _method_name, ...) \
namespace detail { \
decltype(&_class_name<__VA_ARGS__>::_method_name) UNIQUE_IDENTIFIER(instantiate ## _class_name ## _method_name)() \
{ \
	return &_class_name<__VA_ARGS__>::_method_name; \
} \
}

#define INSTANTIATE_FREESTANDING_FUNCTION(_function_name, ...) \
namespace detail { \
decltype(&_function_name<__VA_ARGS__>) UNIQUE_IDENTIFIER(instantiate ## _method_name)() \
{ \
	return &_function_name<__VA_ARGS__>; \
} \
}


// TODO: Perhaps UNDEF the macros here?

#endif /* UTIL_POOR_MANS_REFLECTION_H_ */
