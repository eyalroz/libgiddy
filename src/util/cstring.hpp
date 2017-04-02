#pragma once
#ifndef SRC_UTIL_CSTRING_HPP_
#define SRC_UTIL_CSTRING_HPP_

#include <cstring>
#include <functional>
// This next line may not be very portable... it accesses the C++ library's internal
// function for actually hashing a sequence of bytes - which is not part of the standard,
// nor the technical specifications etc. It works with GCC.
#include <bits/hash_bytes.h> // for std::_Hash_bytes


namespace util {

namespace cstring {

struct equals: public std::binary_function<const char*, const char*, bool> {
	bool operator()(const char* lhs, const char* rhs) const
	{
        return std::strcmp(lhs, rhs) == 0;
	}
};

struct hasher {
	using result_type = std::hash<std::string>::result_type;
	using argument_type = const char*;

	result_type operator()(argument_type const& arg) const
	{
		// The following is what std::hash<std::string>() uses - but I don't
		// know how to acces it in properly - this is fugly.
		size_t seed = static_cast<size_t>(0xc70f6907UL);
		return std::_Hash_bytes(arg, strlen(arg), seed);
	}
};

constexpr bool constexpr_equals(char const * a, char const * b) {
    return *a == *b && (*a == '\0' || constexpr_equals(a + 1, b + 1));
}

} // namespace cstring
} // namespace util

#endif /* SRC_UTIL_CSTRING_HPP_ */
