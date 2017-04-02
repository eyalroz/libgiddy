#pragma once
#ifndef ENDIANNESS_H_
#define ENDIANNESS_H_

#include <endian.h>

namespace util {
enum class terminal_t : bool { start, beginning = start, end };

enum class endianness_t : bool { big, big_endian = big, little, little_endian = little };

inline constexpr terminal_t starts_at(endianness_t endianness) {
	return endianness == endianness_t::little_endian ? terminal_t::beginning : terminal_t::end;
}
inline constexpr terminal_t ends_at(endianness_t endianness) {
	return endianness == endianness_t::little_endian ? terminal_t::end : terminal_t::beginning;
}

inline constexpr terminal_t little_end_at(endianness_t endianness) { return starts_at(endianness); }
inline constexpr terminal_t big_end_at(endianness_t endianness) { return ends_at(endianness); }

inline constexpr endianness_t compilation_target_endianness() {
#if __BYTE_ORDER == __LITTLE_ENDIAN
	return endianness_t::little;
#elif __BYTE_ORDER == __BIG_ENDIAN
	return endianness_t::big;
#else
#error "Non-little, non-big endianness not supported."
#endif
}

}

#endif /* ENDIANNESS_H_ */
