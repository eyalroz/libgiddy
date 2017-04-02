/*
 * Some of this code is derived from the "stack trace" package
 * and is Copyright (c) 2009, Fredrik Orderud
 * License: BSD licence (http://www.opensource.org/licenses/bsd-license.php)
 */

#pragma once
#ifndef UTIL_EXCEPTION_H_
#define UTIL_EXCEPTION_H_

// TODO: Consider moving this into a separate header
#if defined(__GNUC__) && __GNUC__ >= 4
#ifndef UNLIKELY
#define LIKELY(x)   (__builtin_expect((x), 1))
#define UNLIKELY(x) (__builtin_expect((x), 0))
#endif
#else
#ifndef UNLIKELY
#define LIKELY(x)   (x)
#define UNLIKELY(x) (x)
#endif
#endif

#include "util/stack_trace.h"

#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <sstream>
#include <string>
#include <system_error>
#include <cerrno>

namespace util {

namespace stack_trace {

namespace mixins {

class with_stack {
public:

	// Note that the ctors here add 1 to the number of stack frames to ignore,
	// since we want to ignore the ctor's own stack frame as well; the same
	// should apply to any subclass calling this ctor

	with_stack (
		bool should_print_stack_= true,
		size_t stack_frames_to_ignore = 0)
		: call_stack(stack_frames_to_ignore + 1),
		  should_print_stack(should_print_stack_) { }

	with_stack (size_t additional_stack_frames_to_ignore)
		: call_stack(additional_stack_frames_to_ignore + 1),
		  should_print_stack(true) { }

	// virtual const char * what () const throw() = 0;

	call_stack_t call_stack;
	bool should_print_stack;
};

} // namespace mixins

} // namespace stack_trace

/** Template for stack-augmented exception classes. */
template<class ParentException>
class exception : public ParentException, public stack_trace::mixins::with_stack {
public:
	using parent_exception_t = ParentException;

	// Note that the ctors here add 1 to the number of stack frames to ignore,
	// since we want to ignore the ctor's own stack frame as well; the same
	// should apply to any subclass calling this ctor
	exception(const std::string & msg,
		size_t stack_frames_to_discard = 0)
		: ParentException(msg), with_stack(stack_frames_to_discard + 1) { }

	const char* what() const throw() override {
		if (should_print_stack) {
			std::ostringstream oss;
			oss << std::string(ParentException::what()) << '\n';
			if (!call_stack.empty()) {
				oss << "Stack trace:\n" << to_string(call_stack);
			}
			msg_buffer = oss.str();
			return msg_buffer.c_str();
		} else {
			return ParentException::what();
		}
	}
private:
	// Having this as a member allows us to return a char* without
	// the caller taking ownership of it
	mutable std::string msg_buffer;
};

namespace stack_trace {

namespace standard_exception_variants {

using domain_error     = exception<std::domain_error>;
using invalid_argument = exception<std::invalid_argument>;
using length_error     = exception<std::length_error>;
using logic_error      = exception<std::logic_error>;
using out_of_range     = exception<std::out_of_range>;
using overflow_error   = exception<std::overflow_error>;
using range_error      = exception<std::range_error>;
using runtime_error    = exception<std::runtime_error>;
using underflow_error  = exception<std::underflow_error>;

} // namespace standard_exceptions

} // namespace stack_trace

// Now just prefix your favorite exception with util:: rather
// than with std:: , to have it with a stack trace!
using namespace stack_trace::standard_exception_variants;


template <typename Exception, typename Printable>
[[ noreturn ]] static void die(const Exception& exception, const Printable& preceding_message) {
	std::ostringstream oss;
	oss << preceding_message;
	auto str = oss.str();
	auto last_char = oss.str().back();
	if (last_char == '\n' || last_char == '\r') {
		std::cerr << str.substr(0, str.length() - 1) << ": " << last_char;
	}
	else { std::cerr << str << ": "; }
	std::cerr << exception.what() << '\n';
	exit(EXIT_FAILURE);
}

template <typename Printable>
[[ noreturn ]] static void die(const Printable& message) {
	std::cerr << message << '\n';
	exit(EXIT_FAILURE);
}

/**
 * Throw an exception if the first argument is false.  The exception
 * message will contain the argument string as well as any passed-in
 * arguments to enforce, formatted using folly::to<std::string>.
 */
template <typename... Arguments>
inline void enforce(bool condition_check_result, Arguments&&... args) {
	if (UNLIKELY(!condition_check_result)) {
		std::ostringstream oss;
		// This applies oss << arg to all arguments in the pack
		(void) std::initializer_list<int> { (oss << std::forward<Arguments>(args), 0)... };
		throw logic_error(oss.str());
	}
}

template <typename T>
inline void enforceNonNull(const T* ptr, const std::string& message) {
	if (UNLIKELY(ptr == nullptr)) {
		throw logic_error(std::string("Null pointer encountered: ") + message);
	}
}

template <typename T>
inline void enforceNonNull(const T* ptr, const char* message ) {
	enforceNonNull(ptr, std::string(message));
}

template <typename T>
inline void enforceNonNull(const T* ptr) {
	if (UNLIKELY(ptr == nullptr)) {
		throw logic_error(std::string("Null pointer encountered"));
	}
}

template <typename T>
inline void enforceNull(const T* ptr, const std::string& message) {
	if (UNLIKELY(ptr != nullptr)) {
		throw logic_error(std::string("Non-null pointer encountered: ") + message);
	}
}

template <typename T>
inline void enforceNull(const T* ptr, const char* message ) {
	enforceNull(ptr, std::string(message));
}

template <typename T>
inline void enforceNull(const T* ptr) {
	if (UNLIKELY(ptr != nullptr)) {
		throw logic_error(std::string("Non-null pointer encountered"));
	}
}

// TODO: Make this include a stack trace?
inline std::system_error make_system_error(const std::string& message)
{
	return std::system_error(std::error_code(errno, std::system_category()), message);
}

} // namespace util

/*

// No need for these, since our exceptions have stack traces now...
#define ENFORCE_NOT_NULL(ptr) \
	::util::enforceNonNull(ptr, STRINGIZE(ptr) " at " __FILE__ ":" STRINGIZE(__LINE__) )
#define ENFORCE_NULL(ptr) \
	::util::enforceNull(ptr, STRINGIZE(ptr) " at " __FILE__ ":" STRINGIZE(__LINE__) )

*/

#endif /* UTIL_EXCEPTION_H_ */

