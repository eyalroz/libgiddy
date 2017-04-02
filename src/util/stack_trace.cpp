

#ifndef __GNUC__
#error "Stack trace functionality is only currently available through a GCC extension, and you're not using GCC..."
#endif

#include "util/stack_trace.h"

#include <execinfo.h>
#include <cxxabi.h>
#include <dlfcn.h>
#include <cstdlib>

namespace util {

char* demangle(const char* symbol_name) {
	int status;
	char* demangled = abi::__cxa_demangle(symbol_name, nullptr, nullptr, &status);
	if (status == 0) { return demangled; }
	if (demangled != nullptr) { free(demangled); }
	return nullptr;
}

namespace stack_trace {

call_stack_t::call_stack_t (size_t stack_frames_to_discard) {
	using namespace abi;

	// We also need to discard the stack frame for this function,
	// and of the backtrace (I think)
	stack_frames_to_discard += 2;

	// unfortunately, the GCC ABI doesn't provide us with line numbers (or does it?)
	has_line_numbers = false;

	void* return_addresses[MaximumDepth];
	auto stack_depth = backtrace(return_addresses, MaximumDepth);
	entries.reserve(stack_depth - stack_frames_to_discard);

	for (int i = stack_frames_to_discard; i < stack_depth; i++) {

		Dl_info dlinfo;
		if (dladdr(return_addresses[i], &dlinfo) == 0) { break; }

		const char* symbol_name = dlinfo.dli_sname;

		if (dlinfo.dli_fname && symbol_name) {
			// TODO: use unique_ptr maybe and a helper function
			char* demangled = demangle(symbol_name);
			auto maybe_demangled = (demangled != nullptr) ? demangled : symbol_name;
			entries.emplace_back(dlinfo.dli_fname, maybe_demangled);
			if (demangled != nullptr) { free(demangled); }
		} else {
			// These are stack entries below main - they
			// have a symbol, but no function; we can skip them
			break;
		}
	}
}

std::string to_string(const call_stack_t::entry_t& entry, bool with_line_number)
{
	std::ostringstream oss;
	oss << entry.file_name << ':';
	if (with_line_number) { oss << entry.line_num; }
	else { oss << "(??" ")"; }
	oss << " " << entry.function_name;
	return oss.str();
}

std::string to_string (const call_stack_t& call_stack)
{
	std::ostringstream os;
	for (const auto& entry: call_stack.entries)
		os << to_string(entry, call_stack.has_line_numbers) << '\n';
	return os.str();
}

} // namespace stack_trace


} // namespace util

