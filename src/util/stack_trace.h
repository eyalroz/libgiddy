#pragma once
#ifndef UTIL_CALL_STACK_H_
#define UTIL_CALL_STACK_H_

#include <string>
#include <vector>
#include <sstream>

namespace util {

namespace stack_trace {

class call_stack_t {
public:
	enum { MaximumDepth = 32 };

	// Shouldn't this use char*'s and be on the stack?
	struct entry_t {
	    std::string  file_name;
	    std::string  function_name;
	    size_t       line_num;

		entry_t () : line_num(0) { }
	    entry_t(const std::string& file_, const std::string& function_, size_t line_ = 0) :
	    	file_name(file_), function_name(function_), line_num(line_) { }

	};


    call_stack_t (size_t stack_frames_to_discard = 0);

    bool empty() const { return entries.empty(); }

    virtual ~call_stack_t () throw() { }

    /**
     * Call stack... but in what direction does it go? :-(
     *
     * @todo Shouldn't this be something of static size and stack-allocated,
     * for safety? like an std::array? Hmm.
     *
     */
    std::vector<entry_t> entries;
    /**
     * The implementation of the call stack retriever might not be able
     * to retrieve line numbers (and we'll then only have 0's)
     */
    bool has_line_numbers;
};

std::string to_string(const call_stack_t::entry_t& entry, bool with_line_number = false);
std::string to_string(const call_stack_t& call_stack);


} // namespace stack_trace

} // namespace util

#endif /* UTIL_CALL_STACK_H_ */
