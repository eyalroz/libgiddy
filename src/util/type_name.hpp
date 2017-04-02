#pragma once
#ifndef UTIL_TYPE_NAME_HPP_
#define UTIL_TYPE_NAME_HPP_

#include <type_traits>
#include <typeinfo>
#include <iostream>
#ifndef _MSC_VER
#include <cxxabi.h>
#endif
#include <string>
#include <sstream>
#include <memory>

namespace util {
/*
namespace detail {

// You called it, you own it! This allocates space
inline char* demangle_type_name(const char* name)
{
	return abi::__cxa_demangle(name, nullptr, nullptr, nullptr);
}

} // namespace
*/

/**
 * A function for obtaining the string name
 * of a type, using that actual type at compile-time.
 * (The function might have been constexpr, but I doubt
 * so much is acceptable at compile time.) This is an
 * alternative to using type_info<T>.name() which also
 * preserves CV qualifiers (const, volatile, reference,
 *  rvalue-reference)
 *
 * The code was copied from this StackOverflow answer:
 *  http://stackoverflow.com/a/20170989/1593077
 * due to Howard Hinnant
 * ... with some slight modifications by Eyal Rozenberg
 */


template <typename T, bool WithCVCorrections = false>
std::string type_name()
{
	typedef typename std::remove_reference<T>::type TR;

	std::unique_ptr<char, void(*)(void*)> own(
#ifndef _MSC_VER
//		detail::demangle_type_name(typeid(TR).name()),
	abi::__cxa_demangle(typeid(TR).name(), nullptr,	nullptr, nullptr),
#else
	nullptr,
#endif
		std::free
	);
	std::string r = (own != nullptr) ? own.get() : typeid(TR).name();
	if (WithCVCorrections) {
		if (std::is_const<TR>::value)
			r += " const";
		if (std::is_volatile<TR>::value)
			r += " volatile";
		if (std::is_lvalue_reference<T>::value)
			r += "&";
		else if (std::is_rvalue_reference<T>::value)
			r += "&&";
	}
	return r;
}

/**
 * This is a convenience function, so that instead of
 *
 *   util::type_name<decltype(my_value)>()
 *
 * you could use:
 *
 *   util::type_name_of(my_value
 *
 * @param v a value which is only passed to indicate a type
 * @return the string type name of typeof(v)
 */
template <typename T, bool WithCVCorrections = false>
std::string type_name_of(const T& v) { return util::type_name<T>(); }


#ifndef STRING_VIEW_DEFINED
struct string_view
{
    char const* data;
    std::size_t size;
};
#endif /* ifndef STRING_VIEW_DEFINED */

/*

// A C++14isbh constexpr way to get the name of a type
template<class T>
constexpr string_view get_name()
{
    char const* p = __PRETTY_FUNCTION__;
     *
     *
     * The __PRETTY_FUNCTION__ string has a form similar to one of the following:
     *
     *   constexpr string_view util::get_name<T>() [with T=whatever]
     *   constexpr string_view util::get_name<T>() [with T = whatever]
     *
     * so we need to find the = and chomp the space;
     *
     *
    while (*p++ != '=');
    for (; *p == ' '; ++p);
    char const* p2 = p;
    int count = 1;
    for (;;++p2)
    {
        switch (*p2)
        {
        case '[':
            ++count;
            break;
        case ']':
            --count;
            if (!count)
                return {p, std::size_t(p2 - p)};
        }
    }
    return {};
}
*/
/*

namespace detail {

inline std::string className(const std::string& pretty_function_name)
{
    auto first_colons = pretty_function_name.find("::");
    if (first_colons == std::string::npos) {
    	// we're not within a method of a class at all!
        return "";
    }
    auto first_colons = pretty_function_name.find("::");
    auto next_colons = pretty_function_name.find("::");
    while(auto next_colons = )
    size_t begin = pretty_function_name.substr(0, first_colons).rfind(" ") + 1;
    size_t end = first_colons - begin;

    return pretty_function_name.substr(begin,end);
}

#define __CLASS_NAME__ ::util::detail::className(__PRETTY_FUNCTION__)

*/

// TODO: Drop the following two, or redo them using
// the mechanisms in miscellany.h
namespace concat_type_names_detail {
	static inline std::string const& to_string(std::string const& s) { return s; }
}

template <char Separator, typename... Args>
std::string concat_type_names() {
    std::string result;
    using std::to_string;
    using concat_type_names_detail::to_string;
    // Note that we need to_string to be defined for strings here as well!
    // Make sure you've done that
    int dummy[]{0, (
    	result.empty() ? 0 : (result += Separator, 0),
    	result += to_string(type_name<Args>()), 0)...
    };
    static_cast<void>(dummy); // so we don't get a complaint about unpack being unused.
    return result;
}

/**
 * Removed the trailing template parameter listing from a type name, e.g.
 *
 *   foo<int> bar<plus<int>>
 *
 * becomes
 *
 *   foo<int> bar<plus<int>>
 *
 * This is not such useful function, as int bar<int>(double x) will
 * become int bar. So - fix it.
 *
 * @param type_name the name of a type, preferably obtained with
 * util::type_info
 * @return the template-less type name, or the original type name if
 * we could not find anything to remove (doesn't throw)
 */
inline std::string discard_template_parameters(const std::string& type_name)
{
	auto template_rbracket_pos = type_name.rfind('>');
	if (template_rbracket_pos == std::string::npos) {
		return type_name;
	}
	unsigned bracket_depth = 1;
	for (unsigned pos = template_rbracket_pos; pos > 0; pos++) {
		switch(type_name[pos]) {
		case '>': bracket_depth++; break;
		case '<': bracket_depth--; break;
		}
		if (bracket_depth == 0) return type_name.substr(0,pos);
	}
	return type_name;
}


} /* namespace util */

#endif /* UTIL_TYPE_NAME_HPP_ */
