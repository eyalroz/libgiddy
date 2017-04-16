#ifndef SRC_UTIL_BUILTINS_HPP_
#define SRC_UTIL_BUILTINS_HPP_

#include <climits>
#include <cstdint>

// Move this out of here, it should really not be CUDA-specific
// (but should have its own small specific header, perhaps with some other related definitions
namespace util {
namespace builtins {

#ifndef __GNUC__
# error "The population count code is currently GCC-specific. It might actually work with clang, and should be easily adaptable to MSVC, but this has not been done yet."
#endif

template <typename T> int population_count(T x);

template<> inline int population_count<unsigned>(unsigned x) { return __builtin_popcount(x); }
template<> inline int population_count<unsigned long>(unsigned long x) { return __builtin_popcountl(x); }
template<> inline int population_count<unsigned long long>(unsigned long long x) { return __builtin_popcountll(x); }

template <typename T> int find_first_set(T x);

template<> inline int find_first_set<unsigned>(unsigned x) { return __builtin_ffs(x); }
template<> inline int find_first_set<unsigned long>(unsigned long x) { return __builtin_ffsl(x); }
template<> inline int find_first_set<unsigned long long>(unsigned long long x) { return __builtin_ffsll(x); }

template <typename T> int count_leading_zeros(T x);

template<> inline int count_leading_zeros<unsigned>(unsigned x)                     { return __builtin_clz(x);   }
template<> inline int count_leading_zeros<unsigned long>(unsigned long x)           { return __builtin_clzl(x);  }
template<> inline int count_leading_zeros<unsigned long long>(unsigned long long x) { return __builtin_clzll(x); }
template<> inline int count_leading_zeros<unsigned char>(unsigned char x)           { return __builtin_clz(x) - (sizeof(unsigned) - sizeof(unsigned char)) * CHAR_BIT; }
template<> inline int count_leading_zeros<unsigned short>(unsigned short x)         { return __builtin_clz(x) - (sizeof(unsigned) - sizeof(unsigned short)) * CHAR_BIT; }

template <typename T> int count_trailing_zeros(T x);

template<> inline int count_trailing_zeros<unsigned>(unsigned x)                     { return __builtin_ctz(x);   }
template<> inline int count_trailing_zeros<unsigned long>(unsigned long x)           { return __builtin_ctzl(x);  }
template<> inline int count_trailing_zeros<unsigned long long>(unsigned long long x) { return __builtin_ctzll(x); }

template <typename T> int parity(T x);

template<> inline int parity<unsigned>(unsigned x)                     { return __builtin_parity(x);   }
template<> inline int parity<unsigned long>(unsigned long x)           { return __builtin_parityl(x);  }
template<> inline int parity<unsigned long long>(unsigned long long x) { return __builtin_parityll(x); }

template <typename T> T power(T x, int p);

template<> inline float       power<float>(float x, int p)             { return __builtin_powif(x, p);   }
template<> inline double      power<double>(double x, int p)           { return __builtin_powi(x, p);  }
template<> inline long double power<long double>(long double x, int p) { return __builtin_powil(x, p); }

// Note: the byte_swap primitives assume 8-bit chars
template <typename T> T byte_swap(T x);

template<> inline uint8_t  byte_swap<uint8_t >(uint8_t x)  { return x;                      }
template<> inline uint16_t byte_swap<uint16_t>(uint16_t x) { return __builtin_bswap16(x);   }
template<> inline uint32_t byte_swap<uint32_t>(uint32_t x) { return __builtin_bswap32(x);   }
template<> inline uint64_t byte_swap<uint64_t>(uint64_t x) { return __builtin_bswap64(x);   }

template <typename N>
struct number_and_overflow_indication {
	N value;
	bool overflow;
};

template <typename LHS, typename RHS = LHS, typename Result = LHS>
number_and_overflow_indication<Result> add_with_overflow_indication(LHS x, RHS y)
{
	Result result;
	bool operation_overflowed = __builtin_add_overflow(x,y, &result);
	return { result, operation_overflowed };
}

template <typename LHS, typename RHS = LHS, typename Result = LHS>
number_and_overflow_indication<Result> subtract_with_overflow_indication(LHS x, RHS y)
{
	Result result;
	bool operation_overflowed = __builtin_sub_overflow(x,y, &result);
	return { result, operation_overflowed };
}

template <typename LHS, typename RHS = LHS, typename Result = LHS>
number_and_overflow_indication<Result> multiply_with_overflow_indication(LHS x, RHS y)
{
	Result result;
	bool operation_overflowed = __builtin_mul_overflow(x,y, &result);
	return { result, operation_overflowed };
}

template <typename LHS, typename RHS = LHS>
bool addition_will_overflow      (LHS x, RHS y) { return __builtin_add_overflow_p(x,y, decltype(x + y)(0)); }
template <typename LHS, typename RHS = LHS>
bool substraction_will_overflow  (LHS x, RHS y) { return __builtin_sub_overflow_p(x,y, decltype(x - y)(0)); }
template <typename LHS, typename RHS = LHS>
bool multiplication_will_overflow(LHS x, RHS y) { return __builtin_mul_overflow_p(x,y, decltype(x * y)(0)); }


// Missing: floating-point-related primities such as huge_val, nan, etc. - and a few others

} // namespace builtins
} // namespace util

#endif /* SRC_UTIL_BUILTINS_HPP_ */
