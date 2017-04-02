/**
 * A C++14 implementation of std::index_sequence for use with
 * C++11 code; does not implement the more general
 * std::integer_sequence.
 *
 * From the StackOverflow community wiki:
 * http://stackoverflow.com/a/32223343/1593077
 */
#pragma once
#ifndef UTIL_INDEX_SEQUENCE_HPP_
#define UTIL_INDEX_SEQUENCE_HPP_

#include <cstddef> // for size_t

#if __cplusplus == 201103L

namespace std  // WARNING: use at own risk, otherwise use own namespace
{
    template <size_t... Ints>
    struct index_sequence
    {
        using type = index_sequence;
        using value_type = size_t;
        static constexpr std::size_t size() noexcept { return sizeof...(Ints); }
    };

    // --------------------------------------------------------------

    template <class Sequence1, class Sequence2>
    struct _merge_and_renumber;

    template <size_t... I1, size_t... I2>
    struct _merge_and_renumber<index_sequence<I1...>, index_sequence<I2...>>
      : index_sequence<I1..., (sizeof...(I1)+I2)...>
    { };

    // --------------------------------------------------------------

    template <size_t N>
    struct make_index_sequence
      : _merge_and_renumber<typename make_index_sequence<N/2>::type,
                            typename make_index_sequence<N - N/2>::type>
    { };

    template<> struct make_index_sequence<0> : index_sequence<> { };
    template<> struct make_index_sequence<1> : index_sequence<0> { };
}

#endif /* __cplusplus == 201103L */

#endif /* UTIL_INDEX_SEQUENCE_HPP_ */
