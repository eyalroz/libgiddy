#pragma once
#ifndef UTIL_MAPS_HPP_
#define UTIL_MAPS_HPP_

namespace util {

template<typename Map, class ... Needles>
inline typename Map::mapped_type get_first_match(const Map& haystack, Needles... needles)
{
    for (auto& needle : {needles...}) {
        auto it = haystack.find(needle);
        if (it != haystack.end()) { return it->second; }
    }
    throw std::invalid_argument("None of the elements were found in the container");
}

// Iterate a map's keys. This is an ugly but dependency-less alternative
// to using the nice Boost transform_iterator...

template<class M>
class const_key_iterator : public M::const_iterator
{
public:
	typedef typename M::const_iterator map_const_iterator;
	typedef typename map_const_iterator::value_type::first_type key_type;

	const_key_iterator(const map_const_iterator& other) : M::const_iterator(other) {} ;

	const key_type& operator *() const
	{
	    return M::const_iterator::operator*().first;
	}
};

template<class M> const_key_iterator<M> inline key_cbegin(M& m)  { return const_key_iterator<M>(m.cbegin()); }
template<class M> const_key_iterator<M> inline key_cend(M& m)    { return const_key_iterator<M>(m.cend());   }

template<typename Map, typename Function>
inline Function for_each_mapped(Map& map, Function f)
{
	return std::for_each(
		std::begin(map), std::end(map),
		[&f](auto& key_value_pair) { f(key_value_pair.second); }
	);
}

template<typename Map, typename Function>
inline Function for_each_key(Map& map, Function f)
{
	return std::for_each(
		std::begin(map), std::end(map),
		[&f](auto& key_value_pair) { f(key_value_pair.first); }
	);
}

// This next tricky bit is due to:
//
//
namespace detail
{
	// preferred overload: if we can push_back a Container::value_type
	template <class Container>
	auto hopefully_back_inserter(Container& c, int) -> decltype(std::back_inserter(c))
	{
	    return std::back_inserter(c);
	}

	// fallback if we can't
	template <class Container>
	auto hopefully_back_inserter(Container& c, ...) -> decltype(std::inserter(c))
	{
	    return std::inserter(c, c.end());
	}
} // namespace detail

template <class Container>
inline auto hopefully_back_inserter(Container& c)
-> decltype(detail::hopefully_back_inserter(c, 0))
{
	return detail::hopefully_back_inserter(c, 0);
}

// TODO: Try to use hopefully_back_inserter
template <typename Map, typename Container = std::vector<typename Map::key_type>>
inline Container get_keys(const Map& m) {
	Container c;
	std::copy(key_cbegin(m), key_cend(m), std::inserter(c, c.end()));
	return c;
}

template <typename M1, typename M2>
inline bool equal_keys(const M1& m1, const M2& m2)
{
	if (m1.size() != m2.size()) { return false;	}
	if (m1.empty()) { return true; }
	return std::equal(key_cbegin(m1), key_cend(m1), key_cbegin(m2));
}

template<typename Map, typename Predicate>
inline void map_erase_if(Map& m, Predicate const& p)
{
	for(auto it = m.begin(); it != m.end();)
	{
	  if(p(*it)) it = m.erase(it);
	  else ++it;
	}
}

// ----------------------------------------------------------
// Map-related adapters
// ----------------------------------------------------------

// TODO: Use the key iterators, or for_each_key
template<template<typename, typename> class M, typename K, typename V>
inline size_t count_keys(const M<K, V>& map, const K& key)
{
	return std::count(map.cbegin(), map.cend(),
		[&key](const typename M<K,V>::value_type& e) {
			return e.first == key;
		});
}

// TODO: Have value iterators and use them
template<template<typename, typename> class M, typename K, typename V>
inline size_t count_values(const M<K, V>& map, const V& value)
{
	return std::count(map.cbegin(), map.cend(),
		[&value](const typename M<K,V>::value_type& e) {
			return e.second == value;
		});
}

// TODO: Use the key iterators, or for_each_key
template<template<typename, typename> class M, typename K, typename V, typename P>
inline size_t count_if_key(const M<K, V>& map, P pred)
{
	return std::count_if(map.cbegin(), map.cend(),
		[&pred](const typename M<K,V>::value_type& e) {
			return pred(e.first);
		});
}

// TODO: Have value iterators and use them
template<template<typename, typename> class M, typename K, typename V, typename P>
inline size_t count_if_value(const M<K, V>& map, P pred)
{
	return std::count_if(map.cbegin(), map.cend(),
		[&pred](const typename M<K,V>::value_type& e) {
			return pred(e.second);
		});
}


} // namespace util

#endif /* UTIL_MAPS_HPP_ */
