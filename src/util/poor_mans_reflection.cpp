#include "util/poor_mans_reflection.h"
#include "util/exception.h"

#include <functional>

namespace util {

size_t size_of(const std::string& data_type_name)
{
#define SIZEOF_RESOLVER_MAP_ENTRY(_type) { #_type, sizeof(_type) },

	static const std::unordered_map<std::string, size_t> size_by_type = {
		MAP(SIZEOF_RESOLVER_MAP_ENTRY, ALL_FUNDAMENTAL_NONVOID_TYPES)
	};

	auto result = size_by_type.find(data_type_name);
	if (result == size_by_type.end()) {
		throw util::runtime_error("No size registered for type " + std::string(data_type_name));
	}
	return result->second;
}

} // namespace util

