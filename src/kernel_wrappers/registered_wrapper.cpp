#include "registered_wrapper.h"
#include "util/maps.hpp"
#include "util/miscellany.hpp"

#include <map>
#include <unordered_map>

template<class K, class V>
static void print_sorted_keys(std::unordered_map<K,V> const& map, std::ostream& os, const std::string& separator){
	auto keys = util::get_keys(map);
	std::sort(keys.begin(), keys.end());
	util::print_separated(os, separator, keys.cbegin(), keys.cend());
}

namespace cuda {
namespace registered {

void kernel_t::listSubclasses(std::ostream& os, bool be_verbose, const std::string& separator)
{
	if (be_verbose) {
		if (getSubclassFactory().getInsantiators().empty()) {
			os << "There are no registered kernel test adapters!";
		}
		else {
			auto num_insantiators = getSubclassFactory().getInsantiators().size();
			os << "There " << (num_insantiators > 1 ? "is" : "are") << ' '
			   << num_insantiators << " regsitered "
			   << (num_insantiators > 1 ? "kernels" : "kernel") << ": \n\n";
		}
	}
	print_sorted_keys(getSubclassFactory().getInsantiators(), os, "\n");
	os << '\n';
}

} // namespace registered
} // namespace cuda
