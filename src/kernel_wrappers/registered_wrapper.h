
#ifndef SRC_KERNEL_WRAPPERS_REGISTERED_WRAPPER_CUH_
#define SRC_KERNEL_WRAPPERS_REGISTERED_WRAPPER_CUH_

#include "cuda/kernel_wrapper.cuh"
#include "util/FactoryProducible.h"
#include "util/type_name.hpp"

#include <string>
#include <iostream>


namespace cuda {

namespace registered {

class kernel_t :
	public cuda::kernel_t,
	protected util::mixins::FactoryProducible<std::string, kernel_t>
{
protected:
	using factory_key_type = std::string;
	using parent = cuda::kernel_t;
	using factory_producible_mixin_type = util::mixins::FactoryProducible<factory_key_type, parent>;

public:
	static void listSubclasses(std::ostream& os, bool be_verbose, const std::string& separator = "\n");

};

namespace detail {
/**
 * An inlinable constexpr version the strlen() library function (from <cstring>)
 *
 * @note I don't like having this here... but I don't want to depend on the
 * entire string utility header
 * @param str A nul-terminated string (i.e. ends with '\0' = 0)
 * @return the number of characters starting with the one at address @p str
 * which have a non-0 (i.e. non-NUL, non-'\0') value, until the first 0.
 */
size_t constexpr cstring_length(const char* s)
{
    return *s ? 1 + cstring_length(s + 1) : 0;
}

} // namespace detail

} // namespace registered
} // namespace cuda

// All subclasses have to define all of these, unfortunately; And while theoretically
// they could, say, not make the override's final, in practice we don't have that
#define REGISTERED_KERNEL_WRAPPER_BOILERPLATE_DEFINITIONS(subclass_name) \
	using parent          = cuda::registered::kernel_t; \
	using arguments_type  = parent::arguments_type; \
	using parent::resolve_launch_configuration; \
	static void registerInSubclassFactory() { \
		auto subclass_name_str = util::type_name<subclass_name>(); \
		static constexpr const char* prefix = "cuda::kernels::"; \
		auto pos = subclass_name_str.find(prefix); \
		auto key = (pos == std::string::npos) ? subclass_name_str : \
			subclass_name_str.substr(pos + cuda::registered::detail::cstring_length(prefix)); \
		FactoryProducible::registerInSubclassFactory<subclass_name>(key) ; \
	} \
public: \
	launch_configuration_t resolve_launch_configuration( \
		device::properties_t           device_properties, \
		device_function::attributes_t  kerenl_function_attributes, \
		arguments_type                 extra_arguments, \
		launch_configuration_limits_t  limits = { nullopt, nullopt, nullopt }) const override final; \
	\
	/* Note: Stream is assumed to be on the current device */ \
	void enqueue_launch( \
		stream::id_t                   stream, \
		const launch_configuration_t&  launch_config, \
		arguments_type                 arguments) const override final; \
	\
	const device_function_t get_device_function() const override final \


#endif /* SRC_KERNEL_WRAPPERS_REGISTERED_WRAPPER_CUH_ */
