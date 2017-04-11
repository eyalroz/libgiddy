#pragma once
#ifndef CUDA_KERNEL_WRAPPER_CUH_
#define CUDA_KERNEL_WRAPPER_CUH_

#include "cuda/api/types.h"
#include "cuda/api/device_function.hpp"
#include "cuda/api/device.hpp"
#include "cuda/api/kernel_launch.cuh"
#include "cuda/optional_and_any.hpp"

#include "util/macro.h" // for MAP()

#include <string>
#include <tuple>
#include <unordered_map>
#include <exception>

namespace cuda {

struct launch_configuration_limits_t {
	optional<cuda::grid_block_dimension_t> block_size;
	optional<cuda::grid_dimension_t>       grid_size;
	optional<cuda::shared_memory_size_t>   dynamic_shared_memory;

	static launch_configuration_limits_t limit_block_size(optional<cuda::grid_block_dimension_t> limit) {
		return { limit, nullopt, nullopt };
	}

	bool none() const
	{
		return !block_size and !grid_size and !dynamic_shared_memory;
	}
};

// Note this class has no data members (just a vtable)

/**
 * This is an abstraction of a kernel. Bringing together its actual device function
 * and the additional code it requires for coming up with the launch configuration.
 *
 * @note I would have liked to say it's a very thin low-level abstraction, but it's actually
 * a bit high-level in the sense of using the multi-type boost::any construct, as well as
 * maps. It's still thin though - There's no state, and incurs (relatively) little overhead;
 * also, the benefit of being somewhat higher-level is that there is no need for the code
 * using a kernel_t* to "know" the executing code, compilation-wise - just pass it the arguments
 *
 * @note the API is _very_ sensitive and flakey, since arguments are checked at run time,
 * not compile time; and casts are impossible, i.e. you have to pass in the exact right
 * type. So expect map::at and boost::any_cast to throw
 *
 * @todo expose the required arguments via a set (or a map from string to typeid)
 */
struct kernel_t {
public:
	/**
	 * Type-erased (or type-hidden) arguments for either the launch config
	 * resolution and the actual kernel enqueue.
	 *
	 * @note May be replaced (or augmented) in the future with something
	 * less costly
	 */
	using arguments_type = std::unordered_map<std::string, boost::any>;

	// Only two methods for subclasses to override - one for the device function
	// (the "kernel itself"), one for the launch config resolution. The override
	// launches on the current device.
	//
	virtual void enqueue_launch(
		stream::id_t                    stream_id,
		const launch_configuration_t&   launch_config,
		arguments_type                  args) const = 0;

	/**
	 * Uses meta-data other than the actual (large and on-device) inputs
	 * to resolve the parameters for launching the wrapped kernel. These
	 * can then be used with the @ref launch function, or one can launch
	 * on his/her own using @ref get_device_function.
	 *
	 * @todo: I don't like how we pass the attributes here, but
	 * otherwise we would have to get them with every implementation.
	 *
	 * @param device_properties relevant properties of the CUDA device on
	 * which the kernel is to be launched
	 * @param kernel_function_attributes function-specific attributes
	 * @param extra_arguments the resolver might take custom arguments
	 * which this, the base class, is not aware of - yet the user may be;
	 * they are therefore passed anonymously and with type 'hiding' via a
	 * map of boost::any's.
	 * @param limits arbitrarily-imposed limits on the lauch configuration
	 * @return a valid launch configuration, in the sense that this kernel
	 * wrapper will not balk on getting it as input. Specifically, a
	 * configuration which requires _no_ launch may also be returned if
	 * the launcher supports that.
	 */
	virtual launch_configuration_t resolve_launch_configuration(
		device::properties_t            device_properties,
		device_function::attributes_t   kernel_function_attributes,
		arguments_type                  extra_arguments,
		launch_configuration_limits_t   limits = {} ) const = 0;

	virtual const cuda::device_function_t get_device_function() const = 0;

	virtual ~kernel_t() { }

};

namespace kernel {

// Note that the following overload the enqueue_launch() method that's
// a part of the CUDA API wrappers

template <typename... KernelParameters>
inline void enqueue_launch(
	const kernel_t&                  kernel_wrapper,
	stream::id_t                     stream_id,
	const launch_configuration_t&    launch_config,
	KernelParameters...              parameters)
{
	using kernel_function_type = void (*)(KernelParameters...);
	auto kernel_function = reinterpret_cast<kernel_function_type>(
		kernel_wrapper.get_device_function().ptr());
	cuda::enqueue_launch<kernel_function_type, KernelParameters...>(
		kernel_function, stream_id, launch_config, parameters...);
}

template <typename... KernelParameters>
inline void enqueue_launch(
	const kernel_t&                  kernel_wrapper,
	device::id_t                     device_id,
	stream::id_t                     stream_id,
	const launch_configuration_t&    launch_config,
	KernelParameters...              parameters)
{
	device::current::scoped_override_t<false> using_device(device_id);
	enqueue_launch(kernel_wrapper, stream_id, launch_config, parameters...);
}

inline launch_configuration_t resolve_launch_configuration(
	kernel_t&                       kernel_wrapper,
	device::properties_t            device_properties,
	kernel_t::arguments_type        extra_arguments,
	launch_configuration_limits_t   limits = {})
{
	return kernel_wrapper.resolve_launch_configuration(
		device_properties,
		kernel_wrapper.get_device_function().attributes(),
		extra_arguments,
		limits);
}

inline launch_configuration_t resolve_launch_configuration(
	kernel_t&                       kernel_wrapper,
	device::id_t                    device_id,
	kernel_t::arguments_type        extra_arguments,
	launch_configuration_limits_t   limits = {})
{
	auto device_properties = cuda::device::get(device_id).properties();
	return resolve_launch_configuration(
		kernel_wrapper, device_properties, extra_arguments, limits);
}

} // namespace kernel

#define MAKE_KERNEL_ARGUMENTS_ELEMENT(_kernel_arg_identifier) \
	{ QUOTE(_kernel_arg_identifier), _kernel_arg_identifier },

/**
 * This convenience macro will save you the trouble of writing:
 *
 *   {
 *     { "argument1", argument1 },
 *     { "argument2", argument2 }
 *   }
 *
 * for initializing a kernel_t::arguments_type you're passing to one
 * of the methods.
 *
 * @note --== !!REALLY IMPORTANT!! ==--  since we're using boost::any,
 * the data type has to be _exactly_ the type expected by the
 * kernel wrapper, otherwise you'll get a Boost error. Also, for
 * this macro to work, your variable names must be _exactly_ the same
 * as the key names the wrapper is looking for (so, config.test_length
 * will definitely not do, and you will need to define some temporaries
 * of the right types and names; watch out for 'auto's, they might
 * bite you if you're not careful to coerce their type).
 */
#define MAKE_KERNEL_ARGUMENTS(...) \
{ \
	MAP(MAKE_KERNEL_ARGUMENTS_ELEMENT, __VA_ARGS__) \
}


} // namespace cuda

#endif /* CUDA_KERNEL_WRAPPER_CUH_ */
