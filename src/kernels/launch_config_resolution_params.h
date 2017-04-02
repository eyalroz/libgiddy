#pragma once
#ifndef LAUNCH_CONFIG_RESOLUTION_PARAMS_H_
#define LAUNCH_CONFIG_RESOLUTION_PARAMS_H_

#include "cuda/api/types.h"
#include "cuda/api/device_properties.hpp"
#include "cuda/api/device_function.hpp"
#include "util/exception.h"
#include "cuda/optional_and_any.hpp"

namespace cuda {

class launch_config_resolution_params_t {

public: // types

	// This type signifies what part of a grid handles a single input unit;
	// consequently, launch configuration must respect the atomicity of these
	enum resolution_t {
		thread,                      // the resolution code is (mostly) unrestricted in
		                             // dividing threads into blocks
		warp,                        // every block must be made up of full warps,
		                             // but the resolution code can set that number (mostly)
		                             // arbitrarily
		block                        // the kernel has strict expectations regarding what
		                             // happens inside a block, so the resolution code can't
		                             // do much inside - nor can it change the number of blocks
		                             // to schedule, except by serialization
	};
	// Does each thread/warp/block be made to process more than one length unit, and if so - how?
	enum serialization_option_t {
		none,                        // No, one length unit per min-resolution computational unit
		fixed_factor,                // A serialization factor is hard-coded (though possibly templated)
		                             // into the kernel
		runtime_specified_factor,    // The kernel takes a serialization factor as a run-time parameter
		auto_maximized,              // The kernels tries to 'serialize' as much as possible, i.e.
		                             // each block in the grid continues processing data at grid stride
		                             // until getting to the end of the input.
		keep_gpu_busy = auto_maximized, 
		                             // ... and actually, the maximum effective serialization factor is
		                             // where it can't be increased without making the GPU go idle
		                             // occasionally, which we obviously don't want happening
	};

public: // data members
	device::properties_t              device_properties;
	device_function::attributes_t   kernel_function_attributes;
	size_t                            length { 0 }; // in units of the appropriate resolution
	resolution_t                      grid_construction_resolution { thread };
		// i.e. "each input element is handled by a single thread / a single warp / etc."

	serialization_option_t            serialization_option { serialization_option_t::none };
	serialization_factor_t            default_serialization_factor { 1 };
	struct {
		grid_block_dimension_t                                threads_in_block  { 1 };
		optional<grid_block_dimension_t>   warps_in_block    { nullopt };
		optional<grid_dimension_t>         blocks_in_grid    { nullopt };
	} quanta;
		// e.g. number of threads (in a block) needs to be an integral multiple
		// of quanta.threads_in_block

	struct {
		optional<grid_block_dimension_t>   threads_in_block  { nullopt };
		optional<grid_block_dimension_t>   warps_in_block    { nullopt };
	} upper_limits;

	bool                              must_have_power_of_2_threads_per_block { false };

	struct {
		// If both of these are unset, the kernel doesn't use dynamic shared memory
		// (and in fact that is also the case if they're both zero, but that shouldn't happen)
		optional<shared_memory_size_t> per_length_unit       { nullopt };
		optional<shared_memory_size_t> per_block             { nullopt };
	} dynamic_shared_memory_requirement;

	struct {
		// these should only be set when grid_construction_resolution == resolution_t::block;
		// if we had std::variant's I might have only had them _defined_ in that case
		optional<grid_block_dimension_t>  fixed_threads_per_block { nullopt };
		optional<grid_block_dimension_t>  max_threads_per_block   { nullopt };
	} block_resolution_constraints;

	// The following figure is _after_ allowing for a limiting value - even though
	// the assisted_resolution_kernel_t class passes both the limits and a params class
	// instance to the final resolution logic. This inelegant kludge prevents params
	// from having to include a lambda for later work as this value becomes available
	shared_memory_size_t              available_dynamic_shared_memory_per_block;

	// Since we lack a more refined and informed mechanism for deciding how many
	// blocks/warps/threads are necessary to keep the GPU busy running kernels
	// with long-running threads, we'll allow the kernel-specific launch params
	// to tweak this for their own needs using this factor; of course this will
	// probably be quite inaccurate, since the actual value depends on
	// microarchitecture-specific features. see also the static method
	// resolve_num_block_to_keep_gpu_busy the implementation of
	// cuda:: assisted_resolve_kernel_t .
	unsigned                          keep_gpu_busy_factor { 50 };

public:
	/**
	 * True when the associated kernel requires at least some dynamic shared memory
	 */
	bool use_dynamic_shared_memory() const {
		return dynamic_shared_memory_requirement.per_block ||
			dynamic_shared_memory_requirement.per_length_unit;
	}

protected: // constructors
	launch_config_resolution_params_t(
		device::properties_t device_properites_,
		device_function::attributes_t kernel_function_attributes_,
		optional<shared_memory_size_t> shared_memory_size_limit = nullopt) :
		device_properties(device_properites_),
		kernel_function_attributes(kernel_function_attributes_)
	{
		available_dynamic_shared_memory_per_block =
			device_function::maximum_dynamic_shared_memory_per_block(
				kernel_function_attributes,
				device_properties.compute_capability());
		if (shared_memory_size_limit &&
			shared_memory_size_limit < available_dynamic_shared_memory_per_block) {
			available_dynamic_shared_memory_per_block = shared_memory_size_limit.value();
		}
	}

	launch_config_resolution_params_t(
		device::properties_t device_properites_,
		device_function_t device_function,
		optional<shared_memory_size_t> shared_memory_size_limit = nullopt) :
		launch_config_resolution_params_t(
			device_properites_,
			device_function.attributes(),
			shared_memory_size_limit) { }

};


} // namespace cuda

#endif /* LAUNCH_CONFIG_RESOLUTION_PARAMS_H_ */
