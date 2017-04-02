#pragma once
#ifndef RESOLVE_LAUNCH_CONFIGURATION_H
#define RESOLVE_LAUNCH_CONFIGURATION_H

#include "launch_config_resolution_params.h"
#include "cuda/kernel_wrapper.cuh" // for launch_configuration_limits_t

namespace cuda {

/**
 * Resolves a launch configuration based on the configuration parameters passed to it.
 *
 * TODO: Making the limits optionall as well
 *
 * @param params (mostly) declarative configuration parameters, which are sufficient
 * for determining the exact launch configuration for all DBMS-related kernels
 * in this repository
 * @param limits one-time imposed limits on the grid (used for testing essentially)
 * @param serialization_factor see @ref launch_config_resolution_params_t::serialization_option_t
 * @return the (hopefully) best possible launch configuration for the kernel and device
 * combination described by @p params
 */
launch_configuration_t resolve_launch_configuration(
	const launch_config_resolution_params_t&  params,
	const launch_configuration_limits_t&      limits,
	optional<serialization_factor_t>          serialization_factor = nullopt);

} // namespace cuda

#endif /* RESOLVE_LAUNCH_CONFIGURATION_H */
