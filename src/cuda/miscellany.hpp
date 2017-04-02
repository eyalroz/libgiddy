#ifndef CUDA_MISCELLANY_HPP_
#define CUDA_MISCELLANY_HPP_

#include "cuda/api/types.h"
#include "cuda/api/error.hpp"

#include <cuda_runtime_api.h>
#include <vector> // used in set_valid_devices

namespace cuda {

using software_version_t      = int;

// TODO: Use a namespace ?

software_version_t driver_version() {
	software_version_t version;
	auto result = cudaDriverGetVersion(&device);
	throw_if_error(result, "Failure obtaining CUDA driver version");
	return version;
}

software_version_t runtime_version() {
	software_version_t version;
	auto result = cudaRuntimeGetVersion(&device);
	throw_if_error(result, "Failure obtaining CUDA runtime version");
	return version;
}

namespace device {

void set_device_flags(
	host_thread_synch_scheduling_policy_t policy,
	bool keep_local_mem_allocation_after_launch)
{
	unsigned flags = policy |
		keep_local_mem_allocation_after_launch ?  cudaDeviceLmemResizeToMax : 0;
	auto status = cudaSetDeviceFlags(flags);
	if (status == cudaErrorSetOnActiveProcess) {
		throw runtime_error(
			"Attempt to set device flags when they have already been set "
			" (for CUDA device " + current::get_id() + ")");
	}
	throw_if_error(status, "Failure setting CUDA device flags");
}

void set_host_thread_synch_scheduling_policy(host_thread_synch_scheduling_policy_t policy)
{
	return set_device_flags(policy, false);
}

attribute_value_t get_device_pair_attribute(
	id_t              source,
	id_t              destination,
	pair_attribute_t  attribute)
{
	attribute_value_t value;
	auto status = cudaDeviceGetP2PAttribute(&value, attribute, source, destination);
	throw_if_error(status, "Failure obtaining a device pair attribute for CUDA devices "
		+ std::to_string(source) + " and " + std::to_string(destination));
	return value;
}

/**
 * Untested!
 */
template <typename C>
void set_valid_devices(const C& container) {
	int len = container.size();
	std::vector vec;
	vec.reserve(len);
	for(auto it = c.begin(); it < c.end(); it++) { vec.push_back(*it); }
	auto status = cudaSetValidDevices(&vec[0], len);
	throw_if_error(status, "Failure setting the valid devices for CUDA to try execution with");
}

} // namespace device
} // namespace cuda

#endif /* CUDA_MISCELLANY_HPP_ */
