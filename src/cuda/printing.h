#pragma once
#ifndef CUDA_PRINTING_H_
#define CUDA_PRINTING_H_

#include "cuda/api/kernel_launch.cuh"
#include "cuda/api/types.h"
#include "cuda_runtime.h"
#include "cuda/api/device_function.hpp"

#include <iostream>


namespace cuda {

void list_device_properties(device::id_t  device, std::ostream& os);
void list_all_devices(std::ostream& os, unsigned indent = 0);


void print_kernel_launch_configuration(
	std::ostream&                  os,
	launch_configuration_t         config);

void print_kernel_launch_configuration(
	std::ostream&                  os,
	const std::string&             kernel_name,
	launch_configuration_t         config);

void print_device_function_attributes(
	std::ostream&                  os,
	device_function::attributes_t  attributes);
void print_device_function_attributes(
	std::ostream&                  os,
	const std::string&             function_name,
	device_function::attributes_t  attributes);


/**
 * The CUDA device properties have a field with the supposed maximum amount of shared
 * memory per block you can use, but - that field is a dirty lie, is what it is!
 * On Kepler, you can only get 48 KiB shared memory and the rest is always reserved
 * for L1...
 * so, if you call this function, you'll get the proper effective maximum figure.
 */

} // namespace cuda

inline std::ostream& operator<<(std::ostream& os, cuda::device::compute_capability_t cc)
{
	return os << cc.major << '.' << cc.minor;
}

inline std::ostream& operator<<(std::ostream &out, cuda::dimensions_t dims)
{
	out << '(' << dims.x << ", " << dims.y << ", " << dims.z << ")";
	return out;
}



#endif /* CUDA_PRINTING_H_ */
