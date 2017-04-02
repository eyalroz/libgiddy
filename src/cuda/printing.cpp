
#include "cuda/api/types.h"
#include "cuda/api/device.hpp"
#include "cuda/api/device_count.hpp"

#include "cuda/printing.h"

#include "util/miscellany.hpp"
#include "util/string.hpp"

#include <string>
#include <sstream>
#include <iomanip>
#include <utility>
#include <climits> // for CHAR_BIT

using std::setw;
using std::left;
using std::setprecision;
using std::setw;
using std::setprecision;


namespace cuda {

// TODO: Convert to a streaming operator
void list_device_properties(cudaDeviceProp properties_, std::ostream& os)
{
	util::ios_flags_saver flag_saver(os);
	cuda::device::properties_t properties(properties_);
	auto memory_bandwidth = properties.memoryClockRate * (properties.memoryBusWidth / CHAR_BIT);
	os << left;
	os << "Name:                            " << properties.name << '\n'
	   << "Microarchitecture generation:    " << properties.architecture_name() << '\n'
	   << "PCI Location:                    " << "Bus " << properties.pciBusID << ", device " << properties.pciDeviceID << '\n'
	   << "CUDA Compute Capability:         " << properties.major << '.' << properties.minor << '\n'
	   << "Global memory bandwidth:         " << setw(8) << std::fixed << setprecision(0) << (double) memory_bandwidth / 1024 << " MiB/sec " " = " << setw(6) << std::fixed << setprecision(1) << (double) memory_bandwidth / (1024*1024) << " GiB/sec\n"
	   << "Total global memory:             " << left << setw(17) << properties.totalGlobalMem    << " = " << setw(6) << properties.totalGlobalMem / (1024*1024) << " MiB\n"
	   << "Total constant memory:           " << left << setw(17) << properties.totalConstMem     << " = " << setw(6) << properties.totalConstMem / 1024 << " KiB\n"
	   << "Effective Shared memory / block: " << left << setw(17) << properties.compute_capability().max_shared_memory_per_block() << " = " << setw(6) << properties.sharedMemPerBlock / 1024 << " KiB\n"
	   << "Total registers per block:       " << properties.regsPerBlock << '\n'
	   << "Warp size:                       " << properties.warpSize << '\n'
	   << "Memory error correction (ECC):   " << (properties.ECCEnabled ? "Enabled" : "Disabled") << '\n'
	   << "Maximum memory pitch:            " << left << setw(17) << properties.memPitch          << " = " << setw(6) << properties.memPitch / (1024*1024) << " MiB\n"
	   << "Maximum threads per block:       " << properties.maxThreadsPerBlock << '\n'
	   << "Maximum block dimensions:        " << properties.maxThreadsDim[0] << " x "
		                                      << properties.maxThreadsDim[1] << " x "
		                                      << properties.maxThreadsDim[2] << '\n'

	   << "Maximum grid dimensions:         " << properties.maxGridSize[0] << " x "
	                                          << properties.maxGridSize[1] << " x "
	                                          << properties.maxGridSize[2] << '\n'
	   << "Max. regs/thread in full block:  " << properties.regsPerBlock / properties.maxThreadsPerBlock << '\n'
	   << "Multiprocessor clock rate:       " << setw(10) << std::fixed << properties.clockRate << " Hz     = " << setw(6) << std::fixed << setprecision(1) << (double) properties.clockRate / (1024*1024) << " GiHz\n"
	   << "Concurrent copy & execution:     " << (properties.deviceOverlap ? "Supported" : "Unsupported") << '\n'
	   << "Number of multiprocessors:       " << properties.multiProcessorCount << '\n'
	   << "Kernel execution timeout:        " << (properties.kernelExecTimeoutEnabled ? "Enabled" : "Disabled") << '\n'
	   << '\n';
	return;
}


void list_device_properties(device::id_t  device, std::ostream& os)
{
	// Get device properties
	os	<< "\n"
	    << "Properties of CUDA Device " << device << '\n'
	    << "-----------------------------\n";
	cudaDeviceProp device_properties;
	cudaGetDeviceProperties(&device_properties, device);
	list_device_properties(device_properties, os);
}

void list_all_devices(std::ostream& os, unsigned indent)
{
	auto num_devices = cuda::device::count();
	os << left;

	if (num_devices == 0) {
		os << "No CUDA devices found on this system.\n";
		return;
	}

	if (indent > 0) { os << std::setw(indent) << ' '; }
	os	<< "CUDA devices\n"
		<< "-------------\n";

	// Device handles are actually just integers, let's iterate over them

	for (cuda::device::id_t  device_index = 0; device_index < num_devices; ++device_index) {
		cuda::device::properties_t properties = cuda::device::get(device_index).properties();
		os << left;
		if (indent > 0) { os << std::setw(indent) << ' '; }
		os	<< left << std::setw(std::max(2,util::log10_constexpr(num_devices-1))) << device_index
			<< ": (bus " << properties.pciBusID << ", dev " << properties.pciDeviceID << ")  "
			<< properties.name << '\n';
	}
}

std::ostream& print_dimensions(std::ostream& os, dimensions_t dims) {
	// A very rough upper bound. Plus, actually, for 2+ -dimensional grids,,
	// five digits of width are enough, since the total grid size cannot exceed
	// 2^{31} apparently
	if (dims.empty()) {
		os  << "(empty)";
	}
	else {
		os  << setw(sizeof(dims.x) * CHAR_BIT / 3.33 + 1) << left;
		switch(dims.dimensionality()) {
		case 3: os  << dims.x << " x " << dims.y << " x " << dims.z << " (3-dimensional)"; break;
		case 2: os  << dims.x << " x " << dims.y << " (2-dimensional)"; break;
		case 1: os  << dims.x << " (1-dimensional)"; break;
		case 0: os  << 1 << " (0-dimensional)";
		}
	}
	return os;
}

static std::string format_dimensions(dimensions_t dims) {
	std::stringstream ss;
	print_dimensions(ss, dims);
	return ss.str();
}

void print_kernel_launch_configuration(
	std::ostream&           os,
	const std::string&      kernel_name,
	launch_configuration_t  config)
{
	os 	<< "Kernel name                            " << kernel_name                                << '\n';
	print_kernel_launch_configuration(os, config);
}

void print_kernel_launch_configuration(
	std::ostream&           os,
	launch_configuration_t  config)
{
	os	<< "Grid dimensions                        " << format_dimensions(config.grid_dimensions)  << '\n'
		<< "Block dimensions                       " << format_dimensions(config.block_dimensions) << '\n'
		<< "Total number of threads                " << config.grid_dimensions.volume() *
		    config.block_dimensions.volume()                                                       << '\n'
		<< "Dynamic shared memory size             " << config.dynamic_shared_memory_size          << '\n';
}

void print_device_function_attributes(
	std::ostream&                  os,
	const std::string&             function_name,
	device_function::attributes_t  attributes)
{
	os 	<< "Device function name                   " << function_name << '\n';
	print_device_function_attributes(os, attributes);
}

void print_device_function_attributes(
	std::ostream&                  os,
	device_function::attributes_t  attributes)
{
	os 	<< "Static shared memory size (per block)  " << attributes.sharedSizeBytes            << '\n'
		<< "Constant memory size                   " << attributes.constSizeBytes             << '\n'
		<< "Local memory size (per thread)         " << attributes.localSizeBytes             << '\n'
		<< "Maximum launchable threads per block   " << attributes.maxThreadsPerBlock         << '\n'
		<< "Number of registers used (per thread)  " << attributes.numRegs                    << '\n'
		<< "PTX version used in compilation        "
		<< cuda::device::compute_capability_t::from_combined_number(attributes.ptxVersion)    << '\n'
		<< "Binary compilation target architecture "
		<< cuda::device::compute_capability_t::from_combined_number(attributes.binaryVersion) << '\n'
		<< "Cache mode set to 'ca'?                "
		<< (attributes.cacheModeCA ? "Yes" : "No")                                            << '\n'
	;
}

} // namespace cuda
