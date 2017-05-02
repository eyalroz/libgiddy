
#include "resolve_launch_configuration.h"

#include "kernel_wrappers/common.h"

#include "cuda/api/constants.h"
#include "cuda/api/kernel_launch.cuh"
#include "cuda/linear_grid.h"
#include "cuda/optional_and_any.hpp"
#include "util/math.hpp" // for lcm

#ifdef DEBUG
#include "cuda/printing.h"
#include <ostream>
#include <string>
#include <boost/optional/optional_io.hpp> // do we _really_ need this?
#endif

namespace cuda {

#ifdef DEBUG
std::ostream& operator<<(std::ostream& os, launch_configuration_limits_t limits)
{
	if (limits.none()) { return os << "(unlimited)"; }
	if (limits.block_size) {
		os << "block size <= " << limits.block_size.value();
	}
	if (limits.grid_size) {
		os << (limits.block_size  ? ", " : "") << "grid size <= " << limits.grid_size.value();
	}
	if (limits.dynamic_shared_memory) {
		os	<< (limits.block_size || limits.grid_size ? ", " : "")
			<< "dynamic shared memory <= " << limits.grid_size.value();
	}
	return os;
}
#endif

namespace kernels {

#ifdef DEBUG
std::ostream& operator<<(std::ostream& os, const launch_config_resolution_params_t& p)
{
	os << "[ (device_properties)\n";
	print_device_function_attributes(os, p.kernel_function_attributes);
	os << "length = " << p.length << "\n";
	os << "grid_construction_resolution = ";
	switch(p.grid_construction_resolution) {
	case launch_config_resolution_params_t::resolution_t::thread: os << "thread"; break;
	case launch_config_resolution_params_t::resolution_t::warp: os << "thread"; break;
	case launch_config_resolution_params_t::resolution_t::block: os << "thread"; break;
	}
	os << "\n";
	os << "serialization_option = ";
	switch(p.serialization_option) {
	case launch_config_resolution_params_t::serialization_option_t::none: os << "none"; break;
	case launch_config_resolution_params_t::serialization_option_t::fixed_factor: os << "fixed_factor"; break;
	case launch_config_resolution_params_t::serialization_option_t::runtime_specified_factor: os << "runtime_specified_factor"; break;
	case launch_config_resolution_params_t::serialization_option_t::auto_maximized: os << "auto_maximized"; break;
	}
	os << "\n";
	os << "default_serialization_factor = " << p.default_serialization_factor << "\n";
	os
		<< "quanta: threads_in_block = " << p.quanta.threads_in_block << ", "
		<< "warps_in_block = " << p.quanta.warps_in_block << ", "
		<< "blocks_in_grid = " << p.quanta.blocks_in_grid << "\n";
	os
		<< "upper_limits: threads_in_block = " << p.upper_limits.threads_in_block << ", "
		<< "warps_in_block = " << p.upper_limits.warps_in_block << "\n";

	os << "must_have_power_of_2_threads_per_block = " <<
		(p.must_have_power_of_2_threads_per_block ? "yes" : "no") << "\n";

	/*
	os
		<< "dynamic_shared_memory_requirement: per_length_unit = "
		<< p.dynamic_shared_memory_requirement.per_length_unit << ", "
		<< "per_block = " << p.dynamic_shared_memory_requirement.per_block << "\n";

	os	<< "block_resolution_constraints = ";
	if (!p.block_resolution_constraints.fixed_threads_per_block and
		!p.block_resolution_constraints.max_threads_per_block) {
		os << "(unconstrained)";
	}
	else {
		if (p.block_resolution_constraints.fixed_threads_per_block) {
			os << "fixed_threads_per_block  = "	<< p.block_resolution_constraints.fixed_threads_per_block
				<< (p.block_resolution_constraints.max_threads_per_block ? ", " : "");
		}
		if (p.block_resolution_constraints.max_threads_per_block ) {
			os << "max_threads_per_block = " << p.block_resolution_constraints.max_threads_per_block;
		}
	}
*/
	os << "\n";
	os	<< "available_dynamic_shared_memory_per_block = "
		<< p.available_dynamic_shared_memory_per_block << "\n";
	os << "keep_gpu_busy_factor = " << p.keep_gpu_busy_factor  << "]";
	return os;
}
#endif

} // namespace kernels
} // namespace cuda

namespace cuda {
namespace kernels {


using params_t = launch_config_resolution_params_t;
using limits_t = launch_configuration_limits_t;
using resolution_t = params_t::resolution_t;

static grid_block_dimension_t determine_block_length_quantum(const params_t& params)
{
	if (params.grid_construction_resolution == resolution_t::block &&
		params.block_resolution_constraints.fixed_threads_per_block) {
		return params.block_resolution_constraints.fixed_threads_per_block.value();
	}
	size_t common_quantum_for_threads_in_block = params.quanta.threads_in_block;
	if (params.quanta.warps_in_block) {
		common_quantum_for_threads_in_block =
			util::lcm<size_t>(common_quantum_for_threads_in_block,
				params.quanta.warps_in_block.value() * warp_size);
	}

	if (params.grid_construction_resolution == resolution_t::warp) {
		common_quantum_for_threads_in_block = util::lcm<size_t>(
			common_quantum_for_threads_in_block, warp_size);
	}

	return common_quantum_for_threads_in_block;
}


grid_block_dimension_t determine_max_threads_per_block(
	const params_t&  params,
	const limits_t&  limits,
	size_t           effective_length)
{
	constexpr const size_t size_t_maximum_value = std::numeric_limits<size_t>::max();

	struct {
		size_t by_device;
		size_t by_num_length_units;
		size_t by_params_thread_limit;
		size_t by_params_warp_limit   { size_t_maximum_value };
		size_t by_one_time_limits     { size_t_maximum_value };
		size_t by_dynamic_shared_mem  { size_t_maximum_value };
		size_t by_device_function;
	} threads_per_block_constraints;

	threads_per_block_constraints.by_device =
		params.device_properties.maxThreadsDim[0]; // ... and not params.device_properties.maxThreadsPerBlock

	grid_block_dimension_t max_threads_for_single_length_unit;
	switch(params.grid_construction_resolution) {
	case resolution_t:: thread:
		max_threads_for_single_length_unit = 1; break;
	case resolution_t:: warp:
		max_threads_for_single_length_unit = warp_size; break;
	case resolution_t:: block:
		max_threads_for_single_length_unit =
			params.block_resolution_constraints
			      .max_threads_per_block.value_or(params.device_properties.maxThreadsDim[0]);
		break;
	default: // can't get here
		throw std::invalid_argument("Unsupported grid construction resolution");

	}
	threads_per_block_constraints.by_num_length_units =
		util::round_up(determine_block_length_quantum(params),
			effective_length * max_threads_for_single_length_unit);
	threads_per_block_constraints.by_one_time_limits =
		limits.block_size.value_or(size_t_maximum_value);

	threads_per_block_constraints.by_params_thread_limit =
		params.upper_limits.threads_in_block.value_or(size_t_maximum_value);

	threads_per_block_constraints.by_params_warp_limit =
		params.upper_limits.warps_in_block ?
			params.upper_limits.warps_in_block.value() * warp_size :
			size_t_maximum_value;

	if (params.use_dynamic_shared_memory()) {
		if (params.dynamic_shared_memory_requirement.per_block &&
			params.available_dynamic_shared_memory_per_block < params.dynamic_shared_memory_requirement.per_block) {
			threads_per_block_constraints.by_dynamic_shared_mem = 0;
		}
		else if (params.dynamic_shared_memory_requirement.per_length_unit.value_or(0) == 0) {
			threads_per_block_constraints.by_dynamic_shared_mem =
				threads_per_block_constraints.by_device; // i.e. no constraint
		} else {
			threads_per_block_constraints.by_dynamic_shared_mem =
				(params.available_dynamic_shared_memory_per_block -
				 params.dynamic_shared_memory_requirement.per_block.value_or(0)) /
				params.dynamic_shared_memory_requirement.per_length_unit.value();
		}
	}

	switch(params.grid_construction_resolution) {
	case resolution_t::thread:
		threads_per_block_constraints.by_device_function =
			params.kernel_function_attributes.maxThreadsPerBlock; break;
	case resolution_t::warp:
		threads_per_block_constraints.by_device_function =
			util::round_down_to_power_of_2(
				params.kernel_function_attributes.maxThreadsPerBlock,
				warp_size);
		break;
	default:
		threads_per_block_constraints.by_device_function =
			threads_per_block_constraints.by_device;
	}

	size_t max_num_threads = threads_per_block_constraints.by_device;
	max_num_threads = std::min(max_num_threads, threads_per_block_constraints.by_dynamic_shared_mem);
	max_num_threads = std::min(max_num_threads, threads_per_block_constraints.by_device_function);
	max_num_threads = std::min(max_num_threads, threads_per_block_constraints.by_num_length_units);
	max_num_threads = std::min(max_num_threads, threads_per_block_constraints.by_one_time_limits);
	max_num_threads = std::min(max_num_threads, threads_per_block_constraints.by_params_thread_limit);
	max_num_threads = std::min(max_num_threads, threads_per_block_constraints.by_params_warp_limit);
	if (params.must_have_power_of_2_threads_per_block) {
		max_num_threads = util::round_down_to_power_of_2<size_t>(max_num_threads);
	}
	return max_num_threads;
}

static grid_block_dimension_t resolve_block_length(
	const params_t&         params,
	const limits_t&         limits,
	size_t                  effective_length,
	grid_block_dimension_t  max_threads_per_block)
{
	grid_block_dimension_t threads_per_full_block;

	if (params.grid_construction_resolution == resolution_t::block &&
		params.block_resolution_constraints.fixed_threads_per_block) {
		if (params.block_resolution_constraints.fixed_threads_per_block.value() > max_threads_per_block) {
			throw std::invalid_argument("Kernel requires "
				+ std::to_string(params.block_resolution_constraints.fixed_threads_per_block.value())
				+ " threads per block, while  the different constraints on the number of threads per "
				"block only allow " + std::to_string(max_threads_per_block) + ".");
		}
		threads_per_full_block = params.block_resolution_constraints.fixed_threads_per_block.value();
	}
	else {
		size_t common_quantum_for_threads_in_block = determine_block_length_quantum(params);

		threads_per_full_block =
			util::round_down(max_threads_per_block, common_quantum_for_threads_in_block);

		if (threads_per_full_block == 0) {
			throw std::invalid_argument(
				"The combined constraints on the number of threads per block do "
				"not allow any threads in a block");
		}
	}
	return threads_per_full_block;
}

static size_t resolve_num_blocks_to_keep_gpu_busy(
	const params_t&         params,
	grid_block_dimension_t  num_threads_per_block)
{
	// TODO: This is really not the right rule-of-thumb. This should take
	// into account the ratios between memory access latencies
	// and computation between memory accesses, over the execution of the kernel.
	// Now, that's hard to calculate, so we'll just let let the kernel set an
	// arbitrary factor. Of course, that's not such a great idea either, since
	// that factor will depend at the very least on the microarchitecture.
	//
	// Anyway, do not be surprised to keep_gpu_busy_factor be upwards of 50 and
	// maybe even 100
	//
	// TODO 2: Consider just making this a virtual method

	auto warps_per_block = util::div_rounding_up(num_threads_per_block, warp_size);

	return
		((size_t) params.device_properties.multiProcessorCount *
		params.device_properties.compute_capability().max_warp_schedulings_per_processor_cycle() *
		params.keep_gpu_busy_factor /
		warps_per_block);
}

static size_t resolve_num_blocks(
	const params_t&             params,
	size_t                      effective_length,
	grid_block_dimension_t      num_threads_per_block,
	optional<grid_dimension_t>  max_blocks_in_grid)
{
	size_t num_units_covered_by_a_single_full_block;
	switch(params.grid_construction_resolution) {
	case resolution_t::thread:
		num_units_covered_by_a_single_full_block = num_threads_per_block; break;
	case resolution_t::warp:
		num_units_covered_by_a_single_full_block = num_threads_per_block / warp_size; break;
	case resolution_t::block:
		num_units_covered_by_a_single_full_block = 1; break;
	default: // can't get here
		throw std::invalid_argument("Unsupported grid construction resolution");
	}
	auto num_blocks_covering_all_length_units =
		util::div_rounding_up(effective_length, num_units_covered_by_a_single_full_block);
	size_t limit =
		max_blocks_in_grid ? max_blocks_in_grid.value() : num_blocks_covering_all_length_units;
	if (params.serialization_option == params_t::serialization_option_t::auto_maximized) {
		// In this case, "num_blocks_covering_all_length_units" is the value you get
		// with no serialization at all. We would like to serialize as much as is possible
		// to keep the GPU busy, i.e. have less blocks; but if there's not enough input
		// we might not be able to do better than no-serialization.k

		return std::min(std::min(
			num_blocks_covering_all_length_units,
			resolve_num_blocks_to_keep_gpu_busy(params, num_threads_per_block)),
			limit);
	}
	else {
		if (num_blocks_covering_all_length_units > limit) {
			if (params.serialization_option != params_t::serialization_option_t::auto_maximized &&
				params.serialization_option != params_t::serialization_option_t::keep_gpu_busy) {
				throw std::invalid_argument(
						"Required grid length exceeds specified limit: "
						+ std::to_string(limit) + " > "
						+ std::to_string(num_blocks_covering_all_length_units));
			}
			return limit;
		}
		return num_blocks_covering_all_length_units;
	}
}

static shared_memory_size_t resolve_dynamic_shared_memory_size(
	const params_t&         params,
	grid_block_dimension_t  num_threads_per_block)
{
	// This is an easy one, since we've already ensured we're satisfying any constraints
	// on shared memory
	return
		params.dynamic_shared_memory_requirement.per_block.value_or(0) +
		num_threads_per_block * params.dynamic_shared_memory_requirement.per_length_unit.value_or(0);
}

static size_t resolve_effective_length(
	const params_t&                   params,
	optional<serialization_factor_t>  serialization_factor = nullopt)
{
	if (params.serialization_option == params_t::serialization_option_t::auto_maximized) {
		return params.length;
	}
	serialization_factor_t effective_serialization_factor;
	switch(params.serialization_option) {
	case params_t::serialization_option_t::none:
		effective_serialization_factor = serialization_factor.value_or(1);
		if (effective_serialization_factor == 1) { return params.length; }
		throw std::logic_error("kernel does not allow for serialization");
	case params_t::serialization_option_t::fixed_factor:
		effective_serialization_factor = serialization_factor.value_or(params.default_serialization_factor);
		if (effective_serialization_factor != params.default_serialization_factor) {
			throw std::logic_error(
				"kernel cannot admit serialization at a factor other than the fixed default");
		}; break;
	case params_t::serialization_option_t::runtime_specified_factor:
		effective_serialization_factor = serialization_factor.value_or(params.default_serialization_factor);
		break;
	case params_t::serialization_option_t::auto_maximized:
		throw std::logic_error("kernel auto-serializes, making it "
			"meaningless to set apply a serialization factor");
	default: // can't get here
		throw std::invalid_argument("Unsupported serialization option");
	}
	// note: using the safer rather than the faster but overflowing variant
	return (params.length > effective_serialization_factor) ?
		1 + (params.length - 1) / effective_serialization_factor :
		(params.length + effective_serialization_factor - 1) / effective_serialization_factor;
}

// TODO: Reduce serialization factor if we end up with fewer blocks than there
// are processors on the GPU
launch_configuration_t resolve_launch_configuration(
	const params_t&                   params,
	const limits_t&                   limits,
	optional<serialization_factor_t>  serialization_factor)
{
	// TODO: Perhaps push this logic down into apply_serialization and call
	// it "apply_preserialization" or some such name?
	auto effective_length = resolve_effective_length(params, serialization_factor);

	// Resolving the launch configuration for our strategy consists of:
	// 1. Maximizing the number of threads per block - by starting with the
	//    hardware maximum and applying constraints
	// 2. Figuring out how many blocks that means we'll have (and deciding what
	//    we do with slack; but currently we only support one option for that)
	//
	// While theoretically we could say we "optimize" the number of threads
	// per block, practically and for now that always means maximization, at
	// least under

	auto max_threads_per_block = determine_max_threads_per_block(params, limits, effective_length);

	cuda::linear_grid::launch_configuration_t launch_config;

	launch_config.block_length = resolve_block_length(params, limits, effective_length, max_threads_per_block);
	launch_config.grid_length = resolve_num_blocks(params, effective_length, launch_config.block_length, limits.grid_size);
	launch_config.dynamic_shared_memory_size = resolve_dynamic_shared_memory_size(params, launch_config.block_length);

	return launch_config;
}

} // namespace kernels
} // namespace cuda
