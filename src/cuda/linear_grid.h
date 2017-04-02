/**
 * Launch configurations and other definitions specific to
 * kernels which are linear, i.e. should be scheduled with
 * a uni-dimensional grid
 */
#ifndef CUDA_LINEAR_GRID_H_
#define CUDA_LINEAR_GRID_H_

#include "cuda/api/kernel_launch.cuh"

namespace cuda {
namespace linear_grid {

enum : grid_dimension_t	{ single_block = 1 };
enum : grid_block_dimension_t { single_thread_per_block = 1 };

class launch_configuration_t {
public:
	grid_dimension_t	    grid_length; // in threads
	grid_block_dimension_t  block_length; // in threads
	shared_memory_size_t    dynamic_shared_memory_size; // in bytes

	launch_configuration_t()
	 : grid_length(single_block), block_length(single_thread_per_block),
	   dynamic_shared_memory_size(no_shared_memory) { }
	launch_configuration_t(
		grid_dimension_t grid_length_,
	    grid_block_dimension_t block_length_,
	    shared_memory_size_t dynamic_shared_memory_size = no_shared_memory)
	: grid_length(grid_length_), block_length(block_length_),
	  dynamic_shared_memory_size(dynamic_shared_memory_size) { }

	// Allows linear launch configs to be passed
	// to the launcher function
	operator cuda::launch_configuration_t()
	{
		 return { grid_length, block_length, dynamic_shared_memory_size };
	}
};

} // linear_grid
} // namespace cuda




#endif /* CUDA_LINEAR_GRID_H_ */
