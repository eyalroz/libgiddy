#include "kernels/common.cuh"
#include "cuda/on_device/primitives/block.cuh"

namespace cuda {
namespace kernels {
namespace gather {

// Do we even use this at all now?
constexpr serialization_factor_t get_default_serialization_factor(bool cache_all_data_in_shared_memory) {
	return cache_all_data_in_shared_memory ? 128 : 16;
}

namespace detail {

// Note: reordered_data is assumed to be 4-byte(=sizeof(unsigned))-aligned
template <unsigned OutputIndexSize, unsigned ElementSize, unsigned InputIndexSize>
__forceinline__ __device__ void gather(
	uint_t<ElementSize>*           __restrict__  reordered_data,
	const uint_t<ElementSize>*     __restrict__  data,
	const uint_t<InputIndexSize>*  __restrict__  indices,
	size_t                                             data_length,
	const uint_t<OutputIndexSize>                num_indices,
	bool                                               cache_data_in_shared_memory,
	serialization_factor_t                             serialization_factor) // TODO: It looks like we're not using this right now
{
	using element_type = uint_t<ElementSize>;
	using input_index_type = uint_t<InputIndexSize>;
	using output_index_type = uint_t<InputIndexSize>;

	auto num_grid_threads = grid_info::linear::grid::num_threads();

	using namespace grid_info::linear;
/*	if (cache_data_in_shared_memory) {
		auto data_in_shared_mem = primitives::block::set_shared_memory_to(data, data_length);
		__syncthreads();
 		auto lookup_table = data_in_shared_mem;

		if ((ElementSize == 1) or (ElementSize == 2)) {
			enum { num_elements_per_memory_write =
				(ElementSize > sizeof(unsigned) ? 1 : (sizeof(unsigned) / ElementSize))
				// this contrivance is necessary since the if() block needs to be compiled
				// also for types which are larger than 4 bytes - even though the code
				// path will never be followed for them
			};
			array<element_type, num_elements_per_memory_write> a;
			// Note: sizeof(a) == 4; it's a way to fill a 32-bit value
			promoted_size<output_index_type> pos =
				::grid_info::linear::thread::global_index() * a.size();
			#pragma unroll
			for(;
				pos + sizeof(unsigned) <= num_indices;
				pos += a.size() * num_grid_threads)
			{
				#pragma unroll
				for(auto i = 0; i < a.size(); i++) {
					a[i] = lookup_table[indices[pos + i]];
				}
				*(reinterpret_cast<decltype(a)*>(reordered_data + pos)) = a;
			};
			// Since the loop processes multiple elements at a time (up to 4)
			// we may need to take care of this remainder. This remainder
			// can only be a small number of consecutive elements (up to 3)
			// near the end of the indices range. Note that there are
			// potential alignment issues here so we need to be careful
			if ((ElementSize == 1) and (pos + 2 == num_indices)) {
				array<element_type, 2> pair; // so this array is of size 2 and stored in a half-register (hopefully)
				pair[0] = lookup_table[indices[pos]];
				pair[1] = lookup_table[indices[pos + 1]];
				*(reinterpret_cast<decltype(pair)*>(reordered_data + pos)) = pair;
			}
			else {
				// steady as she goes
				for(; pos + 1 <= num_indices; pos++)
				{
					reordered_data[pos] = lookup_table[indices[pos]];
				}
			}
		}
		else {
			// All sizes other than 1 and 2
			promoted_size<output_index_type> pos = ::grid_info::linear::thread::global_index();
			for(;
				pos < num_indices;
				pos += num_grid_threads)
			{
				reordered_data[pos] = lookup_table[indices[pos]];
			}
		}
	}
	else */ {
		// not caching data in shared memory
		auto lookup_table = data;

		// Yes, the following is an exact duplicate of the am-caching code above; I tried
		// putting that into a function but nvcc had trouble inlining it effectively

		if ((ElementSize == 1) or (ElementSize == 2)) {
			enum { num_elements_per_memory_write =
				(ElementSize > sizeof(unsigned) ? 1 : (sizeof(unsigned) / ElementSize))
				// this contrivance is necessary since the if() block needs to be compiled
				// also for types which are larger than 4 bytes - even though the code
				// path will never be followed for them
			};
			array<element_type, num_elements_per_memory_write> a;
			// Note: sizeof(a) == 4; it's a way to fill a 32-bit value
			promoted_size<output_index_type> pos =
				::grid_info::linear::thread::global_index() * a.size();
			#pragma unroll
			for(;
				pos + sizeof(unsigned) <= num_indices;
				pos += a.size() * num_grid_threads)
			{
				#pragma unroll
				for(auto i = 0; i < a.size(); i++) {
					a[i] = lookup_table[indices[pos + i]];
				}
				*(reinterpret_cast<decltype(a)*>(reordered_data + pos)) = a;
			};
			// Since the loop processes multiple elements at a time (up to 4)
			// we may need to take care of this remainder. This remainder
			// can only be a small number of consecutive elements (up to 3)
			// near the end of the indices range. Note that there are
			// potential alignment issues here so we need to be careful
			if ((ElementSize == 1) and (pos + 2 == num_indices)) {
				array<element_type, 2> pair; // so this array is of size 2 and stored in a half-register (hopefully)
				pair[0] = lookup_table[indices[pos]];
				pair[1] = lookup_table[indices[pos + 1]];
				*(reinterpret_cast<decltype(pair)*>(reordered_data + pos)) = pair;
			}
			else {
				// steady as she goes
				for(; pos + 1 <= num_indices; pos++)
				{
					reordered_data[pos] = lookup_table[indices[pos]];
				}
			}
		}
		else {
			// All other sizes
			auto f = [&](promoted_size<output_index_type> pos) {
				reordered_data[pos] = lookup_table[indices[pos]];
			};
			primitives::grid::linear::at_grid_stride(num_indices, f);
		}
	}
}

} // namespace detail


template <unsigned OutputIndexSize, unsigned ElementSize, unsigned InputIndexSize>
__global__ void gather(
	uint_t<ElementSize>*           __restrict__  reordered_data,
	const uint_t<ElementSize>*     __restrict__  data,
	const uint_t<InputIndexSize>*  __restrict__  indices,
	size_t                                             data_length,
	const uint_t<OutputIndexSize>                num_indices,
	bool                                               cache_data_in_shared_memory,
	serialization_factor_t                             serialization_factor)
{
	detail::gather<OutputIndexSize, ElementSize, InputIndexSize> (
		reordered_data, data, indices, data_length, num_indices,
		cache_data_in_shared_memory, serialization_factor);
}

template <unsigned OutputIndexSize, unsigned ElementSize, unsigned InputIndexSize>
class launch_config_resolution_params_t final : public cuda::launch_config_resolution_params_t {
public:
	using parent = cuda::launch_config_resolution_params_t;

protected:
	// TODO: Reduce the visibility of this function
	bool should_cache_all_input_data_in_shared_memory(size_t input_data_length)
	{
		auto num_indices = length;
		auto input_data_sizes_in_bytes = input_data_length * ElementSize;

		return
			available_dynamic_shared_memory_per_block >= input_data_sizes_in_bytes and
			num_indices * ElementSize > input_data_sizes_in_bytes * 4;
				// I just picked a factor here, it doesn't really matter
				// since if the first condition holds than the overall
				// time is short anyway
	}

public:
	launch_config_resolution_params_t(
		device::properties_t            device_properties_,
		size_t                          input_data_length,
		size_t                          num_indices,
		optional<bool>                  cache_input_data_in_shared_memory_override = nullopt,
		optional<shared_memory_size_t>  dynamic_shared_mem_limit = nullopt) :
		cuda::launch_config_resolution_params_t(
			device_properties_,
			device_function_t(gather<OutputIndexSize, ElementSize, InputIndexSize>),
			dynamic_shared_mem_limit
		)
	{
		grid_construction_resolution            = thread;
		length                                  = num_indices;

		bool cache_input_data_in_shared_memory =
			cache_input_data_in_shared_memory_override.value_or(
//				should_cache_all_input_data_in_shared_memory(input_data_length));
				false); // The caching seems to mostly hurt us; maybe if we did it some other way?

		serialization_option                    = runtime_specified_factor;
		dynamic_shared_memory_requirement.per_block =
			cache_input_data_in_shared_memory ? input_data_length * ElementSize : 0;
		default_serialization_factor =
			get_default_serialization_factor(cache_input_data_in_shared_memory);
	};

	launch_config_resolution_params_t(
		device::properties_t            device_properties_,
		size_t                          input_data_length,
		size_t                          num_indices,
		optional<shared_memory_size_t>  dynamic_shared_mem_limit) :
		launch_config_resolution_params_t(
			device_properties_, input_data_length,
			num_indices, nullopt, dynamic_shared_mem_limit) { };

};


} // namespace gather
} // namespace kernels
} // namespace cuda

