
#include "kernels/common.cuh"
#include "cuda/on_device/primitives/block.cuh"
#include "cuda/bit_vector.cuh"

namespace cuda {
namespace kernels {
namespace gather_bits {

namespace detail {

// TODO: Currently ignoring this and always reading from main device memory
inline bool should_cache_all_input_data_in_shared_memory(
	size_t    available_shared_memory,
	size_t    input_bit_vector_length,
	size_t    num_bit_indices,
	size_t    size_of_index_into_input,
	size_t    size_of_bit_container_type)
{
	auto total_size_of_input_data = input_bit_vector_length / size_of_bit_container_type;
	return
		available_shared_memory >= total_size_of_input_data &&
		num_bit_indices * size_of_index_into_input > total_size_of_input_data * 4;
			// I just picked a factor here, it doesn't really matter
			// much, since if the first condition holds then the overall
			// time is short anyway
}

namespace warp {

enum WarpIntervalCompleteness : bool { possibly_incomplete_interval = false, complete_interval = true} ;

template <unsigned OutputIndexSize, unsigned InputIndexSize, WarpIntervalCompleteness IntervalIsComplete>
__forceinline__ __device__
void gather_interval_bits(
	typename bit_vector<uint_t<OutputIndexSize>>::container_type*        __restrict__  gathered,
	const typename bit_vector<uint_t<OutputIndexSize>>::container_type*  __restrict__  input,
	const uint_t<InputIndexSize>*                                        __restrict__  indices,
	uint_t<OutputIndexSize>                                                            num_bit_indices,
	uint_t<OutputIndexSize>                                                            warp_base_pos)
{
	using namespace grid_info::linear;
	using input_index_type = uint_t<InputIndexSize>;
	using output_index_type = uint_t<OutputIndexSize>;
	using bit_container_type = typename bit_vector<output_index_type>::container_type;

	// Note that we may safely write complete bit containers - as bit vectors
	// have 'slack' bits in the last container even if they could theoretically
	// 'save' a byte or two at the end; but the same is not true for reading
	// from indices (which is not a bit vector)

	unsigned num_warp_reads = bit_vector<output_index_type>::bits_per_element;
	if (!IntervalIsComplete) {
		num_warp_reads = num_warp_sizes_to_cover(num_bit_indices - warp_base_pos);
			// ... and remember num_bit_indices - warp_base_pos >= 0
	}


	bit_container_type thread_output;
	// TODO: Can I trust NVCC to always use num_warp_reads and for it to realize,
	// when PossiblyIncomplete = false, that it's actually compile-time constant?
	// Let's assume that I can't.
	#pragma unroll
	for(auto i = 0; i < num_warp_reads; i++) {
		output_index_type thread_pos = warp_base_pos + lane::index() + i * warp_size;
		// TODO: Do we need to explictly use __ldg here? I hope not...
		auto input_bit_index = IntervalIsComplete ?
			indices[thread_pos] :
			(thread_pos < num_bit_indices ? indices[thread_pos] : 0);
		auto input_bit_container =
			IntervalIsComplete ? input[bit_vector<input_index_type>::element_index_for(input_bit_index)] :
			(thread_pos < num_bit_indices ? input[bit_vector<input_index_type>::element_index_for(input_bit_index)] : 0);
		auto gathered_bit = bit_vector<input_index_type>::is_set_uncast(
			input_bit_container, bit_vector<input_index_type>::intra_element_index_for(input_bit_index));
//		if (thread_pos < num_bit_indices) {
//			thread_printf("In warp read %2u, my pos was %6u . I read container %8X and gathered the %2u'th bit, which was %u", i, thread_pos, input_bit_container, bit_vector<input_index_type>::intra_element_index_for(input_bit_index), gathered_bit);
//		}
		auto gathered_bit_container = builtins::warp_ballot(gathered_bit);
//		warp_printf("the gathered_bit_container for this warp is %8X", gathered_bit_container);
			// Since the threads begin f with consecutive thread_base_pos,
			// they keep that property throughout this loop, meaning that the bits
			// in the gathered bit container are consecutive output bits
			// (LSB to MSB according to the ballot semantics IIANM), and we
			// could just write them here. But - that would be a waste, since
			// we want to coalesce writes. That's what the unrolled loop is for -
			// to give every thread some output to write
		if (i == lane::index()) { thread_output = gathered_bit_container; }
	}
	if (IntervalIsComplete || lane::index() < num_warp_reads) {
		// each thread kept one container, and the warp will now write all
		// these consecutively
		auto output_container_index =
			bit_vector<output_index_type>::element_index_for(warp_base_pos) + lane::index();
		gathered[output_container_index] = thread_output;
	}
};

} // namespace warp

} // namespace detail


// TODO: Try using the actual bit_vector class for parameters; that should really
// not be a problem (but test to verify). Or maybe - using spans and add
// as_span to the bit_vector class ? Naah, that might be a useful idea elsewhere though
// TODO: Consider templatizing on the last parameter
// Note: Only works with 32-bit-size container  type (i.e. warp-size bit container type)
// Note: block size must be a power of 2
template <unsigned OutputIndexSize, unsigned InputIndexSize>
__global__ void gather_bits(
	typename bit_vector<uint_t<OutputIndexSize>>::container_type*        __restrict__  gathered,
	const typename bit_vector<uint_t<InputIndexSize>>::container_type*   __restrict__  input,
	const uint_t<InputIndexSize>*                                        __restrict__  indices,
	uint_t<InputIndexSize>                                                             num_input_bits,
	uint_t<OutputIndexSize>                                                            num_bit_indices)
{
	using input_index_type = uint_t<InputIndexSize>;
	using output_index_type = uint_t<OutputIndexSize>;
	static_assert(std::is_same<
		typename bit_vector<input_index_type>::container_type,
		typename bit_vector<output_index_type>::container_type>::value,
		"Bit container type mismatch");
	using bit_container_type = typename bit_vector<output_index_type>::container_type;
	static_assert(sizeof(bit_container_type) * CHAR_BIT == warp_size,
			"Only warp-size-bit-per-container bit vectors are supported");
	using namespace grid_info::linear;
	enum { bits_per_container = bit_vector<output_index_type>::bits_per_element };
	using detail::warp::gather_interval_bits;


	//	if (cache_data_in_shared_memory) {
	//		auto data_in_shared_mem = shared_memory_proxy<Datum>();
	//		primitives::block::copy(data_in_shared_mem, input_data, input_data_length);
	//		__syncthreads();
	//
	// ... and probably call an auxiliary function. Or make this boolean
	// into a template parameter? also possible.


	enum { bits_covered_by_each_warp = warp_size * bits_per_container };
		// I know, I know, abusing enums. But I don't trust NVCC enough to use static constxpr const

	output_index_type warp_base_pos   = warp::global_index() * bits_covered_by_each_warp;

	if (num_bit_indices >  warp_base_pos + bits_covered_by_each_warp) {
		gather_interval_bits<OutputIndexSize, InputIndexSize, detail::warp::complete_interval>(
			gathered, input, indices, num_bit_indices, warp_base_pos);
	}
	else if (num_bit_indices >  warp_base_pos) {
		// this warp still has output bits to cover - but it's the last one
		gather_interval_bits<OutputIndexSize, InputIndexSize, detail::warp::possibly_incomplete_interval>(
			gathered, input, indices, num_bit_indices, warp_base_pos);
	}
}

template <unsigned OutputIndexSize, unsigned InputIndexSize>
class launch_config_resolution_params_t final : public cuda::launch_config_resolution_params_t {
public:
	using parent = cuda::launch_config_resolution_params_t;
public:
	launch_config_resolution_params_t(
		device::properties_t            device_properties_,
		size_t                          num_input_bits_,
		size_t                          num_bit_indices_,
		optional<shared_memory_size_t>  dynamic_shared_mem_limit = nullopt) :
		cuda::launch_config_resolution_params_t(
			device_properties_,
			device_function_t(gather_bits<OutputIndexSize, InputIndexSize>),
			dynamic_shared_mem_limit
		)
	{
		using input_index_type = uint_t<InputIndexSize>;
		using output_index_type = uint_t<OutputIndexSize>;
		using bit_container_type = typename cuda::bit_vector<output_index_type>::container_type;
		num_input_bits                          = num_input_bits_;
		num_bit_indices                         = num_bit_indices_;
		grid_construction_resolution            = warp;
		size_t indices_covered_by_each_warp     = warp_size * warp_size;
		length                                  = util::div_rounding_up(num_bit_indices, indices_covered_by_each_warp);
		serialization_option                    = none;
		bool cache_input_data_in_shared_memory = false; // change this when caching is supported
//			detail::should_cache_all_input_data_in_shared_memory(
//				available_dynamic_shared_memory_per_block,
//				data_length_,
//				num_indices_, sizeof(Datum), sizeof(InputIndexSize));
		dynamic_shared_memory_requirement.per_block =
			cache_input_data_in_shared_memory ? num_input_bits * sizeof(bit_container_type): 0;
		dynamic_shared_memory_requirement.per_length_unit = 0;
	};

public:
	size_t                          num_input_bits;
	size_t                          num_bit_indices;
};


} // namespace gather_bits
} // namespace kernels
} // namespace cuda

