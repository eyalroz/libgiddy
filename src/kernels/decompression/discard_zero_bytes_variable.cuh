#include "util/endianness.h"
#include "kernels/elementwise/unary_operation.cuh"
#include "discard_zero_bytes_fixed.cuh"
#include "cuda/on_device/primitives/warp.cuh"
#include "cuda/on_device/miscellany.cuh"
#include "cuda/stl_array.cuh"
#include <functional>

namespace cuda {
namespace kernels {
namespace decompression {
namespace discard_zero_bytes {
namespace variable_width {

// This could also very well be a template parameter
using element_size_t = unsigned;


using namespace grid_info::linear;

namespace detail {

/**
 * DZB-V element sizes are stored as sequences of consecutive bits within
 * 'bit container' data - typically, say, 2-3 bit sequences within a
 * 32-bit or 64-bit unsigned. Also, what's stored is not the actual
 * element size, but only the relative index of the size within the
 * range of represented size, i.e. the extra size beyond the minimum
 * represented. Length bit sequences do not overflow from one container
 * to another, i.e. the last several bits may be slack (e.g. for 3-bit
 * lengths in a 32-bit container the last 2 bits are slack). This
 * function applies the above to extract the element size from the size
 * containers buffer, given the element's position (in the uncompressed
 * data) and other relevant parameters.
 *
 * @param element_size_containers
 * @param element_index
 * @param element_sizes_per_container
 * @param bits_per_element_size
 * @param min_represented_element_size
 */
template <unsigned IndexSize, unsigned ElementSizesContainerSize>
__forceinline__ __device__  unsigned get_element_size(
	const uint_t<ElementSizesContainerSize>*  __restrict__  element_size_containers,
	uint_t<IndexSize>              element_index,
	unsigned                             element_sizes_per_container,
	unsigned                             bits_per_element_size,
	unsigned                             min_represented_element_size)
{
	auto sizes_container =  element_size_containers[element_index / element_sizes_per_container];
	auto size_representation =
		bit_subsequence(
			sizes_container, element_index % element_sizes_per_container * bits_per_element_size,
			bits_per_element_size);
//	thread_printf("My sizes container is %X, getting %d bits starting at %d yields size representation is %X",
//		(unsigned) sizes_container, (int) bits_per_element_size,
//		(int) (element_index % element_sizes_per_container), (unsigned) size_representation);
	return size_representation + min_represented_element_size;
}

using util::endianness_t;

/*
 * The copy implementations here are very ugly and have too much code duplication,
 * but they should be ok for the smaller sizes at least (ignoring alignment anyway).
 * Also, this assumes little-endianness is assumed.
 */
namespace impl {

//template <endianness_t OutputEndianness>
__device__ __forceinline__
void pad_high_bytes_1_byte(
	unsigned char*       __restrict__  result,
	const unsigned char* __restrict__  low_bytes,
	unsigned                           num_low_bytes)
{
	*result = 0;
	if (num_low_bytes > 0) { *result = *low_bytes; }
}

//template <endianness_t OutputEndianness>
__device__ __forceinline__
void pad_high_bytes_2_byte(
	unsigned char*       __restrict__  result,
	const unsigned char* __restrict__  low_bytes,
	unsigned                           num_low_bytes)
{
	*result = 0;
	switch(num_low_bytes) {
	case 1: *reinterpret_cast<unsigned char*> (result) = *reinterpret_cast<const unsigned char*> (low_bytes); break;
	case 2: *reinterpret_cast<unsigned short*>(result) = *reinterpret_cast<const unsigned short*>(low_bytes); break;
	}
}


//template <endianness_t OutputEndianness>
__device__ __forceinline__
void pad_high_bytes_4_byte(
	unsigned char*       __restrict__  result,
	const unsigned char* __restrict__  low_bytes,
	unsigned                           num_low_bytes)
{
	*result = 0;
	switch(num_low_bytes) {
	case 1: *reinterpret_cast<unsigned char*> (result) = *reinterpret_cast<const unsigned char*> (low_bytes); break;
	case 2: *reinterpret_cast<unsigned short*>(result) = *reinterpret_cast<const unsigned short*>(low_bytes); break;
	case 3: *reinterpret_cast<uchar3*>        (result) = *reinterpret_cast<const uchar3*>        (low_bytes); break;
	case 4: *reinterpret_cast<unsigned int*>  (result) = *reinterpret_cast<const unsigned int*>  (low_bytes); break;
	}
}

//template <endianness_t OutputEndianness>
__device__ __forceinline__
void pad_high_bytes_3_byte(
	unsigned char*       __restrict__  result,
	const unsigned char* __restrict__  low_bytes,
	unsigned                           num_low_bytes)
{
	unsigned staging;
	pad_high_bytes_4_byte(reinterpret_cast<unsigned char*>(&staging), low_bytes, num_low_bytes);
	*reinterpret_cast<uchar3*>(result) = reinterpret_cast<uchar3&>(staging);
}


//template <endianness_t OutputEndianness>
__device__ __forceinline__
void pad_high_bytes_5_byte(
	unsigned char*       __restrict__  result,
	const unsigned char* __restrict__  low_bytes,
	unsigned                           num_low_bytes)
{
	if (num_low_bytes > 4) {
		*reinterpret_cast<unsigned int*>(result) = *reinterpret_cast<const unsigned int*>(low_bytes);
		pad_high_bytes_1_byte(result + 4, low_bytes + 4, num_low_bytes - 4);
		return;
	}
	pad_high_bytes_4_byte(result, low_bytes, num_low_bytes);
}

//template <endianness_t OutputEndianness>
__device__ __forceinline__
void pad_high_bytes_6_byte(
	unsigned char*       __restrict__  result,
	const unsigned char* __restrict__  low_bytes,
	unsigned                           num_low_bytes)
{
	if (num_low_bytes > 4) {
		*reinterpret_cast<unsigned int*>  (result) = *reinterpret_cast<const unsigned int*>(low_bytes);
		pad_high_bytes_2_byte(result + 4, low_bytes + 4, num_low_bytes - 4);
		return;
	}
	pad_high_bytes_4_byte(result, low_bytes, num_low_bytes);
}

//template <endianness_t OutputEndianness>
__device__ __forceinline__
void pad_high_bytes_7_byte(
	unsigned char*       __restrict__  result,
	const unsigned char* __restrict__  low_bytes,
	unsigned                           num_low_bytes)
{
	if (num_low_bytes > 4) {
		*reinterpret_cast<unsigned int*>  (result) = *reinterpret_cast<const unsigned int*>(low_bytes);
		pad_high_bytes_3_byte(result + 4, low_bytes + 4, num_low_bytes - 4);
		return;
	}
	pad_high_bytes_4_byte(result, low_bytes, num_low_bytes);
}

//template <endianness_t OutputEndianness>
__device__ __forceinline__
void pad_high_bytes_8_byte(
	unsigned char*       __restrict__  result,
	const unsigned char* __restrict__  low_bytes,
	unsigned                           num_low_bytes)
{
	if (num_low_bytes > 4) {
		*reinterpret_cast<unsigned int*>  (result) = *reinterpret_cast<const unsigned int*>(low_bytes);
		pad_high_bytes_4_byte(result + 4, low_bytes + 4, num_low_bytes - 4);
		return;
	}
	pad_high_bytes_4_byte(result, low_bytes, num_low_bytes);
}

template <typename T>
__device__ __forceinline__
void assign_zero_if_possible(typename std::enable_if<std::is_arithmetic<T>::value, T>::type& x) { x = 0; }

template <typename T>
__device__ __forceinline__
void assign_zero_if_possible(typename std::enable_if<!std::is_arithmetic<T>::value, T>::type& x) { }

} // namespace impl

template <typename T/*, endianness_t OutputEndianness*/>
__device__ __forceinline__
void pad_high_bytes(
	T&                                 result,
	const unsigned char* __restrict__  low_bytes,
	unsigned                           num_low_bytes)
{
	if (std::is_arithmetic<T>::value) {
		impl::assign_zero_if_possible<T>(result);
	}
	auto result_bytes = reinterpret_cast<unsigned char*>(&result);
	if (num_low_bytes > 0) {
		memcpy(result_bytes, low_bytes, num_low_bytes);
	}
	if (!std::is_arithmetic<T>::value) {
		memzero(result_bytes + num_low_bytes, sizeof(result) - num_low_bytes);
	}
}


namespace warp {

/**
 * Decompresses - independently by each warp - data compressed with the
 * DZB-V/NSV compression sceheme  (drop zero bytes - variable, a.k.a. null
 * suppression - variable), with no position anchors available. It is
 * expected that either the number of elements to decompress not be
 * so high, or that the distribution of element lengths be close enough
 * to uniformity, so as to avoid warp workload imbalance and potential
 * idling of the GPU.
 *
 * @note This implementation may be rather inefficient when the element
 * sizes are non-uniform, as some threads will be done with their single-element
 * decompression sooner than their fellow threads. However, I doubt this
 * become significant enough to merit another strategy with sizes smaller than,
 * say, 8. And for larger sizes still we might need a non-templated
 * uncompressed size, which is currently not supported.
 *
 *
 * @param decompressed
 * @param compressed_input
 * @param uncompressed_element_sizes
 * @param position_anchors
 * @param num_elements
 */
template<unsigned IndexSize, /* util::terminal_t EndToPad, */unsigned UncompressedSize, unsigned ElementSizesContainerSize>
__device__ void decompress(
	uint_t<UncompressedSize>*  __restrict__  decompressed,
	const unsigned char*             __restrict__  compressed_data,
	const uint_t<ElementSizesContainerSize>*
	                                 __restrict__  packed_element_sizes,
	unsigned                                       offset_into_first_element_sizes_container,
	uint_t<IndexSize>                        num_compressed_elements,
	element_size_t                                 min_represented_element_size,
	unsigned                                       bits_per_element_size,
	unsigned                                       element_sizes_per_container,
	unsigned                                       element_sizes_offset)
{
	using uncompressed_type = uint_t<UncompressedSize>;

	uint_t<IndexSize> output_pos_past_warp = warp_size;

	// The number of elements in uncompressed_elements_sizes which we will be
	// reading data from (which, effectively, is its length as far as this
	// function is concerned)
	auto num_element_size_containers = div_rounding_up(
		num_compressed_elements + element_sizes_offset, element_sizes_per_container);

	uncompressed_type decompressed_element;

//	warp_printf(
//		"This warp will decompress %llu elements. Represented sizes are %d..%d. "
//		"Each sizes container holds the sizes of %d elements (%d bits each). "
//		"First container has %d sizes regarding a previous input range.",
//		(size_t) num_compressed_elements,
//		(int) min_represented_element_size, (int) min_represented_element_size + (1 << bits_per_element_size) - 1,
//		(int) element_sizes_per_container, (int) bits_per_element_size, (int) element_sizes_offset
//	);

	uint_t<IndexSize> output_pos = lane::index();
	uint_t<IndexSize> warp_input_pos = 0;
	for(; output_pos_past_warp <= num_compressed_elements;
		output_pos += warp_size, output_pos_past_warp += warp_size) {
		// In this case, all warp lanes can write an element to the output

//		if (grid_info::linear::warp::is_last_in_grid()) {
//			warp_printf("Will perform a full lane write to (warp-relative) output positions %d - %d of "
//				"%d total ", (int)(output_pos_past_warp - warp_size), (int) output_pos_past_warp - 1,
//				(int) num_compressed_elements);
//		}

		// TODO: We could probably avoid the division and the modulo in there if we
		// always move forward using a fixed pattern; and that pattern's length
		// can only be as long as bits_per_element_size, so under usual circumstances
		// the pattern length is under 7
		auto element_size = get_element_size<IndexSize, ElementSizesContainerSize>(
			packed_element_sizes,
			output_pos + element_sizes_offset,
			element_sizes_per_container,
			bits_per_element_size,
			min_represented_element_size);

		auto compressed_element_pos_in_input = warp_input_pos +
			primitives::warp::exclusive_prefix_sum<unsigned>(element_size);

		pad_high_bytes(decompressed_element, compressed_data + compressed_element_pos_in_input, element_size);
		decompressed[output_pos] = decompressed_element;

		// Advance the warp's base position in the compressed input beyond the last byte the
		// last write has used (i.e. the last byte the last lane has used)
		warp_input_pos = primitives::warp::have_last_lane_compute(
			[&]() { return compressed_element_pos_in_input + element_size; }
		);
	}

	// Note: The whole warp exits the loop at the same iteration

	auto element_size = output_pos < num_compressed_elements ?
		get_element_size<IndexSize, ElementSizesContainerSize>(
			packed_element_sizes,
			output_pos + element_sizes_offset,
			element_sizes_per_container,
			bits_per_element_size,
			min_represented_element_size) :
		0;

	auto compressed_element_pos_in_input = warp_input_pos +
		primitives::warp::exclusive_prefix_sum<unsigned>(element_size);

	if (output_pos >= num_compressed_elements) { return; }

	pad_high_bytes(decompressed_element, compressed_data + compressed_element_pos_in_input, element_size);
	decompressed[output_pos] = decompressed_element;
}

} // namespace warp
} // namespace detail

/**
 * Decompress data compressed with the DZB-V/NSV compression sceheme
 * (drop zero bytes - variable, a.k.a. null suppression - variable),
 * when position anchors are available.
 *
 * @param decompressed
 * @param compressed_input
 * @param uncompressed_element_sizes
 * @param position_anchors
 * @param position_anchoring_period
 * @param num_elements
 * @param min_represented_element_size
 * @param bits_per_element_size Number of bits used to represented the
 * size of an uncompressed elements.
 */
template<unsigned IndexSize, /* util::terminal_t EndToPad, */unsigned UncompressedSize, unsigned ElementSizesContainerSize>
__global__ void decompress(
	uint_t<UncompressedSize>*  __restrict__  decompressed,
	const unsigned char*             __restrict__  compressed_data,
	const uint_t<ElementSizesContainerSize>*
	                                 __restrict__  packed_element_sizes,
	const uint_t<IndexSize>*                      __restrict__  position_anchors,
	uint_t<IndexSize>                        position_anchoring_period,
	uint_t<IndexSize>                        num_elements,
	element_size_t                                 min_represented_element_size,
	unsigned                                       bits_per_element_size)
{
	using element_sizes_container_type = uint_t<ElementSizesContainerSize>;
	static_assert(ElementSizesContainerSize == 4 or ElementSizesContainerSize == 8,
		"Only container types with 32 or 64 bits are currently supported");

	auto warp_start_output_pos = position_anchoring_period * warp::global_index();

	if (warp_start_output_pos >= num_elements) { return; }

	auto element_sizes_per_size_container = size_in_bits<element_sizes_container_type>::value / bits_per_element_size;

	// TODO: If anchors are very close for some reason (and remember, this kernel itself
	// does  not control the anchoring period) - we might want to consider having a single
	// anchor process more than a single period. For now that doesn't happen
	//
	// Each warp will decompress the elements between consecutive anchor positions (or
	// all elements starting at the last anchor, if it's the last warp in the grid).
	// If we had chosen to do that with a full block we would have had to prefix-sum
	// the lengths so as to allow latter warps to know where to start.


	auto is_last_active_warp_in_grid = (warp_start_output_pos + warp_size >= num_elements);

	auto warp_start_pos_in_element_sizes         = warp_start_output_pos / element_sizes_per_size_container;
	auto offset_in_first_element_sizes_container = warp_start_output_pos % element_sizes_per_size_container;
	auto num_elements_for_warp_decompression     = is_last_active_warp_in_grid ?
		(num_elements - warp_start_output_pos) : position_anchoring_period;

//	grid_printf(
//		"This grid will decompress %llu elements, represented sizes %d..%d, sizes "
//		"per container %d, bits per element size %d",
//		(size_t) num_elements,
//		(int) min_represented_element_size, (int) min_represented_element_size + (1 << bits_per_element_size) - 1,
//		(int) element_sizes_per_size_container, (int) bits_per_element_size
//	);

	detail::warp::decompress<IndexSize, UncompressedSize, ElementSizesContainerSize>(
		decompressed + warp_start_output_pos,
		compressed_data + position_anchors[warp::index_in_grid()],
		packed_element_sizes + warp_start_pos_in_element_sizes,
		offset_in_first_element_sizes_container,
		num_elements_for_warp_decompression,
		min_represented_element_size,
		bits_per_element_size,
		element_sizes_per_size_container,
		offset_in_first_element_sizes_container
	);
}

template<unsigned IndexSize, /* util::terminal_t EndToPad, */unsigned UncompressedSize, unsigned ElementSizesContainerSize>
class launch_config_resolution_params_t final : public kernels::launch_config_resolution_params_t {
public:
	using parent = kernels::launch_config_resolution_params_t;
public:
	launch_config_resolution_params_t(
		device::properties_t            device_properties_,
		size_t                          length_in_elements,
		size_t                          position_anchoring_period) :
		parent(
			device_properties_,
			device_function_t(decompress<IndexSize, UncompressedSize, ElementSizesContainerSize>)
		)
	{
		grid_construction_resolution            = warp;
		length                                  = util::div_rounding_up(length_in_elements, position_anchoring_period);
		serialization_option                    = none;
	};
};

} // namespace variable_width
} // namespace discard_zero_bytes
} // namespace decompression
} // namespace kernels
} // namespace cuda
