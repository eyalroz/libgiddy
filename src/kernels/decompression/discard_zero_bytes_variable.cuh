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
using element_size_t = native_word_t;


using namespace grid_info::linear;

namespace detail {

/**
 * DZB-V element sizes are stored as sequences of consecutive bits within
 * 'bit container' data - typically, say, 2-3 bit sequences within a
 * 32-bit or 64-bit value. Also, what's stored is not the actual
 * element size, but only the relative index of the size within the
 * range of represented sizes, i.e. the extra size beyond the minimum
 * represented. Length bit sequences do not overflow from one container
 * to another, i.e. the last several bits of a lengths container may be
 * unused slack; for example, if we encodes lengths using 3 bits each,
 * in a 32-bit container, the last 2 bits are unused.
 *
 * This function applies the above scheme to extract the element size from
 * the size containers buffer, given the element's position (in the
 * uncompressed data) and other relevant parameters.
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
	uint_t<IndexSize>                                       element_index,
	element_size_t                                          element_sizes_per_container,
	native_word_t                                           bits_per_element_size,
	native_word_t                                           min_represented_element_size)
{
	auto sizes_container =  element_size_containers[element_index / element_sizes_per_container];
	auto size_representation =
		bit_subsequence(
			sizes_container, element_index % element_sizes_per_container * bits_per_element_size,
			bits_per_element_size);
	return size_representation + min_represented_element_size;
}

using util::endianness_t;

/*
 * The copy implementations here are very ugly and have too much code duplication,
 * but they should be ok for the smaller sizes at least (ignoring alignment anyway).
 * Also, this assumes little-endianness is assumed.
 */
namespace impl {

static_assert(sizeof(native_word_t) == 4, "This code assumes the native GPU word has 4 bytes");

//template <endianness_t OutputEndianness>
__device__ __forceinline__
void pad_high_bytes_1_byte(
	uint8_t*       __restrict__  result,
	const uint8_t* __restrict__  low_bytes,
	element_size_t               num_low_bytes)
{
	*result = 0;
	if (num_low_bytes > 0) { *result = *low_bytes; }
}

//template <endianness_t OutputEndianness>
__device__ __forceinline__
void pad_high_bytes_2_byte(
	uint8_t*       __restrict__  result,
	const uint8_t* __restrict__  low_bytes,
	element_size_t               num_low_bytes)
{
	*result = 0;
	switch(num_low_bytes) {
	case 1: *reinterpret_cast<uint8_t*> (result) = *reinterpret_cast<const uint8_t*> (low_bytes); break;
	case 2: *reinterpret_cast<uint16_t*>(result) = *reinterpret_cast<const uint16_t*>(low_bytes); break;
	}
}


//template <endianness_t OutputEndianness>
__device__ __forceinline__
void pad_high_bytes_4_byte(
	uint8_t*       __restrict__  result,
	const uint8_t* __restrict__  low_bytes,
	element_size_t               num_low_bytes)
{
	*result = 0;
	switch(num_low_bytes) {
	case 1: *reinterpret_cast<uint8_t* >(result) = *reinterpret_cast<const uint8_t* >(low_bytes); break;
	case 2: *reinterpret_cast<uint16_t*>(result) = *reinterpret_cast<const uint16_t*>(low_bytes); break;
	case 3: *reinterpret_cast<uchar3*  >(result) = *reinterpret_cast<const uchar3*  >(low_bytes); break;
	case 4: *reinterpret_cast<uint32_t*>(result) = *reinterpret_cast<const uint32_t*>(low_bytes); break;
	}
}

//template <endianness_t OutputEndianness>
__device__ __forceinline__
void pad_high_bytes_3_byte(
	uint8_t*       __restrict__  result,
	const uint8_t* __restrict__  low_bytes,
	native_word_t num_low_bytes)
{
	uint32_t staging;
	pad_high_bytes_4_byte(reinterpret_cast<uint8_t*>(&staging), low_bytes, num_low_bytes);
	*reinterpret_cast<uchar3*>(result) = reinterpret_cast<uchar3&>(staging);
}


//template <endianness_t OutputEndianness>
__device__ __forceinline__
void pad_high_bytes_5_byte(
	uint8_t*       __restrict__  result,
	const uint8_t* __restrict__  low_bytes,
	element_size_t               num_low_bytes)
{
	if (num_low_bytes > 4) {
		*reinterpret_cast<uint32_t*>(result) = *reinterpret_cast<const uint32_t*>(low_bytes);
		pad_high_bytes_1_byte(result + 4, low_bytes + 4, num_low_bytes - 4);
		return;
	}
	pad_high_bytes_4_byte(result, low_bytes, num_low_bytes);
}

//template <endianness_t OutputEndianness>
__device__ __forceinline__
void pad_high_bytes_6_byte(
	uint8_t*       __restrict__  result,
	const uint8_t* __restrict__  low_bytes,
	element_size_t               num_low_bytes)
{
	if (num_low_bytes > 4) {
		*reinterpret_cast<uint32_t*>  (result) = *reinterpret_cast<const uint32_t*>(low_bytes);
		pad_high_bytes_2_byte(result + 4, low_bytes + 4, num_low_bytes - 4);
		return;
	}
	pad_high_bytes_4_byte(result, low_bytes, num_low_bytes);
}

//template <endianness_t OutputEndianness>
__device__ __forceinline__
void pad_high_bytes_7_byte(
	uint8_t*       __restrict__  result,
	const uint8_t* __restrict__  low_bytes,
	element_size_t               num_low_bytes)
{
	if (num_low_bytes > 4) {
		*reinterpret_cast<uint32_t*>  (result) = *reinterpret_cast<const uint32_t*>(low_bytes);
		pad_high_bytes_3_byte(result + 4, low_bytes + 4, num_low_bytes - 4);
		return;
	}
	pad_high_bytes_4_byte(result, low_bytes, num_low_bytes);
}

//template <endianness_t OutputEndianness>
__device__ __forceinline__
void pad_high_bytes_8_byte(
	uint8_t*       __restrict__  result,
	const uint8_t* __restrict__  low_bytes,
	element_size_t               num_low_bytes)
{
	if (num_low_bytes > 4) {
		*reinterpret_cast<unsigned int*>  (result) = *reinterpret_cast<const uint32_t*>(low_bytes);
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
	T&                           result,
	const uint8_t* __restrict__  low_bytes,
	element_size_t               num_low_bytes)
{
	if (std::is_arithmetic<T>::value) {
		impl::assign_zero_if_possible<T>(result);
	}
	auto result_bytes = reinterpret_cast<uint8_t*>(&result);
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
 * @param length
 */
template<unsigned IndexSize, /* util::terminal_t EndToPad, */unsigned UncompressedSize, unsigned ElementSizesContainerSize>
__device__ void decompress(
	uint_t<UncompressedSize>*  __restrict__  decompressed,
	const uint8_t*             __restrict__  compressed_data,
	const uint_t<ElementSizesContainerSize>*
	                           __restrict__  packed_element_sizes,
	native_word_t                            offset_into_first_element_sizes_container,
	size_type_by_index_size<IndexSize>       num_compressed_elements,
	element_size_t                           min_represented_element_size,
	native_word_t                            bits_per_element_size,
	native_word_t                            element_sizes_per_container,
	native_word_t                            element_sizes_offset)
{
	using uncompressed_type = uint_t<UncompressedSize>;
	using size_type = size_type_by_index_size<IndexSize>;

	size_type output_pos_past_warp = warp_size;

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

	size_type output_pos = lane::index();
	size_type warp_input_pos = 0;
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
			primitives::warp::exclusive_prefix_sum(element_size);

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
		primitives::warp::exclusive_prefix_sum(element_size);

	if (output_pos >= num_compressed_elements) { return; }

	pad_high_bytes(decompressed_element, compressed_data + compressed_element_pos_in_input, element_size);
	decompressed[output_pos] = decompressed_element;
}

} // namespace warp
} // namespace detail

/**
 * @brief Decompresses data compressed with the DZB-V/NSV compression scheme
 * (drop zero bytes - variable, a.k.a. null suppression - variable),
 * in anchored segments.
 *
 * In this scheme, each element of the uncompressed data is represented
 * using a variable number of bytes within @p compressed_data : When
 * some of its prefix bytes are 0, we skip their representation. It
 * is assumed all elements have a number of non-zero bytes (their
 * "element size") equal to @p min_represented_element_size + a number
 * in the range [ 0 .. 1 << bits_per_element_size - 1). Thus,
 * typically, we might represent elements of sizes 1, 2, 3, 4 with
 * a minimum length of 1 and 2 bits of size information.
 *
 * The element sizes are themselves stored in _fixed-width_ form,
 * so we can easily tell where they're located without having to sum
 * anything up. They're bunched up within element-size containers,
 * which form the @p uncompressed_element_sizes array
 * (see @ref get_element_size for more details).
 *
 * Now, just from an element's size you can't tell where exactly
 * its actual data is located - since data is variable-size. To do
 * so we have to sum up consecutive element lengths from some
 * starting point up until our element of interest. Since we
 * do not wish to always sum up from the beginning of the entire column,
 * we decompress in <i>anchored segments</i>, i.e. for every certain
 * number of elements (the @p segment_length) we have a <i>position
 * anchor</i> which indicates where the data for the first of these
 * elements starts with the @p compressed data array. In a sense, this
 * is like a pre-computed first phase of a "prefix sum" over the element
 * sizes.
 *
 * @note The data is not <i>actually</i> broken up into segments. One
 * can decompress it without using the anchors at all, with no change
 * of semantics for the @compressed_input and @uncompressed_element_sizes
 * arrays.
 *
 * @param decompressed
 * @param compressed_data
 * @param packed_element_sizes
 * @param position_anchors
 * @param segment_length
 * @param length overall
 * @param min_represented_element_size
 * @param bits_per_element_size
 */
template<unsigned IndexSize, /* util::terminal_t EndToPad, */unsigned UncompressedSize, unsigned ElementSizesContainerSize>
__global__ void decompress(
	uint_t<UncompressedSize>*  __restrict__  decompressed,
	const uint8_t*             __restrict__  compressed_data,
	const uint_t<ElementSizesContainerSize>*
	                           __restrict__  packed_element_sizes,
	const uint_t<IndexSize>*   __restrict__  position_anchors,
	size_type_by_index_size<IndexSize>       segment_length,
	size_type_by_index_size<IndexSize>       length,
	element_size_t                           min_represented_element_size,
	native_word_t                            bits_per_element_size)
{
	using element_sizes_container_type = uint_t<ElementSizesContainerSize>;
	static_assert(ElementSizesContainerSize == 4 or ElementSizesContainerSize == 8,
		"Only container types with 32 or 64 bits are currently supported");

	auto warp_start_output_pos = segment_length * warp::global_index();

	if (warp_start_output_pos >= length) { return; }

	auto element_sizes_per_size_container = size_in_bits<element_sizes_container_type>::value / bits_per_element_size;

	// TODO: If anchors are very close for some reason (and remember, this kernel itself
	// does  not control the segment length) - we might want to consider having a single
	// anchor process more than a single segment. For now that doesn't happen
	//
	// Each warp will decompress the elements between consecutive anchor positions (or
	// all elements starting at the last anchor, if it's the last warp in the grid).
	// If we had chosen to do that with a full block we would have had to prefix-sum
	// the lengths so as to allow latter warps to know where to start.


	auto is_last_active_warp_in_grid = (warp_start_output_pos + warp_size >= length);

	auto warp_start_pos_in_element_sizes         = warp_start_output_pos / element_sizes_per_size_container;
	auto offset_in_first_element_sizes_container = warp_start_output_pos % element_sizes_per_size_container;
	auto num_elements_for_warp_decompression     = is_last_active_warp_in_grid ?
		(length - warp_start_output_pos) : segment_length;

//	grid_printf(
//		"This grid will decompress %llu elements, represented sizes %d..%d, sizes "
//		"per container %d, bits per element size %d",
//		(size_t) length,
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
		size_t                          segment_length) :
		parent(
			device_properties_,
			device_function_t(decompress<IndexSize, UncompressedSize, ElementSizesContainerSize>)
		)
	{
		grid_construction_resolution            = warp;
		length                                  = util::div_rounding_up(length_in_elements, segment_length);
		serialization_option                    = none;
	};
};

} // namespace variable_width
} // namespace discard_zero_bytes
} // namespace decompression
} // namespace kernels
} // namespace cuda
