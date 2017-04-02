#pragma once
#ifndef SRC_KERNELS_DECOMPRESSION_INCIDENCE_BITMAPS_CUH
#define SRC_KERNELS_DECOMPRESSION_INCIDENCE_BITMAPS_CUH

#include "kernels/common.cuh"
#include "cuda/bit_operations.cuh"
#include "cuda/on_device/primitives/warp.cuh"
#include "cuda/on_device/primitives/block.cuh"
#include "cuda/on_device/math.cuh"
#include "cuda/on_device/printing.cuh"
#include "util/miscellany.hpp" // for size_in_bits

namespace cuda {
namespace kernels {
namespace decompression {
namespace incidence_bitmaps {

using namespace grid_info::linear;

using bit_container_t = unsigned;
using bitmap_index_t  = unsigned char;

namespace detail {

namespace thread {

template<unsigned IndexSize> // the decompression here is to bitmap indices, which are unsigned char
__device__ void decompress_to_shared_mem_without_translation(
	bitmap_index_t*         __restrict__  thread_decompressed,
	const bit_container_t*  __restrict__  bitmaps,
	uint_t<IndexSize>                     position_in_bitmaps,
	uint_t<IndexSize>                     aligned_bitmap_size_in_ints,
	bitmap_index_t                        num_bitmaps)
{
	// In each call to the function f,
	// the warp will process warp_size unsigned's from each of the bitmasks - one
	// container element per each thread in the warp -
	// thus decompressing sizeof(unsigned) * warp_size elements, which should
	// mean decompressing 1024 elements (into shared memory)

	for (promoted_size<bitmap_index_t> bitmap_index = 0; bitmap_index < num_bitmaps; bitmap_index++) {
		auto relevant_bitmap = bitmaps + aligned_bitmap_size_in_ints * bitmap_index;
		bit_container_t bit_container_element = relevant_bitmap[position_in_bitmaps];

		// TODO: Consider dropping all the conditions, looping over all 32 elements,
		// but ensuring we don't get bank conflicts by starting each threads at a
		// different index depending on sizeof(T) (consecutive indices for 32-bit
		// types, to ensure no conflicts)
		while (bit_container_element != 0) {
			auto lowest_bit_index = count_trailing_zeros(bit_container_element);
			thread_decompressed[lowest_bit_index] = bitmap_index;
			bit_container_element = clear_bit(bit_container_element, lowest_bit_index);
		}
	}
}

} // namespace thread_to_thread

} // namespace detail

// Note: There are at most 256 bitmaps, so if we use a dictionary, it is
// indexed by an uint8_t / unsigned char.
template<
	unsigned IndexSize, unsigned UncompressedSize, // the DictionaryIndex is unsigned char!
	unsigned BitmapAlignmentInInts,	bool UseDictionary = true>
__global__ void decompress(
	uint_t<UncompressedSize>*        __restrict__  decompressed,
	const bit_container_t*           __restrict__  bitmaps,
	const uint_t<UncompressedSize>*  __restrict__  dictionary_entries,
	uint_t<IndexSize>                              bitmap_length,
	unsigned char                                  num_bitmaps)
{
	enum { bits_per_container = size_in_bits<bit_container_t>::value };
	using index_type = uint_t<IndexSize>;
	using uncompressed_type = uint_t<UncompressedSize>;

	static_assert(warp_size == bits_per_container,
		"We assume warp size = number of bits in an unsigned int - but it isn't");

	// Note these areas in shared memory will hold indices into the range of
	// of bitmaps (i.e. values in the range {0..num_bitmaps - 1}, not
	// dictionary-translated uncompressed values - even if we _are_
	// going to use the dictionary eventually.
	bitmap_index_t* warp_decompressed =
		primitives::warp::get_warp_specific_shared_memory<unsigned char>(
			warp_size * bits_per_container);
	bitmap_index_t* thread_decompressed =
		warp_decompressed + bits_per_container * lane::index();
	uint_t<IndexSize> aligned_bitmap_length =
		round_up_to_multiple_of_power_of_2(bitmap_length, BitmapAlignmentInInts);

	const auto& f = [&](index_type pos) {

		if (pos < bitmap_length) {
			detail::thread::decompress_to_shared_mem_without_translation<IndexSize>(
				thread_decompressed, bitmaps, pos,
				aligned_bitmap_length, num_bitmaps);
		}

		// now the entire warp has decompressed (or part of the warp, if
		// we're at the end of the grid)

		// a shuffle is faster than the calculation, I think
		uint_t<IndexSize> warp_base_pos =
			//clear_lower_k_bits(pos, log_warp_size);
			primitives::warp::get_from_first_lane(pos);
		uncompressed_type* warp_output_start =
			decompressed + warp_base_pos * bits_per_container;
		auto num_threads_in_warp_which_decompressed =
			primitives::warp::vote_and_tally(pos < bitmap_length);
		unsigned warp_output_length =
			bits_per_container * num_threads_in_warp_which_decompressed;
		if (UseDictionary) {
			primitives::warp::lookup(
				warp_output_start, dictionary_entries,
				warp_decompressed, warp_output_length);
		}
		else {
			primitives::warp::cast_and_copy(
				warp_output_start, warp_decompressed, warp_output_length);
		}
	};

	primitives::grid::linear::at_block_stride(
		round_up_to_multiple_of_power_of_2(bitmap_length, (index_type) warp_size), f);

}

template<
	unsigned IndexSize, unsigned UncompressedSize, // the DictionaryIndex is unsigned char!
	unsigned BitmapAlignmentInInts,
	bool UseDictionary = true>
class launch_config_resolution_params_t final : public cuda::launch_config_resolution_params_t {
public:
	using parent = cuda::launch_config_resolution_params_t;
public:
	launch_config_resolution_params_t(
		device::properties_t            device_properties_,
		size_t                          uncompressed_length,
		optional<shared_memory_size_t>  dynamic_shared_mem_limit = nullopt) :
		cuda::launch_config_resolution_params_t(
			device_properties_,
			device_function_t(decompress<IndexSize, UncompressedSize, BitmapAlignmentInInts, UseDictionary>),
			dynamic_shared_mem_limit
		)
	{
		auto bitmap_length_in_containers =
			util::round_up(uncompressed_length, util::size_in_bits<unsigned>::value);
		// TODO:
		grid_construction_resolution            = thread;
		serialization_option                    = none;
		dynamic_shared_memory_requirement.per_length_unit =
			size_in_bits<bit_container_t>::value * sizeof(bitmap_index_t);
		length                                  = uncompressed_length;
		quanta.threads_in_block                 = warp_size;
	};
};

} // namespace incidence_bitmaps
} // namespace decompression
} // namespace kernels
} // namespace cuda

#endif /* SRC_KERNELS_DECOMPRESSION_INCIDENCE_BITMAPS_CUH */
