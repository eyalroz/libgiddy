#include "cuda/api_wrappers.h"
#include "cuda/model_functors.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <algorithm>
#include <numeric>

template <typename T>
constexpr inline T div_rounding_up(const T& dividend, const T& divisor) {
	return (dividend + divisor - 1) / divisor;
}

using index_type        = unsigned;
using uncompressed_type = int;
using compressed_type   = short;
using model_type = cuda::functors::unary::parametric_model::constant<sizeof(index_type), uncompressed_type>;
using model_coefficients_type = typename model_type::coefficients_type;

void decompress_on_device(
	uncompressed_type*              __restrict__  decompressed,
	const compressed_type*          __restrict__  compressed,
	const model_coefficients_type*  __restrict__  segment_model_coefficients,
	index_type                                    length,
	index_type                                    segment_length);

int main(void)
{
	size_t length =  2 << 12;
	size_t segment_length = 2 << 10;
	size_t num_segments = div_rounding_up(length, segment_length);
	enum : uncompressed_type { amplitude = 2 << (sizeof(uncompressed_type) * (CHAR_BIT - 2)) };

	auto h_decompressed = std::unique_ptr<uncompressed_type>(new uncompressed_type[length]);
	auto h_compressed = std::unique_ptr<compressed_type>(new compressed_type[length]);
	auto h_model_coefficients = std::unique_ptr<model_coefficients_type>(new model_coefficients_type[num_segments]);


	for(size_t segment_index = 0; segment_index < num_segments; segment_index ++) {
		uncompressed_type segment_reference = segment_index * 100;
		h_model_coefficients.get()[segment_index] = { segment_reference };
			// it's an array of size one - the coefficients of
			// a degree-0 polynomial, i.e. a constant function

		auto segment_compressed_start = h_compressed.get() + segment_index * segment_length;
		auto segment_compressed_end = segment_compressed_start + segment_length;
		std::iota(segment_compressed_start, segment_compressed_end, 0);
	}

	// Let sl = segment_length. At this point, the compressed data is repeating
	// rising sequence:
	//
	//     |
	// sl-1|    *    *     *     *
	//     |   *    *     *     *
	//     |  *    *     *     *
	//     | *    *     *     *
	//     |*    *     *     *
	//     *-------------------------------
	//     0    sl   2*sl  3*sl
	//
	// but with the added reference values, each rise is offset by 100, so we get:
	//
	//     |
	//     |                      *
	//     |                *    *
	//     |          *    *    *
	// sl-1|    *    *    *    *
	// ... |   *    *    *    *
	// 300 |  *    *    *    *
	// 200 | *    *    *
	// 100 |*    *
	//   0 *-------------------------------
	//     0    sl   2*sl  3*sl
	//

	auto current_device       = cuda::device::current::get();
	auto d_decompressed       = cuda::memory::device::make_unique<uncompressed_type[]>(current_device, length);
	auto d_compressed         = cuda::memory::device::make_unique<compressed_type[]>(current_device, length);
	auto d_model_coefficients = cuda::memory::device::make_unique<model_coefficients_type[]>(current_device, length);

	cuda::memory::copy(d_compressed.get(), h_compressed.get(), length * sizeof(*d_compressed.get()));
	cuda::memory::copy(d_model_coefficients.get(), h_model_coefficients.get(), length * sizeof(*d_model_coefficients.get()));

	decompress_on_device(d_decompressed.get(), d_compressed.get(), d_model_coefficients.get(), length, segment_length);

	cuda::memory::copy(h_decompressed.get(), d_decompressed.get(), length * sizeof(*d_decompressed.get()));

	for(size_t segment_index = 0; segment_index < num_segments - 1; segment_index ++) {
		auto segment_reference = h_model_coefficients.get()[segment_index][0];
		auto segment_decompressed_start = h_decompressed.get() + segment_index * segment_length;
		for(size_t i = 0; i < segment_length; i++) {
			if (segment_decompressed_start[i] != (uncompressed_type) (segment_reference + i) ) {
				std::cerr << "Unexpected value at index " << i + (segment_index * segment_length) << " ( segment "
					<< segment_index << ", offset " << i << "): Got " << segment_decompressed_start[i]
					<< " instead of " << segment_reference + i << "." << std::endl;
				return EXIT_FAILURE;
			}
		}
	}

	return EXIT_SUCCESS;
}
