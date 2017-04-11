#pragma once
#ifndef SRC_KERNELS_DECOMPRESSION_DISCARD_ZERO_BYTES_FIXED_CUH
#define SRC_KERNELS_DECOMPRESSION_DISCARD_ZERO_BYTES_FIXED_CUH


#include "kernels/common.cuh"
#include "cuda/api/constants.h"
// #include "compression.h"
#include "cuda/on_device/primitives/grid.cuh"
#include "cuda/stl_array.cuh"
#include "kernels/elementwise/unary_operation.cuh"

namespace cuda {
namespace kernels {
namespace decompression {
namespace discard_zero_bytes {

/**
 * The bytes cut from the uncompressed data are
 * the higher one (even though theoretically that could be the
 * wrong choice, but that's not too likely). Still, the question
 * is - which bytes in the result should we write to? Which are
 * considered the low bytes? Let's make the endianness
 * a template parameter (and maybe move this enum up)
 *
 * @note For now we assume the compressor has, at least, arranged
 * the low bytes to avoid having to switch their order on
 * decompression; that might mean more compression work, or
 * alternatively adding the feature of byte reordering here
 *
 * @todo take the endianness value from someplace when CUDA
 * exposes it.
 */

namespace fixed_width {

enum { DefaultSerializationFactor = 32 };

/*******************************************/

template<unsigned UncompressedSize, unsigned CompressedSize, cuda::endianness_t OutputEndianness>
struct pad_high_bytes:
	public std::unary_function<uint_t<UncompressedSize>, util::uint_t<CompressedSize>>
{
	using argument_type = uint_t<CompressedSize>;
	using result_type = uint_t<UncompressedSize>;

	static_assert(UncompressedSize > CompressedSize,
		"Uncompressed size isn't higher than Compressed size");

	enum { pad_length = UncompressedSize - CompressedSize };


	__host__ __device__ __forceinline__
	result_type operator()(const argument_type& x) const {
		result_type result;
		if (OutputEndianness == endianness_t::little) {
			// Copying to the beginning, padding the trailing
			auto& low_bytes = reinterpret_cast<argument_type&>(result);
			// TODO: Perhaps adopt alias.cuh 's "working type" mechanism?
			// that should ensure optimization for 2 byte and 4 byte
			// assignments
			auto& high_bytes = reinterpret_cast<cuda::array<char,pad_length>&>((&low_bytes)[1]);
			low_bytes = x;
			high_bytes.fill(0);
		}
		else {
			// Copying to the trailing, padding the beginning
			// TODO: Perhaps adopt alias.cuh 's "working type" mechanism?
			auto& low_bytes = reinterpret_cast<cuda::array<char,pad_length>&>(result);
			auto& high_bytes = reinterpret_cast<argument_type&>((&low_bytes)[1]);
			low_bytes.fill(0);
			high_bytes = x;
		}
		return result;
	}
};

// TODO: Specialize the above template for known types; and perhaps don't always
// call it with sized_anonymous and only cast it thus in the default version


using namespace grid_info::linear;

/*
template<unsigned SourceSize, unsigned TargetSize, serialization_factor_t SerializationFactor = DefaultSerialization>
using drop_leading_zero_bytes = elementwise::unary::unary_operation<cuda::drop_leading_bytes<SourceSize, TargetSize>, SerializationFactor>;

template<unsigned SourceSize, unsigned TargetSize, serialization_factor_t SerializationFactor = DefaultSerialization>
using drop_trailing_zero_bytes = elementwise::unary::unary_operation<cuda::drop_trailing_bytes<SourceSize, TargetSize>, SerializationFactor>;
*/


/**
 * Decompress data which was compressed by dropping off several (all-zero) bytes (either
 * from the beginning or from the end of each data element).
 *
 * @note Endianness matters. Specifically, the endinanness of the GPU and the CPU may
 * be different!
 *
 * @tparam ZerosAtWhichEnd determines whether the zero bytes should be
 * added before the actual data bytes or after them. This should be set according to
 * the relevant machine's endianness
 * @param[out] decompressed The data with the zeros at one end
 * @param[in] compressed_input The data without the zeros
 * @param[in] length Number of elements in each of the input and output arrays
 */
template<
	unsigned IndexSize, unsigned UncompressedSize, unsigned CompressedSize,
	cuda::endianness_t UncompressedEndianness  = cuda::compilation_target_endianness,
	serialization_factor_t SerializationFactor = DefaultSerializationFactor>
__global__ void decompress(
	uint_t<UncompressedSize>*      __restrict__  decompressed,
	const uint_t<CompressedSize>*  __restrict__  compressed_input,
	uint_t<IndexSize>                            length)
{
	using index_type = uint_t<IndexSize>;
	auto f = [&decompressed, &compressed_input](index_type pos) {
		decompressed[pos] =
			pad_high_bytes<UncompressedSize, CompressedSize, UncompressedEndianness>()(
				compressed_input[pos]
			);
	};
	primitives::grid::linear::at_block_stride(length, f, SerializationFactor);
}

template<
	unsigned IndexSize, unsigned UncompressedSize, unsigned CompressedSize,
	cuda::endianness_t UncompressedEndianness  = cuda::compilation_target_endianness,
	serialization_factor_t SerializationFactor = DefaultSerializationFactor>
class launch_config_resolution_params_t final : public kernels::launch_config_resolution_params_t {
public:
	using parent = kernels::launch_config_resolution_params_t;
public:
	launch_config_resolution_params_t(
		device::properties_t            device_properties_,
		size_t                          length_)
		:
		parent(
			device_properties_,
			device_function_t(decompress<
				IndexSize, UncompressedSize, CompressedSize,
				UncompressedEndianness, SerializationFactor>)
		)
	{
		grid_construction_resolution            = thread;
		length                                  = length_;
		serialization_option                    = fixed_factor;
		default_serialization_factor            = DefaultSerializationFactor;
	};
};

} // namespace fixed_width
} // namespace discard_zero_bytes
} // namespace decompression
} // namespace kernels
} // namespace cuda

#endif /* SRC_KERNELS_DECOMPRESSION_DISCARD_ZERO_BYTES_FIXED_CUH */
