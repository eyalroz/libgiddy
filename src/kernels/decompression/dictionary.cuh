#pragma once
#ifndef SRC_KERNELS_DECOMPRESSION_DICTIONARY_CUH
#define SRC_KERNELS_DECOMPRESSION_DICTIONARY_CUH


#include "kernels/common.cuh"
#include "kernels/data_layout/gather.cuh"

namespace cuda {
namespace kernels {
namespace decompression {
namespace dictionary {


/*
 *  TODO:
 *  - Consider whether we want to support variable-length dictionaries here already
 *    (might be a good idea if we think of strings
 *  - Unify decompress_small_dict and decompress_large_dict
 *
 */

template <typename T, typename Index>
struct Dictionary {
	using index_type = Index;
	using entry_type = T;
		// That's very simply. It could have been something else, e.g. for a string -
		// an std::array of characters and the actual length of the string in case
		// it's shorter
public:
	const Index     num_entries;
	const T* const  entries;
};



/**
 * Decompress data which was compressed by dropping off several (all-zero) bytes (either
 * from the beginning or from the end of each data element).
 *
 * @note Endianness matters. Specifically, the endinanness of the GPU and the CPU may
 * be different!
 *
 * @param[out] decompressed The data with the zeros at one end
 * @param[in] compressed_input The data without the zeros
 * @param[in] length Number of elements in each of the input and output arrays
 */

/**
 * Decompress data compressed with a Dictionary, i.e. in which every original element
 * was replaced with an index into a dictionary, with its value appearing at that index
 * (upto casting).
 *
 * @note gather, like this function, should distinguish between the cases where
 * the dictionary can be stored entirely in a block's shared memory, entirely in the
 * shared memory of several blocks (maybe), and the large-dictionary case
 *
 * @todo Consider passing an actual dictionary struct
 *
 * @tparam Uncompressed The type of the uncompressed array to produce. Ostensibly,
 * this should not be necessary, since the dictionary gives us some typed value already,
 * but we opt to integrate a potential cast (e.g. in case the dictionary produces
 * int's and we want long int's)
 * @tparam Dictionary the kind of dictionary this is; at the moment, this is only
 * used to determine dictionary-related types
 * @param[out] decompressed The data with the zeros at one end
 * @param[in] compressed_input The data without the zeros
 * @param[in] dictionary_entries the 'raw' dictionary, i.e. for every allowed index,
 * a value in the original uncompressed type (up to casting)
 * @param[in] length Number of elements in each of the input and output arrays
 * (it's the same, since the DICT compression mode is one-to-one)
 * @param[in] num_dictionary_entries Number of entries in the dictionary, which is
 * also the length of the dictionary_entries array
 */
template<
	unsigned IndexSize, unsigned UncompressedSize, unsigned DictionaryIndexSize
> __global__ void decompress(
	uint_t<UncompressedSize>*           __restrict__  decompressed,
	const uint_t<DictionaryIndexSize>*  __restrict__  compressed_input,
	const uint_t<UncompressedSize>*     __restrict__  dictionary_entries,
	uint_t<IndexSize>                                 length, // of compressed and uncompressed data - it's the same value
	size_t                                                  num_dictionary_entries,
	bool                                                    cache_dictionary_in_shared_memory,
	serialization_factor_t                                  serialization_factor)
{
	using dictionary_index_type = uint_t<DictionaryIndexSize>;
	using uncompressed_type = uint_t<UncompressedSize>;

	gather::detail::gather<IndexSize, UncompressedSize, DictionaryIndexSize>(
		decompressed, dictionary_entries, compressed_input, num_dictionary_entries, length,
		cache_dictionary_in_shared_memory, serialization_factor);
}

template<unsigned IndexSize, unsigned UncompressedSize, unsigned DictionaryIndexSize>
using launch_config_resolution_params_t =
	gather::launch_config_resolution_params_t<IndexSize, UncompressedSize, DictionaryIndexSize>;

} // namespace dictionary
} // namespace decompression
} // namespace kernels
} // namespace cuda

#endif /* SRC_KERNELS_DECOMPRESSION_DICTIONARY_CUH */
