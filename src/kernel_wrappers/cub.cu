
#include "util/integer.h"

#include <cub/device/device_partition.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cuda/api/stream.hpp>
#include <cuda/functors.hpp>


namespace cub {
namespace device_partition {

template <typename SelectionOp, unsigned IndexSize>
size_t get_scratch_size(util::uint_t<IndexSize> num_items)
{
	using Datum = typename SelectionOp::result_type;
	using index_type = util::uint_t<IndexSize>;

	size_t scratch_size = 0;
	auto cub_primitive = &::cub::DevicePartition::If<const Datum*, Datum*, index_type*, SelectionOp>;
	auto status = cub_primitive(
		nullptr,               // d_temp_storage
		scratch_size,          // temp_storage_bytes,
		nullptr,               // d_in,
		nullptr,               // d_out,
		nullptr,               // d_num_selected_out,
		num_items,             // num_items,
		SelectionOp(),         // select_op,
		cuda::stream::default_stream_id,
		                       // stream             = 0  ; should not matter since nothing is launched
		false                  // debug_synchronous  = false  ; should not matter since nothing is launched)
	);
	cuda::throw_if_error(status, "cub::DevicePartition::If() (scratch size determination) failed");
	return scratch_size;
}

template <typename SelectionOp, unsigned IndexSize>
void partition(
	void*                                     device_scratch_area,
	size_t                                    scratch_size,
	typename SelectionOp::result_type const*  data,
	typename SelectionOp::result_type*        partitioned_data,
	util::uint_t<IndexSize>*                  num_selected,
	util::uint_t<IndexSize>                   num_items,
	cudaStream_t                              stream_id)
{
	using Datum = typename SelectionOp::result_type;
	using index_type = util::uint_t<IndexSize>;
	// auto cub_primitive = &::cub::DevicePartition::If<const Datum*, Datum*, index_type*, SelectionOp>;
	auto status = //cub_primitive(
		::cub::DevicePartition::If<const Datum*, Datum*, index_type*, SelectionOp>(
			device_scratch_area, // d_temp_storage
			scratch_size,        // temp_storage_bytes,
			data,                // d_in,
			partitioned_data,    // d_out,
			num_selected,        // d_num_selected_out,
			num_items,           // num_items,
			SelectionOp(),       // select_op,
			stream_id,              // stream             = 0,
			false                // debug_synchronous  = false)
		);
	cuda::throw_if_error(status, "cub::DevicePartition::If() failed");
};

namespace detail {


template <typename SelectionOp, unsigned IndexSize>
size_t instantiate() {
	auto ptr1 = &get_scratch_size<SelectionOp, IndexSize>;
	auto ptr2 = &partition<SelectionOp, IndexSize>;

	return reinterpret_cast<size_t>(ptr1) +
			reinterpret_cast<size_t>(ptr2);
}

size_t instantiate_all() {
	using namespace ::cuda::functors;
	return
		instantiate< is_non_negative<int>,    4 >() +
		instantiate< is_non_negative<float>,  4 >() +
		instantiate< is_non_negative<double>, 4 >() +
		instantiate< is_non_negative<int>,    8 >() +
		instantiate< is_non_negative<float>,  8 >() +
		instantiate< is_non_negative<double>, 8 >();
}

}


} // namespace device_partition

namespace radix_sort {

template <typename Datum>
size_t get_scratch_size(size_t num_items)
{
	size_t scratch_size = 0;
	auto status = ::cub::DeviceRadixSort::SortKeys<Datum>(
		nullptr,               // d_temp_storage
		scratch_size,          // temp_storage_bytes
		nullptr,               // d_keys_out
		nullptr,               // d_keys_in
		num_items              // num_items
	);
	cuda::throw_if_error(status,
		"cub::DeviceRadixSort::SortKeys() (scratch size determination) failed");

	return scratch_size;
}

template <typename Datum>
void sort(
	void*                       device_scratch_area,
	size_t                      scratch_size,
	Datum*        __restrict__  sorted_data,
	const Datum*  __restrict__  input_data,
	size_t                      num_items,
	cudaStream_t                stream_id,
	bool                        ascending)
{

	cuda::status_t status;
	status = ascending ?
		::cub::DeviceRadixSort::SortKeys<Datum>(
			device_scratch_area,       // d_temp_storage
			scratch_size,              // temp_storage_bytes
			(Datum*) input_data,       // d_keys_in
			sorted_data,               // d_keys_out
			num_items,                 // num_items
			0,                         // begin_bit
			(int) sizeof(Datum) * 8,   // end_bit
			stream_id,                 // stream
			false                      // debug_synchronous
		) :
		::cub::DeviceRadixSort::SortKeysDescending<Datum>(
			device_scratch_area,       // d_temp_storage
			scratch_size,              // temp_storage_bytes
			(Datum*) input_data,       // d_keys_in
			sorted_data,               // d_keys_out
			num_items,                 // num_items
			0,                         // begin_bit
			(int) sizeof(Datum) * 8,   // end_bit
			stream_id,                 // stream
			false                      // debug_synchronous
		);
	cuda::throw_if_error(status, "cub::RadixSort::SortKeys() failed");
};

namespace detail {


template <typename Datum>
size_t instantiate() {
	auto ptr1 = &get_scratch_size<Datum>;
	auto ptr2 = &sort<Datum>;

	return
		reinterpret_cast<size_t>(ptr1) +
		reinterpret_cast<size_t>(ptr2);
}

size_t instantiate_all() {
	using namespace ::cuda::functors;
	return
		instantiate< char               >() +
		instantiate< unsigned char      >() +
		instantiate< short              >() +
		instantiate< unsigned short     >() +
		instantiate< int                >() +
		instantiate< unsigned int       >() +
		instantiate< long               >() +
		instantiate< unsigned long      >() +
		instantiate< long long          >() +
		instantiate< unsigned long long >() +

		0;
}

} // namespace detail

} // namespace radix_sort
} // namespace cub
