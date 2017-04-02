
#include "kernel_wrappers/common.h"
#ifdef __CUDACC__
#include "kernels/data_layout/compressed_indices_scatter.cuh"
#endif

namespace cuda {
namespace kernels {
namespace scatter {
namespace compressed_indices {

template<unsigned OutputIndexSize, unsigned ElementSize, unsigned InputIndexSize, unsigned RunLengthSize = InputIndexSize>
class kernel: public cuda::registered::kernel_t {
public:
	REGISTERED_KERNEL_WRAPPER_BOILERPLATE_DEFINITIONS(kernel);

	using element_type      = util::uint_t<ElementSize>;
	using input_index_type  = util::uint_t<InputIndexSize>;
	using output_index_type = util::uint_t<OutputIndexSize>;
	using run_length_type   = util::uint_t<RunLengthSize>;

};

#ifdef __CUDACC__

template<unsigned OutputIndexSize, unsigned ElementSize, unsigned InputIndexSize, unsigned RunLengthSize>
launch_configuration_t kernel<OutputIndexSize, ElementSize, InputIndexSize, RunLengthSize>::resolve_launch_configuration(
	device::properties_t             device_properties,
	device_function::attributes_t    kernel_function_attributes,
	arguments_type                   extra_arguments,
	launch_configuration_limits_t    limits) const
{
	namespace kernel_ns = cuda::kernels::scatter::compressed_indices;

	auto input_data_length = any_cast<size_t>(extra_arguments.at("input_data_length"));
	auto anchoring_period  = any_cast<size_t>(extra_arguments.at("anchoring_period"));

	if (input_data_length == 0) {
		throw std::invalid_argument("Zero-length scatters not currently supported");
	}

	kernel_ns::launch_config_resolution_params_t<OutputIndexSize, ElementSize, InputIndexSize, RunLengthSize> params(
		device_properties, input_data_length, anchoring_period);

	return cuda::resolve_launch_configuration(params, limits);
}

template<unsigned OutputIndexSize, unsigned ElementSize, unsigned InputIndexSize, unsigned RunLengthSize>
void kernel<OutputIndexSize, ElementSize, InputIndexSize, RunLengthSize>::launch(
	stream::id_t                      stream,
	const launch_configuration_t&    launch_config,
	arguments_type                   arguments) const
{
	if (launch_config.grid_dimensions == 0) {
		// No patches, so nothing to do
		// TODO: Is this reasonable behavior? Or should we expect not to receive empty grids?
		return;
	}


	auto target                                       = any_cast<element_type*            >(arguments.at("target"                                      ));
	auto data_to_scatter                              = any_cast<const element_type*      >(arguments.at("data_to_scatter"                             ));
	auto scatter_position_run_lengths                 = any_cast<const run_length_type*   >(arguments.at("scatter_position_run_lengths"                ));
	auto scatter_position_run_individual_offset_sizes = any_cast<const unsigned char*     >(arguments.at("scatter_position_run_individual_offset_sizes"));
	auto scatter_position_run_baseline_values         = any_cast<const output_index_type* >(arguments.at("scatter_position_run_baseline_values"        ));
	auto scatter_position_run_offsets_start_pos       = any_cast<const input_index_type*  >(arguments.at("scatter_position_run_offsets_start_pos"      ));
	auto scatter_position_offset_bytes                = any_cast<const unsigned char*     >(arguments.at("scatter_position_offset_bytes"               ));
	auto scatter_position_anchors                     = any_cast<const input_index_type*  >(arguments.at("scatter_position_anchors"                    ));
	auto anchoring_period                             = any_cast<input_index_type         >(arguments.at("anchoring_period"                            ));
	auto num_scatter_position_runs                    = any_cast<input_index_type         >(arguments.at("num_scatter_position_runs"                   ));
	auto input_data_length                            = any_cast<size_t                   >(arguments.at("input_data_length"                           ));

	cuda::enqueue_launch(
		cuda::kernels::scatter::compressed_indices::scatter<OutputIndexSize, ElementSize, InputIndexSize, RunLengthSize>,
		launch_config, stream,
		target,
		data_to_scatter,
		scatter_position_run_lengths,
		scatter_position_run_individual_offset_sizes,
		scatter_position_run_baseline_values,
		scatter_position_run_offsets_start_pos,
		scatter_position_offset_bytes,
		scatter_position_anchors,
		anchoring_period,
		num_scatter_position_runs,
		input_data_length
	);
}

template<unsigned OutputIndexSize, unsigned ElementSize, unsigned InputIndexSize, unsigned RunLengthSize>
const cuda::device_function_t kernel<OutputIndexSize, ElementSize, InputIndexSize, RunLengthSize>::get_device_function() const
{
	return {
		cuda::kernels::scatter::compressed_indices::scatter
			<OutputIndexSize, ElementSize, InputIndexSize, RunLengthSize>
	};
}


static_block {
	//      OutputIndexSize  ElementSize  InputIndexSize
	//-----------------------------------------------------------------------
	kernel< 4,               1,           1 >::registerInSubclassFactory();
	kernel< 4,               1,           2 >::registerInSubclassFactory();
	kernel< 4,               1,           4 >::registerInSubclassFactory();
	kernel< 4,               1,           8 >::registerInSubclassFactory();
	kernel< 4,               2,           1 >::registerInSubclassFactory();
	kernel< 4,               2,           2 >::registerInSubclassFactory();
	kernel< 4,               2,           4 >::registerInSubclassFactory();
	kernel< 4,               2,           8 >::registerInSubclassFactory();
	kernel< 4,               4,           1 >::registerInSubclassFactory();
	kernel< 4,               4,           2 >::registerInSubclassFactory();
	kernel< 4,               4,           4 >::registerInSubclassFactory();
	kernel< 4,               4,           8 >::registerInSubclassFactory();
	kernel< 4,               8,           1 >::registerInSubclassFactory();
	kernel< 4,               8,           2 >::registerInSubclassFactory();
	kernel< 4,               8,           4 >::registerInSubclassFactory();
	kernel< 4,               8,           8 >::registerInSubclassFactory();

	kernel< 8,               1,           1 >::registerInSubclassFactory();
	kernel< 8,               1,           2 >::registerInSubclassFactory();
	kernel< 8,               1,           4 >::registerInSubclassFactory();
	kernel< 8,               1,           8 >::registerInSubclassFactory();
	kernel< 8,               2,           1 >::registerInSubclassFactory();
	kernel< 8,               2,           2 >::registerInSubclassFactory();
	kernel< 8,               2,           4 >::registerInSubclassFactory();
	kernel< 8,               2,           8 >::registerInSubclassFactory();
	kernel< 8,               4,           1 >::registerInSubclassFactory();
	kernel< 8,               4,           2 >::registerInSubclassFactory();
	kernel< 8,               4,           4 >::registerInSubclassFactory();
	kernel< 8,               4,           8 >::registerInSubclassFactory();
	kernel< 8,               8,           1 >::registerInSubclassFactory();
	kernel< 8,               8,           2 >::registerInSubclassFactory();
	kernel< 8,               8,           4 >::registerInSubclassFactory();
	kernel< 8,               8,           8 >::registerInSubclassFactory();

}

#endif /* __CUDACC__ */

} // namespace compressed_indices
} // namespace scatter
} // namespace kernels
} // namespace cuda
