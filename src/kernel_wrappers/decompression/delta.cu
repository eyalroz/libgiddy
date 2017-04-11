
#include "kernel_wrappers/common.h"
#ifdef __CUDACC__
#include "kernels/decompression/delta.cuh"
#endif

namespace cuda {
namespace kernels {
namespace decompression {
namespace delta {

// TODO: This currently ignores the possibility of a sorted variant of the kernel

template<unsigned IndexSize, unsigned UncompressedSize, unsigned CompressedSize>
class kernel: public cuda::registered::kernel_t {
public:
	REGISTERED_KERNEL_WRAPPER_BOILERPLATE_DEFINITIONS(kernel);

	using compressed_type   = util::uint_t<CompressedSize>;
	using uncompressed_type = util::uint_t<UncompressedSize>;
};

#ifdef __CUDACC__

template<unsigned IndexSize, unsigned UncompressedSize, unsigned CompressedSize>
inline launch_configuration_t kernel<IndexSize, UncompressedSize, CompressedSize>::resolve_launch_configuration(
		device::properties_t           device_properties,
		device_function::attributes_t  kernel_function_attributes,
		arguments_type                 extra_arguments,
		launch_configuration_limits_t  limits) const
{
	namespace kernel_ns = cuda::kernels::decompression::delta;

	auto length            = any_cast<size_t>(extra_arguments.at("length"));
	auto baselining_period = any_cast<size_t>(extra_arguments.at("baselining_period"));

	kernel_ns::launch_config_resolution_params_t<IndexSize, UncompressedSize, CompressedSize> params(
		device_properties,
		length, baselining_period,
		limits.dynamic_shared_memory);

	return cuda::kernels::resolve_launch_configuration(params, limits);
}

template<unsigned IndexSize, unsigned UncompressedSize, unsigned CompressedSize>
inline void kernel<IndexSize, UncompressedSize, CompressedSize>::enqueue_launch(
		stream::id_t                   stream,
		const launch_configuration_t&  launch_config,
		arguments_type                 arguments) const
{
	using compressed_type   = util::uint_t<CompressedSize>;
	using uncompressed_type = util::uint_t<UncompressedSize>;

	auto decompressed      = any_cast<uncompressed_type*       >(arguments.at("decompressed"      ));
	auto compressed_input  = any_cast<const compressed_type*   >(arguments.at("compressed_input"  ));
	auto baseline_values   = any_cast<const uncompressed_type* >(arguments.at("baseline_values"   ));
	auto length            = any_cast<util::uint_t<IndexSize>  >(arguments.at("length"            ));
	auto baselining_period = any_cast<util::uint_t<IndexSize>  >(arguments.at("baselining_period" ));

	cuda::kernel::enqueue_launch(
		*this, stream, launch_config,
		decompressed, compressed_input, baseline_values, length, baselining_period
	);
}

template<unsigned IndexSize, unsigned UncompressedSize, unsigned CompressedSize>
inline const device_function_t kernel<IndexSize, UncompressedSize, CompressedSize>::get_device_function() const
{
	return cuda::kernels::decompression::delta::decompress<IndexSize, UncompressedSize, CompressedSize>;
}

static_block {
	//       IndexSize   UncompressedSize   CompressedSize
	// ----------------------------------------------------
	kernel < 4,          2,                 1 >::registerInSubclassFactory();
	kernel < 4,          4,                 1 >::registerInSubclassFactory();
	kernel < 4,          4,                 2 >::registerInSubclassFactory();
	kernel < 4,          8,                 1 >::registerInSubclassFactory();
	kernel < 4,          8,                 2 >::registerInSubclassFactory();
	kernel < 4,          8,                 4 >::registerInSubclassFactory();

	kernel < 8,          2,                 1 >::registerInSubclassFactory();
	kernel < 8,          4,                 1 >::registerInSubclassFactory();
	kernel < 8,          4,                 2 >::registerInSubclassFactory();
	kernel < 8,          8,                 1 >::registerInSubclassFactory();
	kernel < 8,          8,                 2 >::registerInSubclassFactory();
	kernel < 8,          8,                 4 >::registerInSubclassFactory();

}

#endif /* __CUDACC__ */

} // namespace delta
} // namespace decompression
} // namespace kernels
} // namespace cuda
