# Giddy <br> A lightweight GPU decompression library

Described in [this mini-paper](https://www.researchgate.net/publication/315834231_Faster_across_the_PCIe_bus_A_GPU_library_for_lightweight_decompression) in the [DaMoN 2017](http://daslab.seas.harvard.edu/damon2017/) workshop.

* [Why lightweight compression for GPU work?](#why)
* [What does this library comprise?](#what)
* [Which compression schemes are supported?](#which)
* [Examples of use](#examples)
* [Performance](#performance)

For questions, requests or bug reports - either use the [Issues page](https://github.com/eyalroz/libgiddy/issues) or [email me](mailto:eyalroz@technion.ac.il).


## <a name="why">Why lightweight compression for GPU work?</a>


Discrete GPUs are powerful beasts, with numerous cores and high-bandwidth memory, often capable of 10x the throughput in crunching data relative to the maximum achievable on a CPU. Perhaps the main obstacle, however, to utilizing them, is that data usually resides in main system memory, close to the CPU - and to have the GPU process it, we must send it over a PCIe bus. Thus the CPU has a potential of processing in-memory data at (typically in 2017) 30-35 GB/sec, and a discrete GPU at no more than 12 GB/sec.

One way of counteracting this handicap is using compression. The GPU can afford expending more effort in decompressing data arriving over the bus than the CPU; thus if we the data is available apriori in system memory, and is amenable to compression, using it may increase the GPU's effective bandwidth more than it would the CPU's.

Compression schemes come in many shapes and sizes, but it is customary to distinguish "heavy-weight" schemes (such as those based on [Lempel-Ziv](https://en.wikipedia.org/wiki/LZ77_and_LZ78)) from "lightweight" schemes, involving small amounts of computation per element, few accesses to the compressed data for decompressing any single element.

This library 

## <a name="what">What does this library comprise?</a>

This is essentially a library of GPU kernels and logic for configuring their launch (grid dimensions, block dimensions, dynamic shared memory size). The kernels can be used in one of three ways, described below. Let's suppose we want to use compression scheme "Foo"

1. Use of the kernel code itself. In this case you would need to take care of the launch configuration

   * include the relevant kernel file itself: `src/kernels/decompression/foo.cuh`;
   * include `src/kernels/resolve_launch_configuration.h`;
   * instantiate a launch configuration resultion parameters object (the class ould be `cuda::kernels::decompression::foo::launch_config_resolution_params_t`,  defined in the `foo.cuh` file);
   * call `resolve_launch_configuration()` function with the object you instantiated;
   * use the resulting `launch_configuration_t` for a launch - either immediately (with the [C++'ish CUDA API wrappers](https://github.com/eyalroz/cuda-api-wrappers/)), or by using its data members in a plain vanilla CUDA kernel launch statement.

2. Use of kernel wrapper objects, with a class corresponding to each scheme. These wrappers are pretty thin, and do not perform any scheduling or resource allocation; they're just intended to provide a uniform interface for launching. This uniformity is achieved by type-erasure: The wrappers' launch method takes a map of [`boost::any`](http://www.boost.org/doc/libs/1_63_0/doc/html/any.html) objects. It is up to the client code to pass the appropriate parameters in that map. e-specific methods).

   * Include the kernel's wrapper class definition, in `src/kernel_wrappers/decompression/foo.cu`
   * Instantiate `cuda::kernels::decompression::foo::kernel_t`
   * Create an std::unordered_map of the launch arguments
   * Use the instance's `launch()` method

3. Use of the kernel wrappers via a factory. Another feature of the wrapper class is being factory-producible: Each compression scheme's wrapper class registers several available template instantiations of itself in a factory, at program load time; these can then be retrieved without actually including any of the compression-scheme-specific code - which means, in particular, not including any CUDA-specific code that requires nvcc to compile.

   To use the wrapper for class :

   * Include [`src/kernel_wrappers/registered_wrappers.h`](https://github.com/eyalroz/libgiddy/blob/master/src/kernel_wrappers/registered_wrapper.h)
   * Use the `cuda::registered::kernel_t` class' static methods - `listSubclasses()` and `produceSubclass()` - to instantiate specific scheme wrappers. The instantiated wrappers don't have any state; the instantiation is necessary simply for their vtables (i.e. to be able to call the compression-scheme-specific methods).

## <a name="which">Supported compression schemes</a>

The following compression schemes are currently included:

* Delta
* Dictionary
* DiscardZeroBytesFixed
* DiscardZeroBytesVariable
* FrameOfReference
* IncidenceBitmaps
* Model
* RunLengthEncoding
* RunPositionEncoding

*(A specification of each of these, its semantics and exact parameters is forthcoming; for now, please consult the sources themselves.)*

Additionally, two patching schemes are supported: Naive patching and Compressed-Indices patching. As these are "aposteriori" patching schemes, you apply them by simply decompressing using some base scheme, then using one of the two kernels `data_layout::scatter` or `data_layout::compressed_indices_scatter` on the initial decompression result. You will not find specific kernels, kernel wrappers or factory entries for the "combined" patched scheme, only for its components.

## <a name="why">Exaples of use</a>

Example code is forthcoming. For now, you can find a rather complicated example in the form of the test harness used to develop these kernels, available in [this repository](https://bitbucket.org/eyalroz/db-kernel-testbench/) on BitBucket.

## <a name="performance">Performance</a>

Some of the decompressors are well-optimized, some need more work. The most recent (and only) performance analysis is in the  [mini-paper](https://www.researchgate.net/publication/315834231_Faster_across_the_PCIe_bus_A_GPU_library_for_lightweight_decompression) mentioned above. *Step-by-step instructions for measuring performance (using well-known data sets) are forthcoming.*

## <a name="credits">Credits</a>

This endevor was made possible by:

* [CWI Amsterdam](http://www.cwi.nl/) - the research institute where I'm employed
* [Prof. Peter Boncz](http://homepages.cwi.nl/~boncz/), co-author of the above-mentioned paper
* The [MonetDB](http://www.monetdb.org/) DBMS project - which got me into DBMSes and GPUs in the first place, and which I (partially) use for performance testing

... do check all of these out.
