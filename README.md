# Giddy - A GPU lightweight decompression library

(presented in [this mini-paper](https://www.researchgate.net/publication/315834231_Faster_across_the_PCIe_bus_A_GPU_library_for_lightweight_decompression) in the [DaMoN 2017](http://daslab.seas.harvard.edu/damon2017/) workshop.)

| Table of contents|
|:----------------|
|- [Why lightweight compression for GPU work?](#why)<br>- [What does this library comprise?](#what)<br>- [Which compression schemes are supported?](#which)<br>- [How to decompress data using Giddy?](#examples)<br>- [Performance](#performance)<br>- [Acknowledgements](#acks)|


For questions, requests or bug reports - either use the [Issues page](https://github.com/eyalroz/libgiddy/issues) or [email me](mailto:eyalroz@technion.ac.il).


## <a name="why">Why lightweight compression for GPU work?</a>


Discrete GPUs are powerful beasts, with numerous cores and high-bandwidth memory, often capable of 10x the throughput in crunching data relative to the maximum achievable on a CPU. Perhaps the main obstacle, however, to utilizing them, is that data usually resides in main system memory, close to the CPU - and to have the GPU process it, we must send it over a PCIe bus. Thus the CPU has a potential of processing in-memory data at (typically in 2017) 30-35 GB/sec, and a discrete GPU at no more than 12 GB/sec.

One way of counteracting this handicap is using compression. The GPU can afford expending more effort in decompressing data arriving over the bus than the CPU; thus if we the data is available apriori in system memory, and is amenable to compression, using it may increase the GPU's effective bandwidth more than it would the CPU's.

Compression schemes come in many shapes and sizes, but it is customary to distinguish "heavy-weight" schemes (such as those based on [Lempel-Ziv](https://en.wikipedia.org/wiki/LZ77_and_LZ78)) from "lightweight" schemes, involving small amounts of computation per element, few accesses to the compressed data for decompressing any single element.

Giddy enables the use of lightweight compressed data on the GPU by providing decompressor implementations for a plethora of compression schemes.

## <a name="what">What does the library comprise?</a>

Giddy comprises:

* **CUDA kernel source code** for decompressing data using each of the compression schemes listed [below](#which). The kernels are **templated**, and one may instantiate them for a **variety of combinations of types** and some compression scheme parameters which it would not be efficient to pass at run-time. 
    * ... and source code for auxiliary kernels required for decompression (e.g. for the scattering of patch data).
* A uniform **mechanism for configuring launches** of these kernels (grid dimensions, block dimensions and dynamic shared memory size). 
* A kernel wrapper abstraction class --- which is not specific to decompression work, but rather general --- and individual **kernel wrappers** for each decompression scheme (templated similarly to the kernels themselves). Instead of dealing directly with the kernels at the lower level, making CUDA API calls yourself, you can instead use the associated wrapper.
* The kernel wrapper class also registers itself in a **factory**, which you can use **for instantiate wrappers without having compiled against their code**. The factory provides us with instances of a common base class - and their virtual methods are used to pass scheme-specific arguments.

If this sounds a bit confusing, scroll down to the [examples](#examples) section.

## <a name="which">Supported compression schemes</a>

The following compression schemes are currently included:

* Delta
* Dictionary
* Null Suppression (Discard Zero-Bytes):
   * Fixed width
   * Variable width
* (Generalized) Frame of Reference
* Incidence Bitmaps
* Model
* Run Length Encoding
* Run Position Encoding

*(A specification of each of these, its semantics and exact parameters is forthcoming; for now, please consult the sources themselves.)*

Additionally, two patching schemes are supported: 

* Naive patching 
* Compressed-Indices patching

As these are "aposteriori" patching schemes, you apply them by simply decompressing using some base scheme, then using one of the two kernels `data_layout::scatter` or `data_layout::compressed_indices_scatter` on the initial decompression result. You will not find specific kernels, kernel wrappers or factory entries for the "combined" patched scheme, only for its components.

## <a name="examples">How to decompress data using Giddy?</a>

*Note: The examples use the  [C++'ish CUDA API wrappers](https://github.com/eyalroz/cuda-api-wrappers/)), making the host-side code somewhat clearer and shorter.*

Suppose we're presented with compressed data with the following characteristics, which for simplicity is already in GPU memory:

| Parameter                      | Value               |
|:-------------------------------|:--------------------|
|Decompression scheme            | Frame of Reference  |
|width of size/index type        | 32 bits             |
|Uncompressed data type          | uncompressed_type   |
|type of offsets from FOR value  | compressed_type     |
|segment length                  | (runtime variable)  |
|total length of compressed data | (runtime variable)  |

in other words, we want to implement the following function:

```
void decompress_on_device(
	uncompressed_type*              __restrict__  decompressed,
	const compressed_type*          __restrict__  compressed,
	const model_coefficients_type*  __restrict__  segment_model_coefficients,
	index_type                                    length,
	index_type                                    segment_length);
```

We can do this with Giddy in one of three ways.

### <a name="direct-use-of-kernel">Direct use of the kernel source code</a>

The example code for this mode of use is found in [`examples/src/direct_use_of_kernel.cu`](examples/src/modes_of_use/direct_use_of_kernel.cu).

In this mode, we

   * Include the [kernel source file](https://github.com/eyalroz/libgiddy/blob/master/src/kernels/decompression/frame_of_reference.cuh); we now have a pointer to the kernel's device-side function.
   * Include the launch config resolution mechanism header.
   * Instantiate a launch configuration resolution parameters object, with the parameters specific to our launch.
   * Call `resolve_launch_configuration()` function with the object we instantiated, obtaining a [`launch_configuration_t` struct](https://codedocs.xyz/eyalroz/cuda-api-wrappers/structcuda_1_1launch__configuration__t.html).
   * Perform a CUDA kernel launch, either using the API wrapper (which takes the device function pointer and a `launch_configuration_t`) or the plain vanilla way, extracting the fields of the `launch_configuration_t`.
   
### <a name="instantiation-of-wrapper">Instantiation of the specific kernel launch wrapper</a>

The example code for this mode of use is found in [`examples/src/instantiation_of_wrapper.cu`](examples/src/modes_of_use/instantiation_of_wrapper.cu).

Each decompression kernel has a corresponding thin wrapper class. An instance of the wrapper class has no state - no data members; we only use it for its vtable - its virtual methods, specific to the decompression scheme. Thus, in this mode of use, we:

   * Include the kernel's wrapper class [definition](https://github.com/eyalroz/libgiddy/blob/master/src/kernel_wrappers/decompression/frame_of_reference.cu).
   * Instantiate the wrapper class `cuda::kernels::decompression::frame_of_reference::kernel_t`
   * Call the wrapper's  `resolve_launch_configuration()` method with the appropriate parameters, obtaining a `launch_configuration_t` structure.
   * Call the freestanding function `cuda::kernel::enqueue_launch()` with our wrapper instance, the launch configuration, and the arguments we need to pass the kernel
   
### <a name="factory-provided-type-erased-wrapper">Use of factory-provided, type-erased wrapper</a>

The example code for this mode of use is found in [`examples/src/factory_provided_type_erased_wrapper.cu`](examples/src/modes_of_use/factory_provided_type_erased_wrapper.cu).

The kernel wrappers are intended to allow a uniform interface for launching kernels. This uniformity is achieved by type-erasure: The wrappers' base class virtual methods wrappers' all take a map of [`boost::any`](http://www.boost.org/doc/libs/1_63_0/doc/html/any.html) objects; and it is up to the caller to pass the appropriate parameters in that map. Thus, in this mode, we:

   * Include just the [common base class header](https://github.com/eyalroz/libgiddy/blob/master/src/kernel_wrappers/registered_wrapper.h) for the kernel wrappers.
   * Use the `cuda::registered::kernel_t` class' static method `produceSubclass()` - to instantiate specific the wrapper relevant to our scenario (named `"decompression::frame_of_reference::kernel_t<4u, int, short, cuda::functors::unary::parametric_model::constant<4u, int> >"`). What we actually hold is an `std::unique_ptr()` to such an instance.   
   * Prepare a type-erased map of parameters, and pass it to the `resolve_launch_configuration()` method of our isntance, obtaining a `launch_configuration_t` structure.
   * Prepare a second type-erased map of parameters, and pass it to the `enqueue_launch()` method of our isntance, along with the launch configuration structure we've just obtained.

### No facility for compression!

No code is currently provided for *compressing* data - neither on the device nor on the host side. This is [Issue #3](https://github.com/eyalroz/libgiddy/issues/3).

## <a name="performance">Performance</a>

Some of the decompressors are well-optimized, some need more work. The most recent (and only) performance analysis is in the  [mini-paper](https://www.researchgate.net/publication/315834231_Faster_across_the_PCIe_bus_A_GPU_library_for_lightweight_decompression) mentioned above. *Step-by-step instructions for measuring performance (using well-known data sets) are forthcoming.*

## <a name="acks">Acknowledgements</a>

This endevor was made possible with the help of:

* [CWI Amsterdam](http://www.cwi.nl/)
* Prof. [Peter Boncz](http://homepages.cwi.nl/~boncz/), co-author of the above-mentioned paper
* The [MonetDB](http://www.monetdb.org/) DBMS project - which got me into DBMSes and GPUs in the first place, and which I (partially) use for performance testing
