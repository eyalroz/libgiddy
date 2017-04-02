# CMakeLists.txt for the giddy GPU decompression library
#
cmake_minimum_required(VERSION 2.8)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# -----------------------------------
# Project name, version & build type
# -----------------------------------

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # This is the top-level project.
	PROJECT(giddy)

#	# No versioning for now
#
#	set(PROJECT_MAJOR_VERSION 0)
#	set(PROJECT_MINOR_VERSION 1)
#	set(PROJECT_PATCH_VERSION 0)
#	set(PROJECT_VERSION ${PROJECT_MAJOR_VERSION}.${PROJECT_MINOR_VERSION}.${PROJECT_PATCH_VERSION})
endif()

set(DEFAULT_BUILD_TYPE "RELEASE")

include (ReleaseOrDebug)

# ----------------------------
# General C/C++ build settings
# ----------------------------

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++14 -g" )
if (NOT "$ENV{GCC_COLORS}" STREQUAL "")
	# This will force color output from GCC even when you pipe through less
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
else (NOT "$ENV{GCC_COLORS}" STREQUAL "") 
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-diagnostics-color")
endif (NOT "$ENV{GCC_COLORS}" STREQUAL "")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG -Og" )
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -O3" )
include_directories( src/ )

# -------------
# Boost
# -------------
find_package(Boost 1.54.0 QUIET COMPONENTS filesystem program_options regex filesystem REQUIRED)
include_directories( ${Boost_INCLUDE_DIR} )
# See http://stackoverflow.com/a/30877725/1593077
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_SYSTEM_NO_DEPRECATED")
# These next two lines should not be necessary, but I don't get the proper -L switch
# without them... see http://stackoverflow.com/q/40208960/1593077 
set (CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS} "-L${Boost_LIBRARY_DIRS}")
set (CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} "-L${Boost_LIBRARY_DIRS}")

# -------------
# CUDA
# -------------

find_package(CUDA 7.5 QUIET REQUIRED)
set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY} ${CUDA_cupti_LIBRARY})

include(HandleCUDAComputeCapability)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --std=c++11 --keep  --generate-line-info --source-in-ptx -Xcompiler -D__CORRECT_ISO_CPP11_MATH_H_PROTO -Xcompiler -Wall )
# Note: Adding "-G" to the next line might give you debugging info, but it switches off all optimizers, so you're not really debugging what you're running; and it also increases the register pressure, and the local memory pressure tremendously, changing the behavior even further; finally, it _may_ cause linking trouble with CUDA 7.5 and Fedora 23 (or equivalent build environments)
set(CUDA_NVCC_FLAGS_DEBUG -g -lineinfo -DDEBUG)
# Note: Using fast math has drawbacks, of course; you might want to remove this, or better yet, add the use of the fast math switch as a kernel template instantiation option, and compile just the appropriate instantiations with the switch turned on (i.e. different object files)
set(CUDA_NVCC_FLAGS_RELEASE -O3 --use_fast_math -DNDEBUG)

set(CUDA_SEPARABLE_COMPILATION OFF)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

find_library(CUDA_OPENCL_LIBRARY
  NAMES OpenCL opencl
  PATHS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES "lib64" "lib"
  DOC "Location of the CUDA OpenCL support library"
  NO_DEFAULT_PATH
  )
mark_as_advanced(CUDA_OPENCL_LIBRARY)
set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_OPENCL_LIBRARY})

find_library(CUDA_NVTX_LIBRARY
  NAMES nvToolsExt nvTools nvtoolsext nvtools nvtx NVTX
  PATHS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES "lib64" "common/lib64" "common/lib" "lib"
  DOC "Location of the CUDA Toolkit Extension (NVTX) library"
  NO_DEFAULT_PATH
  )
mark_as_advanced(CUDA_NVTX_LIBRARY)
set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_NVTX_LIBRARY})


# -------------
# CUB
# -------------

find_package(CUB 1.5.0 REQUIRED)
include_directories(${CUB_INCLUDE_DIR})

# -----------------------
# Miscellaneous targets
# -----------------------


if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)	
    # This is the top-level project.

	if(NOT TARGET tags)
	set_source_files_properties( tags PROPERTIES GENERATED true )
	add_custom_target(tags 
		COMMAND ctags --langmap=c++:+.cu.cuh.hpp -R src/ 
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
	endif()

	if(NOT TARGET cclean)
	add_custom_target(cclean COMMAND rm -r CMakeCache.txt cmake_install.cmake CMakeFiles/ Makefile)
	add_custom_target(distclean ALIAS cclean)
	endif()

endif()

# -----------------------------------
# Subrepository: CUDA API wrappers
# -----------------------------------

set(CUDA_API_WRAPPERS_SUBDIR "external/cuda-api-wrappers")
add_subdirectory(${CUDA_API_WRAPPERS_SUBDIR})
include_directories("${CUDA_API_WRAPPERS_SUBDIR}/src")
link_directories("${CMAKE_CURRENT_SOURCE_DIR}/${CUDA_API_WRAPPERS_SUBDIR}/lib")

# -----------------------
# Main target(s)
# -----------------------

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin/")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "lib/")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "lib/")

cuda_add_library(
	giddy-kernels
	src/kernel_wrappers/data_layout/gather.cu
	src/kernel_wrappers/data_layout/gather_bits.cu
	src/kernel_wrappers/data_layout/scatter.cu
	src/kernel_wrappers/data_layout/compressed_indices_scatter.cu
	src/kernel_wrappers/reduction/scan.cu
	src/kernel_wrappers/reduction/reduce.cu
	src/kernel_wrappers/reduction/count_if.cu
	src/kernel_wrappers/decompression/dictionary.cu
	src/kernel_wrappers/decompression/incidence_bitmaps.cu
	src/kernel_wrappers/decompression/delta.cu
	src/kernel_wrappers/decompression/frame_of_reference.cu
	src/kernel_wrappers/decompression/discard_zero_bytes_fixed.cu
	src/kernel_wrappers/decompression/run_length_encoding.cu
	src/kernel_wrappers/decompression/run_position_encoding.cu
	src/kernel_wrappers/decompression/discard_zero_bytes_variable.cu
	src/kernel_wrappers/decompression/model.cu
#	OPTIONS RELEASE -O3 --use_fast_math
)

#Note: When you link a binary with giddy, you will need to link with:
#
# * The Boost libraries
# * The compiled part of the CUDA API wrapper library
# * The CUDA libraries
# * Giddy itself
#
# or, in CMake terms:
#
#  target_link_libraries(some_executable_binary ${Boost_LIBRARIES} cuda-api-wrappers monetdb-buffer-pool cpp-util ${CUDA_LIBRARIES} ktkernels)
#

# We don't do installation for now

