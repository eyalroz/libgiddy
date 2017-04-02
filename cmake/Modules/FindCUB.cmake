# Find the nVIDIA CUB header-only library's include directory
#
# The following variables are set if CUB is found.
#
#  CUB_FOUND        - True when the CUB include directory is found.
#                     If CUB is not found, CUB_FOUND is set to false.
#
#  CUB_INCLUDE_DIR  - The path to where the CUB include files are
#                     located within a cub/ subdirectory.
#
#  CUB_VERSION      - The CUB library's version string (which appears
#                     in the changelog within the library headers)

find_package(PkgConfig)

if(NOT EXISTS "${CUB_INCLUDE_DIR}")
	find_path(
		CUB_INCLUDE_DIR 
		cub/cub.cuh
		HINTS
			${CUDA_INCLUDE_DIRS}
			${CMAKE_SOURCE_DIR}/include
			${CMAKE_SOURCE_DIR}
			${PROJECT_SOURCE_DIR}
			${PROJECT_SOURCE_DIR}/include 
			/opt 
			$ENV{HOME}/opt 
			ENV CUB_DIR 
			ENV CUB_INCLUDE_DIR 
			ENV CUB_PATH
		DOC "nVIDIA CUB GPU primitives header-only CUDA library"
		PATH_SUFFIXES cub libcub nvidia-cub 
	)
endif()

if(EXISTS "${CUB_INCLUDE_DIR}")
	include(FindPackageHandleStandardArgs) # I think this is a CMake v3 thing
	mark_as_advanced(CUB_INCLUDE_DIR)
else()
	include(ExternalProject)
	ExternalProject_Add(
		CUB
		GIT_REPOSITORY https://github.com/NVlabs/cub
		TIMEOUT 5
		CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
		PREFIX "${CMAKE_CURRENT_BINARY_DIR}"
		CONFIGURE_COMMAND "" # Disable configure step
		BUILD_COMMAND "" # Disable build step
		INSTALL_COMMAND "" # Disable install step
		UPDATE_COMMAND "" # Disable update step: clones the project only once
	)
	
	# Specify include dir
	ExternalProject_Get_Property(CUB source_dir)
	set(CUB_INCLUDE_DIR ${source_dir})
endif()

if(EXISTS "${CUB_INCLUDE_DIR}")
	set(CUB_FOUND 1)
	execute_process(COMMAND bash "-c" "egrep \"^\\s*[0-9]\" -m1 \"${CUB_INCLUDE_DIR}/CHANGE_LOG.TXT\" | cut -d\\  -f1 | xargs echo -n" OUTPUT_VARIABLE CUB_VERSION)
else()
	set(CUB_FOUND 0)
endif()

