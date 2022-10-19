###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and other RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if (ENABLE_OPENMP)
  if(OPENMP_FOUND)
    list(APPEND RAJA_EXTRA_NVCC_FLAGS -Xcompiler ${OpenMP_CXX_FLAGS})
    message(STATUS "OpenMP Enabled")
  else()
    message(WARNING "OpenMP NOT FOUND")
    set(ENABLE_OPENMP Off)
  endif()
endif()

if (ENABLE_HPX)
  if(EXISTS "${HPX_DIR}")
    set(__hpx_dir ${HPX_DIR})
    find_package(HPX REQUIRED NO_CMAKE_PACKAGE_REGISTRY)

    if(NOT HPX_FOUND)
      message(FATAL_ERROR "HPX could not be found, please set HPX_DIR to help locating it.")
    endif()

    # HPX_DIR is being reset by find_packe *sigh*
    set(HPX_DIR ${__hpx_dir})

    # make sure that configured build type for Phylanx matches the one used for HPX
    get_property(_GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if (NOT _GENERATOR_IS_MULTI_CONFIG AND
        NOT (${HPX_BUILD_TYPE} STREQUAL ${CMAKE_BUILD_TYPE}))
      list(FIND ${CMAKE_BUILD_TYPE} ${HPX_BUILD_TYPE} __pos)
      if(${__pos} EQUAL -1)
        message(
          "The configured CMAKE_BUILD_TYPE (${CMAKE_BUILD_TYPE}) is "
          "different from the build type used for the found HPX "
          "(HPX_BUILD_TYPE: ${HPX_BUILD_TYPE})")
      endif()
    endif()

    if(HPX_CXX_STANDARD)
      set(__hpx_standard "using C++${HPX_CXX_STANDARD}")
    endif()

    blt_register_library(
      NAME HPX::hpx
      INCLUDES ${HPX_INCLUDE_DIRS}
      LIBRARIES ${HPX_LIBRARIES})
    message(STATUS "HPX Enabled")

  else()
    message(FATAL_ERROR "HPX_DIR has not been specified, please set it to help locating HPX")
  endif()

endif()

if (ENABLE_TBB)
  find_package(TBB)
  if(TBB_FOUND)
    blt_register_library(
      NAME tbb
      INCLUDES ${TBB_INCLUDE_DIRS}
      LIBRARIES ${TBB_LIBRARIES})
    message(STATUS "TBB Enabled")
  else()
    message(WARNING "TBB NOT FOUND")
    set(ENABLE_TBB Off)
  endif()
endif ()

if (ENABLE_CUDA OR ENABLE_EXTERNAL_CUB)
  find_package(CUB)
  if (CUB_FOUND)
    set(ENABLE_EXTERNAL_CUB On)
    blt_import_library(
      NAME cub
      INCLUDES ${CUB_INCLUDE_DIRS}
      TREAT_INCLUDES_AS_SYSTEM ON
      EXPORTABLE ON)
  elseif(ENABLE_EXTERNAL_CUB)
    message(FATAL_ERROR "External CUB not found, CUB_DIR=${CUB_DIR}.")
  else()
    message(STATUS "Using RAJA CUB submodule.")
  endif()
endif ()

if (ENABLE_CUDA AND ENABLE_NV_TOOLS_EXT)
  find_package(NvToolsExt)
  if (NVTOOLSEXT_FOUND)
    blt_import_library( NAME       nvtoolsext
                        TREAT_INCLUDES_AS_SYSTEM ON
                        INCLUDES   ${NVTOOLSEXT_INCLUDE_DIRS}
                        LIBRARIES  ${NVTOOLSEXT_LIBRARY}
                        EXPORTABLE ON
                      )
  else()
    message(FATAL_ERROR "NvToolsExt not found, NVTOOLSEXT_DIR=${NVTOOLSEXT_DIR}.")
  endif()
endif ()

if (ENABLE_HIP OR ENABLE_EXTERNAL_ROCPRIM)
  find_package(RocPRIM)
  if (ROCPRIM_FOUND)
    set(ENABLE_EXTERNAL_ROCPRIM On)
    blt_import_library(
      NAME rocPRIM
      INCLUDES ${ROCPRIM_INCLUDE_DIRS}
      TREAT_INCLUDES_AS_SYSTEM ON
      EXPORTABLE ON)
  elseif (ENABLE_EXTERNAL_ROCPRIM)
      message(FATAL_ERROR "External rocPRIM not found, ROCPRIM_DIR=${ROCPRIM_DIR}.")
  else()
    message(STATUS "Using RAJA rocPRIM submodule.")
  endif()
endif ()

if (ENABLE_HIP AND ENABLE_ROCTX)
  include(FindRoctracer)
  blt_import_library(NAME roctx
                     INCLUDES ${ROCTX_INCLUDE_DIRS}
                     LIBRARIES ${ROCTX_LIBRARIES})
endif ()

set(TPL_DEPS)
blt_list_append(TO TPL_DEPS ELEMENTS cuda cuda_runtime IF ENABLE_CUDA)
blt_list_append(TO TPL_DEPS ELEMENTS nvtoolsext IF ENABLE_NV_TOOLS_EXT)
blt_list_append(TO TPL_DEPS ELEMENTS cub IF ENABLE_EXTERNAL_CUB)
blt_list_append(TO TPL_DEPS ELEMENTS hip hip_runtime IF ENABLE_HIP)
blt_list_append(TO TPL_DEPS ELEMENTS rocPRIM IF ENABLE_EXTERNAL_ROCPRIM)
blt_list_append(TO TPL_DEPS ELEMENTS openmp IF ENABLE_OPENMP)
blt_list_append(TO TPL_DEPS ELEMENTS HPX::hpx IF ENABLE_HPX)
blt_list_append(TO TPL_DEPS ELEMENTS mpi IF ENABLE_MPI)

foreach(dep ${TPL_DEPS})
    # If the target is EXPORTABLE, add it to the export set
    get_target_property(_is_imported ${dep} IMPORTED)
    if(NOT ${_is_imported})
        install(TARGETS              ${dep}
                EXPORT               RAJA
                DESTINATION          lib)
        # Namespace target to avoid conflicts
        set_target_properties(${dep} PROPERTIES EXPORT_NAME RAJA::${dep})
    endif()
endforeach()
