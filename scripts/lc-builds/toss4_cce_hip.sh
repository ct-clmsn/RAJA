#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if [[ $# -lt 3 ]]; then
  echo
  echo "You must pass 3 or more arguments to the script (in this order): "
  echo "   1) compiler version number"
  echo "   2) HIP version"
  echo "   3) HIP compute architecture"
  echo "   4...) optional arguments to cmake"
  echo
  echo "For example: "
  echo "    toss4_cce_hip.sh 14.0.3 5.2.3 gfx90a"
  exit
fi

COMP_VER=$1
HIP_VER=$2
HIP_ARCH=$3
shift 3

HOSTCONFIG="hip_3_X"

BUILD_SUFFIX=lc_toss4-cce-${COMP_VER}-hip-${HIP_VER}-${HIP_ARCH}

echo
echo "Creating build directory ${BUILD_SUFFIX} and generating configuration in it"
echo "Configuration extra arguments:"
echo "   $@"
echo

rm -rf build_${BUILD_SUFFIX} >/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}


module load cmake/3.24.2
module load cce/${COMP_VER}

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=$(which cc) \
  -DCMAKE_CXX_COMPILER=$(which CC) \
  -DHIP_PATH=/opt/rocm-${HIP_VER}/hip \
  -DCMAKE_HIP_ARCHITECTURES=${HIP_ARCH} \
  -DGPU_TARGETS=${HIP_ARCH} \
  -DAMDGPU_TARGETS=${HIP_ARCH} \
  -C "../host-configs/lc-builds/toss4/${HOSTCONFIG}.cmake" \
  -DENABLE_HIP=ON \
  -DENABLE_OPENMP=ON \
  -DENABLE_CUDA=OFF \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..

echo
echo "***********************************************************************"
echo
echo "cd into directory build_${BUILD_SUFFIX} and run make to build RAJA"
echo
echo "  Please note that you have to have a consistent build environment"
echo "  when you make RAJA as cmake may reconfigure; load the appropriate"
echo "  cce module (${COMP_VER}) when building."
echo
echo "    module load cce/${COMP_VER}"
echo "    srun -n1 make"
echo
echo "***********************************************************************"
