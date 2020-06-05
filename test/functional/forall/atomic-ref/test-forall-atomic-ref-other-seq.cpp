//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for logical, accessor, min/max, and cas atomic operations
///

#include "tests/test-forall-atomic-ref-other.hpp"

#include "RAJA_test-forall-execpol.hpp"

#include "../test-forall-atomic-utils.hpp"

using SeqAtomicForallRefOtherTypes = 
  Test< camp::cartesian_product< SequentialForallAtomicExecPols,
                                 SequentialAtomicPols,
                                 HostResourceList,
                                 AtomicDataTypeList> >::Types;

INSTANTIATE_TYPED_TEST_SUITE_P( SeqTest,
                                ForallAtomicRefOtherFunctionalTest,
                                SeqAtomicForallRefOtherTypes );
