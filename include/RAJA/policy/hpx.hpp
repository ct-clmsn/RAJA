/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA headers for HPX execution.
 *
 *          These methods work only on platforms that support HPX.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2022, Tactical Computing Labs, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#ifndef RAJA_hpx_HPP
#define RAJA_hpx_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HPX)

#include <hpx.hpp>
#include <iostream>
#include <thread>

#if !defined(RAJA_ENABLE_DESUL_ATOMICS)
    #include "RAJA/policy/hpx/atomic.hpp"
#endif

#include "RAJA/policy/hpx/forall.hpp"
#include "RAJA/policy/hpx/kernel.hpp"
#include "RAJA/policy/hpx/policy.hpp"
#include "RAJA/policy/hpx/reduce.hpp"
#include "RAJA/policy/hpx/region.hpp"
#include "RAJA/policy/hpx/scan.hpp"
#include "RAJA/policy/hpx/sort.hpp"
#include "RAJA/policy/hpx/synchronize.hpp"
#include "RAJA/policy/hpx/teams.hpp"
#include "RAJA/policy/hpx/WorkGroup.hpp"


#endif  // closing endif for if defined(RAJA_ENABLE_HPX)

#endif  // closing endif for header file include guard
