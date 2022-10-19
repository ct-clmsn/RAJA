/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run kernel
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_hpx_kernel_hpxsyncthreads_HPP
#define RAJA_policy_hpx_kernel_hpxsyncthreads_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HPX)

#include "RAJA/pattern/kernel/internal.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/policy/hpx/policy.hpp"

namespace RAJA
{

namespace statement
{
struct HPXSyncThreads : public internal::Statement<camp::nil> {
  static hpx::barrier<> bar;
};

hpx::barrier<> HPXSyncThreads::bar = hpx::barrier<>{};

} // namespace statement

namespace internal
{



//Statement executor to synchronize hpx threads inside a kernel region
template<typename Types>
struct StatementExecutor<statement::HPXSyncThreads, Types> {

  template<typename Data>
  static RAJA_INLINE void exec(Data &&)
  {
    statement::HPXSyncThreads::bar.arrive_and_wait();
  }

};


}  // namespace internal
}  // namespace RAJA


#endif  // closing endif for RAJA_ENABLE_HPX guard

#endif  // closing endif for header file include guard
