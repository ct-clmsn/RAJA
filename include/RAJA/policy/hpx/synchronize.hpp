/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for HPX synchronization.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_synchronize_hpx_HPP
#define RAJA_synchronize_hpx_HPP

namespace RAJA
{

namespace policy
{

namespace hpx
{

/*!
 * \brief Synchronize all HPX threads and tasks.
 */
RAJA_INLINE
void synchronize_impl(const hpx_synchronize& sync)
{
   sync.bar->arrive_and_wait();
}


}  // end of namespace hpx
}  // namespace policy
}  // end of namespace RAJA

#endif  // RAJA_synchronize_hpx_HPP
