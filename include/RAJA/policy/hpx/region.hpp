//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2022, Tactical Computing Labs, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
#ifndef RAJA_region_hpx_HPP
#define RAJA_region_hpx_HPP

#include <hpx/local/task_block.hpp>

namespace RAJA
{
namespace policy
{
namespace hpx
{

/*!
 * \brief RAJA::region implementation for HPX.
 *
 * Generates an HPX parallel region
 *
 * \code
 *
 * RAJA::region<hpx_parallel_region>([=](){
 *
 *  // region body - may contain multiple loops
 *
 *  });
 *
 * \endcode
 *
 * \tparam Policy region policy
 *
 */

template <typename Func>
RAJA_INLINE void region_impl(const hpx_parallel_region &, Func &&body)
{
    ::hpx::parallel::define_task_block(::hpx::execution::par(::hpx::execution::task), [&](auto& trh) {
        // curly brackets to ensure body() is encapsulated in hpx parallel region
        //thread private copy of body
        auto loopbody = body;

        trh.run([&]() {
            loopbody();
        });
    });
}

}  // namespace hpx

}  // namespace policy

}  // namespace RAJA

#endif  // closing endif for header file include guard
