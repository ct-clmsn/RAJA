/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing RAJA scan declarations.
*
******************************************************************************
*/

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_scan_hpx_HPP
#define RAJA_scan_hpx_HPP

#include "RAJA/config.hpp"

#include <algorithm>
#include <functional>
#include <iterator>
#include <type_traits>
#include <vector>

#include <hpx/parallel/algorithms/exclusive_scan.hpp>
#include <hpx/parallel/algorithms/inclusive_scan.hpp>

#include "RAJA/policy/hpx/policy.hpp"
#include "RAJA/policy/loop/scan.hpp"
#include "RAJA/pattern/detail/algorithm.hpp"

namespace RAJA
{
namespace impl
{
namespace scan
{

/*!
        \brief explicit inclusive inplace scan given range, function, and
   initial value
*/
template <typename Policy, typename Iter, typename BinFn>
RAJA_INLINE
concepts::enable_if_t<resources::EventProxy<resources::Host>,
                      type_traits::is_hpx_policy<Policy>>
inclusive_inplace(
    resources::Host host_res,
    const Policy&,
    Iter begin,
    Iter end,
    BinFn f)
{
  hpx::parallel::algorithms::inclusive_scan(begin, end, begin, f);
  return resources::EventProxy<resources::Host>(host_res);
}

/*!
        \brief explicit exclusive inplace scan given range, function, and
   initial value
*/
template <typename Policy, typename Iter, typename BinFn, typename ValueT>
RAJA_INLINE
concepts::enable_if_t<resources::EventProxy<resources::Host>,
                      type_traits::is_hpx_policy<Policy>>
exclusive_inplace(
    resources::Host host_res,
    const Policy&,
    Iter begin,
    Iter end,
    BinFn f,
    ValueT v)
{
  hpx::parallel::algorithms::exclusive_scan(begin, end, begin, f, v);
  return resources::EventProxy<resources::Host>(host_res);
}

/*!
        \brief explicit inclusive scan given input range, output, function, and
   initial value
*/
template <typename Policy, typename Iter, typename OutIter, typename BinFn>
RAJA_INLINE
concepts::enable_if_t<resources::EventProxy<resources::Host>,
                      type_traits::is_hpx_policy<Policy>>
inclusive(
    resources::Host host_res,
    const Policy& exec,
    Iter begin,
    Iter end,
    OutIter out,
    BinFn f)
{
  hpx::parallel::algorithms::inclusive_scan(begin, end, out, f);
  return resources::EventProxy<resources::Host>(host_res);
}

/*!
        \brief explicit exclusive scan given input range, output, function, and
   initial value
*/
template <typename Policy,
          typename Iter,
          typename OutIter,
          typename BinFn,
          typename ValueT>
RAJA_INLINE
concepts::enable_if_t<resources::EventProxy<resources::Host>,
                      type_traits::is_hpx_policy<Policy>>
exclusive(
    resources::Host host_res,
    const Policy& exec,
    Iter begin,
    Iter end,
    OutIter out,
    BinFn f,
    ValueT v)
{
  hpx::parallel::algorithms::exclusive_scan(begin, end, out, f, v);
  return resources::EventProxy<resources::Host>(host_res);
}

}  // namespace scan

}  // namespace impl

}  // namespace RAJA

#endif
