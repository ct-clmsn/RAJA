/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing RAJA sort declarations.
*
******************************************************************************
*/

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2022, Tactical Computing Labs, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_sort_hpx_HPP
#define RAJA_sort_hpx_HPP

#include "RAJA/config.hpp"

#include <algorithm>
#include <functional>
#include <iterator>

#include <hpx/parallel/algorithms/sort.hpp>

#include "RAJA/util/macros.hpp"

#include "RAJA/util/concepts.hpp"

#include "RAJA/policy/hpx/policy.hpp"
#include "RAJA/policy/loop/sort.hpp"
#include "RAJA/pattern/detail/algorithm.hpp"

namespace RAJA
{
namespace impl
{
namespace sort
{

namespace detail
{
namespace hpx
{

/*!
        \brief sort given range using sorter and comparison function
*/
template <typename Sorter, typename Iter, typename Compare>
inline
void sort(Sorter,
          Iter begin,
          Iter end,
          Compare comp)
{
    ::hpx::sort(::hpx::execution::par, begin, end, comp);
}

} // namespace hpx

} // namespace detail

/*!
        \brief sort given range using comparison function
*/
template <typename ExecPolicy, typename Iter, typename Compare>
concepts::enable_if_t<resources::EventProxy<resources::Host>,
                      type_traits::is_hpx_policy<ExecPolicy>>
unstable(
    resources::Host host_res,
    const ExecPolicy&,
    Iter begin,
    Iter end,
    Compare comp)
{
  detail::hpx::sort(detail::UnstableSorter{}, begin, end, comp);

  return resources::EventProxy<resources::Host>(host_res);
}

/*!
        \brief stable sort given range using comparison function
*/
template <typename ExecPolicy, typename Iter, typename Compare>
concepts::enable_if_t<resources::EventProxy<resources::Host>,
                      type_traits::is_hpx_policy<ExecPolicy>>
stable(
    resources::Host host_res,
    const ExecPolicy&,
    Iter begin,
    Iter end,
    Compare comp)
{
  detail::hpx::sort(detail::StableSorter{}, begin, end, comp);

  return resources::EventProxy<resources::Host>(host_res);
}

/*!
        \brief sort given range of pairs using comparison function on keys
*/
template <typename ExecPolicy, typename KeyIter, typename ValIter, typename Compare>
concepts::enable_if_t<resources::EventProxy<resources::Host>,
                      type_traits::is_hpx_policy<ExecPolicy>>
unstable_pairs(
    resources::Host host_res,
    const ExecPolicy&,
    KeyIter keys_begin,
    KeyIter keys_end,
    ValIter vals_begin,
    Compare comp)
{
  auto begin  = RAJA::zip(keys_begin, vals_begin);
  auto end    = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
  using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
  detail::hpx::sort(detail::UnstableSorter{}, begin, end, RAJA::compare_first<zip_ref>(comp));

  return resources::EventProxy<resources::Host>(host_res);
}

/*!
        \brief stable sort given range of pairs using comparison function on keys
*/
template <typename ExecPolicy, typename KeyIter, typename ValIter, typename Compare>
concepts::enable_if_t<resources::EventProxy<resources::Host>,
                      type_traits::is_hpx_policy<ExecPolicy>>
stable_pairs(
    resources::Host host_res,
    const ExecPolicy&,
    KeyIter keys_begin,
    KeyIter keys_end,
    ValIter vals_begin,
    Compare comp)
{
  auto begin  = RAJA::zip(keys_begin, vals_begin);
  auto end    = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
  using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
  detail::hpx::sort(detail::StableSorter{}, begin, end, RAJA::compare_first<zip_ref>(comp));

  return resources::EventProxy<resources::Host>(host_res);
}

}  // namespace sort

}  // namespace impl

}  // namespace RAJA

#endif
