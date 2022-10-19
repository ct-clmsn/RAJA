/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA index set and segment iteration
 *          template methods for HPX.
 *
 *          These methods should work on any platform that supports HPX.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2022, Tactical Computing Labs, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_forall_hpx_HPP
#define RAJA_forall_hpx_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HPX)

#include <iostream>
#include <type_traits>

#include <hpx.h>

#include "RAJA/util/types.hpp"

#include "RAJA/internal/fault_tolerance.hpp"

#include "RAJA/index/IndexSet.hpp"
#include "RAJA/index/ListSegment.hpp"
#include "RAJA/index/RangeSegment.hpp"

#include "RAJA/policy/hpx/policy.hpp"

#include "RAJA/pattern/forall.hpp"
#include "RAJA/pattern/region.hpp"


namespace RAJA
{

namespace policy
{
namespace hpx
{
///
/// HPX parallel policy implementation
///
template <typename Iterable, typename Func, typename InnerPolicy>
RAJA_INLINE resources::EventProxy<resources::Host> forall_impl(resources::Host host_res,
                                                    const hpx_parallel_exec<InnerPolicy>&,
                                                    Iterable&& iter,
                                                    Func&& loop_body)
{
  RAJA::region<RAJA::hpx_parallel_region>([&]() {
    using RAJA::internal::thread_privatize;
    auto body = thread_privatize(loop_body);
    forall_impl(host_res, InnerPolicy{}, iter, body.get_priv());
  });
  return resources::EventProxy<resources::Host>(host_res);
}


///
/// HPX parallel for schedule policy implementation
///

namespace internal
{

  /// Tag dispatch for hpx forall

  //
  // hpx for (Auto)
  //
  template <typename Iterable, typename Func>
  RAJA_INLINE void forall_impl(const ::RAJA::policy::hpx::Auto&,
                               Iterable&& iter,
                               Func&& loop_body)
  {
    RAJA_EXTRACT_BED_IT(iter);
    hpx::for_each(hpx::execution::par, begin_it, end_it, [](decltype(distance_it) i) {
      loop_body(begin_it[i]);
    });
  }

  //
  // hpx for schedule(static)
  //
  template <typename Iterable, typename Func, int ChunkSize,
    typename std::enable_if<(ChunkSize <= 0)>::type* = nullptr>
  RAJA_INLINE void forall_impl(const ::RAJA::policy::hpx::Static<ChunkSize>&,
                               Iterable&& iter,
                               Func&& loop_body)
  {
    RAJA_EXTRACT_BED_IT(iter);
    hpx::for_each(hpx::execution::par, begin_it, end_it, [](decltype(distance_it) i) {
      loop_body(begin_it[i]);
    });
  }

  //
  // hpx for schedule(static, ChunkSize)
  //
  template <typename Iterable, typename Func, int ChunkSize,
    typename std::enable_if<(ChunkSize > 0)>::type* = nullptr>
  RAJA_INLINE void forall_impl(const ::RAJA::policy::hpx::Static<ChunkSize>&,
                               Iterable&& iter,
                               Func&& loop_body)
  {
    RAJA_EXTRACT_BED_IT(iter);
    hpx::for_each(hpx::execution::par, begin_it, end_it, [](decltype(distance_it) i) {
      loop_body(begin_it[i]);
    });
  }

  //
  // hpx for schedule(dynamic)
  //
  template <typename Iterable, typename Func, int ChunkSize,
    typename std::enable_if<(ChunkSize <= 0)>::type* = nullptr>
  RAJA_INLINE void forall_impl(const ::RAJA::policy::hpx::Dynamic<ChunkSize>&,
                               Iterable&& iter,
                               Func&& loop_body)
  {
    RAJA_EXTRACT_BED_IT(iter);
    hpx::for_each(hpx::execution::par, begin_it, end_it, [](decltype(distance_it) i) {
      loop_body(begin_it[i]);
    });
  }

  //
  // hpx for schedule(dynamic, ChunkSize)
  //
  template <typename Iterable, typename Func, int ChunkSize,
    typename std::enable_if<(ChunkSize > 0)>::type* = nullptr>
  RAJA_INLINE void forall_impl(const ::RAJA::policy::hpx::Dynamic<ChunkSize>&,
                               Iterable&& iter,
                               Func&& loop_body)
  {
    RAJA_EXTRACT_BED_IT(iter);
    hpx::for_each(hpx::execution::par, begin_it, end_it, [](decltype(distance_it) i) {
      loop_body(begin_it[i]);
    });
  }

  //
  // hpx for schedule(guided)
  //
  template <typename Iterable, typename Func, int ChunkSize,
    typename std::enable_if<(ChunkSize <= 0)>::type* = nullptr>
  RAJA_INLINE void forall_impl(const ::RAJA::policy::hpx::Guided<ChunkSize>&,
                               Iterable&& iter,
                               Func&& loop_body)
  {
    RAJA_EXTRACT_BED_IT(iter);
    hpx::for_each(hpx::execution::par, begin_it, end_it, [](decltype(distance_it) i) {
      loop_body(begin_it[i]);
    });
  }

  //
  // hpx for schedule(guided, ChunkSize)
  //
  template <typename Iterable, typename Func, int ChunkSize,
    typename std::enable_if<(ChunkSize > 0)>::type* = nullptr>
  RAJA_INLINE void forall_impl(const ::RAJA::policy::hpx::Guided<ChunkSize>&,
                               Iterable&& iter,
                               Func&& loop_body)
  {
    RAJA_EXTRACT_BED_IT(iter);
    hpx::for_each(hpx::execution::par, begin_it, end_it, [](decltype(distance_it) i) {
      loop_body(begin_it[i]);
    });
  }

  //
  // hpx for schedule(runtime)
  //
  template <typename Iterable, typename Func>
  RAJA_INLINE void forall_impl(const ::RAJA::policy::hpx::Runtime& rt,
                               Iterable&& iter,
                               Func&& loop_body)
  {
    RAJA_EXTRACT_BED_IT(iter);
    hpx::for_each(rt.exec, begin_it, end_it, [](decltype(distance_it) i) {
      loop_body(begin_it[i]);
    });
  }

  #if !defined(RAJA_COMPILER_MSVC)
  // dynamic & guided
  template <typename Policy, typename Iterable, typename Func>
  RAJA_INLINE void forall_impl(const Policy&,
                               Iterable&& iter,
                               Func&& loop_body)
  {
    auto ex = hpx::execution::par.on(hpx::execution::parallel_executor{}).with(Policy::chunk_size);
    forall_impl(::RAJA::policy::hpx::Runtime{ex}, std::forward<Iterable>(iter), std::forward<Func>(loop_body));
    hpx_set_schedule(prev_sched, prev_chunk);
  }
  #endif


  /// Tag dispatch for hpx forall with nowait

  //
  // hpx for nowait (Auto)
  //
  template <typename Iterable, typename Func>
  RAJA_INLINE void forall_impl_nowait(const ::RAJA::policy::hpx::Auto&,
                               Iterable&& iter,
                               Func&& loop_body)
  {
    RAJA_EXTRACT_BED_IT(iter);
    hpx::for_each(hpx::execution::par, begin_it, end_it, [](decltype(distance_it) i) {
      loop_body(begin_it[i]);
    });
  }

  //
  // hpx for schedule(static) nowait
  //
  template <typename Iterable, typename Func, int ChunkSize,
    typename std::enable_if<(ChunkSize <= 0)>::type* = nullptr>
  RAJA_INLINE void forall_impl_nowait(const ::RAJA::policy::hpx::Static<ChunkSize>&,
                               Iterable&& iter,
                               Func&& loop_body)
  {
    RAJA_EXTRACT_BED_IT(iter);
    hpx::for_each(hpx::execution::par, begin_it, end_it, [](decltype(distance_it) i) {
      loop_body(begin_it[i]);
    });
  }

  //
  // hpx for schedule(static, ChunkSize) nowait
  //
  template <typename Iterable, typename Func, int ChunkSize,
    typename std::enable_if<(ChunkSize > 0)>::type* = nullptr>
  RAJA_INLINE void forall_impl_nowait(const ::RAJA::policy::hpx::Static<ChunkSize>&,
                               Iterable&& iter,
                               Func&& loop_body)
  {
    RAJA_EXTRACT_BED_IT(iter);
    hpx::for_each(hpx::execution::par, begin_it, end_it, [](decltype(distance_it) i) {
      loop_body(begin_it[i]);
    });
  }

  #if !defined(RAJA_COMPILER_MSVC)
  // dynamic & guided
  template <typename Policy, typename Iterable, typename Func>
  RAJA_INLINE void forall_impl_nowait(const Policy&,
                               Iterable&& iter,
                               Func&& loop_body)
  {
    auto ex = hpx::execution::par.on(hpx::execution::parallel_executor{}).with(Policy::chunk_size);
    forall_impl_nowait(::RAJA::policy::hpx::Runtime{ex}, std::forward<Iterable>(iter), std::forward<Func>(loop_body));
    hpx_set_schedule(prev_sched, prev_chunk);
  }
  #endif

} // end namespace internal

template <typename Schedule, typename Iterable, typename Func>
RAJA_INLINE resources::EventProxy<resources::Host> forall_impl(resources::Host host_res,
                                                               const hpx_for_schedule_exec<Schedule>&,
                                                               Iterable&& iter,
                                                               Func&& loop_body)
{
  internal::forall_impl(Schedule{}, std::forward<Iterable>(iter), std::forward<Func>(loop_body));
  return resources::EventProxy<resources::Host>(host_res);
}

template <typename Schedule, typename Iterable, typename Func>
RAJA_INLINE resources::EventProxy<resources::Host> forall_impl(resources::Host host_res,
                                                               const hpx_for_nowait_schedule_exec<Schedule>&,
                                                               Iterable&& iter,
                                                               Func&& loop_body)
{
  internal::forall_impl_nowait(Schedule{}, std::forward<Iterable>(iter), std::forward<Func>(loop_body));
  return resources::EventProxy<resources::Host>(host_res);
}

}  // namespace hpx

}  // namespace policy

}  // namespace RAJA

#endif  // closing endif for if defined(RAJA_ENABLE_HPX)

#endif  // closing endif for header file include guard
