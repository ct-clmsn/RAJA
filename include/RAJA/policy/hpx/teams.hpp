/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing user interface for RAJA::Teams::openmp
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_teams_openmp_HPP
#define RAJA_pattern_teams_openmp_HPP

#include <hpx/local/execution.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>

#include "RAJA/pattern/teams/teams_core.hpp"
#include "RAJA/policy/openmp/policy.hpp"


namespace RAJA
{

namespace expt
{

template <>
struct LaunchExecute<RAJA::expt::hpx_launch_t> {


  template <typename BODY>
  static void exec(LaunchContext const &ctx, BODY const &body)
  {
    RAJA::region<RAJA::hpx_parallel_region>([&]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);
      loop_body.get_priv()(ctx);
    });
  }

  template <typename BODY>
  static resources::EventProxy<resources::Resource>
  exec(RAJA::resources::Resource res, LaunchContext const &ctx, BODY const &body)
  {
    RAJA::region<RAJA::hpx_parallel_region>([&]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);
      loop_body.get_priv()(ctx);
    });

    return resources::EventProxy<resources::Resource>(res);
  }

};


template <typename SEGMENT>
struct LoopExecute<hpx_parallel_for_exec, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    int len = segment.end() - segment.begin();
    RAJA::region<RAJA::hpx_parallel_region>([&]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

      auto irange = hpx::util::detail::make_counting_shape(len);

      hpx::for_each(hpx::execution::par, hpx::util::begin(irange), hpx::util::end(irange), [&](const int i) {
          loop_body.get_priv()(*(segment.begin() + i));
      });
    });
  }

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      BODY const &body)
  {

    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    RAJA::region<RAJA::hpx_parallel_region>([&]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

      auto irange = hpx::util::detail::make_counting_shape(len1);
      hpx::for_each(hpx::execution::par, hpx::util::begin(irange), hpx::util::end(irange), [&](const int j) {
        for (int i = 0; i < len0; i++) {
          loop_body.get_priv()(*(segment0.begin() + i),
                               *(segment1.begin() + j));
        }
      });
    });
  }

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      SEGMENT const &segment2,
      BODY const &body)
  {

    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    RAJA::region<RAJA::hpx_parallel_region>([&]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

      auto irange = hpx::util::detail::make_counting_shape(len2);
      hpx::for_each(hpx::execution::par, hpx::util::begin(irange), hpx::util::end(irange), [&](const int k) {
        for (int j = 0; j < len1; j++) {
          for (int i = 0; i < len0; i++) {
            loop_body.get_priv()(*(segment0.begin() + i),
                                 *(segment1.begin() + j),
                                 *(segment2.begin() + k));
          }
        }
      });
    });
  }
};

template <typename SEGMENT>
struct LoopExecute<hpx_for_exec, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    int len = segment.end() - segment.begin();
    auto irange = hpx::util::detail::make_counting_shape(len);

    hpx::for_each(hpx::execution::par, hpx::util::begin(irange), hpx::util::end(irange), [&](const int i) {
      body(*(segment.begin() + i));
    });
  }

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      BODY const &body)
  {

    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    auto irange = hpx::util::detail::make_counting_shape(len1);

    hpx::for_each(hpx::execution::par, hpx::util::begin(irange), hpx::util::end(irange), [&](const int j) {
      for (int i = 0; i < len0; i++) {
        body(*(segment0.begin() + i), *(segment1.begin() + j));
      }
    });
  }

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      SEGMENT const &segment2,
      BODY const &body)
  {

    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    auto irange = hpx::util::detail::make_counting_shape(len2);

    hpx::for_each(hpx::execution::par, hpx::util::begin(irange), hpx::util::end(irange), [&](const int k) {
      for (int j = 0; j < len1; j++) {
        for (int i = 0; i < len0; i++) {
          body(*(segment0.begin() + i),
               *(segment1.begin() + j),
               *(segment2.begin() + k));
        }
      }
    });
  }
};

//
// Return local index
//
template <typename SEGMENT>
struct LoopICountExecute<hpx_parallel_for_exec, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    int len = segment.end() - segment.begin();
    RAJA::region<RAJA::hpx_parallel_region>([&]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

      auto irange = hpx::util::detail::make_counting_shape(len);

      hpx::for_each(hpx::execution::par, hpx::util::begin(irange), hpx::util::end(irange), [&](const int i) {
        loop_body.get_priv()(*(segment.begin() + i), i);
      });
    });
  }

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      BODY const &body)
  {

    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    RAJA::region<RAJA::hpx_parallel_region>([&]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

      auto irange = hpx::util::detail::make_counting_shape(len1);
      hpx::for_each(hpx::execution::par, hpx::util::begin(irange), hpx::util::end(irange), [&](const int j) {
        for (int i = 0; i < len0; i++) {

          loop_body.get_priv()(*(segment0.begin() + i),
                               *(segment1.begin() + j),
                               i,
                               j);
        }
      });
    });
  }

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      SEGMENT const &segment2,
      BODY const &body)
  {

    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    RAJA::region<RAJA::hpx_parallel_region>([&]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

      auto irange = hpx::util::detail::make_counting_shape(len2);
      hpx::for_each(hpx::execution::par, hpx::util::begin(irange), hpx::util::end(irange), [&](const int k) {
        for (int j = 0; j < len1; j++) {
          for (int i = 0; i < len0; i++) {
            loop_body.get_priv()(*(segment0.begin() + i),
                                 *(segment1.begin() + j),
                                 *(segment2.begin() + k),
                                 i,
                                 j,
                                 k);
          }
        }
      });
    });
  }
};

// policy for perfectly nested loops
struct hpx_parallel_nested_for_exec;

template <typename SEGMENT>
struct LoopExecute<hpx_parallel_nested_for_exec, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      BODY const &body)
  {
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();
    auto i0 = len0;
    auto i1 = len1;

    const auto len = len0 * len1;

    RAJA::region<RAJA::hpx_parallel_region>([&]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

      auto rng = hpx::util::detail::make_counting_shape(len);

      hpx::foreach(hpx::execution::par,
          std::begin(rng), std::end(rng), [&](const auto i) {
              const auto x = i / len1;
              const auto y = i % len1;

              loop_body.get_priv()(*(segment0.begin() + x),
                                   *(segment1.begin() + y),
                                   x, y);
          }
      );
    });
  }

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      SEGMENT const &segment2,
      BODY const &body)
  {

    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();
    auto i0 = len0;
    auto i1 = len1;
    auto i2 = len2;
    const auto len = len0 * len1 * len2;

    RAJA::region<RAJA::hpx_parallel_region>([&]() {
      auto rng = hpx::util::detail::make_counting_shape(len);
      const auto YZ = len1 * len2;

      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

      hpx::foreach(hpx::execution::par,
          std::begin(rng), std::end(rng), [&](const auto i) {
              const auto modiYZ = i % YZ;
              const auto z = modiYZ % l2;
              const auto y = modiYZ / l2;
              const auto x = i / YZ; 

              loop_body.get_priv()(*(segment0.begin() + x),
                                   *(segment1.begin() + y),
                                   *(segment2.begin() + z),
                                   x, y, z);
          }
      );
    });
  }
};

// Return local index
template <typename SEGMENT>
struct LoopICountExecute<hpx_parallel_nested_for_exec, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      BODY const &body)
  {

    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();
    auto i0 = len0;
    auto i1 = len1;

    const auto len = len0 * len1;

    RAJA::region<RAJA::hpx_parallel_region>([&]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

      auto rng = hpx::util::detail::make_counting_shape(len);

      hpx::foreach(hpx::execution::par,
          std::begin(rng), std::end(rng), [&](const auto i) {
              const auto x = i / len1;
              const auto y = i % len1;

              loop_body.get_priv()(*(segment0.begin() + x),
                                   *(segment1.begin() + y),
                                   x,
                                   y);
          }
      );
  }

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      SEGMENT const &segment2,
      BODY const &body)
  {

    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();
    auto i0 = len0;
    auto i1 = len1;
    auto i2 = len2;
    const auto len = len0 * len1 * len2;

    RAJA::region<RAJA::hpx_parallel_region>([&]() {
      auto rng = hpx::util::detail::make_counting_shape(len);
      const auto YZ = len1 * len2;

      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

      hpx::foreach(hpx::execution::par,
          std::begin(rng), std::end(rng), [&](const auto i) {
              const auto modiYZ = i % YZ;
              const auto z = modiYZ % l2;
              const auto y = modiYZ / l2;
              const auto x = i / YZ; 
              loop_body.get_priv()(*(segment0.begin() + x),
                                   *(segment1.begin() + y),
                                   *(segment2.begin() + z),
                                   x,
                                   y,
                                   z);
          }
      );
    });
  }
};


template <typename SEGMENT>
struct TileExecute<hpx_parallel_for_exec, SEGMENT> {

  template <typename BODY, typename TILE_T>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    int len = segment.end() - segment.begin();

    RAJA::region<RAJA::hpx_parallel_region>([&]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

      auto ex =
          hpx::execution::par.on(hpx::execution::parallel_executor{}).with(tile_size);

      auto irange = hpx::util::detail::make_counting_shape(len);

      hpx::for_each(hpx::util::begin(irange), hpx::util::end(irange), [&](const int i) {
        loop_body.get_priv()(segment.slice(i, tile_size));
      });
    });
  }
};

template <typename SEGMENT>
struct TileICountExecute<hpx_parallel_for_exec, SEGMENT> {

  template <typename BODY, typename TILE_T>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    const int numTiles = (len - 1) / tile_size + 1;

    RAJA::region<RAJA::hpx_parallel_region>([&]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

      auto irange = hpx::util::detail::make_counting_shape(numTiles);
      hpx::for_each(hpx::execution::par, hpx::util::begin(irange), hpx::util::end(irange), [&](const int i) {
        const int i_tile_size = i * tile_size;
        loop_body.get_priv()(segment.slice(i_tile_size, tile_size), i);
      });
    });
  }
};

template <typename SEGMENT>
struct TileExecute<hpx_for_exec, SEGMENT> {

  template <typename BODY, typename TILE_T>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    int len = segment.end() - segment.begin();

    auto ex =
        hpx::execution::par.on(hpx::execution::parallel_executor{}).with(tile_size);

    auto irange = hpx::util::detail::make_counting_shape(len);

    hpx::for_each(hpx::execution::par, hpx::util::begin(irange), hpx::util::end(irange), [&](const int i) {
      body(segment.slice(i, tile_size));
    });
  }
};

template <typename SEGMENT>
struct TileICountExecute<hpx_for_exec, SEGMENT> {

  template <typename BODY, typename TILE_T>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    const int numTiles = (len - 1) / tile_size + 1;

    auto irange = hpx::util::detail::make_counting_shape(numTiles);
    hpx::for_each(hpx::execution::par, hpx::util::begin(irange), hpx::util::end(irange), [&](const int i) {
      const int i_tile_size = i * tile_size;
      body.get_priv()(segment.slice(i_tile_size, tile_size), i);
    });
  }
};

}  // namespace expt

}  // namespace RAJA
#endif
