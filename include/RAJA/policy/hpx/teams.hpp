/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing user interface for RAJA::Teams::hpx
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2022, Tactical Computing Labs, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_teams_hpx_HPP
#define RAJA_pattern_teams_hpx_HPP

#include <hpx/local/execution.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>

#include "RAJA/pattern/teams/teams_core.hpp"
#include "RAJA/policy/hpx/policy.hpp"


namespace RAJA
{

namespace expt
{

template <>
struct LaunchExecute<RAJA::expt::hpx_launch_t> {


  template <typename BODY>
  static void exec(LaunchContext const &ctx, BODY const &body)
  {
std::cout << "launchexecute:exec:1" << std::endl;
    RAJA::region<RAJA::hpx_parallel_region>([&ctx, &body]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);
      loop_body.get_priv()(ctx);
    });
  }

  template <typename BODY>
  static resources::EventProxy<resources::Resource>
  exec(RAJA::resources::Resource res, LaunchContext const &ctx, BODY const &body)
  {
std::cout << "launchexecute:exec:2" << std::endl;
    RAJA::region<RAJA::hpx_parallel_region>([&ctx, &body]() {
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
std::cout << "LoopExecute:exec:1" << std::endl;

    int len = segment.end() - segment.begin();
    RAJA::region<RAJA::hpx_parallel_region>([len, &segment, &body]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

      auto irange = ::hpx::util::detail::make_counting_shape(len);

      ::hpx::for_each(::hpx::execution::par, ::hpx::util::begin(irange), ::hpx::util::end(irange), [&segment, &loop_body](const int i) {
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
std::cout << "LoopExecute:exec:2" << std::endl;

    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    RAJA::region<RAJA::hpx_parallel_region>([&segment0, &segment1, &body, &len0, &len1]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

      auto irange = ::hpx::util::detail::make_counting_shape(len1);
      ::hpx::for_each(::hpx::execution::par, ::hpx::util::begin(irange), ::hpx::util::end(irange), [&segment0, &segment1, len0, &loop_body](const int j) {
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
std::cout << "LoopExecute:exec:3" << std::endl;

    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    RAJA::region<RAJA::hpx_parallel_region>([len2, len1, len0, &segment2, &segment1, &segment0, &body]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

      auto irange = ::hpx::util::detail::make_counting_shape(len2);
      ::hpx::for_each(::hpx::execution::par, ::hpx::util::begin(irange), ::hpx::util::end(irange), [len1, len0, &loop_body, &segment0, &segment1, &segment2](const int k) {
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
std::cout << "LoopExecute:exec1:1" << std::endl;

    int len = segment.end() - segment.begin();
    auto irange = ::hpx::util::detail::make_counting_shape(len);

    ::hpx::for_each(::hpx::execution::par, ::hpx::util::begin(irange), ::hpx::util::end(irange), [&segment, &body](const int i) {
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
std::cout << "LoopExecute:exec1:2" << std::endl;

    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    auto irange = ::hpx::util::detail::make_counting_shape(len1);

    ::hpx::for_each(::hpx::execution::par, ::hpx::util::begin(irange), ::hpx::util::end(irange), [len0, &segment0, &segment1, &body](const int j) {
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
std::cout << "LoopExecute:exec1:3" << std::endl;

    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    auto irange = ::hpx::util::detail::make_counting_shape(len2);

    ::hpx::for_each(::hpx::execution::par, ::hpx::util::begin(irange), ::hpx::util::end(irange), [len1, len0, &segment0, &segment1, &segment2, &body](const int k) {
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
std::cout << "LoopICountExecute:exec:1" << std::endl;

    int len = segment.end() - segment.begin();
    RAJA::region<RAJA::hpx_parallel_region>([len, &segment, &body]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

      auto irange = ::hpx::util::detail::make_counting_shape(len);

      ::hpx::for_each(::hpx::execution::par, ::hpx::util::begin(irange), ::hpx::util::end(irange), [&segment, &loop_body](const int i) {
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
std::cout << "LoopICountExecute:exec:2" << std::endl;

    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    RAJA::region<RAJA::hpx_parallel_region>([&segment0, &segment1, len1, len0, &body]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

      auto irange = ::hpx::util::detail::make_counting_shape(len1);
      ::hpx::for_each(::hpx::execution::par, ::hpx::util::begin(irange), ::hpx::util::end(irange), [len0, &segment0, &segment1, &loop_body](const int j) {
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
std::cout << "LoopICountExecute:exec:3" << std::endl;

    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    RAJA::region<RAJA::hpx_parallel_region>([len0, len1, len2, &segment0, &segment1, &segment2, &body]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

      auto irange = ::hpx::util::detail::make_counting_shape(len2);
      ::hpx::for_each(::hpx::execution::par, ::hpx::util::begin(irange), ::hpx::util::end(irange), [len1, len0, &loop_body, &segment0, &segment1, &segment2](const int k) {
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
std::cout << "LoopExecute:exec:2:1" << std::endl;
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    const auto len = len0 * len1;

    RAJA::region<RAJA::hpx_parallel_region>([len, len1, &segment0, &segment1, &body]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

      auto rng = ::hpx::util::detail::make_counting_shape(len);

      ::hpx::for_each(::hpx::execution::par,
          std::begin(rng), std::end(rng), [len1, &segment0, &segment1, &loop_body](const auto i) {
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
std::cout << "LoopExecute:exec:2:2" << std::endl;

    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();
    const auto len = len0 * len1 * len2;

    RAJA::region<RAJA::hpx_parallel_region>([len2, len1, len, &body, &segment0, &segment1, &segment2]() {
      auto rng = ::hpx::util::detail::make_counting_shape(len);
      const auto YZ = len1 * len2;

      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

      ::hpx::for_each(::hpx::execution::par,
          std::begin(rng), std::end(rng), [&loop_body, len2, &YZ, &segment0, &segment1, &segment2](const auto i) {
              const auto modiYZ = i % YZ;
              const auto z = modiYZ % len2;
              const auto y = modiYZ / len2;
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
std::cout << "LoopICountExecute:exec:3:1" << std::endl;

    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    const auto len = len0 * len1;

    RAJA::region<RAJA::hpx_parallel_region>([&body, len1, len, &segment0, &segment1]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

      auto rng = ::hpx::util::detail::make_counting_shape(len);

      ::hpx::for_each(::hpx::execution::par,
          std::begin(rng), std::end(rng), [len1, &segment0, &segment1, &loop_body](const auto i) {
              const auto x = i / len1;
              const auto y = i % len1;

              loop_body.get_priv()(*(segment0.begin() + x),
                                   *(segment1.begin() + y),
                                   x,
                                   y);
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
std::cout << "LoopICountExecute:exec:3:2" << std::endl;

    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();
    const auto len = len0 * len1 * len2;

    RAJA::region<RAJA::hpx_parallel_region>([len2, len1, &body, &segment0, &segment1, &segment2]() {
      auto rng = ::hpx::util::detail::make_counting_shape(len);
      const auto YZ = len1 * len2;

      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

      ::hpx::for_each(::hpx::execution::par,
          std::begin(rng), std::end(rng), [YZ, len2, &loop_body, &segment0, &segment1, &segment2](const auto i) {
              const auto modiYZ = i % YZ;
              const auto z = modiYZ % len2;
              const auto y = modiYZ / len2;
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
std::cout << "TileExecute:exec:1" << std::endl;

    int len = segment.end() - segment.begin();

    RAJA::region<RAJA::hpx_parallel_region>([len, tile_size, &segment, &body]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

      auto ex =
          ::hpx::execution::par.on(::hpx::execution::parallel_executor{}).with(tile_size);

      auto irange = ::hpx::util::detail::make_counting_shape(len);

      ::hpx::for_each(::hpx::util::begin(irange), ::hpx::util::end(irange), [&loop_body, &segment, tile_size](const int i) {
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
std::cout << "TileICountExecute:exec:1" << std::endl;

    const int len = segment.end() - segment.begin();
    const int numTiles = (len - 1) / tile_size + 1;

    RAJA::region<RAJA::hpx_parallel_region>([&body, &segment, tile_size, numTiles]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

      auto irange = ::hpx::util::detail::make_counting_shape(numTiles);
      ::hpx::for_each(::hpx::execution::par, ::hpx::util::begin(irange), ::hpx::util::end(irange), [&loop_body, &segment, tile_size](const int i) {
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
std::cout << "TileExecute:exec:3" << std::endl;

    int len = segment.end() - segment.begin();

    auto ex =
        ::hpx::execution::par.on(::hpx::execution::parallel_executor{}).with(tile_size);

    auto irange = ::hpx::util::detail::make_counting_shape(len);

    ::hpx::for_each(::hpx::execution::par, ::hpx::util::begin(irange), ::hpx::util::end(irange), [&body, &segment, tile_size](const int i) {
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
std::cout << "TileICountExecute:exec:2" << std::endl;

    const int len = segment.end() - segment.begin();
    const int numTiles = (len - 1) / tile_size + 1;

    auto irange = ::hpx::util::detail::make_counting_shape(numTiles);
    ::hpx::for_each(::hpx::execution::par, ::hpx::util::begin(irange), ::hpx::util::end(irange), [tile_size, &segment, &body](const int i) {
      const int i_tile_size = i * tile_size;
      body.get_priv()(segment.slice(i_tile_size, tile_size), i);
    });
  }
};

}  // namespace expt

}  // namespace RAJA
#endif
