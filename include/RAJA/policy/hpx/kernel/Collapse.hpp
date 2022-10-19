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

#ifndef RAJA_policy_hpx_kernel_collapse_HPP
#define RAJA_policy_hpx_kernel_collapse_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HPX)

#include <hpx/local/execution.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>

#include "RAJA/pattern/detail/privatizer.hpp"

#include "RAJA/pattern/kernel/Collapse.hpp"
#include "RAJA/pattern/kernel/internal.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/policy/hpx/policy.hpp"

namespace RAJA
{

struct hpx_parallel_collapse_exec
    : make_policy_pattern_t<RAJA::Policy::hpx,
                            RAJA::Pattern::forall,
                            RAJA::policy::hpx::For> {
};

namespace internal
{

/////////
// Collapsing two loops
/////////

template <camp::idx_t Arg0, camp::idx_t Arg1, typename... EnclosedStmts, typename Types>
struct StatementExecutor<statement::Collapse<hpx_parallel_collapse_exec,
                                             ArgList<Arg0, Arg1>,
                                             EnclosedStmts...>, Types> {


  template <typename Data>
  static RAJA_INLINE void exec(Data&& data)
  {
    const auto l0 = segment_length<Arg0>(data);
    const auto l1 = segment_length<Arg1>(data);

    auto i0 = l0;
    auto i1 = l1;

    const auto len = l0 * l1;

    std::shared_ptr<Data> data_ptr(&data);
    auto privatizer = thread_privatize(data_ptr);

    // Set the argument types for this loop
    using NewTypes0 = setSegmentTypeFromData<Types, Arg0, Data>;
    using NewTypes1 = setSegmentTypeFromData<NewTypes0, Arg1, Data>;

    auto rng = ::hpx::util::detail::make_counting_shape(len);

    ::hpx::for_each(::hpx::execution::par,
        std::begin(rng), std::end(rng), [&](const auto i) {
            const auto x = i / l1;
            const auto y = i % l1;
            auto& private_data = privatizer.get_priv();
            private_data.template assign_offset<Arg0>(x);
            private_data.template assign_offset<Arg1>(y);
            execute_statement_list<camp::list<EnclosedStmts...>, NewTypes1>(private_data);
        }
    );
  }
};

template <camp::idx_t Arg0,
          camp::idx_t Arg1,
          camp::idx_t Arg2,
          typename... EnclosedStmts,
          typename Types>
struct StatementExecutor<statement::Collapse<hpx_parallel_collapse_exec,
                                             ArgList<Arg0, Arg1, Arg2>,
                                             EnclosedStmts...>, Types> {


  template <typename Data>
  static RAJA_INLINE void exec(Data&& data)
  {
    const auto l0 = segment_length<Arg0>(data);
    const auto l1 = segment_length<Arg1>(data);
    const auto l2 = segment_length<Arg2>(data);
    auto i0 = l0;
    auto i1 = l1;
    auto i2 = l2;

    const auto len = l0 * l1 * l2;

    std::shared_ptr<Data> data_ptr(&data);
    auto privatizer = thread_privatize(data_ptr);

    // Set the argument types for this loop
    using NewTypes0 = setSegmentTypeFromData<Types, Arg0, Data>;
    using NewTypes1 = setSegmentTypeFromData<NewTypes0, Arg1, Data>;
    using NewTypes2 = setSegmentTypeFromData<NewTypes1, Arg2, Data>;

    auto rng = ::hpx::util::detail::make_counting_shape(len);

    const auto YZ = l1 * l2;

    ::hpx::for_each(::hpx::execution::par,
        std::begin(rng), std::end(rng), [&](const auto i) {
            const auto modiYZ = i % YZ;
            const auto z = modiYZ % l2;
            const auto y = modiYZ / l2;
            const auto x = i / YZ; 
            auto& private_data = privatizer.get_priv();
            private_data.template assign_offset<Arg0>(x);
            private_data.template assign_offset<Arg1>(y);
            private_data.template assign_offset<Arg2>(z);
            execute_statement_list<camp::list<EnclosedStmts...>, NewTypes2>(private_data);
        }
    );
  }
};

}  // namespace internal
}  // namespace RAJA

#undef RAJA_COLLAPSE

#endif  // closing endif for RAJA_ENABLE_HPX guard

#endif  // closing endif for header file include guard
