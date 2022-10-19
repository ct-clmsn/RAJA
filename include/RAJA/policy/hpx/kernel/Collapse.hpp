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

#ifndef RAJA_policy_openmp_kernel_collapse_HPP
#define RAJA_policy_openmp_kernel_collapse_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_OPENMP)

#include "RAJA/pattern/detail/privatizer.hpp"

#include "RAJA/pattern/kernel/Collapse.hpp"
#include "RAJA/pattern/kernel/internal.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/policy/openmp/policy.hpp"

namespace RAJA
{

struct omp_parallel_collapse_exec
    : make_policy_pattern_t<RAJA::Policy::openmp,
                            RAJA::Pattern::forall,
                            RAJA::policy::omp::For> {
};

namespace internal
{

/////////
// Collapsing two loops
/////////

template <camp::idx_t Arg0, camp::idx_t Arg1, typename... EnclosedStmts, typename Types>
struct StatementExecutor<statement::Collapse<omp_parallel_collapse_exec,
                                             ArgList<Arg0, Arg1>,
                                             EnclosedStmts...>, Types> {


  template <typename Data>
  static RAJA_INLINE void exec(Data&& data)
  {
    using namespace hpx::parallel::util::range;

    const auto len = segment_length<Arg0*Arg1>(data);

    std::shared_ptr<Data> data_ptr(&data);
    auto privatizer = thread_privatize(data_ptr);

    // Set the argument types for this loop
    using NewTypes0 = setSegmentTypeFromData<Types, Arg0, Data>;
    using NewTypes1 = setSegmentTypeFromData<NewTypes0, Arg1, Data>;

    using par_t = decltype(hpx::execution::par);
    range<iter_t> rng(0, len);

    hpx::foreach(std::forward<par_t>(hpx::execution::par),
        std::begin(rng), std::end(rng), [&](const auto i) {
            auto& private_data = privatizer.get_priv();
            private_data.template assign_offset<Arg0*Arg1>(i);
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
struct StatementExecutor<statement::Collapse<omp_parallel_collapse_exec,
                                             ArgList<Arg0, Arg1, Arg2>,
                                             EnclosedStmts...>, Types> {


  template <typename Data>
  static RAJA_INLINE void exec(Data&& data)
  {
    using namespace hpx::parallel::util::range;

    const auto len = segment_length<Arg0*Arg1*Arg2>(data);

    std::shared_ptr<Data> data_ptr(&data);
    auto privatizer = thread_privatize(data_ptr);

    // Set the argument types for this loop
    using NewTypes0 = setSegmentTypeFromData<Types, Arg0, Data>;
    using NewTypes1 = setSegmentTypeFromData<NewTypes0, Arg1, Data>;
    using NewTypes2 = setSegmentTypeFromData<NewTypes1, Arg2, Data>;

    using par_t = decltype(hpx::execution::par);
    range<iter_t> rng(0, len);

    hpx::foreach(std::forward<par_t>(hpx::execution::par),
        std::begin(rng), std::end(rng), [&](const auto i) {
            auto& private_data = privatizer.get_priv();
            private_data.template assign_offset<Arg0*Arg1*Arg2>(i);
            execute_statement_list<camp::list<EnclosedStmts...>, NewTypes2>(private_data);
        }
    );
  }
};

}  // namespace internal
}  // namespace RAJA

#undef RAJA_COLLAPSE

#endif  // closing endif for RAJA_ENABLE_OPENMP guard

#endif  // closing endif for header file include guard
