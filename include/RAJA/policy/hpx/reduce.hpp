/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA reduction templates for
 *          OpenMP execution.
 *
 *          These methods should work on any platform that supports OpenMP.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2022, Tactical Computing Labs, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_hpx_reduce_HPP
#define RAJA_hpx_reduce_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HPX)

#include <hpx/mutex.hpp>

#include <memory>
#include <vector>
#include <mutex>

#include "RAJA/util/types.hpp"

#include "RAJA/pattern/detail/reduce.hpp"
#include "RAJA/pattern/reduce.hpp"

#include "RAJA/policy/hpx/policy.hpp"

namespace RAJA
{

namespace detail
{
//class ReduceHPXOrdered
template <typename T, typename Reduce>
class ReduceHPX
    : public reduce::detail::
          BaseCombinable<T, Reduce, ReduceHPX<T, Reduce>>
{
  using Base = reduce::detail::BaseCombinable<T, Reduce, ReduceHPX>;
  std::shared_ptr<std::vector<T>> data;

public:
  ReduceHPX() { reset(T(), T()); }

  //! constructor requires a default value for the reducer
  explicit ReduceHPX(T init_val, T identity_)
  {
    reset(init_val, identity_);
  }

  void reset(T init_val, T identity_)
  {
    Base::reset(init_val, identity_);
    data = std::shared_ptr<std::vector<T>>(
        std::make_shared<std::vector<T>>(::hpx::get_num_worker_threads(), identity_));
  }

  ~ReduceHPX()
  {
    Reduce{}((*data)[::hpx::get_worker_thread_num()], Base::my_data);
    Base::my_data = Base::identity;
  }

  T get_combined() const
  {
    if (Base::my_data != Base::identity) {
      Reduce{}((*data)[::hpx::get_worker_thread_num()], Base::my_data);
      Base::my_data = Base::identity;
    }

    T res = Base::identity;
    for (size_t i = 0; i < data->size(); ++i) {
      Reduce{}(res, (*data)[i]);
    }
    return res;
  }
};

}  // namespace detail

RAJA_DECLARE_ALL_REDUCERS(hpx_reduce, detail::ReduceHPX)

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_HPX guard

#endif  // closing endif for header file include guard
