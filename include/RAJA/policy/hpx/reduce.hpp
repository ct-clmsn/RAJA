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
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_hpx_reduce_HPP
#define RAJA_hpx_reduce_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HPX)

#include <memory>
#include <vector>
#include <mutex>

#include <hpx/local/shared_mutex.hpp>
#include <hpx/local/thread.hpp>
#include <hpx/modules/synchronization.hpp>

#include "RAJA/util/types.hpp"

#include "RAJA/pattern/detail/reduce.hpp"
#include "RAJA/pattern/reduce.hpp"

#include "RAJA/policy/hpx/policy.hpp"

namespace RAJA
{

namespace detail
{
template <typename T, typename Reduce>
class ReduceHPX
{
private:
  static ::hpx::mutex mtx_;

  //! HPX native per-thread container
  std::shared_ptr<T> data;

public:
  //! default constructor calls the reset method
  ReduceHPX() { reset(T(), T()); }

  //! constructor requires a default value for the reducer
  explicit ReduceHPX(T init_val, T initializer)
  {
    reset(init_val, initializer);
  }

  void reset(T init_val, T initializer)
  {
    data = std::shared_ptr<T>(
        std::make_shared<T>( initializer ));
    (*data) = init_val;
  }

  /*!
   *  \return the calculated reduced value
   */
  T get() const { return local(); }

  /*!
   *  \return update the local value
   */
  void combine(const T& other) {
     std::unique_lock<::hpx::mutex> l{mtx_};
     Reduce{}(*(data), other);
  }

  /*!
   *  \return reference to the local value
   */
  T& local() const { return (*data); }
};

template <typename T, typename Reduce>
::hpx::mutex ReduceHPX<T, Reduce>::mtx_{};

}  // namespace detail

RAJA_DECLARE_ALL_REDUCERS(hpx_reduce, detail::ReduceHPX)

///////////////////////////////////////////////////////////////////////////////
//
// Old ordered reductions are included below.
//
///////////////////////////////////////////////////////////////////////////////

namespace detail
{
template <typename T, typename Reduce>
class ReduceHPXOrdered
    : public reduce::detail::
          BaseCombinable<T, Reduce, ReduceHPXOrdered<T, Reduce>>
{
  using Base = reduce::detail::BaseCombinable<T, Reduce, ReduceHPXOrdered>;
  std::shared_ptr<std::vector<T>> data;

public:
  ReduceHPXOrdered() { reset(T(), T()); }

  //! constructor requires a default value for the reducer
  explicit ReduceHPXOrdered(T init_val, T identity_)
  {
    reset(init_val, identity_);
  }

  void reset(T init_val, T identity_)
  {
    Base::reset(init_val, identity_);
    data = std::shared_ptr<std::vector<T>>(
        std::make_shared<std::vector<T>>(::hpx::threads::get_thread_data(::hpx::threads::get_self_id()), identity_));
  }

  ~ReduceHPXOrdered()
  {
    Reduce{}((*data)[::hpx::threads::get_thread_data(::hpx::threads::get_self_id())], Base::my_data);
    Base::my_data = Base::identity;
  }

  T get_combined() const
  {
    if (Base::my_data != Base::identity) {
      Reduce{}((*data)[::hpx::threads::get_thread_data(::hpx::threads::get_self_id())], Base::my_data);
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

RAJA_DECLARE_ALL_REDUCERS(hpx_reduce_ordered, detail::ReduceHPXOrdered)

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_HPX guard

#endif  // closing endif for header file include guard
