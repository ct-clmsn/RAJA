/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining OpenMP atomic operations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_hpx_atomic_HPP
#define RAJA_policy_hpx_atomic_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HPX)

#include "RAJA/policy/hpx/policy.hpp"

#include "RAJA/util/macros.hpp"


namespace RAJA
{

// Relies on builtin_atomic when OpenMP can't do the job
#if !defined(RAJA_COMPILER_MSVC)

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicAdd(hpx_atomic, T volatile *acc, T value)
{
  std::atomic<T> val{*acc};
  val.store(*acc);
  return val.fetch_add(value);
}


RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicSub(hpx_atomic, T volatile *acc, T value)
{
  std::atomic<T> val{*acc};
  val.store(*acc);
  return val.fetch_sub(value);
}


RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicMin(hpx_atomic, T volatile *acc, T value)
{
  // OpenMP doesn't define atomic trinary operators so use builtin atomics
  return atomicMin(builtin_atomic{}, acc, value);
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicMax(hpx_atomic, T volatile *acc, T value)
{
  // OpenMP doesn't define atomic trinary operators so use builtin atomics
  return atomicMax(builtin_atomic{}, acc, value);
}


RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicInc(hpx_atomic, T volatile *acc)
{
  std::atomic<T> val{*acc};
  val.store(*acc);
  return val.fetch_add(1);
}


RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicInc(hpx_atomic, T volatile *acc, T val)
{
  // OpenMP doesn't define atomic trinary operators so use builtin atomics
  return RAJA::atomicInc(builtin_atomic{}, acc, val);
}


RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicDec(hpx_atomic, T volatile *acc)
{
  std::atomic<T> val{*acc};
  val.store(*acc);
  return val.fetch_sub(1);
}


RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicDec(hpx_atomic, T volatile *acc, T val)
{
  // OpenMP doesn't define atomic trinary operators so use builtin atomics
  return RAJA::atomicDec(builtin_atomic{}, acc, val);
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicAnd(hpx_atomic, T volatile *acc, T value)
{
  std::atomic<T> val{*acc};
  val.store(*acc);
  return val.fetch_and(value);
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicOr(hpx_atomic, T volatile *acc, T value)
{
  std::atomic<T> val{*acc};
  val.store(*acc);
  return val.fetch_or(value);
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicXor(hpx_atomic, T volatile *acc, T value)
{
  std::atomic<T> val{*acc};
  val.store(*acc);
  return val.fetch_xor(value);
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicExchange(hpx_atomic, T volatile *acc, T value)
{
  std::atomic<T> val{*acc};
  val.store(*acc);
  return val.exchange(value);
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicCAS(hpx_atomic, T volatile *acc, T chpxare, T value)
{
  // OpenMP doesn't define atomic trinary operators so use builtin atomics
  return RAJA::atomicCAS(builtin_atomic{}, acc, chpxare, value);
}

#endif // not defined RAJA_COMPILER_MSVC


}  // namespace RAJA

#endif  // RAJA_ENABLE_HPX
#endif  // guard
