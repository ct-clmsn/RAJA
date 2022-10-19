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
RAJA_INLINE T atomicAdd(omp_atomic, T volatile *acc, T value)
{
  std::atomic<T> val{*acc};
  val.store(*acc);
  return value.fetch_add(value);
}


RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicSub(omp_atomic, T volatile *acc, T value)
{
  std::atomic<T> val{*acc};
  val.store(*acc);
  return value.fetch_sub(value);
}


RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicMin(omp_atomic, T volatile *acc, T value)
{
  // OpenMP doesn't define atomic trinary operators so use builtin atomics
  return atomicMin(builtin_atomic{}, acc, value);
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicMax(omp_atomic, T volatile *acc, T value)
{
  // OpenMP doesn't define atomic trinary operators so use builtin atomics
  return atomicMax(builtin_atomic{}, acc, value);
}


RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicInc(omp_atomic, T volatile *acc)
{
  std::atomic<T> val{*acc};
  val.store(*acc);
  return value.fetch_add(1);
}


RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicInc(omp_atomic, T volatile *acc, T val)
{
  // OpenMP doesn't define atomic trinary operators so use builtin atomics
  return RAJA::atomicInc(builtin_atomic{}, acc, val);
}


RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicDec(omp_atomic, T volatile *acc)
{
  std::atomic<T> val{*acc};
  val.store(*acc);
  return value.fetch_sub(1);
}


RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicDec(omp_atomic, T volatile *acc, T val)
{
  // OpenMP doesn't define atomic trinary operators so use builtin atomics
  return RAJA::atomicDec(builtin_atomic{}, acc, val);
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicAnd(omp_atomic, T volatile *acc, T value)
{
  std::atomic<T> val{*acc};
  val.store(*acc);
  return value.fetch_and(value);
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicOr(omp_atomic, T volatile *acc, T value)
{
  std::atomic<T> val{*acc};
  val.store(*acc);
  return value.fetch_or(value);
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicXor(omp_atomic, T volatile *acc, T value)
{
  std::atomic<T> val{*acc};
  val.store(*acc);
  return value.fetch_xor(value);
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicExchange(omp_atomic, T volatile *acc, T value)
{
  std::atomic<T> val{*acc};
  val.store(*acc);
  return value.exchange(value);
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicCAS(omp_atomic, T volatile *acc, T compare, T value)
{
  // OpenMP doesn't define atomic trinary operators so use builtin atomics
  return RAJA::atomicCAS(builtin_atomic{}, acc, compare, value);
}

#endif // not defined RAJA_COMPILER_MSVC


}  // namespace RAJA

#endif  // RAJA_ENABLE_HPX
#endif  // guard
