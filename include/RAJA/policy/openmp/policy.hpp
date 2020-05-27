/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA OpenMP policy definitions.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef policy_openmp_HPP
#define policy_openmp_HPP

#include <type_traits>

#include "RAJA/policy/PolicyBase.hpp"

namespace RAJA
{
namespace policy
{

namespace omp
{

namespace internal
{
    struct Schedule {};
}

struct Parallel {
};

struct Collapse {
};

struct For {
};

struct NoWait {
};

template <unsigned int ChunkSize>
struct Static : std::integral_constant<unsigned int, ChunkSize>, ::RAJA::policy::omp::internal::Schedule {
};

template <unsigned int ChunkSize>
struct Dynamic : std::integral_constant<unsigned int, ChunkSize>, ::RAJA::policy::omp::internal::Schedule {
};

template <unsigned int ChunkSize>
struct Guided : std::integral_constant<unsigned int, ChunkSize>, ::RAJA::policy::omp::internal::Schedule {
};

struct Auto : ::RAJA::policy::omp::internal::Schedule {
};

struct Runtime : ::RAJA::policy::omp::internal::Schedule {
};

//
//////////////////////////////////////////////////////////////////////
//
// Execution policies
//
//////////////////////////////////////////////////////////////////////
//

struct omp_parallel_region
    : make_policy_pattern_launch_platform_t<Policy::openmp,
                                            Pattern::region,
                                            Launch::undefined,
                                            Platform::host> {
};

template <typename Schedule>
struct omp_for_nowait_schedule : make_policy_pattern_launch_platform_t<Policy::openmp,
                                                              Pattern::forall,
                                                              Launch::undefined,
                                                              Platform::host,
                                                              omp::For,
                                                              omp::NoWait,
                                                              Schedule> {
    static_assert(std::is_base_of<::RAJA::policy::omp::internal::Schedule, Schedule>,
        "Schedule must be one of:\nRuntime|Auto|{Default,}{Static,Dynamic,Guided}");
};


template <typename Schedule>
struct omp_for_schedule : make_policy_pattern_launch_platform_t<Policy::openmp,
                                                              Pattern::forall,
                                                              Launch::undefined,
                                                              Platform::host,
                                                              omp::For,
                                                              Schedule> {
    static_assert(std::is_base_of<::RAJA::policy::omp::internal::Schedule, Schedule>,
        "Schedule must be one of:\nRuntime|Auto|{Default,}{Static,Dynamic,Guided}");
};

struct omp_for_exec
    : make_policy_pattern_t<Policy::openmp, Pattern::forall, omp::For> {
};

struct omp_for_nowait_exec : 
    : make_policy_pattern_launch_platform_t<Policy::openmp,
                                            Pattern::forall,
                                            Launch::undefined,
                                            Platform::host,
                                            omp::For,
                                            omp::NoWait> {
};

template <unsigned int N>
struct omp_for_static : omp_for_schedule<omp::Static<N>> {
};


template <typename InnerPolicy>
struct omp_parallel_exec
    : make_policy_pattern_launch_platform_t<Policy::openmp,
                                            Pattern::forall,
                                            Launch::undefined,
                                            Platform::host,
                                            omp::Parallel,
                                            wrapper<InnerPolicy>> {
};

struct omp_parallel_for_exec : omp_parallel_exec<omp_for_exec> {
};

template <unsigned int N>
struct omp_parallel_for_static : omp_parallel_exec<omp_for_static<N>> {
};


///
/// Index set segment iteration policies
///

using omp_parallel_for_segit = omp_parallel_for_exec;

using omp_parallel_segit = omp_parallel_for_segit;

struct omp_taskgraph_segit
    : make_policy_pattern_t<Policy::openmp, Pattern::taskgraph, omp::Parallel> {
};

struct omp_taskgraph_interval_segit
    : make_policy_pattern_t<Policy::openmp, Pattern::taskgraph, omp::Parallel> {
};

///
///////////////////////////////////////////////////////////////////////
///
/// Reduction execution policies
///
///////////////////////////////////////////////////////////////////////
///

struct omp_reduce : make_policy_pattern_t<Policy::openmp, Pattern::reduce> {
};

struct omp_reduce_ordered
    : make_policy_pattern_t<Policy::openmp, Pattern::reduce, reduce::ordered> {
};

struct omp_synchronize : make_policy_pattern_launch_t<Policy::openmp,
                                                      Pattern::synchronize,
                                                      Launch::sync> {
};

}  // namespace omp
}  // namespace policy

using policy::omp::omp_for_exec;
using policy::omp::omp_for_nowait_exec;
using policy::omp::omp_for_static;
using policy::omp::omp_parallel_exec;
using policy::omp::omp_parallel_for_exec;
using policy::omp::omp_parallel_for_segit;
using policy::omp::omp_parallel_region;
using policy::omp::omp_parallel_segit;
using policy::omp::omp_reduce;
using policy::omp::omp_reduce_ordered;
using policy::omp::omp_synchronize;




}  // namespace RAJA


#endif
