/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA HPX policy definitions.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2022, Tactical Computing Labs, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef policy_hpx_HPP
#define policy_hpx_HPP

#include <type_traits>

#include <hpx/local/execution.hpp>
#include <hpx/local/barrier.hpp>

#include "RAJA/policy/PolicyBase.hpp"

// Rely on builtin_atomic when HPX can't do the job
#include "RAJA/policy/atomic_builtin.hpp"

#if defined(RAJA_COMPILER_MSVC)
typedef enum hpx_sched_t { 
    // schedule kinds 
    hpx_sched_static = 0x1, 
    hpx_sched_dynamic = 0x2, 
    hpx_sched_guided = 0x3, 
    hpx_sched_auto = 0x4, 
    
    // schedule modifier 
    hpx_sched_monotonic = 0x80000000u 
} hpx_sched_t;
#else
#include <hpx.hpp>
#endif

namespace RAJA
{
namespace policy
{
namespace hpx
{

namespace internal
{
    struct ScheduleTag {};

    template <hpx_sched_t Sched, int Chunk>
    struct Schedule : public ScheduleTag {
        constexpr static hpx_sched_t schedule = Sched;
        constexpr static int chunk_size = Chunk;
    };
}  // namespace internal

//
//////////////////////////////////////////////////////////////////////
//
// Basic tag types
//
//////////////////////////////////////////////////////////////////////
//

struct Parallel {
};

struct For {
};

struct NoWait {
};

static constexpr int default_chunk_size = -1;

struct Auto : private internal::Schedule<hpx_sched_auto, default_chunk_size>{
};

template <int ChunkSize = default_chunk_size>
struct Static : public internal::Schedule<hpx_sched_static, ChunkSize> {
};

template <int ChunkSize = default_chunk_size>
using Dynamic = internal::Schedule<hpx_sched_dynamic, ChunkSize>;

template <int ChunkSize = default_chunk_size>
using Guided = internal::Schedule<hpx_sched_guided, ChunkSize>;

struct Runtime : private internal::Schedule<static_cast<hpx_sched_t>(-1), default_chunk_size> {
    Runtime() : exec(hpx::execution::par) {}
    Runtime(hpx::execution::parallel_executor && ex) : exec(ex) {}
    Runtime(hpx::execution::parallel_executor&) = delete;
    Runtime(hpx::execution::parallel_executor) = delete;
    hpx::execution::parallel_executor exec;
};

//
//////////////////////////////////////////////////////////////////////
//
// Execution policies
//
//////////////////////////////////////////////////////////////////////
//

///
///  Struct supporting HPX parallel region. 
///
struct hpx_parallel_region
    : make_policy_pattern_launch_platform_t<Policy::hpx,
                                            Pattern::region,
                                            Launch::undefined,
                                            Platform::host> {
};

///
///  Struct supporting HPX parallel region for Teams
///
struct hpx_launch_t
    : make_policy_pattern_launch_platform_t<Policy::hpx,
                                            Pattern::region,
                                            Launch::undefined,
                                            Platform::host> {
};


///
///  Struct supporting HPX 'for nowait schedule( )'
///
template <typename Sched>
struct hpx_for_nowait_schedule_exec : make_policy_pattern_launch_platform_t<Policy::hpx,
                                                              Pattern::forall,
                                                              Launch::undefined,
                                                              Platform::host,
                                                              hpx::For,
                                                              hpx::NoWait,
                                                              Sched> {
    static_assert(std::is_base_of<::RAJA::policy::hpx::internal::ScheduleTag, Sched>::value,
        "Schedule type must be one of: Auto|Runtime|Static|Dynamic|Guided");
};


///
///  Struct supporting HPX 'for schedule( )'
///
template <typename Sched>
struct hpx_for_schedule_exec : make_policy_pattern_launch_platform_t<Policy::hpx,
                                                              Pattern::forall,
                                                              Launch::undefined,
                                                              Platform::host,
                                                              hpx::For,
                                                              Sched> {
    static_assert(std::is_base_of<::RAJA::policy::hpx::internal::ScheduleTag, Sched>::value,
        "Schedule type must be one of: Auto|Runtime|Static|Dynamic|Guided");
};

///
///  Internal type aliases supporting 'hpx for schedule( )' for specific
///  schedule types.
///
using hpx_for_exec = hpx_for_schedule_exec<Auto>;

///
template <int ChunkSize = default_chunk_size>
using hpx_for_static_exec = hpx_for_schedule_exec<hpx::Static<ChunkSize>>;

///
template <int ChunkSize = default_chunk_size>
using hpx_for_dynamic_exec = hpx_for_schedule_exec<hpx::Dynamic<ChunkSize>>;

///
template <int ChunkSize = default_chunk_size>
using hpx_for_guided_exec = hpx_for_schedule_exec<hpx::Guided<ChunkSize>>;

///
using hpx_for_runtime_exec = hpx_for_schedule_exec<hpx::Runtime>;


///
///  Internal type aliases supporting 'hpx for schedule( ) nowait' for specific
///  schedule types. 
///
///  IMPORTANT: We only provide a nowait policy option for static scheduling
///             since that is the only scheduling case that can be used with
///             nowait and be correct in general. Paraphrasing the HPX 
///             standard:
///             
///             Programs that depend on which thread executes a particular 
///             iteration under any circumstance other than static schedule
///             are non-conforming.
///
template <int ChunkSize = default_chunk_size>
using hpx_for_nowait_static_exec = hpx_for_nowait_schedule_exec<hpx::Static<ChunkSize>>;

///
///  Struct supporting HPX 'parallel' region containing an inner loop
///  execution construct.
///
template <typename InnerPolicy>
using hpx_parallel_exec = make_policy_pattern_launch_platform_t<Policy::hpx,
                                            Pattern::forall,
                                            Launch::undefined,
                                            Platform::host,
                                            hpx::Parallel,
                                            wrapper<InnerPolicy>>;

///
///  Internal type aliases supporting 'hpx parallel for schedule( )' for 
///  specific schedule types.
///
using hpx_parallel_for_exec = hpx_parallel_exec<hpx_for_exec>;

///
template <int ChunkSize = default_chunk_size>
using hpx_parallel_for_static_exec = hpx_parallel_exec<hpx_for_schedule_exec<hpx::Static<ChunkSize>> >;

///
template <int ChunkSize = default_chunk_size>
using hpx_parallel_for_dynamic_exec = hpx_parallel_exec<hpx_for_schedule_exec<hpx::Dynamic<ChunkSize>> >;

///
template <int ChunkSize = default_chunk_size>
using hpx_parallel_for_guided_exec = hpx_parallel_exec<hpx_for_schedule_exec<hpx::Guided<ChunkSize>> >;

///
using hpx_parallel_for_runtime_exec = hpx_parallel_exec<hpx_for_schedule_exec<hpx::Runtime>>;


///
///////////////////////////////////////////////////////////////////////
///
/// Basic Indexset segment iteration policies
///
///////////////////////////////////////////////////////////////////////
///
using hpx_parallel_for_segit = hpx_parallel_for_exec;

///
using hpx_parallel_segit = hpx_parallel_for_segit;


///
///////////////////////////////////////////////////////////////////////
///
/// Taskgraph Indexset segment iteration policies
///
///////////////////////////////////////////////////////////////////////
///
struct hpx_taskgraph_segit
    : make_policy_pattern_t<Policy::hpx, Pattern::taskgraph, hpx::Parallel> {
};

///
struct hpx_taskgraph_interval_segit
    : make_policy_pattern_t<Policy::hpx, Pattern::taskgraph, hpx::Parallel> {
};


///
///////////////////////////////////////////////////////////////////////
///
/// WorkGroup execution policies
///
///////////////////////////////////////////////////////////////////////
///
struct hpx_work : make_policy_pattern_launch_platform_t<Policy::hpx,
                                                        Pattern::workgroup_exec,
                                                        Launch::sync,
                                                        Platform::host> {
};

///
///////////////////////////////////////////////////////////////////////
///
/// Reduction execution policies
///
///////////////////////////////////////////////////////////////////////
///
struct hpx_reduce : make_policy_pattern_t<Policy::hpx, Pattern::reduce> {
};

///
struct hpx_reduce_ordered
    : make_policy_pattern_t<Policy::hpx, Pattern::reduce, reduce::ordered> {
};

///
struct hpx_synchronize : make_policy_pattern_launch_t<Policy::hpx,
                                                      Pattern::synchronize,
                                                      Launch::sync> {
    hpx_synchronize() : bar() {}

    void operator()() { bar.arrive_and_wait(); }

    hpx::barrier<> bar;
};

#if defined(RAJA_COMPILER_MSVC)

// For MS Visual C, just default to builtin_atomic for everything
using hpx_atomic = builtin_atomic;

#else  // RAJA_COMPILER_MSVC not defined

struct hpx_atomic {};

#endif

}  // namespace hpx
}  // namespace policy


///
///////////////////////////////////////////////////////////////////////
///
/// Type aliases exposed to users in the RAJA namespace.
///
///////////////////////////////////////////////////////////////////////
///

///
/// Type alias for atomics
///
using policy::hpx::hpx_atomic;

///
/// Type aliases to simplify common hpx parallel for loop execution
///
using policy::hpx::hpx_parallel_for_exec;
///
using policy::hpx::hpx_parallel_for_static_exec;
///
using policy::hpx::hpx_parallel_for_dynamic_exec;
///
using policy::hpx::hpx_parallel_for_guided_exec;
///
using policy::hpx::hpx_parallel_for_runtime_exec;

///
/// Type aliases for hpx parallel for iteration over indexset segments
///
using policy::hpx::hpx_parallel_for_segit;
///
using policy::hpx::hpx_parallel_segit;

///
/// Type alias for hpx parallel region containing an inner 'hpx for' loop 
/// execution policy. Inner policy types follow.
///
using policy::hpx::hpx_parallel_exec;

///
/// Type alias for 'hpx for' loop execution within an hpx_parallel_exec construct
///
using policy::hpx::hpx_for_exec;

///
/// Type aliases for 'hpx for' and 'hpx for nowait' loop execution with a 
/// scheduling policy within an hpx_parallel_exec construct
/// Scheduling policies are near the top of this file and include:
/// RAJA::policy::hpx::{Auto, Static, Dynamic, Guided, Runtime}
///
/// Helper aliases to make usage less verbose for common use cases follow these.
///
/// Important: 'nowait' schedule must be used with care to guarantee code
///             correctness.
///
using policy::hpx::hpx_for_schedule_exec;
///
using policy::hpx::hpx_for_nowait_schedule_exec;

///
/// Type aliases for 'hpx for' and 'hpx for nowait' loop execution with a 
/// static scheduling policy within an hpx_parallel_exec construct
///
using policy::hpx::hpx_for_static_exec;
///
using policy::hpx::hpx_for_nowait_static_exec;
///
using policy::hpx::hpx_for_dynamic_exec;
///
using policy::hpx::hpx_for_guided_exec;
///
using policy::hpx::hpx_for_runtime_exec;

///
/// Type aliases for hpx parallel region
///
using policy::hpx::hpx_parallel_region;

namespace expt
{
  using policy::hpx::hpx_launch_t;
}

///
/// Type aliases for hpx reductions
///
using policy::hpx::hpx_reduce;
///
using policy::hpx::hpx_reduce_ordered;

///
/// Type aliases for hpx reductions
///
using policy::hpx::hpx_synchronize;

///
using policy::hpx::hpx_work;

}  // namespace RAJA

#endif
