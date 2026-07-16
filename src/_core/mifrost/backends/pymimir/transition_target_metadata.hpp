/**
 * @file transition_target_metadata.hpp
 * @brief Pymimir transition-DAG conversion into neutral target rows.
 */
#pragma once

#include <optional>

#include "mifrost/backends/pymimir/encoders/common/transition_dag.hpp"
#include "mifrost/core/common_types.hpp"
#include "mifrost/core/encoders/common/root_policy.hpp"
#include "mifrost/core/encoders/common/target_metadata.hpp"

namespace mifrost::pymimir_backend {

MIFROST_API std::vector< TargetCandidateRow > collect_transition_dag_target_candidate_rows(
   const TransitionDAG& dag,
   const hash_map< int64_t, int64_t >& positions_by_index,
   RootPolicy root_policy,
   std::optional< int64_t > group_id,
   bool include_names = true
);

}  // namespace mifrost::pymimir_backend
