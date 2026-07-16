#pragma once

// Pymimir graph context for the compatibility flat relation engine.

#include <map>
#include <optional>
#include <span>

#include "flat_relation_encoder.hpp"

namespace mifrost {

/**
 * @brief Inputs needed to assemble one flat relation encoding context.
 *
 * This keeps the relation encoder smaller while moving the target and history
 * setup into one focused helper file.
 */
struct FlatRelationContextBuildConfig {
   FlatRelationConfigView relation_config;
   bool ignore_zero_arity_relations = true;
   bool supports_target_metadata = false;
   size_t predicate_symbol_capacity = 0;
   const std::map< TargetSource, int64_t >* target_entity_group_ids = nullptr;
   const std::map< TargetSource, int64_t >* target_metadata_group_ids = nullptr;
};

/// Build all per-graph rows and maps needed before flat relation tuple emission.
FlatRelationEncoderEngine::EncodingContext build_flat_relation_encoding_context(
   const mimir::search::State& state,
   const GoalInputs& goals,
   std::span< const mimir::formalism::GroundAction > actions,
   std::span< const FlatRelationEncoderEngine::HistorySubgoal > history_subgoals,
   std::optional< int > history_max_steps,
   const FlatRelationContextBuildConfig& config
);

}  // namespace mifrost
