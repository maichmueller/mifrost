#pragma once

#include <map>
#include <optional>
#include <span>

#include "flat_relation_encoder.hpp"

namespace mifrost {

/**
 * @brief Inputs needed to assemble one flat relation encoding context.
 *
 * This keeps the relation encoder orchestration small while leaving the
 * relation-specific target/history semantics in a dedicated implementation unit.
 */
struct FlatRelationContextBuildConfig {
   FlatRelationConfigView relation_config;
   bool ignore_zero_arity_relations = true;
   bool supports_target_metadata = false;
   size_t predicate_symbol_capacity = 0;
   const std::map< TargetSource, int64_t >* target_entity_group_ids = nullptr;
   const std::map< TargetSource, int64_t >* target_metadata_group_ids = nullptr;
};

FlatRelationEncoderEngine::EncodingContext build_flat_relation_encoding_context(
   const mimir::search::State& state,
   const GoalInputs& goals,
   std::span< const mimir::formalism::GroundAction > actions,
   std::span< const FlatRelationEncoderEngine::HistorySubgoal > history_subgoals,
   std::optional< int > history_max_steps,
   const FlatRelationContextBuildConfig& config
);

}  // namespace mifrost
