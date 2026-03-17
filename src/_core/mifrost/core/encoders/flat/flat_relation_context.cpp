/**
 * @file flat_relation_context.cpp
 * @brief Per-graph setup for the flat relation encoder.
 *
 * The relation encoder uses this file to fill all node rows and lookup maps
 * before tuple emission starts. Emission code assumes the returned context is complete.
 */
#include "flat_relation_context.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "flat_entity_context.hpp"

namespace mifrost {

namespace {

int64_t lookup_group_id(
   const std::map< TargetSource, int64_t >& group_ids,
   TargetSource source,
   std::string_view error_prefix
)
{
   const auto it = group_ids.find(source);
   if(it == group_ids.end()) {
      throw std::invalid_argument(
         std::string(error_prefix) + std::string(target_source_group_name(source)) + "'"
      );
   }
   return it->second;
}

}  // namespace

FlatRelationEncoderEngine::EncodingContext build_flat_relation_encoding_context(
   const mimir::search::State& state,
   const GoalInputs& goals,
   std::span< const mimir::formalism::GroundAction > actions,
   std::span< const FlatRelationEncoderEngine::HistorySubgoal > history_subgoals,
   std::optional< int > history_max_steps,
   const FlatRelationContextBuildConfig& config
)
{
   FlatRelationEncoderEngine::EncodingContext context;
   const auto& objects = state.get_problem().get_problem_and_domain_objects();
   std::vector< mimir::formalism::Object > ordered(objects.begin(), objects.end());
   std::ranges::sort(ordered, [](const auto& lhs, const auto& rhs) {
      return lhs->get_index() < rhs->get_index();
   });
   const auto prepared_history = prepare_history_entries(history_subgoals, history_max_steps);

   reserve_common_entity_context(
      context,
      ordered.size(),
      actions.size() + prepared_history.size(),
      config.predicate_symbol_capacity
   );
   context.history_entity_indices.reserve(prepared_history.size());
   context.history_entity_dt.reserve(prepared_history.size());
   context.target_entity_indices.reserve(ordered.size() + actions.size() + prepared_history.size());
   context.target_entity_group_ids.reserve(actions.size() + prepared_history.size());
   context.target_entity_index_by_key.reserve(actions.size() + prepared_history.size());
   context.history_entries.reserve(prepared_history.size());
   context.unique_actions.reserve(actions.size());
   if(config.supports_target_metadata) {
      const size_t total_goal_literals = goals.static_goals.size() + goals.fluent_goals.size()
                                         + goals.derived_goals.size();
      size_t total_history_literals = 0;
      for(const auto& entry : prepared_history) {
         total_history_literals += entry.literals.size();
      }
      context.target_columns.reserve(
         total_goal_literals + actions.size() + total_history_literals,
         /*include_depth=*/false,
         /*include_group=*/true
      );
   }

   append_object_entities(context, ordered);

   auto ensure_target_entity =
      [&](
         const FlatTargetEntityKey& key,
         TargetSource source,
         const std::string& name,
         const std::optional< mimir::formalism::GroundAction >& action = std::nullopt
      ) {
         if(const auto it = context.target_entity_index_by_key.find(key);
            it != context.target_entity_index_by_key.end()) {
            return it->second;
         }
         const int64_t local_index = static_cast< int64_t >(context.entity_names.size());
         context.target_entity_index_by_key.emplace(key, local_index);
         context.entity_names.push_back(name);
         context.entity_role_ids.push_back(
            static_cast< int64_t >(entity_role_for_target_source(source))
         );
         context.target_entity_indices.push_back(local_index);
         context.target_entity_group_ids.push_back(lookup_group_id(
            *config.target_entity_group_ids,
            source,
            "FlatRelationEncoder does not define a target-entity group for source '"
         ));
         if(action.has_value()) {
            context.unique_actions.push_back(*action);
         }
         return local_index;
      };

   auto append_history_entity = [&](int dt, size_t entry_idx) {
      const int64_t local_index = static_cast< int64_t >(context.entity_names.size());
      context.entity_names.push_back(fmt::format("history:{}#{}", dt, entry_idx));
      context.entity_role_ids.push_back(static_cast< int64_t >(FlatEntityRole::history));
      context.history_entity_indices.push_back(local_index);
      context.history_entity_dt.push_back(static_cast< int64_t >(dt));
      return local_index;
   };

   auto append_target_row = [&](TargetSource source, int64_t position, const std::string& name) {
      const int64_t target_index = static_cast< int64_t >(context.target_columns.size());
      append_target_candidate_row(
         context.target_columns,
         TargetCandidateRow{
            .position = position,
            .index = target_index,
            .candidate_id = target_index,
            .depth = std::nullopt,
            .group_id = lookup_group_id(
               *config.target_metadata_group_ids,
               source,
               "FlatRelationEncoder does not define target metadata for source '"
            ),
            .name = name,
         },
         TargetCandidateAppendConfig{
            .include_depth = false,
            .include_group = true,
         }
      );
   };

   auto collect_goal_targets =
      [&]< typename GoalTag >(
         std::span< const mimir::formalism::GroundLiteral< GoalTag > > literals,
         const auto& goal_levels,
         TargetSource source
      ) {
         for(const auto& literal : literals) {
            const auto predicate = literal->get_atom()->get_predicate();
            const int arity = static_cast< int >(predicate->get_arity());
            if(config.ignore_zero_arity_relations && arity == 0) {
               continue;
            }
            const auto goal_level = goal_level_for(goal_levels, literal);
            const bool is_subgoal = goal_level.has_value() && *goal_level > 0;
            if((source == TargetSource::goals && is_subgoal)
               || (source == TargetSource::subgoals && ! is_subgoal)) {
               continue;
            }
            const auto display_name = goal_target_display_name(literal, goal_level);
            const auto local_index = ensure_target_entity(
               goal_target_entity_key(source, literal, goal_level), source, display_name
            );
            if(config.relation_config.target_sources.contains(source)) {
               append_target_row(source, local_index, display_name);
            }
         }
      };

   if(has_anchor_entity_source(config.relation_config, TargetSource::goals)) {
      collect_goal_targets(
         std::span{goals.static_goals}, goals.static_goal_levels, TargetSource::goals
      );
      collect_goal_targets(
         std::span{goals.fluent_goals}, goals.fluent_goal_levels, TargetSource::goals
      );
      collect_goal_targets(
         std::span{goals.derived_goals}, goals.derived_goal_levels, TargetSource::goals
      );
   }

   if(has_anchor_entity_source(config.relation_config, TargetSource::subgoals)) {
      collect_goal_targets(
         std::span{goals.static_goals}, goals.static_goal_levels, TargetSource::subgoals
      );
      collect_goal_targets(
         std::span{goals.fluent_goals}, goals.fluent_goal_levels, TargetSource::subgoals
      );
      collect_goal_targets(
         std::span{goals.derived_goals}, goals.derived_goal_levels, TargetSource::subgoals
      );
   }

   for(const auto& action : actions) {
      const auto action_name = RelationFormatter::format_action(action);
      const auto local_index = ensure_target_entity(
         action_target_entity_key(action), TargetSource::actions, action_name, action
      );
      if(config.relation_config.target_sources.contains(TargetSource::actions)) {
         append_target_row(TargetSource::actions, local_index, action_name);
      }
   }

   for(const auto& entry : prepared_history) {
      const int64_t history_entity_index = append_history_entity(entry.dt, entry.entry_idx);
      context.history_entries.push_back(
         FlatRelationEncoderEngine::EncodingContext::HistoryEntry{
            .dt = entry.dt,
            .entry_idx = entry.entry_idx,
            .entity_index = history_entity_index,
            .literals = entry.literals,
         }
      );
   }

   if(has_anchor_entity_source(config.relation_config, TargetSource::history)) {
      for(const auto& entry : context.history_entries) {
         for(const auto& literal_variant : entry.literals) {
            std::visit(
               [&]< typename HistoryTag >(
                  const mimir::formalism::GroundLiteral< HistoryTag >& literal
               ) {
                  const auto predicate = literal->get_atom()->get_predicate();
                  const int arity = static_cast< int >(predicate->get_arity());
                  if(config.ignore_zero_arity_relations && arity == 0) {
                     return;
                  }
                  const auto display_name = history_target_display_name(
                     entry.dt, entry.entry_idx, literal
                  );
                  const auto local_index = ensure_target_entity(
                     history_target_entity_key(entry.dt, entry.entry_idx, literal),
                     TargetSource::history,
                     display_name
                  );
                  if(config.relation_config.target_sources.contains(TargetSource::history)) {
                     append_target_row(TargetSource::history, local_index, display_name);
                  }
               },
               literal_variant
            );
         }
      }
   }

   return context;
}

}  // namespace mifrost
