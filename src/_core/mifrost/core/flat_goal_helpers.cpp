#include "flat_goal_helpers.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace mifrost {

GoalInputs default_goal_inputs_for_state(const mimir::search::State& state)
{
   GoalInputs inputs;
   const auto& problem = state.get_problem();
   for(const auto& goal : problem.get_goal_literals< mimir::formalism::StaticTag >()) {
      inputs.append(goal, 0);
   }
   for(const auto& goal : problem.get_goal_literals< mimir::formalism::FluentTag >()) {
      inputs.append(goal, 0);
   }
   for(const auto& goal : problem.get_goal_literals< mimir::formalism::DerivedTag >()) {
      inputs.append(goal, 0);
   }
   return inputs;
}

bool has_lgan_anchor_source(const FlatRelationConfigView& config, TargetSource source)
{
   return config.include_lgan_edges && config.lgan_anchor_sources.contains(source);
}

bool has_anchor_entity_source(const FlatRelationConfigView& config, TargetSource source)
{
   return config.target_sources.contains(source) || has_lgan_anchor_source(config, source);
}

std::optional< TargetSource > anchor_source_for_goal_level(
   const FlatRelationConfigView& config,
   const std::optional< size_t >& goal_level
)
{
   if(goal_level.has_value() && *goal_level > 0) {
      if(has_anchor_entity_source(config, TargetSource::subgoals)) {
         return TargetSource::subgoals;
      }
      return std::nullopt;
   }
   if(has_anchor_entity_source(config, TargetSource::goals)) {
      return TargetSource::goals;
   }
   return std::nullopt;
}

FlatTupleLayout goal_relation_layout(
   const FlatRelationConfigView& config,
   int logical_arity,
   const std::optional< size_t >& goal_level
)
{
   std::vector< FlatSlotRole > auxiliary_slot_roles;
   if(const auto source = anchor_source_for_goal_level(config, goal_level); source.has_value()) {
      auxiliary_slot_roles.push_back(slot_role_for_target_source(*source));
   }
   return make_predicate_tuple_layout(
      logical_arity, std::span{auxiliary_slot_roles}, config.use_predicate_virtual_nodes
   );
}

FlatTupleLayout history_relation_layout(const FlatRelationConfigView& config, int logical_arity)
{
   std::vector< FlatSlotRole > auxiliary_slot_roles;
   if(has_anchor_entity_source(config, TargetSource::history)) {
      auxiliary_slot_roles.push_back(FlatSlotRole::history_target_slot);
   }
   auxiliary_slot_roles.push_back(FlatSlotRole::history_slot);
   return make_predicate_tuple_layout(
      logical_arity, std::span{auxiliary_slot_roles}, config.use_predicate_virtual_nodes
   );
}

FlatTargetEntityKey action_target_entity_key(const mimir::formalism::GroundAction& action)
{
   return FlatTargetEntityKey{
      .source = TargetSource::actions,
      .discriminator = 0,
      .primary = static_cast< int64_t >(action->get_index()),
      .secondary = 0,
      .tertiary = 0,
      .quaternary = 0,
   };
}

std::vector< PreparedHistoryEntry > prepare_history_entries(
   std::span< const std::pair< int, std::vector< LiteralVariant > > > history_subgoals,
   std::optional< int > history_max_steps
)
{
   std::vector< PreparedHistoryEntry > entries;
   entries.reserve(history_subgoals.size());
   for(const auto& [dt, literals] : history_subgoals) {
      if(dt >= 0) {
         throw std::invalid_argument("history_subgoals expects negative dt values");
      }
      if(history_max_steps.has_value() && std::abs(dt) > *history_max_steps) {
         continue;
      }
      entries.push_back(
         PreparedHistoryEntry{
            .dt = dt,
            .entry_idx = entries.size(),
            .literals = literals,
         }
      );
   }

   std::ranges::stable_sort(entries, [](const auto& lhs, const auto& rhs) {
      return lhs.dt < rhs.dt;
   });
   for(size_t idx = 0; idx < entries.size(); ++idx) {
      entries[idx].entry_idx = idx;
   }
   return entries;
}

auto FlatTargetEntityKeyHash::operator()(const FlatTargetEntityKey& key) const noexcept -> uint64_t
{
   uint64_t out = static_cast< uint64_t >(key.source);
   out = (out * 1315423911ULL) ^ static_cast< uint64_t >(key.discriminator);
   out = (out * 1315423911ULL) ^ static_cast< uint64_t >(key.primary);
   out = (out * 1315423911ULL) ^ static_cast< uint64_t >(key.secondary);
   out = (out * 1315423911ULL) ^ static_cast< uint64_t >(key.tertiary);
   out = (out * 1315423911ULL) ^ static_cast< uint64_t >(key.quaternary);
   return out;
}

}  // namespace mifrost
