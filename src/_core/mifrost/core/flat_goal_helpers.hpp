#pragma once

#include <fmt/format.h>

#include <mimir/formalism/ground_action.hpp>
#include <mimir/search/state.hpp>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <vector>

#include "common_types.hpp"
#include "default_relations.hpp"
#include "flat_tuple_layout.hpp"
#include "goal_inputs.hpp"
#include "relation_formatter.hpp"
#include "state_fact_iteration.hpp"
#include "target_source.hpp"

namespace mifrost {

struct FlatRelationConfigView {
   bool include_lgan_edges = false;
   bool use_predicate_virtual_nodes = false;
   bool ignore_zero_arity_relations = true;
   std::set< TargetSource > lgan_anchor_sources;
   std::set< TargetSource > target_sources;
};

struct FlatTargetEntityKey {
   TargetSource source = TargetSource::actions;
   int64_t discriminator = 0;
   int64_t primary = 0;
   int64_t secondary = 0;
   int64_t tertiary = 0;
   int64_t quaternary = 0;

   auto operator==(const FlatTargetEntityKey& other) const -> bool = default;
};

struct FlatTargetEntityKeyHash {
   using is_avalanching = void;

   [[nodiscard]] auto operator()(const FlatTargetEntityKey& key) const noexcept -> uint64_t;
};

struct PreparedHistoryEntry {
   int dt = 0;
   size_t entry_idx = 0;
   std::vector< LiteralVariant > literals;
};

GoalInputs default_goal_inputs_for_state(const mimir::search::State& state);
bool has_lgan_anchor_source(const FlatRelationConfigView& config, TargetSource source);
bool has_anchor_entity_source(const FlatRelationConfigView& config, TargetSource source);
std::optional< TargetSource > anchor_source_for_goal_level(
   const FlatRelationConfigView& config,
   const std::optional< size_t >& goal_level
);
FlatTupleLayout goal_relation_layout(
   const FlatRelationConfigView& config,
   int logical_arity,
   const std::optional< size_t >& goal_level
);
FlatTupleLayout history_relation_layout(const FlatRelationConfigView& config, int logical_arity);
FlatTargetEntityKey action_target_entity_key(const mimir::formalism::GroundAction& action);
std::vector< PreparedHistoryEntry > prepare_history_entries(
   std::span< const std::pair< int, std::vector< LiteralVariant > > > history_subgoals,
   std::optional< int > history_max_steps
);

/// Template implementations

template < typename GoalTag >
FlatTargetEntityKey goal_target_entity_key(
   TargetSource source,
   const mimir::formalism::GroundLiteral< GoalTag >& literal,
   const std::optional< size_t >& goal_level
)
{
   return FlatTargetEntityKey{
      .source = source,
      .discriminator = static_cast< int64_t >(state_fact_tag_id< GoalTag >()),
      .primary = static_cast< int64_t >(literal->get_atom()->get_index()),
      .secondary = literal->get_polarity() ? 1 : 0,
      .tertiary = goal_level.has_value() ? static_cast< int64_t >(*goal_level) : -1,
      .quaternary = 0,
   };
}

template < typename HistoryTag >
FlatTargetEntityKey history_target_entity_key(
   int dt,
   size_t entry_idx,
   const mimir::formalism::GroundLiteral< HistoryTag >& literal
)
{
   return FlatTargetEntityKey{
      .source = TargetSource::history,
      .discriminator = static_cast< int64_t >(state_fact_tag_id< HistoryTag >()),
      .primary = static_cast< int64_t >(dt),
      .secondary = static_cast< int64_t >(entry_idx),
      .tertiary = static_cast< int64_t >(literal->get_atom()->get_index()),
      .quaternary = literal->get_polarity() ? 1 : 0,
   };
}

template < typename GoalTag >
std::string goal_target_display_name(
   const mimir::formalism::GroundLiteral< GoalTag >& literal,
   const std::optional< size_t >& goal_level
)
{
   if(goal_level.has_value()) {
      return RelationFormatter::format_literal< GoalTag >(literal, GoalLevel(*goal_level));
   }
   return RelationFormatter::format_literal< GoalTag >(literal, std::nullopt);
}

template < typename HistoryTag >
std::string history_target_display_name(
   int dt,
   size_t entry_idx,
   const mimir::formalism::GroundLiteral< HistoryTag >& literal
)
{
   return fmt::format(
      "history:{}#{}:{}",
      dt,
      entry_idx,
      RelationFormatter::format_literal< HistoryTag >(literal, std::nullopt)
   );
}

template < typename GoalLevelsMap, typename LiteralTag >
std::optional< size_t > goal_level_for(
   const GoalLevelsMap& goal_levels,
   const mimir::formalism::GroundLiteral< LiteralTag >& literal
)
{
   if(const auto it = goal_levels.find(literal); it != goal_levels.end()) {
      return it->second;
   }
   return std::nullopt;
}

template < typename HistoryTag >
std::string
history_relation_name(const mimir::formalism::Predicate< HistoryTag >& predicate, bool polarity)
{
   return RelationFormatter::format_predicate(
      predicate, std::nullopt, std::nullopt, polarity, "[hist]"
   );
}

}  // namespace mifrost
