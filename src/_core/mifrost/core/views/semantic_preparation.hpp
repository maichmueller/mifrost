/**
 * @file semantic_preparation.hpp
 * @brief Compile-time semantic View adapters retaining graph keys only.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <utility>

#include "mifrost/core/encoders/flat/semantic_flat_relation_encoder.hpp"

namespace mifrost::canonical::detail {

using GraphInput = ::mifrost::detail::GraphInput;
using GraphAtomKey = ::mifrost::detail::GraphAtomKey;
using GraphLiteralKey = ::mifrost::detail::GraphLiteralKey;
using GraphActionKey = ::mifrost::detail::GraphActionKey;
using GraphHistoryEntry = ::mifrost::detail::GraphHistoryEntry;

template < views::AtomView Atom >
[[nodiscard]] GraphAtomKey graph_atom_key(const Atom& atom)
{
   GraphAtomKey result;
   result.predicate = static_cast< int64_t >(atom.predicate_id());
   if(result.predicate < 0) {
      throw std::invalid_argument("semantic View contains an invalid predicate ID");
   }
   for(const auto object : atom.arguments()) {
      const auto id = static_cast< int64_t >(object);
      if(id < 0) {
         throw std::invalid_argument("semantic View contains an invalid object ID");
      }
      result.arguments.push_back(id);
   }
   return result;
}

template < views::LiteralView Literal >
[[nodiscard]] GraphLiteralKey graph_literal_key(const Literal& literal)
{
   return GraphLiteralKey{
      graph_atom_key(literal.atom()), not static_cast< bool >(literal.is_negated())
   };
}

template < views::GroundActionView Action >
[[nodiscard]] GraphActionKey graph_action_key(const Action& action)
{
   GraphActionKey result;
   result.action = static_cast< int64_t >(action.schema_id());
   if(result.action < 0) {
      throw std::invalid_argument("semantic View contains an invalid action schema ID");
   }
   for(const auto object : action.arguments()) {
      const auto id = static_cast< int64_t >(object);
      if(id < 0) {
         throw std::invalid_argument("semantic View contains an invalid action object ID");
      }
      result.arguments.push_back(id);
   }
   return result;
}

inline GraphAtomKey graph_atom_key(const SemanticAtom& atom)
{
   return GraphAtomKey{atom.predicate, atom.arguments};
}
inline GraphLiteralKey graph_literal_key(const SemanticLiteral& literal)
{
   return GraphLiteralKey{graph_atom_key(literal.atom), literal.positive};
}
inline GraphActionKey graph_action_key(const SemanticGroundAction& action)
{
   return GraphActionKey{action.action, action.arguments};
}

inline const std::vector< std::string >& semantic_objects(const GraphInput& input)
{
   if(input.task_context) {
      return input.task_context->objects;
   }
   if(input.objects == nullptr) {
      throw std::invalid_argument("semantic graph input requires an object table");
   }
   return *input.objects;
}

inline const std::vector< GraphLiteralKey >& semantic_goals(const GraphInput& input)
{
   return input.goals;
}

inline const std::vector< SemanticAtom >& semantic_static_facts(const GraphInput& input)
{
   static const std::vector< SemanticAtom > empty;
   return input.task_context ? input.task_context->static_facts : empty;
}

inline GraphInput make_graph_input(const SemanticFlatRelationInput& input)
{
   GraphInput result{
      .task_context = input.task_context,
      .objects = &input.objects,
      .history_max_steps = input.history_max_steps,
   };
   for(const auto& atom : input.state_facts) {
      result.state_facts.push_back(graph_atom_key(atom));
   }
   const auto& goals = input.task_context and input.use_default_goals
                          ? input.task_context->default_goals
                          : input.goals;
   for(const auto& goal : goals) {
      result.goals.push_back(graph_literal_key(goal));
   }
   for(const auto& action : input.actions) {
      result.actions.push_back(graph_action_key(action));
   }
   for(const auto& layer : input.subgoal_layers) {
      auto& target = result.subgoal_layers.emplace_back();
      for(const auto& goal : layer) {
         target.push_back(graph_literal_key(goal));
      }
   }
   for(const auto& entry : input.history) {
      auto& target = result.history.emplace_back();
      target.dt = entry.dt;
      for(const auto& literal : entry.literals) {
         target.literals.push_back(graph_literal_key(literal));
      }
   }
   return result;
}

template < views::StateView State, views::GroundActionRange Actions >
[[nodiscard]] GraphInput make_graph_input(
   const std::shared_ptr< const SemanticTaskContext >& context,
   const State& state,
   Actions&& actions
)
{
   if(not context) {
      throw std::invalid_argument("semantic View input requires a task context");
   }
   GraphInput result{.task_context = context};
   for(const auto atom : state.fluent_atoms()) {
      result.state_facts.push_back(graph_atom_key(atom));
   }
   for(const auto atom : state.derived_atoms()) {
      result.state_facts.push_back(graph_atom_key(atom));
   }
   for(const auto& goal : context->default_goals) {
      result.goals.push_back(graph_literal_key(goal));
   }
   for(const auto& action : actions) {
      result.actions.push_back(graph_action_key(action));
   }
   return result;
}

template < views::StateView State, views::LiteralRange Goals, views::GroundActionRange Actions >
[[nodiscard]] GraphInput make_graph_input(
   const std::shared_ptr< const SemanticTaskContext >& context,
   const State& state,
   Goals&& goals,
   Actions&& actions
)
{
   auto result = make_graph_input(context, state, std::forward< Actions >(actions));
   result.goals.clear();
   for(const auto& goal : goals) {
      result.goals.push_back(graph_literal_key(goal));
   }
   return result;
}

template <
   views::StateView State,
   views::LiteralRange Goals,
   views::LiteralLayerRange Layers,
   views::GroundActionRange Actions >
[[nodiscard]] GraphInput make_graph_input(
   const std::shared_ptr< const SemanticTaskContext >& context,
   const State& state,
   Goals&& goals,
   Layers&& layers,
   Actions&& actions
)
{
   auto result = make_graph_input(
      context, state, std::forward< Goals >(goals), std::forward< Actions >(actions)
   );
   for(const auto& layer : layers) {
      auto& target = result.subgoal_layers.emplace_back();
      for(const auto& goal : layer) {
         target.push_back(graph_literal_key(goal));
      }
   }
   return result;
}

template <
   views::StateView State,
   views::LiteralRange Goals,
   views::LiteralLayerRange Layers,
   views::GroundActionRange Actions,
   views::HistoryRange History >
[[nodiscard]] GraphInput make_graph_input(
   const std::shared_ptr< const SemanticTaskContext >& context,
   const State& state,
   Goals&& goals,
   Layers&& layers,
   Actions&& actions,
   History&& history,
   std::optional< int64_t > history_max_steps = std::nullopt
)
{
   auto result = make_graph_input(
      context,
      state,
      std::forward< Goals >(goals),
      std::forward< Layers >(layers),
      std::forward< Actions >(actions)
   );
   result.history_max_steps = history_max_steps;
   for(const auto& entry : history) {
      auto& target = result.history.emplace_back();
      target.dt = static_cast< int64_t >(entry.dt());
      for(const auto& literal : entry.literals()) {
         target.literals.push_back(graph_literal_key(literal));
      }
   }
   return result;
}

}  // namespace mifrost::canonical::detail

namespace mifrost::detail {

inline const std::vector< std::string >& semantic_objects(const GraphInput& input)
{
   return canonical::detail::semantic_objects(input);
}
inline const std::vector< GraphLiteralKey >& semantic_goals(const GraphInput& input)
{
   return canonical::detail::semantic_goals(input);
}
inline const std::vector< SemanticAtom >& semantic_static_facts(const GraphInput& input)
{
   return canonical::detail::semantic_static_facts(input);
}

}  // namespace mifrost::detail

namespace mifrost {

[[nodiscard]] inline std::vector< SemanticGoalLevel > semantic_goal_levels(
   const ::mifrost::detail::GraphInput& input
)
{
   std::vector< SemanticGoalLevel > levels;
   levels.reserve(input.goals.size());
   for(const auto& goal : input.goals) {
      levels.push_back({static_cast< SemanticLiteral >(goal), 0});
   }
   size_t layer = 0;
   for(const auto& goals : input.subgoal_layers) {
      for(const auto& goal : goals) {
         levels.push_back({static_cast< SemanticLiteral >(goal), layer + 1});
      }
      ++layer;
   }
   std::ranges::sort(levels);
   return levels;
}

}  // namespace mifrost
