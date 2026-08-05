/**
 * @file semantic_flat_relation_view_bridge.hpp
 * @brief Templated semantic View entry points for the flat encoders.
 */
#pragma once

#include <memory>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mifrost/core/api.hpp"
#include "mifrost/core/encoders/flat/semantic_flat_relation_encoder.hpp"
#include "mifrost/core/views/concepts.hpp"

namespace mifrost::canonical {

/**
 * Validate the context shared by the Views and the owned semantic input.
 *
 * This is out-of-line so a caller gets one stable ABI boundary for the
 * required context check while all backend-specific traversal remains in the
 * constrained templates below.
 */
MIFROST_API void require_semantic_view_context(
   const std::shared_ptr< const SemanticTaskContext >& context
);

namespace detail {

template < views::AtomView Atom >
[[nodiscard]] SemanticAtom materialize_atom(const Atom& atom)
{
   SemanticAtom result;
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
[[nodiscard]] SemanticLiteral materialize_literal(const Literal& literal)
{
   return {
      materialize_atom(literal.atom()),
      not static_cast< bool >(literal.is_negated()),
   };
}

template < views::GroundActionView Action >
[[nodiscard]] SemanticGroundAction materialize_action(const Action& action)
{
   SemanticGroundAction result;
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

template < views::StateView State, typename Target >
void append_state(const State& state, Target& result)
{
   for(const auto atom : state.fluent_atoms()) {
      result.state_facts.push_back(materialize_atom(atom));
   }
   for(const auto atom : state.derived_atoms()) {
      result.state_facts.push_back(materialize_atom(atom));
   }
}

template < views::StateView State, views::GroundActionRange Actions >
[[nodiscard]] SemanticFlatRelationInput make_input(
   const std::shared_ptr< const SemanticTaskContext >& context,
   const State& state,
   Actions&& actions
)
{
   require_semantic_view_context(context);
   SemanticFlatRelationInput result;
   result.task_context = context;
   result.use_default_goals = true;
   append_state(state, result);
   for(const auto& action : actions) {
      result.actions.push_back(materialize_action(action));
   }
   return result;
}

}  // namespace detail

template < views::AtomView Atom >
[[nodiscard]] SemanticAtom materialize_semantic_atom(const Atom& atom)
{
   return detail::materialize_atom(atom);
}

template < views::LiteralView Literal >
[[nodiscard]] SemanticLiteral materialize_semantic_literal(const Literal& literal)
{
   return detail::materialize_literal(literal);
}

template < views::GroundActionView Action >
[[nodiscard]] SemanticGroundAction materialize_semantic_action(const Action& action)
{
   return detail::materialize_action(action);
}

/**
 * Build the neutral traversal sink used by the semantic encoders.
 */
template < views::StateView State, views::LiteralRange Goals, views::GroundActionRange Actions >
[[nodiscard]] SemanticFlatRelationInput make_semantic_flat_relation_input(
   const std::shared_ptr< const SemanticTaskContext >& context,
   const State& state,
   Goals&& goals,
   Actions&& actions
)
{
   auto result = detail::make_input(context, state, std::forward< Actions >(actions));
   result.use_default_goals = false;
   for(const auto& goal : goals) {
      result.goals.push_back(detail::materialize_literal(goal));
   }
   return result;
}

/**
 * Materialize a state and actions while selecting the task context's default
 * goals. This overload is the normal path for a planning task adapter.
 */
template < views::StateView State, views::GroundActionRange Actions >
[[nodiscard]] SemanticFlatRelationInput make_semantic_flat_relation_input(
   const std::shared_ptr< const SemanticTaskContext >& context,
   const State& state,
   Actions&& actions
)
{
   return detail::make_input(context, state, std::forward< Actions >(actions));
}

template <
   views::StateView State,
   views::LiteralRange Goals,
   views::LiteralLayerRange SubgoalLayers,
   views::GroundActionRange Actions,
   views::HistoryRange History >
[[nodiscard]] SemanticFlatRelationInput make_semantic_flat_relation_input(
   const std::shared_ptr< const SemanticTaskContext >& context,
   const State& state,
   Goals&& goals,
   SubgoalLayers&& subgoal_layers,
   Actions&& actions,
   History&& history,
   std::optional< int64_t > history_max_steps = std::nullopt
)
{
   auto result = make_semantic_flat_relation_input(
      context, state, std::forward< Actions >(actions)
   );
   result.use_default_goals = false;
   for(const auto& goal : goals) {
      result.goals.push_back(detail::materialize_literal(goal));
   }
   for(const auto& layer : subgoal_layers) {
      auto& target = result.subgoal_layers.emplace_back();
      for(const auto& goal : layer) {
         target.push_back(detail::materialize_literal(goal));
      }
   }
   for(const auto& entry : history) {
      SemanticHistoryEntry target;
      target.dt = static_cast< int64_t >(entry.dt());
      for(const auto& literal : entry.literals()) {
         target.literals.push_back(detail::materialize_literal(literal));
      }
      result.history.push_back(std::move(target));
   }
   result.history_max_steps = history_max_steps;
   return result;
}

}  // namespace mifrost::canonical

namespace mifrost {

template < views::StateView State, views::GroundActionRange Actions >
BatchBuilder::BatchEncoding
SemanticFlatRelationEncoderEngine::encode(const State& state, Actions&& actions) const
{
   BatchBuilder builder;
   encode(state, std::forward< Actions >(actions), builder);
   builder.next_graph();
   return builder.build();
}

template < views::StateView State, views::GroundActionRange Actions >
void SemanticFlatRelationEncoderEngine::encode(
   const State& state,
   Actions&& actions,
   BatchBuilder& builder
) const
{
   encode(
      canonical::make_semantic_flat_relation_input(
         get_task_context(), state, std::forward< Actions >(actions)
      ),
      builder
   );
}

template < views::StateView State, views::LiteralRange Goals, views::GroundActionRange Actions >
BatchBuilder::BatchEncoding SemanticFlatRelationEncoderEngine::encode(
   const State& state,
   Goals&& goals,
   Actions&& actions
) const
{
   BatchBuilder builder;
   encode(state, std::forward< Goals >(goals), std::forward< Actions >(actions), builder);
   builder.next_graph();
   return builder.build();
}

template < views::StateView State, views::LiteralRange Goals, views::GroundActionRange Actions >
void SemanticFlatRelationEncoderEngine::encode(
   const State& state,
   Goals&& goals,
   Actions&& actions,
   BatchBuilder& builder
) const
{
   encode(
      canonical::make_semantic_flat_relation_input(
         get_task_context(), state, std::forward< Goals >(goals), std::forward< Actions >(actions)
      ),
      builder
   );
}

template <
   views::StateView State,
   views::LiteralRange Goals,
   views::LiteralLayerRange SubgoalLayers,
   views::GroundActionRange Actions,
   views::HistoryRange History >
BatchBuilder::BatchEncoding SemanticFlatRelationEncoderEngine::encode(
   const State& state,
   Goals&& goals,
   SubgoalLayers&& subgoal_layers,
   Actions&& actions,
   History&& history,
   std::optional< int64_t > history_max_steps
) const
{
   BatchBuilder builder;
   encode(
      state,
      std::forward< Goals >(goals),
      std::forward< SubgoalLayers >(subgoal_layers),
      std::forward< Actions >(actions),
      std::forward< History >(history),
      history_max_steps,
      builder
   );
   builder.next_graph();
   return builder.build();
}

template <
   views::StateView State,
   views::LiteralRange Goals,
   views::LiteralLayerRange SubgoalLayers,
   views::GroundActionRange Actions,
   views::HistoryRange History >
void SemanticFlatRelationEncoderEngine::encode(
   const State& state,
   Goals&& goals,
   SubgoalLayers&& subgoal_layers,
   Actions&& actions,
   History&& history,
   std::optional< int64_t > history_max_steps,
   BatchBuilder& builder
) const
{
   encode(
      canonical::make_semantic_flat_relation_input(
         get_task_context(),
         state,
         std::forward< Goals >(goals),
         std::forward< SubgoalLayers >(subgoal_layers),
         std::forward< Actions >(actions),
         std::forward< History >(history),
         history_max_steps
      ),
      builder
   );
}

}  // namespace mifrost
