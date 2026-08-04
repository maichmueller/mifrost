/**
 * @file semantic_flat_relation_view_bridge.hpp
 * @brief Materialize backend-neutral planning Views for semantic encoders.
 */
#pragma once

#include <algorithm>
#include <cstdint>
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

template < views::StateView State >
void materialize_state(const State& state, SemanticFlatRelationInput& result)
{
   std::vector< SemanticAtom > fluent;
   for(const auto atom : state.fluent_atoms()) {
      fluent.push_back(materialize_atom(atom));
   }
   std::ranges::sort(fluent);

   std::vector< SemanticAtom > derived;
   for(const auto atom : state.derived_atoms()) {
      derived.push_back(materialize_atom(atom));
   }
   std::ranges::sort(derived);

   result.state_facts.reserve(fluent.size() + derived.size());
   result.state_facts.insert(result.state_facts.end(), fluent.begin(), fluent.end());
   result.state_facts.insert(result.state_facts.end(), derived.begin(), derived.end());
}

template < views::StateView State, views::GroundActionRange Actions >
[[nodiscard]] SemanticFlatRelationInput materialize(
   const std::shared_ptr< const SemanticTaskContext >& context,
   const State& state,
   Actions&& actions
)
{
   require_semantic_view_context(context);
   SemanticFlatRelationInput result;
   result.task_context = context;
   result.use_default_goals = true;
   materialize_state(state, result);
   for(const auto& action : actions) {
      result.actions.push_back(materialize_action(action));
   }
   return result;
}

}  // namespace detail

/**
 * Materialize a state, explicit goals, and ground actions into the compact
 * transport consumed by every semantic encoder. Backend values never cross
 * this boundary; only the IDs exposed by the Views are copied.
 */
template < views::StateView State, views::LiteralRange Goals, views::GroundActionRange Actions >
[[nodiscard]] SemanticFlatRelationInput make_semantic_flat_relation_input(
   const std::shared_ptr< const SemanticTaskContext >& context,
   const State& state,
   Goals&& goals,
   Actions&& actions
)
{
   auto result = detail::materialize(context, state, std::forward< Actions >(actions));
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
   return detail::materialize(context, state, std::forward< Actions >(actions));
}

/**
 * Encode granular Views without requiring callers to assemble a semantic
 * input record. The engine supplies the shared task context.
 */
template <
   typename Encoder,
   views::StateView State,
   views::LiteralRange Goals,
   views::GroundActionRange Actions >
   requires requires(const Encoder& encoder, const SemanticFlatRelationInput& input) {
      encoder.encode(input);
   }
[[nodiscard]] auto encode_semantic_views(
   const Encoder& encoder,
   const std::shared_ptr< const SemanticTaskContext >& context,
   const State& state,
   Goals&& goals,
   Actions&& actions
)
{
   return encoder.encode(make_semantic_flat_relation_input(
      context, state, std::forward< Goals >(goals), std::forward< Actions >(actions)
   ));
}

template <
   typename Encoder,
   views::StateView State,
   views::LiteralRange Goals,
   views::GroundActionRange Actions >
   requires requires(const Encoder& encoder) { encoder.get_task_context(); }
[[nodiscard]] auto
encode_semantic_views(const Encoder& encoder, const State& state, Goals&& goals, Actions&& actions)
{
   return encode_semantic_views(
      encoder,
      encoder.get_task_context(),
      state,
      std::forward< Goals >(goals),
      std::forward< Actions >(actions)
   );
}

template < typename Encoder, views::StateView State, views::GroundActionRange Actions >
   requires requires(const Encoder& encoder) { encoder.get_task_context(); }
[[nodiscard]] auto
encode_semantic_views(const Encoder& encoder, const State& state, Actions&& actions)
{
   return encoder.encode(make_semantic_flat_relation_input(
      encoder.get_task_context(), state, std::forward< Actions >(actions)
   ));
}

}  // namespace mifrost::canonical
