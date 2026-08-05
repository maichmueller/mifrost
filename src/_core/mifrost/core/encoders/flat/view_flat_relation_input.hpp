/**
 * @file view_flat_relation_input.hpp
 * @brief Non-owning, type-erased View lanes for direct flat encoding.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>

#include "mifrost/core/views/concepts.hpp"

namespace mifrost {

struct SemanticTaskContext;

namespace canonical {

/**
 * A callback-based flat input keeps backend Views at the encoder boundary.
 *
 * The callbacks are invoked synchronously by the encoder. They expose compact
 * IDs and argument iteration without allocating a SemanticFlatRelationInput or
 * retaining a backend value beyond the encode call.
 */
struct FlatRelationViewInput {
   using ObjectVisitor = std::function< void(views::ObjectId) >;
   using ObjectRangeVisitor = std::function< void(const ObjectVisitor&) >;
   using AtomVisitor = std::function< void(views::PredicateId, const ObjectRangeVisitor&) >;
   using LiteralVisitor = std::function<
      void(views::PredicateId, bool, const ObjectRangeVisitor&) >;
   using ActionVisitor = std::function< void(views::ActionSchemaId, const ObjectRangeVisitor&) >;
   using AtomRangeVisitor = std::function< void(const AtomVisitor&) >;
   using LiteralRangeVisitor = std::function< void(const LiteralVisitor&) >;
   using LayerVisitor = std::function< void(std::size_t, const LiteralRangeVisitor&) >;
   using LayerRangeVisitor = std::function< void(const LayerVisitor&) >;
   using ActionRangeVisitor = std::function< void(const ActionVisitor&) >;
   using HistoryVisitor = std::function< void(std::int64_t, const LiteralRangeVisitor&) >;
   using HistoryRangeVisitor = std::function< void(const HistoryVisitor&) >;

   std::shared_ptr< const SemanticTaskContext > task_context;
   AtomRangeVisitor state_atoms;
   LiteralRangeVisitor goals;
   LayerRangeVisitor subgoal_layers;
   ActionRangeVisitor actions;
   HistoryRangeVisitor history;
   std::optional< std::int64_t > history_max_steps = std::nullopt;
   bool use_default_goals = false;
};

namespace detail {

template < views::AtomView Atom >
void visit_flat_atom(const Atom& atom, const FlatRelationViewInput::AtomVisitor& visitor)
{
   const auto arguments = atom.arguments();
   visitor(
      static_cast< views::PredicateId >(atom.predicate_id()),
      [&](const FlatRelationViewInput::ObjectVisitor& object_visitor) {
         for(const auto object : arguments) {
            object_visitor(static_cast< views::ObjectId >(object));
         }
      }
   );
}

template < views::LiteralView Literal >
void visit_flat_literal(
   const Literal& literal,
   const FlatRelationViewInput::LiteralVisitor& visitor
)
{
   const auto atom = literal.atom();
   const auto arguments = atom.arguments();
   visitor(
      static_cast< views::PredicateId >(atom.predicate_id()),
      not static_cast< bool >(literal.is_negated()),
      [&](const FlatRelationViewInput::ObjectVisitor& object_visitor) {
         for(const auto object : arguments) {
            object_visitor(static_cast< views::ObjectId >(object));
         }
      }
   );
}

template < views::GroundActionView Action >
void visit_flat_action(const Action& action, const FlatRelationViewInput::ActionVisitor& visitor)
{
   const auto arguments = action.arguments();
   visitor(
      static_cast< views::ActionSchemaId >(action.schema_id()),
      [&](const FlatRelationViewInput::ObjectVisitor& object_visitor) {
         for(const auto object : arguments) {
            object_visitor(static_cast< views::ObjectId >(object));
         }
      }
   );
}

}  // namespace detail

/**
 * Build a direct View input with default task goals and no action lane.
 */
template < views::StateView State >
[[nodiscard]] FlatRelationViewInput make_flat_relation_view_input(
   std::shared_ptr< const SemanticTaskContext > task_context,
   const State& state
)
{
   FlatRelationViewInput result;
   result.task_context = std::move(task_context);
   result.use_default_goals = true;
   result.state_atoms = [state](const auto& visitor) {
      for(const auto atom : state.fluent_atoms()) {
         detail::visit_flat_atom(atom, visitor);
      }
      for(const auto atom : state.derived_atoms()) {
         detail::visit_flat_atom(atom, visitor);
      }
   };
   return result;
}

/**
 * Build a direct View input with default task goals and ground actions.
 */
template < views::StateView State, views::GroundActionRange Actions >
[[nodiscard]] FlatRelationViewInput make_flat_relation_view_input(
   std::shared_ptr< const SemanticTaskContext > task_context,
   const State& state,
   Actions actions
)
{
   auto result = make_flat_relation_view_input(std::move(task_context), state);
   result.actions = [actions](const auto& visitor) {
      for(const auto& action : actions) {
         detail::visit_flat_action(action, visitor);
      }
   };
   return result;
}

/**
 * Build a direct View input with explicit goals and ground actions.
 */
template < views::StateView State, views::LiteralRange Goals, views::GroundActionRange Actions >
[[nodiscard]] FlatRelationViewInput make_flat_relation_view_input(
   std::shared_ptr< const SemanticTaskContext > task_context,
   const State& state,
   Goals goals,
   Actions actions
)
{
   auto result = make_flat_relation_view_input(std::move(task_context), state, actions);
   result.use_default_goals = false;
   result.goals = [goals](const auto& visitor) {
      for(const auto& goal : goals) {
         detail::visit_flat_literal(goal, visitor);
      }
   };
   return result;
}

}  // namespace canonical
}  // namespace mifrost
