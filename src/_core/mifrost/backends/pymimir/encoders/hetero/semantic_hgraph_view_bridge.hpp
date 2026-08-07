/**
 * @file semantic_hgraph_view_bridge.hpp
 * @brief Pymimir View adapters for the backend-neutral HGraph engines.
 */
#pragma once

#include <algorithm>
#include <memory>
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "mifrost/backends/pymimir/views.hpp"
#include "mifrost/core/encoders/flat/semantic_flat_relation_view_bridge.hpp"
#include "mifrost/core/semantic/semantic_transition_dag.hpp"

namespace mifrost::pymimir::hetero_bridge {

struct Schema {
   std::vector< SemanticPredicateSpec > predicates;
   std::vector< SemanticActionSpec > actions;
};

inline Schema schema(const mimir::formalism::DomainImpl& domain)
{
   Schema result;
   auto append_predicates = [&]< typename Tag >(Tag, SemanticPredicateCategory category) {
      auto predicates = domain.get_predicates< Tag >();
      std::ranges::sort(predicates, [](const auto lhs, const auto rhs) {
         return std::tuple{lhs->get_name(), lhs->get_arity(), lhs->get_index()}
                < std::tuple{rhs->get_name(), rhs->get_arity(), rhs->get_index()};
      });
      for(const auto predicate : predicates) {
         result.predicates.push_back(
            SemanticPredicateSpec{
               .category = category,
               .name = std::string(predicate->get_name()),
               .arity = static_cast< int64_t >(predicate->get_arity()),
            }
         );
      }
   };
   append_predicates(mimir::formalism::StaticTag{}, SemanticPredicateCategory::static_predicate);
   append_predicates(mimir::formalism::FluentTag{}, SemanticPredicateCategory::fluent);
   append_predicates(mimir::formalism::DerivedTag{}, SemanticPredicateCategory::derived);

   auto actions = domain.get_actions();
   std::ranges::sort(actions, [](const auto lhs, const auto rhs) {
      return std::tuple{lhs->get_name(), lhs->get_arity(), lhs->get_index()}
             < std::tuple{rhs->get_name(), rhs->get_arity(), rhs->get_index()};
   });
   for(const auto action : actions) {
      result.actions.push_back(
         SemanticActionSpec{
            .name = std::string(action->get_name()),
            .arity = static_cast< int64_t >(action->get_arity()),
         }
      );
   }
   return result;
}

inline std::shared_ptr< SemanticProblemContext >
problem_context(const mimir::formalism::ProblemImpl& problem, const Schema& schema)
{
   auto result = std::make_shared< SemanticProblemContext >();
   result->schema = std::make_shared< SemanticSchemaContext >(
      SemanticSchemaContext{.predicates = schema.predicates, .actions = schema.actions}
   );
   const auto view_context = views::make_context(problem);

   auto objects = problem.get_problem_and_domain_objects();
   std::ranges::sort(objects, [](const auto lhs, const auto rhs) {
      return std::tuple{lhs->get_name(), lhs->get_index()}
             < std::tuple{rhs->get_name(), rhs->get_index()};
   });
   result->objects.reserve(objects.size());
   for(const auto object : objects) {
      result->objects.emplace_back(object->get_name());
   }

   const auto add_static = [&]< typename Tag >(Tag) {
      for(const auto& literal : problem.get_initial_literals< Tag >()) {
         if(not literal->get_polarity()) {
            continue;
         }
         using Native = decltype(literal);
         constexpr auto category = [] {
            if constexpr(std::is_same_v< Tag, mimir::formalism::StaticTag >) {
               return views::Category::static_predicate;
            } else if constexpr(std::is_same_v< Tag, mimir::formalism::FluentTag >) {
               return views::Category::fluent;
            } else {
               return views::Category::derived;
            }
         }();
         result->static_facts.push_back(
            canonical::materialize_semantic_atom(
               views::AtomView< decltype(literal->get_atom()), category >{
                  literal->get_atom(), view_context
               }
            )
         );
      }
   };
   add_static(mimir::formalism::StaticTag{});

   const auto add_goals = [&]< typename Tag >(Tag) {
      for(const auto& literal : problem.get_goal_literals< Tag >()) {
         using Native = decltype(literal);
         constexpr auto category = [] {
            if constexpr(std::is_same_v< Tag, mimir::formalism::StaticTag >) {
               return views::Category::static_predicate;
            } else if constexpr(std::is_same_v< Tag, mimir::formalism::FluentTag >) {
               return views::Category::fluent;
            } else {
               return views::Category::derived;
            }
         }();
         result->default_goals.push_back(
            canonical::materialize_semantic_literal(
               views::LiteralView< Native, category >{literal, view_context}
            )
         );
      }
   };
   add_goals(mimir::formalism::StaticTag{});
   add_goals(mimir::formalism::FluentTag{});
   add_goals(mimir::formalism::DerivedTag{});
   return result;
}

template < typename Tag >
inline SemanticLiteral materialize_literal(
   const mimir::formalism::GroundLiteral< Tag >& literal,
   const views::Context& context
)
{
   constexpr auto category = [] {
      if constexpr(std::is_same_v< Tag, mimir::formalism::StaticTag >) {
         return views::Category::static_predicate;
      } else if constexpr(std::is_same_v< Tag, mimir::formalism::FluentTag >) {
         return views::Category::fluent;
      } else {
         return views::Category::derived;
      }
   }();
   return canonical::materialize_semantic_literal(
      views::LiteralView< mimir::formalism::GroundLiteral< Tag >, category >{literal, context}
   );
}

inline SemanticGroundAction
materialize_action(const mimir::formalism::GroundAction& action, const views::Context& context)
{
   return canonical::materialize_semantic_action(
      views::GroundActionView< mimir::formalism::GroundAction >{action, context}
   );
}

inline SemanticFlatRelationInput state_input(
   const std::shared_ptr< const SemanticProblemContext >& context,
   const mimir::search::State& state,
   std::span< const mimir::formalism::GroundAction > actions,
   const views::Context& view_context
)
{
   const auto action_views = mifrost::views::TransformRange{
      std::span{actions}, [&view_context](const auto& action) {
         return views::GroundActionView< mimir::formalism::GroundAction >{action, view_context};
      }
   };
   auto result = canonical::make_semantic_flat_relation_input(
      context,
      views::make_state_view(state, view_context),
      std::vector< views::LiteralView<
         mimir::formalism::GroundLiteral< mimir::formalism::FluentTag >,
         views::Category::fluent > >{},
      action_views
   );
   result.use_default_goals = true;
   return result;
}

inline SemanticFlatRelationInput state_input(
   const std::shared_ptr< const SemanticProblemContext >& context,
   const mimir::search::State& state,
   std::span< const mimir::formalism::GroundAction > actions = {}
)
{
   const auto view_context = views::make_context(state.get_problem());
   return state_input(context, state, actions, view_context);
}

template < typename Literal >
inline void append_goal(
   SemanticFlatRelationInput& result,
   const Literal& goal,
   size_t level,
   const views::Context& context
)
{
   const auto literal = materialize_literal(goal, context);
   if(level == 0) {
      result.goals.push_back(literal);
   } else {
      result.subgoal_layers.resize(std::max(result.subgoal_layers.size(), level));
      result.subgoal_layers[level - 1].push_back(literal);
   }
}

inline SemanticFlatRelationInput input(
   const std::shared_ptr< const SemanticProblemContext >& context,
   const mimir::search::State& state,
   const GoalInputs& goals,
   std::span< const mimir::formalism::GroundAction > actions,
   const views::Context& view_context
)
{
   auto result = state_input(context, state, actions, view_context);
   for(const auto& goal : goals.static_goals) {
      const auto it = goals.static_goal_levels.find(goal);
      append_goal(
         result, goal, it == goals.static_goal_levels.end() ? 0 : it->second, view_context
      );
   }
   for(const auto& goal : goals.fluent_goals) {
      const auto it = goals.fluent_goal_levels.find(goal);
      append_goal(
         result, goal, it == goals.fluent_goal_levels.end() ? 0 : it->second, view_context
      );
   }
   for(const auto& goal : goals.derived_goals) {
      const auto it = goals.derived_goal_levels.find(goal);
      append_goal(
         result, goal, it == goals.derived_goal_levels.end() ? 0 : it->second, view_context
      );
   }
   result.use_default_goals = false;
   result.history_max_steps = std::nullopt;
   return result;
}

inline SemanticFlatRelationInput input(
   const std::shared_ptr< const SemanticProblemContext >& context,
   const mimir::search::State& state,
   const GoalInputs& goals,
   std::span< const mimir::formalism::GroundAction > actions = {}
)
{
   const auto view_context = views::make_context(state.get_problem());
   return input(context, state, goals, actions, view_context);
}

template < typename Target >
inline void add_history(
   Target& result,
   const std::vector< std::pair< int, std::vector< LiteralVariant > > >& history,
   const views::Context& context
)
{
   for(const auto& [dt, literals] : history) {
      SemanticHistoryEntry entry;
      entry.dt = dt;
      for(const auto& literal : literals) {
         std::visit(
            [&](const auto& native) {
               entry.literals.push_back(materialize_literal(native, context));
            },
            literal
         );
      }
      result.history.push_back(std::move(entry));
   }
}

}  // namespace mifrost::pymimir::hetero_bridge
