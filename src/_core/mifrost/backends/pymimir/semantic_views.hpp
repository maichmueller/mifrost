/**
 * @file semantic_views.hpp
 * @brief Pymimir task/problem adapter for canonical semantic encoders.
 */
#pragma once

#include <algorithm>
#include <memory>
#include <mimir/formalism/problem.hpp>
#include <mimir/search/state.hpp>
#include <ranges>
#include <span>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "mifrost/backends/pymimir/encoders/common/goal_inputs.hpp"
#include "mifrost/backends/pymimir/views.hpp"
#include "mifrost/core/encoders/flat/semantic_flat_relation_view_bridge.hpp"
#include "mifrost/core/encoders/homo/semantic_color_encoder.hpp"

namespace mifrost::pymimir {

template < typename Tag >
void append_semantic_predicates(
   const mimir::formalism::DomainImpl& domain,
   SemanticPredicateCategory category,
   std::vector< SemanticPredicateSpec >& predicates
)
{
   auto values = domain.template get_predicates< Tag >();
   std::ranges::sort(values, [](const auto lhs, const auto rhs) {
      return std::tuple{lhs->get_name(), lhs->get_arity(), lhs->get_index()}
             < std::tuple{rhs->get_name(), rhs->get_arity(), rhs->get_index()};
   });
   for(const auto predicate : values) {
      predicates.push_back(
         SemanticPredicateSpec{
            category,
            std::string(predicate->get_name()),
            static_cast< int64_t >(predicate->get_arity())
         }
      );
   }
}

[[nodiscard]] inline std::vector< SemanticPredicateSpec > make_semantic_predicates(
   const mimir::formalism::DomainImpl& domain
)
{
   std::vector< SemanticPredicateSpec > predicates;
   append_semantic_predicates< mimir::formalism::StaticTag >(
      domain, SemanticPredicateCategory::static_predicate, predicates
   );
   append_semantic_predicates< mimir::formalism::FluentTag >(
      domain, SemanticPredicateCategory::fluent, predicates
   );
   append_semantic_predicates< mimir::formalism::DerivedTag >(
      domain, SemanticPredicateCategory::derived, predicates
   );
   return predicates;
}

class SemanticProblemAdapter {
  public:
   explicit SemanticProblemAdapter(const mimir::formalism::ProblemImpl& problem)
       : view_context_(problem), task_context_(build_task_context(problem))
   {
      build_problem_lanes(problem);
   }

   [[nodiscard]] SemanticFlatRelationInput make_input(const mimir::search::State& state) const
   {
      SemanticFlatRelationInput result;
      result.task_context = task_context_;
      result.use_default_goals = true;
      materialize_state_in_native_order(views::make_state_view(state, view_context_), result);
      return result;
   }

   [[nodiscard]] std::shared_ptr< const SemanticTaskContext > get_task_context() const noexcept
   {
      return task_context_;
   }

   [[nodiscard]] views::StateView make_state_view(const mimir::search::State& state) const
   {
      return views::make_state_view(state, view_context_);
   }

   template < std::ranges::input_range Actions >
   [[nodiscard]] auto make_action_views(Actions&& actions) const
   {
      using NativeAction = std::remove_cvref_t< std::ranges::range_value_t< Actions > >;
      std::vector< views::GroundActionView< NativeAction > > result;
      for(const auto& action : actions) {
         result.emplace_back(action, view_context_);
      }
      return result;
   }

   template < std::ranges::input_range Actions >
   [[nodiscard]] SemanticFlatRelationSink
   make_sink(const mimir::search::State& state, Actions&& actions) const
   {
      const auto action_views = make_action_views(std::forward< Actions >(actions));
      return canonical::make_semantic_flat_relation_sink(
         task_context_, views::make_state_view(state, view_context_), action_views
      );
   }

   template < std::ranges::input_range Actions >
   [[nodiscard]] SemanticFlatRelationSink
   make_sink(const mimir::search::State& state, const GoalInputs& goals, Actions&& actions) const
   {
      auto result = make_sink(state, std::forward< Actions >(actions));
      result.use_default_goals = false;
      std::vector< std::vector< SemanticLiteral > > layers;
      append_goals< mimir::formalism::StaticTag, views::Category::static_predicate >(
         goals.static_goals, goals.static_goal_levels, layers
      );
      append_goals< mimir::formalism::FluentTag, views::Category::fluent >(
         goals.fluent_goals, goals.fluent_goal_levels, layers
      );
      append_goals< mimir::formalism::DerivedTag, views::Category::derived >(
         goals.derived_goals, goals.derived_goal_levels, layers
      );
      if(not layers.empty()) {
         result.goals = std::move(layers.front());
      }
      if(layers.size() > 1) {
         result.subgoal_layers.assign(
            std::make_move_iterator(layers.begin() + 1), std::make_move_iterator(layers.end())
         );
      }
      return result;
   }

   [[nodiscard]] const views::Context& get_view_context() const noexcept { return view_context_; }

   [[nodiscard]] SemanticFlatRelationInput
   make_input(const mimir::search::State& state, const GoalInputs& goals) const
   {
      SemanticFlatRelationInput result;
      result.task_context = task_context_;
      result.use_default_goals = false;
      const auto state_view = views::make_state_view(state, view_context_);
      materialize_state_in_native_order(state_view, result);

      std::vector< std::vector< SemanticLiteral > > layers;
      append_goals< mimir::formalism::StaticTag, views::Category::static_predicate >(
         goals.static_goals, goals.static_goal_levels, layers
      );
      append_goals< mimir::formalism::FluentTag, views::Category::fluent >(
         goals.fluent_goals, goals.fluent_goal_levels, layers
      );
      append_goals< mimir::formalism::DerivedTag, views::Category::derived >(
         goals.derived_goals, goals.derived_goal_levels, layers
      );
      if(not layers.empty()) {
         result.goals = std::move(layers.front());
      }
      if(layers.size() > 1) {
         result.subgoal_layers.assign(
            std::make_move_iterator(layers.begin() + 1), std::make_move_iterator(layers.end())
         );
      }
      return result;
   }

  private:
   template < typename State >
      requires mifrost::views::StateView< State >
   static void
   materialize_state_in_native_order(const State& state, SemanticFlatRelationInput& result)
   {
      for(const auto atom : state.fluent_atoms()) {
         result.state_facts.push_back(canonical::materialize_semantic_atom(atom));
      }
      for(const auto atom : state.derived_atoms()) {
         result.state_facts.push_back(canonical::materialize_semantic_atom(atom));
      }
   }

   static std::shared_ptr< SemanticTaskContext > build_task_context(
      const mimir::formalism::ProblemImpl& problem
   )
   {
      auto context = std::make_shared< SemanticTaskContext >();
      const auto domain = problem.get_domain();
      context->predicates = make_semantic_predicates(*domain);
      auto actions = domain->get_actions();
      std::ranges::sort(actions, [](const auto lhs, const auto rhs) {
         return std::tuple{lhs->get_name(), lhs->get_arity(), lhs->get_index()}
                < std::tuple{rhs->get_name(), rhs->get_arity(), rhs->get_index()};
      });
      context->actions.reserve(actions.size());
      for(const auto action : actions) {
         context->actions.push_back(
            SemanticActionSpec{
               std::string(action->get_name()), static_cast< int64_t >(action->get_arity())
            }
         );
      }

      auto objects = problem.get_problem_and_domain_objects();
      std::ranges::sort(objects, [](const auto lhs, const auto rhs) {
         return std::tuple{lhs->get_name(), lhs->get_index()}
                < std::tuple{rhs->get_name(), rhs->get_index()};
      });
      context->objects.reserve(objects.size());
      for(const auto object : objects) {
         context->objects.emplace_back(object->get_name());
      }
      return context;
   }

   template < typename Tag, views::Category Category, typename Range, typename Map >
   void append_goals(
      const Range& values,
      const Map& levels,
      std::vector< std::vector< SemanticLiteral > >& layers
   ) const
   {
      for(const auto& literal : values) {
         const auto level_it = levels.find(literal);
         if(level_it == levels.end()) {
            throw std::invalid_argument("Pymimir goal input is missing its goal level");
         }
         const auto level = level_it->second;
         if(layers.size() <= level) {
            layers.resize(level + 1);
         }
         using NativeLiteral = std::remove_cvref_t< decltype(literal) >;
         const views::LiteralView< NativeLiteral, Category > view{literal, view_context_};
         layers[level].push_back(canonical::materialize_semantic_literal(view));
      }
   }

   void build_problem_lanes(const mimir::formalism::ProblemImpl& problem)
   {
      for(const auto& literal : problem.get_initial_literals< mimir::formalism::StaticTag >()) {
         if(literal->get_polarity()) {
            using NativeLiteral = std::remove_cvref_t< decltype(literal) >;
            const views::LiteralView< NativeLiteral, views::Category::static_predicate > view{
               literal, view_context_
            };
            task_context_->static_facts.push_back(
               canonical::materialize_semantic_atom(view.atom())
            );
         }
      }
      append_default_goals< mimir::formalism::StaticTag, views::Category::static_predicate >(
         problem.get_goal_literals< mimir::formalism::StaticTag >()
      );
      append_default_goals< mimir::formalism::FluentTag, views::Category::fluent >(
         problem.get_goal_literals< mimir::formalism::FluentTag >()
      );
      append_default_goals< mimir::formalism::DerivedTag, views::Category::derived >(
         problem.get_goal_literals< mimir::formalism::DerivedTag >()
      );
   }

   template < typename Tag, views::Category Category, typename Range >
   void append_default_goals(const Range& values)
   {
      for(const auto& literal : values) {
         using NativeLiteral = std::remove_cvref_t< decltype(literal) >;
         const views::LiteralView< NativeLiteral, Category > view{literal, view_context_};
         task_context_->default_goals.push_back(canonical::materialize_semantic_literal(view));
      }
   }

   views::Context view_context_;
   std::shared_ptr< SemanticTaskContext > task_context_;
};

}  // namespace mifrost::pymimir
