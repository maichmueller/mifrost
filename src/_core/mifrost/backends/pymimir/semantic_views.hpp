/**
 * @file semantic_views.hpp
 * @brief Pymimir task/problem adapter for canonical semantic encoders.
 */
#pragma once

#include <algorithm>
#include <iterator>
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

namespace mifrost::pymimir {

using NativeLiteralVariant = LiteralVariant;
using NativeAtomVariant = std::variant<
   mimir::formalism::GroundAtom< mimir::formalism::StaticTag >,
   mimir::formalism::GroundAtom< mimir::formalism::FluentTag >,
   mimir::formalism::GroundAtom< mimir::formalism::DerivedTag > >;

template < typename Tag >
inline constexpr views::Category
   native_category = std::is_same_v< Tag, mimir::formalism::StaticTag >
                        ? views::Category::static_predicate
                        : (std::is_same_v< Tag, mimir::formalism::FluentTag >
                              ? views::Category::fluent
                              : views::Category::derived);

class NativeArgumentsView {
  public:
   NativeArgumentsView(const mimir::formalism::ObjectList* values, const views::Context& context)
       : values_(values), context_(&context)
   {
   }

   class iterator {
      using Base = mimir::formalism::ObjectList::const_iterator;

     public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = mifrost::views::ObjectId;
      using difference_type = std::ptrdiff_t;

      iterator() = default;
      iterator(Base value, const views::Context& context) : value_(value), context_(&context) {}
      [[nodiscard]] mifrost::views::ObjectId operator*() const noexcept
      {
         return context_->object_id(views::raw_index((*value_)->get_index()));
      }
      iterator& operator++() noexcept
      {
         ++value_;
         return *this;
      }
      iterator operator++(int) noexcept
      {
         auto copy = *this;
         ++value_;
         return copy;
      }
      friend bool operator==(const iterator&, const iterator&) = default;

     private:
      Base value_;
      const views::Context* context_;
   };

   [[nodiscard]] iterator begin() const noexcept { return iterator(values_->begin(), *context_); }
   [[nodiscard]] iterator end() const noexcept { return iterator(values_->end(), *context_); }

  private:
   const mimir::formalism::ObjectList* values_;
   const views::Context* context_;
};

class NativeAtomView: public mifrost::views::AtomViewBase< NativeAtomView > {
  public:
   NativeAtomView() = default;
   NativeAtomView(NativeAtomVariant value, const views::Context& context)
       : value_(std::move(value)), context_(&context)
   {
   }

   [[nodiscard]] auto predicate_id_impl() const
   {
      return std::visit(
         [this](const auto atom) {
            using NativeAtom = std::remove_cvref_t< decltype(atom) >;
            using Tag = typename std::remove_pointer_t< NativeAtom >::Type;
            return context_->predicate_id(
               native_category< Tag >, views::raw_index(atom->get_predicate()->get_index())
            );
         },
         value_
      );
   }

   [[nodiscard]] auto arguments_impl() const
   {
      return NativeArgumentsView{
         std::visit([](const auto atom) { return &atom->get_objects(); }, value_), *context_
      };
   }

  private:
   NativeAtomVariant value_;
   const views::Context* context_ = nullptr;
};

class NativeLiteralView: public mifrost::views::LiteralViewBase< NativeLiteralView > {
  public:
   NativeLiteralView() = default;
   NativeLiteralView(const NativeLiteralVariant* value, const views::Context& context)
       : value_(value), context_(&context)
   {
   }

   [[nodiscard]] bool is_negated_impl() const
   {
      return std::visit([](const auto literal) { return not literal->get_polarity(); }, *value_);
   }

   [[nodiscard]] NativeAtomView atom_impl() const
   {
      return std::visit(
         [this](const auto literal) {
            return NativeAtomView{NativeAtomVariant{literal->get_atom()}, *context_};
         },
         *value_
      );
   }

  private:
   const NativeLiteralVariant* value_ = nullptr;
   const views::Context* context_ = nullptr;
};

class NativeLiteralsView {
  public:
   NativeLiteralsView(std::span< const NativeLiteralVariant > values, const views::Context& context)
       : values_(values), context_(&context)
   {
   }

   class iterator {
      using Base = std::span< const NativeLiteralVariant >::iterator;

     public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = NativeLiteralView;
      using difference_type = std::ptrdiff_t;

      iterator() = default;
      iterator(Base value, const views::Context& context) : value_(value), context_(&context) {}
      [[nodiscard]] NativeLiteralView operator*() const noexcept
      {
         return NativeLiteralView{&*value_, *context_};
      }
      iterator& operator++() noexcept
      {
         ++value_;
         return *this;
      }
      iterator operator++(int) noexcept
      {
         auto copy = *this;
         ++value_;
         return copy;
      }
      friend bool operator==(const iterator&, const iterator&) = default;

     private:
      Base value_;
      const views::Context* context_;
   };

   [[nodiscard]] iterator begin() const noexcept { return iterator(values_.begin(), *context_); }
   [[nodiscard]] iterator end() const noexcept { return iterator(values_.end(), *context_); }
   [[nodiscard]] std::size_t size() const noexcept { return values_.size(); }

  private:
   std::span< const NativeLiteralVariant > values_;
   const views::Context* context_;
};

class NativeSubgoalLayersView {
   using Base = std::span< const std::vector< NativeLiteralVariant > >::iterator;

  public:
   NativeSubgoalLayersView(
      std::span< const std::vector< NativeLiteralVariant > > values,
      const views::Context& context
   )
       : values_(values), context_(&context)
   {
   }

   class iterator {
     public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = NativeLiteralsView;
      using difference_type = std::ptrdiff_t;

      iterator() = default;
      iterator(Base value, const views::Context& context) : value_(value), context_(&context) {}
      [[nodiscard]] NativeLiteralsView operator*() const noexcept
      {
         return NativeLiteralsView(std::span{*value_}, *context_);
      }
      iterator& operator++() noexcept
      {
         ++value_;
         return *this;
      }
      iterator operator++(int) noexcept
      {
         auto copy = *this;
         ++value_;
         return copy;
      }
      friend bool operator==(const iterator&, const iterator&) = default;

     private:
      Base value_;
      const views::Context* context_;
   };

   [[nodiscard]] iterator begin() const noexcept { return iterator(values_.begin(), *context_); }
   [[nodiscard]] iterator end() const noexcept { return iterator(values_.end(), *context_); }
   [[nodiscard]] std::size_t size() const noexcept { return values_.size(); }

  private:
   std::span< const std::vector< NativeLiteralVariant > > values_;
   const views::Context* context_;
};

struct NativeGoalViews {
   std::vector< NativeLiteralVariant > goals;
   std::vector< std::vector< NativeLiteralVariant > > subgoal_layers;
   const views::Context* context = nullptr;

   [[nodiscard]] NativeLiteralsView goals_view() const
   {
      return NativeLiteralsView(std::span{goals}, *context);
   }
   [[nodiscard]] NativeSubgoalLayersView subgoal_layers_view() const
   {
      return NativeSubgoalLayersView(std::span{subgoal_layers}, *context);
   }
};

class NativeHistoryEntryView:
    public mifrost::views::HistoryEntryViewBase< NativeHistoryEntryView > {
  public:
   NativeHistoryEntryView(
      const std::pair< int, std::vector< NativeLiteralVariant > >* value,
      const views::Context& context
   )
       : value_(value), context_(&context)
   {
   }

   [[nodiscard]] int64_t dt_impl() const noexcept { return value_->first; }
   [[nodiscard]] NativeLiteralsView literals_impl() const noexcept
   {
      return NativeLiteralsView(std::span{value_->second}, *context_);
   }

  private:
   const std::pair< int, std::vector< NativeLiteralVariant > >* value_;
   const views::Context* context_;
};

class NativeHistoryView {
   using Entry = std::pair< int, std::vector< NativeLiteralVariant > >;

  public:
   NativeHistoryView(std::span< const Entry > values, const views::Context& context)
       : values_(values), context_(&context)
   {
   }

   class iterator {
      using Base = std::span< const Entry >::iterator;

     public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = NativeHistoryEntryView;
      using difference_type = std::ptrdiff_t;

      iterator() = default;
      iterator(Base value, const views::Context& context) : value_(value), context_(&context) {}
      [[nodiscard]] NativeHistoryEntryView operator*() const noexcept
      {
         return NativeHistoryEntryView{&*value_, *context_};
      }
      iterator& operator++() noexcept
      {
         ++value_;
         return *this;
      }
      iterator operator++(int) noexcept
      {
         auto copy = *this;
         ++value_;
         return copy;
      }
      friend bool operator==(const iterator&, const iterator&) = default;

     private:
      Base value_;
      const views::Context* context_;
   };

   [[nodiscard]] iterator begin() const noexcept { return iterator(values_.begin(), *context_); }
   [[nodiscard]] iterator end() const noexcept { return iterator(values_.end(), *context_); }
   [[nodiscard]] std::size_t size() const noexcept { return values_.size(); }

  private:
   std::span< const Entry > values_;
   const views::Context* context_;
};

[[nodiscard]] inline NativeHistoryView make_history_view(
   std::span< const std::pair< int, std::vector< NativeLiteralVariant > > > values,
   const views::Context& context
)
{
   return NativeHistoryView(values, context);
}

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

   [[nodiscard]] NativeGoalViews make_goal_views(const GoalInputs& goals) const
   {
      std::vector< std::vector< NativeLiteralVariant > > layers;
      append_goals< mimir::formalism::StaticTag, views::Category::static_predicate >(
         goals.static_goals, goals.static_goal_levels, layers
      );
      append_goals< mimir::formalism::FluentTag, views::Category::fluent >(
         goals.fluent_goals, goals.fluent_goal_levels, layers
      );
      append_goals< mimir::formalism::DerivedTag, views::Category::derived >(
         goals.derived_goals, goals.derived_goal_levels, layers
      );

      NativeGoalViews result;
      result.context = &view_context_;
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

   template < std::ranges::input_range Actions >
   [[nodiscard]] auto make_action_views(Actions&& actions) const
   {
      using NativeAction = std::remove_cvref_t< std::ranges::range_value_t< Actions > >;
      return mifrost::views::TransformRange{
         std::forward< Actions >(actions), [context = &view_context_](const auto& action) {
            return views::GroundActionView< NativeAction >{action, *context};
         }
      };
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

      auto goal_views = make_goal_views(goals);
      for(const auto literal : goal_views.goals_view()) {
         result.goals.push_back(canonical::materialize_semantic_literal(literal));
      }
      for(const auto layer : goal_views.subgoal_layers_view()) {
         auto& target = result.subgoal_layers.emplace_back();
         for(const auto literal : layer) {
            target.push_back(canonical::materialize_semantic_literal(literal));
         }
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
      std::vector< std::vector< NativeLiteralVariant > >& layers
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
         layers[level].emplace_back(literal);
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
