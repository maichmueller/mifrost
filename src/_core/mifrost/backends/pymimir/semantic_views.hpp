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
   NativeLiteralView(NativeLiteralVariant value, const views::Context& context)
       : value_(std::move(value)), context_(&context)
   {
   }

   [[nodiscard]] bool is_negated_impl() const
   {
      return std::visit([](const auto literal) { return not literal->get_polarity(); }, value_);
   }

   [[nodiscard]] NativeAtomView atom_impl() const
   {
      return std::visit(
         [this](const auto literal) {
            return NativeAtomView{NativeAtomVariant{literal->get_atom()}, *context_};
         },
         value_
      );
   }

  private:
   NativeLiteralVariant value_;
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
         return NativeLiteralView{*value_, *context_};
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

using NativeStaticGoalLevels = std::remove_cvref_t<
   decltype(std::declval< GoalInputs& >().static_goal_levels) >;
using NativeFluentGoalLevels = std::remove_cvref_t<
   decltype(std::declval< GoalInputs& >().fluent_goal_levels) >;
using NativeDerivedGoalLevels = std::remove_cvref_t<
   decltype(std::declval< GoalInputs& >().derived_goal_levels) >;

struct NativeGoalSources {
   std::span< const StaticLiteral > static_goals;
   std::span< const FluentLiteral > fluent_goals;
   std::span< const DerivedLiteral > derived_goals;
   const NativeStaticGoalLevels* static_goal_levels = nullptr;
   const NativeFluentGoalLevels* fluent_goal_levels = nullptr;
   const NativeDerivedGoalLevels* derived_goal_levels = nullptr;
};

/**
 * A filtered view over the original native goal lists.
 *
 * GoalInputs stores the three literal categories separately, while the
 * semantic encoders consume one level-ordered range. This iterator joins the
 * borrowed lists and filters by the existing level maps without allocating a
 * second native goal container. `NativeGoalLayersView` visits only occupied
 * levels, so a sparse high level does not expand into empty intermediate lanes.
 */
class NativeGoalLiteralsView {
  public:
   NativeGoalLiteralsView(NativeGoalSources sources, const views::Context& context, size_t level)
       : sources_(sources), context_(&context), level_(level)
   {
   }

   class iterator {
     public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = NativeLiteralView;
      using difference_type = std::ptrdiff_t;

      iterator() = default;
      iterator(const NativeGoalLiteralsView* owner, size_t source, size_t index)
          : owner_(owner), source_(source), index_(index)
      {
         skip_unmatched();
      }

      [[nodiscard]] NativeLiteralView operator*() const noexcept
      {
         return NativeLiteralView{owner_->literal(source_, index_), *owner_->context_};
      }

      iterator& operator++() noexcept
      {
         ++index_;
         skip_unmatched();
         return *this;
      }

      iterator operator++(int) noexcept
      {
         auto copy = *this;
         ++*this;
         return copy;
      }

      friend bool operator==(const iterator&, const iterator&) = default;

     private:
      void skip_unmatched() noexcept
      {
         while(source_ < 3) {
            if(index_ >= owner_->source_size(source_)) {
               ++source_;
               index_ = 0;
               continue;
            }
            if(owner_->matches(source_, index_)) {
               return;
            }
            ++index_;
         }
      }

      const NativeGoalLiteralsView* owner_ = nullptr;
      size_t source_ = 3;
      size_t index_ = 0;
   };

   [[nodiscard]] iterator begin() const noexcept { return iterator{this, 0, 0}; }
   [[nodiscard]] iterator end() const noexcept { return iterator{this, 3, 0}; }

   [[nodiscard]] size_t size() const noexcept
   {
      return static_cast< size_t >(std::ranges::distance(*this));
   }
   [[nodiscard]] size_t level() const noexcept { return level_; }

  private:
   [[nodiscard]] size_t source_size(size_t source) const noexcept
   {
      switch(source) {
         case 0: return sources_.static_goals.size();
         case 1: return sources_.fluent_goals.size();
         case 2: return sources_.derived_goals.size();
         default: return 0;
      }
   }

   [[nodiscard]] size_t source_level(size_t source, size_t index) const noexcept
   {
      switch(source) {
         case 0:
            return sources_.static_goal_levels == nullptr
                      ? 0
                      : sources_.static_goal_levels->at(sources_.static_goals[index]);
         case 1:
            return sources_.fluent_goal_levels == nullptr
                      ? 0
                      : sources_.fluent_goal_levels->at(sources_.fluent_goals[index]);
         case 2:
            return sources_.derived_goal_levels == nullptr
                      ? 0
                      : sources_.derived_goal_levels->at(sources_.derived_goals[index]);
         default: return 0;
      }
   }

   [[nodiscard]] bool matches(size_t source, size_t index) const noexcept
   {
      return source_level(source, index) == level_;
   }

   [[nodiscard]] NativeLiteralVariant literal(size_t source, size_t index) const
   {
      switch(source) {
         case 0: return sources_.static_goals[index];
         case 1: return sources_.fluent_goals[index];
         case 2: return sources_.derived_goals[index];
         default: return NativeLiteralVariant{};
      }
   }

   NativeGoalSources sources_;
   const views::Context* context_;
   size_t level_;

   friend class NativeGoalViews;
};

class NativeGoalLayersView {
  public:
   NativeGoalLayersView(
      NativeGoalSources sources,
      const views::Context& context,
      const std::vector< size_t >& occupied_levels
   )
       : sources_(sources), context_(&context), occupied_levels_(&occupied_levels)
   {
   }

   class iterator {
     public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = NativeGoalLiteralsView;
      using difference_type = std::ptrdiff_t;

      iterator() = default;
      iterator(const NativeGoalLayersView* owner, size_t index) : owner_(owner), index_(index) {}

      [[nodiscard]] NativeGoalLiteralsView operator*() const noexcept
      {
         return NativeGoalLiteralsView{
            owner_->sources_, *owner_->context_, owner_->occupied_levels_->at(index_)
         };
      }

      iterator& operator++() noexcept
      {
         ++index_;
         return *this;
      }

      iterator operator++(int) noexcept
      {
         auto copy = *this;
         ++*this;
         return copy;
      }

      friend bool operator==(const iterator&, const iterator&) = default;

     private:
      const NativeGoalLayersView* owner_ = nullptr;
      size_t index_ = 0;
   };

   [[nodiscard]] iterator begin() const noexcept { return iterator{this, 0}; }
   [[nodiscard]] iterator end() const noexcept { return iterator{this, occupied_levels_->size()}; }
   [[nodiscard]] size_t size() const noexcept { return occupied_levels_->size(); }

  private:
   NativeGoalSources sources_;
   const views::Context* context_;
   const std::vector< size_t >* occupied_levels_;
};

struct NativeGoalViews {
   NativeGoalSources sources;
   const views::Context* context = nullptr;
   size_t max_level = 0;
   std::vector< size_t > occupied_levels;

   [[nodiscard]] NativeGoalLiteralsView goals_view() const
   {
      return NativeGoalLiteralsView{sources, *context, 0};
   }

   [[nodiscard]] NativeGoalLayersView subgoal_layers_view() const
   {
      return NativeGoalLayersView{sources, *context, occupied_levels};
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
      return make_goal_views(
         NativeGoalSources{
            .static_goals = std::span{goals.static_goals},
            .fluent_goals = std::span{goals.fluent_goals},
            .derived_goals = std::span{goals.derived_goals},
            .static_goal_levels = &goals.static_goal_levels,
            .fluent_goal_levels = &goals.fluent_goal_levels,
            .derived_goal_levels = &goals.derived_goal_levels,
         }
      );
   }

   [[nodiscard]] NativeGoalViews make_default_goal_views() const
   {
      const auto& problem = view_context_.problem();
      return make_goal_views(
         NativeGoalSources{
            .static_goals = std::span{problem.get_goal_literals< mimir::formalism::StaticTag >()},
            .fluent_goals = std::span{problem.get_goal_literals< mimir::formalism::FluentTag >()},
            .derived_goals = std::span{problem.get_goal_literals< mimir::formalism::DerivedTag >()},
         }
      );
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
      result.subgoal_layers.resize(goal_views.max_level);
      for(const auto layer : goal_views.subgoal_layers_view()) {
         auto& target = result.subgoal_layers.at(layer.level() - 1);
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

   template < typename Range, typename Map >
   static size_t validate_goal_source(const Range& values, const Map* levels)
   {
      size_t max_level = 0;
      for(const auto& literal : values) {
         size_t level = 0;
         if(levels != nullptr) {
            const auto level_it = levels->find(literal);
            if(level_it == levels->end()) {
               throw std::invalid_argument("Pymimir goal input is missing its goal level");
            }
            level = level_it->second;
         }
         max_level = std::max(max_level, level);
      }
      return max_level;
   }

   [[nodiscard]] NativeGoalViews make_goal_views(NativeGoalSources sources) const
   {
      NativeGoalViews result{
         .sources = sources,
         .context = &view_context_,
      };
      result.max_level = std::max({
         validate_goal_source(sources.static_goals, sources.static_goal_levels),
         validate_goal_source(sources.fluent_goals, sources.fluent_goal_levels),
         validate_goal_source(sources.derived_goals, sources.derived_goal_levels),
      });
      const auto append_occupied_levels = [&result](const auto& values, const auto* levels) {
         if(levels == nullptr) {
            return;
         }
         for(const auto& literal : values) {
            const auto level = levels->at(literal);
            if(level > 0) {
               result.occupied_levels.push_back(level);
            }
         }
      };
      append_occupied_levels(sources.static_goals, sources.static_goal_levels);
      append_occupied_levels(sources.fluent_goals, sources.fluent_goal_levels);
      append_occupied_levels(sources.derived_goals, sources.derived_goal_levels);
      std::ranges::sort(result.occupied_levels);
      result.occupied_levels.erase(
         std::ranges::unique(result.occupied_levels).begin(), result.occupied_levels.end()
      );
      return result;
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
