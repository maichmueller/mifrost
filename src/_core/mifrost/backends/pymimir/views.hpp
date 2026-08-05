/**
 * @file views.hpp
 * @brief Lazy, non-owning Views over the Pymimir planning repository.
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mimir/formalism/problem.hpp>
#include <mimir/search/state.hpp>
#include <ranges>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mifrost/core/views/concepts.hpp"
#include "mifrost/core/views/ids.hpp"

namespace mifrost::pymimir::views {

using Category = mifrost::views::PredicateCategory;

template < typename Index >
[[nodiscard]] std::int64_t raw_index(Index index) noexcept
{
   return static_cast< std::int64_t >(index);
}

/**
 * Per-problem compact identity tables. The native domain/problem and all
 * objects referenced by produced Views are borrowed, never copied.
 */
class Context {
  public:
   explicit Context(const mimir::formalism::ProblemImpl& problem) : problem_(&problem)
   {
      build_predicates< mimir::formalism::StaticTag >(Category::static_predicate);
      build_predicates< mimir::formalism::FluentTag >(Category::fluent);
      build_predicates< mimir::formalism::DerivedTag >(Category::derived);
      build_actions();
      build_objects();
   }

   [[nodiscard]] const mimir::formalism::ProblemImpl& problem() const noexcept { return *problem_; }

   [[nodiscard]] mifrost::views::PredicateId
   predicate_id(Category category, std::int64_t raw) const noexcept
   {
      const auto* ids = predicate_ids(category);
      if(ids == nullptr) {
         return -1;
      }
      const auto it = ids->find(raw);
      return it == ids->end() ? -1 : it->second;
   }

   [[nodiscard]] mifrost::views::ActionSchemaId action_id(std::int64_t raw) const noexcept
   {
      const auto it = action_ids_.find(raw);
      return it == action_ids_.end() ? -1 : it->second;
   }

   [[nodiscard]] mifrost::views::ObjectId object_id(std::int64_t raw) const noexcept
   {
      const auto it = object_ids_.find(raw);
      return it == object_ids_.end() ? -1 : it->second;
   }

  private:
   template < typename Tag >
   void build_predicates(Category category)
   {
      using Predicate = mimir::formalism::Predicate< Tag >;
      std::vector< Predicate > predicates = problem_->get_domain()
                                               ->template get_predicates< Tag >();
      std::ranges::sort(predicates, [](const auto lhs, const auto rhs) {
         return std::tuple{lhs->get_name(), lhs->get_arity(), lhs->get_index()}
                < std::tuple{rhs->get_name(), rhs->get_arity(), rhs->get_index()};
      });
      auto& ids = predicate_ids(category);
      for(const auto predicate : predicates) {
         ids.emplace(raw_index(predicate->get_index()), predicate_count_++);
      }
   }

   void build_actions()
   {
      auto actions = problem_->get_domain()->get_actions();
      std::ranges::sort(actions, [](const auto lhs, const auto rhs) {
         return std::tuple{lhs->get_name(), lhs->get_arity(), lhs->get_index()}
                < std::tuple{rhs->get_name(), rhs->get_arity(), rhs->get_index()};
      });
      for(const auto action : actions) {
         action_ids_.emplace(
            raw_index(action->get_index()), static_cast< std::int64_t >(action_ids_.size())
         );
      }
   }

   void build_objects()
   {
      auto objects = problem_->get_problem_and_domain_objects();
      std::ranges::sort(objects, [](const auto lhs, const auto rhs) {
         return std::tuple{lhs->get_name(), lhs->get_index()}
                < std::tuple{rhs->get_name(), rhs->get_index()};
      });
      for(const auto object : objects) {
         object_ids_.emplace(
            raw_index(object->get_index()), static_cast< std::int64_t >(object_ids_.size())
         );
      }
   }

   [[nodiscard]] std::unordered_map< std::int64_t, std::int64_t >& predicate_ids(
      Category category
   ) noexcept
   {
      switch(category) {
         case Category::static_predicate: return static_predicate_ids_;
         case Category::fluent: return fluent_predicate_ids_;
         case Category::derived: return derived_predicate_ids_;
      }
      return fluent_predicate_ids_;
   }
   [[nodiscard]] const std::unordered_map< std::int64_t, std::int64_t >* predicate_ids(
      Category category
   ) const noexcept
   {
      return &const_cast< Context* >(this)->predicate_ids(category);
   }

   const mimir::formalism::ProblemImpl* problem_;
   std::unordered_map< std::int64_t, std::int64_t > static_predicate_ids_;
   std::unordered_map< std::int64_t, std::int64_t > fluent_predicate_ids_;
   std::unordered_map< std::int64_t, std::int64_t > derived_predicate_ids_;
   std::unordered_map< std::int64_t, std::int64_t > action_ids_;
   std::unordered_map< std::int64_t, std::int64_t > object_ids_;
   std::int64_t predicate_count_ = 0;
};

template < typename NativePredicate, Category PredicateCategory >
class PredicateView {
  public:
   PredicateView(NativePredicate value, const Context& context) : value_(value), context_(&context)
   {
   }

   [[nodiscard]] auto id() const noexcept
   {
      return context_->predicate_id(PredicateCategory, raw_index(value_->get_index()));
   }
   [[nodiscard]] std::string_view name() const noexcept { return value_->get_name(); }
   [[nodiscard]] std::size_t arity() const noexcept
   {
      return static_cast< std::size_t >(value_->get_arity());
   }
   [[nodiscard]] Category category() const noexcept { return PredicateCategory; }

  private:
   NativePredicate value_;
   const Context* context_;
};

template < typename NativeObject >
class ObjectView {
  public:
   ObjectView(NativeObject value, const Context& context) : value_(value), context_(&context) {}

   [[nodiscard]] auto id() const noexcept
   {
      return context_->object_id(raw_index(value_->get_index()));
   }
   [[nodiscard]] std::string_view name() const noexcept { return value_->get_name(); }

  private:
   NativeObject value_;
   const Context* context_;
};

template < typename NativeAction >
class ActionSchemaView {
  public:
   ActionSchemaView(NativeAction value, const Context& context) : value_(value), context_(&context)
   {
   }

   [[nodiscard]] auto id() const noexcept
   {
      return context_->action_id(raw_index(value_->get_index()));
   }
   [[nodiscard]] std::string_view name() const noexcept { return value_->get_name(); }
   [[nodiscard]] std::size_t arity() const noexcept
   {
      return static_cast< std::size_t >(value_->get_arity());
   }

  private:
   NativeAction value_;
   const Context* context_;
};

template < typename NativeAtom, Category AtomCategory >
class AtomView: public mifrost::views::AtomViewBase< AtomView< NativeAtom, AtomCategory > > {
  public:
   AtomView(NativeAtom value, const Context& context) : value_(value), context_(&context) {}

   [[nodiscard]] auto predicate_id_impl() const noexcept
   {
      return context_->predicate_id(AtomCategory, raw_index(value_->get_predicate()->get_index()));
   }
   [[nodiscard]] auto arguments_impl() const
   {
      return value_->get_objects() | std::views::transform([context = context_](const auto object) {
                return context->object_id(raw_index(object->get_index()));
             });
   }
   [[nodiscard]] auto predicate() const
   {
      using NativePredicate = decltype(value_->get_predicate());
      return PredicateView< NativePredicate, AtomCategory >{value_->get_predicate(), *context_};
   }

  private:
   NativeAtom value_;
   const Context* context_;
};

template < typename NativeLiteral, Category LiteralCategory >
class LiteralView:
    public mifrost::views::LiteralViewBase< LiteralView< NativeLiteral, LiteralCategory > > {
  public:
   LiteralView(NativeLiteral value, const Context& context) : value_(value), context_(&context) {}

   [[nodiscard]] bool is_negated_impl() const noexcept { return not value_->get_polarity(); }
   [[nodiscard]] auto atom_impl() const
   {
      using NativeAtom = decltype(value_->get_atom());
      return AtomView< NativeAtom, LiteralCategory >{value_->get_atom(), *context_};
   }

  private:
   NativeLiteral value_;
   const Context* context_;
};

template < typename NativeAction >
class GroundActionView:
    public mifrost::views::GroundActionViewBase< GroundActionView< NativeAction > > {
  public:
   GroundActionView(NativeAction value, const Context& context) : value_(value), context_(&context)
   {
   }

   [[nodiscard]] auto schema_id_impl() const noexcept
   {
      return context_->action_id(raw_index(value_->get_action()->get_index()));
   }
   [[nodiscard]] auto arguments_impl() const
   {
      return value_->get_objects() | std::views::transform([context = context_](const auto object) {
                return context->object_id(raw_index(object->get_index()));
             });
   }
   [[nodiscard]] auto schema() const { return value_->get_action(); }
   [[nodiscard]] auto action_schema() const
   {
      return ActionSchemaView{value_->get_action(), *context_};
   }

  private:
   NativeAction value_;
   const Context* context_;
};

class StateView: public mifrost::views::StateViewBase< StateView > {
  public:
   StateView(const mimir::search::State& state, const Context& context)
       : state_(&state), context_(&context)
   {
   }

   [[nodiscard]] auto fluent_atoms_impl() const
   {
      const auto& repositories = state_->get_problem().get_repositories();
      auto atoms = repositories.get_ground_atoms_from_indices< mimir::formalism::FluentTag >(
         state_->get_atoms< mimir::formalism::FluentTag >()
      );
      std::vector<
         AtomView< mimir::formalism::GroundAtom< mimir::formalism::FluentTag >, Category::fluent > >
         result;
      for(const auto atom : atoms) {
         result.emplace_back(atom, *context_);
      }
      return result;
   }
   [[nodiscard]] auto derived_atoms_impl() const
   {
      const auto& repositories = state_->get_problem().get_repositories();
      auto atoms = repositories.get_ground_atoms_from_indices< mimir::formalism::DerivedTag >(
         state_->get_atoms< mimir::formalism::DerivedTag >()
      );
      std::vector< AtomView<
         mimir::formalism::GroundAtom< mimir::formalism::DerivedTag >,
         Category::derived > >
         result;
      for(const auto atom : atoms) {
         result.emplace_back(atom, *context_);
      }
      return result;
   }

  private:
   const mimir::search::State* state_;
   const Context* context_;
};

[[nodiscard]] inline Context make_context(const mimir::formalism::ProblemImpl& problem)
{
   return Context(problem);
}

[[nodiscard]] inline StateView
make_state_view(const mimir::search::State& state, const Context& context)
{
   return StateView(state, context);
}

static_assert(
   mifrost::views::AtomView<
      AtomView< mimir::formalism::GroundAtom< mimir::formalism::FluentTag >, Category::fluent > >
);
static_assert(mifrost::views::LiteralView< LiteralView<
                 mimir::formalism::GroundLiteral< mimir::formalism::FluentTag >,
                 Category::fluent > >);
static_assert(
   mifrost::views::GroundActionView< GroundActionView< mimir::formalism::GroundAction > >
);
static_assert(
   mifrost::views::NamedActionSchemaView< ActionSchemaView< mimir::formalism::Action > >
);
static_assert(mifrost::views::StateView< StateView >);

}  // namespace mifrost::pymimir::views
