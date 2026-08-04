/**
 * @file views.hpp
 * @brief Lazy, non-owning Views over the PyTyr planning repository.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <ranges>
#include <string_view>
#include <type_traits>
#include <tyr/formalism/planning/ground_action_view.hpp>
#include <tyr/formalism/planning/ground_atom_view.hpp>
#include <tyr/formalism/planning/ground_literal_view.hpp>
#include <tyr/formalism/planning/planning_task.hpp>
#include <tyr/planning/ground/state_view.hpp>
#include <tyr/planning/lifted/state_view.hpp>
#include <vector>

#include "mifrost/core/views/concepts.hpp"
#include "mifrost/core/views/ids.hpp"

namespace mifrost::pytyr::views {

using Category = mifrost::views::PredicateCategory;

template < typename Index >
[[nodiscard]] std::size_t raw_index(const Index& index) noexcept
{
   return index.is_max() ? static_cast< std::size_t >(-1)
                         : static_cast< std::size_t >(index.get_value());
}

/**
 * Compact identity tables are built once per task and borrowed by every View.
 * The task itself is also borrowed; callers must keep it alive for all Views.
 */
class Context {
  public:
   Context() = default;

   Context(
      const tyr::formalism::planning::PlanningTask& task,
      const std::vector< std::int64_t >& static_predicate_ids,
      const std::vector< std::int64_t >& fluent_predicate_ids,
      const std::vector< std::int64_t >& derived_predicate_ids,
      const std::vector< std::int64_t >& action_ids,
      const std::vector< std::int64_t >& object_ids
   )
       : task_(&task),
         static_predicate_ids_(&static_predicate_ids),
         fluent_predicate_ids_(&fluent_predicate_ids),
         derived_predicate_ids_(&derived_predicate_ids),
         action_ids_(&action_ids),
         object_ids_(&object_ids)
   {
   }

   [[nodiscard]] const tyr::formalism::planning::PlanningTask& task() const noexcept
   {
      return *task_;
   }

   [[nodiscard]] mifrost::views::PredicateId
   predicate_id(Category category, std::size_t raw) const noexcept
   {
      const auto* ids = predicate_ids(category);
      return ids == nullptr or raw == static_cast< std::size_t >(-1) or raw >= ids->size()
                ? -1
                : (*ids)[raw];
   }

   [[nodiscard]] mifrost::views::ActionSchemaId action_id(std::size_t raw) const noexcept
   {
      return raw == static_cast< std::size_t >(-1) ? -1 : lookup(action_ids_, raw);
   }

   [[nodiscard]] mifrost::views::ObjectId object_id(std::size_t raw) const noexcept
   {
      return raw == static_cast< std::size_t >(-1) ? -1 : lookup(object_ids_, raw);
   }

  private:
   [[nodiscard]] const std::vector< std::int64_t >* predicate_ids(Category category) const noexcept
   {
      switch(category) {
         case Category::static_predicate: return static_predicate_ids_;
         case Category::fluent: return fluent_predicate_ids_;
         case Category::derived: return derived_predicate_ids_;
      }
      return nullptr;
   }

   [[nodiscard]] static std::int64_t
   lookup(const std::vector< std::int64_t >* ids, std::size_t raw) noexcept
   {
      return ids == nullptr or raw >= ids->size() ? -1 : (*ids)[raw];
   }

   const tyr::formalism::planning::PlanningTask* task_ = nullptr;
   const std::vector< std::int64_t >* static_predicate_ids_ = nullptr;
   const std::vector< std::int64_t >* fluent_predicate_ids_ = nullptr;
   const std::vector< std::int64_t >* derived_predicate_ids_ = nullptr;
   const std::vector< std::int64_t >* action_ids_ = nullptr;
   const std::vector< std::int64_t >* object_ids_ = nullptr;
};

template < typename NativePredicate >
class PredicateView {
  public:
   PredicateView(NativePredicate value, const Context& context, Category category)
       : value_(std::move(value)), context_(&context), category_(category)
   {
   }

   [[nodiscard]] auto id() const noexcept
   {
      return context_->predicate_id(category_, raw_index(value_.get_index()));
   }
   [[nodiscard]] std::string_view name() const noexcept { return value_.get_name(); }
   [[nodiscard]] std::size_t arity() const noexcept
   {
      return static_cast< std::size_t >(value_.get_arity());
   }
   [[nodiscard]] Category category() const noexcept { return category_; }

  private:
   NativePredicate value_;
   const Context* context_;
   Category category_;
};

template < typename NativeObject >
class ObjectView {
  public:
   ObjectView(NativeObject value, const Context& context)
       : value_(std::move(value)), context_(&context)
   {
   }

   [[nodiscard]] auto id() const noexcept
   {
      return context_->object_id(raw_index(value_.get_index()));
   }
   [[nodiscard]] std::string_view name() const noexcept { return value_.get_name(); }

  private:
   NativeObject value_;
   const Context* context_;
};

template < typename NativeAtom, Category AtomCategory >
class AtomView {
  public:
   AtomView(NativeAtom value, const Context& context) : value_(std::move(value)), context_(&context)
   {
   }

   [[nodiscard]] auto predicate_id() const noexcept
   {
      return context_->predicate_id(AtomCategory, raw_index(value_.get_predicate().get_index()));
   }
   [[nodiscard]] auto arguments() const
   {
      return value_.get_objects() | std::views::transform([context = context_](const auto object) {
                return context->object_id(raw_index(object.get_index()));
             });
   }
   [[nodiscard]] auto predicate() const
   {
      return PredicateView{value_.get_predicate(), *context_, AtomCategory};
   }

  private:
   NativeAtom value_;
   const Context* context_;
};

template < typename NativeLiteral, Category LiteralCategory >
class LiteralView {
  public:
   LiteralView(NativeLiteral value, const Context& context)
       : value_(std::move(value)), context_(&context)
   {
   }

   [[nodiscard]] bool is_negated() const noexcept
   {
      return not static_cast< bool >(value_.get_polarity());
   }
   [[nodiscard]] auto atom() const
   {
      using NativeAtom = std::remove_cvref_t< decltype(value_.get_atom()) >;
      return AtomView< NativeAtom, LiteralCategory >{value_.get_atom(), *context_};
   }

  private:
   NativeLiteral value_;
   const Context* context_;
};

template < typename NativeAction >
class ActionSchemaView {
  public:
   ActionSchemaView(NativeAction value, const Context& context)
       : value_(std::move(value)), context_(&context)
   {
   }

   [[nodiscard]] auto id() const noexcept
   {
      return context_->action_id(raw_index(value_.get_index()));
   }
   [[nodiscard]] std::string_view name() const noexcept { return value_.get_name(); }
   [[nodiscard]] std::size_t arity() const noexcept
   {
      return static_cast< std::size_t >(value_.get_original_arity());
   }

  private:
   NativeAction value_;
   const Context* context_;
};

template < typename NativeAction >
class GroundActionView {
  public:
   GroundActionView(NativeAction value, const Context& context)
       : value_(std::move(value)), context_(&context)
   {
   }

   [[nodiscard]] auto schema_id() const noexcept
   {
      return context_->action_id(raw_index(value_.get_action().get_index()));
   }
   [[nodiscard]] auto arguments() const
   {
      return value_.get_objects() | std::views::transform([context = context_](const auto object) {
                return context->object_id(raw_index(object.get_index()));
             });
   }
   [[nodiscard]] auto schema() const { return ActionSchemaView{value_.get_action(), *context_}; }

  private:
   NativeAction value_;
   const Context* context_;
};

template < typename NativeState, typename StateKind >
class StateView {
  public:
   StateView(NativeState value, const Context& context)
       : value_(std::move(value)), context_(&context)
   {
   }

   [[nodiscard]] auto fluent_atoms() const
   {
      return value_.get_fluent_facts_view() | std::views::filter([](const auto fact) {
                return static_cast< bool >(fact.get_atom());
             })
             | std::views::transform([context = context_](const auto fact) {
                  return AtomView<
                     std::remove_cvref_t< decltype(*fact.get_atom()) >,
                     Category::fluent >{*fact.get_atom(), *context};
               });
   }
   [[nodiscard]] auto derived_atoms() const
   {
      return value_.get_derived_atoms_view()
             | std::views::transform([context = context_](const auto atom) {
                  return AtomView< std::remove_cvref_t< decltype(atom) >, Category::derived >{
                     atom, *context
                  };
               });
   }

  private:
   NativeState value_;
   const Context* context_;
};

static_assert(
   mifrost::views::PredicateView<
      PredicateView< decltype(std::declval< tyr::formalism::planning::DomainView >()
                                 .template get_predicates< tyr::formalism::FluentTag >()[0]) > >
);
static_assert(
   mifrost::views::NamedActionSchemaView< ActionSchemaView< tyr::formalism::planning::ActionView > >
);

}  // namespace mifrost::pytyr::views
