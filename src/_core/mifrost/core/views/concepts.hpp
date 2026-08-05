/**
 * @file concepts.hpp
 * @brief Small statically dispatched concepts used by canonical encoders.
 *
 * A View is a cheap operation-bearing proxy over a backend value. The concepts
 *  intentionally validate only the operations consumed by an encoder. The
 *  small CRTP bases below provide one statically dispatched interface without
 *  introducing a runtime hierarchy or ownership.
 */
#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <string_view>
#include <type_traits>

#include "ids.hpp"

namespace mifrost::views {

template < typename T >
using view_value_t = std::remove_cvref_t< T >;

template < typename Derived >
class AtomViewBase {
  public:
   [[nodiscard]] constexpr decltype(auto) predicate_id() const noexcept(
      noexcept(derived().predicate_id_impl())
   )
   {
      return derived().predicate_id_impl();
   }

   [[nodiscard]] constexpr decltype(auto) arguments() const noexcept(
      noexcept(derived().arguments_impl())
   )
   {
      return derived().arguments_impl();
   }

  private:
   [[nodiscard]] constexpr const Derived& derived() const noexcept
   {
      return static_cast< const Derived& >(*this);
   }
};

template < typename Derived >
class LiteralViewBase {
  public:
   [[nodiscard]] constexpr decltype(auto) is_negated() const noexcept(
      noexcept(derived().is_negated_impl())
   )
   {
      return derived().is_negated_impl();
   }

   [[nodiscard]] constexpr decltype(auto) atom() const noexcept(noexcept(derived().atom_impl()))
   {
      return derived().atom_impl();
   }

  private:
   [[nodiscard]] constexpr const Derived& derived() const noexcept
   {
      return static_cast< const Derived& >(*this);
   }
};

template < typename Derived >
class GroundActionViewBase {
  public:
   [[nodiscard]] constexpr decltype(auto) schema_id() const noexcept(
      noexcept(derived().schema_id_impl())
   )
   {
      return derived().schema_id_impl();
   }

   [[nodiscard]] constexpr decltype(auto) arguments() const noexcept(
      noexcept(derived().arguments_impl())
   )
   {
      return derived().arguments_impl();
   }

  private:
   [[nodiscard]] constexpr const Derived& derived() const noexcept
   {
      return static_cast< const Derived& >(*this);
   }
};

template < typename Derived >
class StateViewBase {
  public:
   [[nodiscard]] constexpr decltype(auto) fluent_atoms() const noexcept(
      noexcept(derived().fluent_atoms_impl())
   )
   {
      return derived().fluent_atoms_impl();
   }

   [[nodiscard]] constexpr decltype(auto) derived_atoms() const noexcept(
      noexcept(derived().derived_atoms_impl())
   )
   {
      return derived().derived_atoms_impl();
   }

  private:
   [[nodiscard]] constexpr const Derived& derived() const noexcept
   {
      return static_cast< const Derived& >(*this);
   }
};

template < typename T >
concept PredicateView = requires(const view_value_t< T >& predicate) {
   { predicate.id() } -> std::convertible_to< PredicateId >;
   { predicate.arity() } -> std::convertible_to< std::size_t >;
};

template < typename T >
concept NamedPredicateView = PredicateView< T > && requires(const view_value_t< T >& predicate) {
   { predicate.name() } -> std::convertible_to< std::string_view >;
};

template < typename T >
concept ObjectView = requires(const view_value_t< T >& object) {
   { object.id() } -> std::convertible_to< ObjectId >;
};

template < typename T >
concept NamedObjectView = ObjectView< T > && requires(const view_value_t< T >& object) {
   { object.name() } -> std::convertible_to< std::string_view >;
};

template < typename T >
concept AtomView = requires(const view_value_t< T >& atom) {
   { atom.predicate_id() } -> std::convertible_to< PredicateId >;
   { atom.arguments() } -> std::ranges::input_range;
} && requires(const view_value_t< T >& atom) {
   requires std::
      convertible_to< std::ranges::range_value_t< decltype(atom.arguments()) >, ObjectId >;
};

template < typename T >
concept LiteralView = requires(const view_value_t< T >& literal) {
   { literal.is_negated() } -> std::convertible_to< bool >;
   literal.atom();
} && AtomView< view_value_t< decltype(std::declval< const view_value_t< T >& >().atom()) > >;

template < typename T >
concept ActionSchemaView = requires(const view_value_t< T >& action) {
   { action.id() } -> std::convertible_to< ActionSchemaId >;
   { action.arity() } -> std::convertible_to< std::size_t >;
};

template < typename T >
concept NamedActionSchemaView = ActionSchemaView< T > && requires(const view_value_t< T >& action) {
   { action.name() } -> std::convertible_to< std::string_view >;
};

template < typename T >
concept GroundActionView = requires(const view_value_t< T >& action) {
   { action.schema_id() } -> std::convertible_to< ActionSchemaId >;
   { action.arguments() } -> std::ranges::input_range;
} && requires(const view_value_t< T >& action) {
   requires std::
      convertible_to< std::ranges::range_value_t< decltype(action.arguments()) >, ObjectId >;
};

template < typename T >
concept StateView =
   requires(const view_value_t< T >& state) {
      { state.fluent_atoms() } -> std::ranges::input_range;
      { state.derived_atoms() } -> std::ranges::input_range;
   }
   && AtomView< std::ranges::range_value_t<
      decltype(std::declval< const view_value_t< T >& >().fluent_atoms()) > >
   && AtomView< std::ranges::range_value_t<
      decltype(std::declval< const view_value_t< T >& >().derived_atoms()) > >;

template < typename T >
concept TransitionView = requires(const view_value_t< T >& transition) {
   transition.source_state();
   transition.action();
   transition.target_state();
};

template < typename R >
concept AtomRange = std::ranges::input_range< R > && AtomView< std::ranges::range_value_t< R > >;

template < typename R >
concept LiteralRange = std::ranges::input_range< R >
                       && LiteralView< std::ranges::range_value_t< R > >;

template < typename R >
concept GroundActionRange = std::ranges::input_range< R >
                            && GroundActionView< std::ranges::range_value_t< R > >;

template < typename R >
concept LiteralLayerRange = std::ranges::input_range< R >
                            && LiteralRange< std::ranges::range_value_t< R > >;

template < typename T >
concept HistoryEntryView =
   requires(const view_value_t< T >& entry) {
      { entry.dt() } -> std::convertible_to< std::int64_t >;
      entry.literals();
   }
   && LiteralRange<
      view_value_t< decltype(std::declval< const view_value_t< T >& >().literals()) > >;

template < typename R >
concept HistoryRange = std::ranges::input_range< R >
                       && HistoryEntryView< std::ranges::range_value_t< R > >;

}  // namespace mifrost::views
