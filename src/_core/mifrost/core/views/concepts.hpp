/**
 * @file concepts.hpp
 * @brief Small statically dispatched concepts used by canonical encoders.
 *
 * A View is a cheap operation-bearing proxy over a backend value. The concepts
 * intentionally validate only the operations consumed by an encoder; they do
 * not impose ownership, inheritance, or a common runtime base class.
 */
#pragma once

#include <concepts>
#include <cstddef>
#include <ranges>
#include <string_view>
#include <type_traits>

#include "ids.hpp"

namespace mifrost::views {

template < typename T >
using view_value_t = std::remove_cvref_t< T >;

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

}  // namespace mifrost::views
