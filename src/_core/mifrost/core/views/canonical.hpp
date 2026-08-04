/**
 * @file canonical.hpp
 * @brief Backend-independent algorithms expressed solely in View concepts.
 */
#pragma once

#include <algorithm>
#include <ranges>
#include <utility>

#include "concepts.hpp"

namespace mifrost::canonical {

template < views::AtomView Left, views::AtomView Right >
[[nodiscard]] bool same_atom(const Left& left, const Right& right)
{
   return left.predicate_id() == right.predicate_id()
          and std::ranges::equal(left.arguments(), right.arguments());
}

template < views::AtomView Atom, views::AtomRange StateAtoms >
[[nodiscard]] bool contains_atom(const StateAtoms& state_atoms, const Atom& atom)
{
   return std::ranges::any_of(state_atoms, [&](const auto& candidate) {
      return same_atom(candidate, atom);
   });
}

template < views::LiteralView Literal, views::AtomRange StateAtoms >
[[nodiscard]] bool satisfies_literal(const StateAtoms& state_atoms, const Literal& literal)
{
   const bool present = contains_atom(state_atoms, literal.atom());
   return present != literal.is_negated();
}

template < views::StateView State, typename Callback >
void for_each_state_atom(const State& state, Callback&& callback)
{
   for(const auto& atom : state.fluent_atoms()) {
      std::forward< Callback >(callback)(atom);
   }
   for(const auto& atom : state.derived_atoms()) {
      std::forward< Callback >(callback)(atom);
   }
}

}  // namespace mifrost::canonical
