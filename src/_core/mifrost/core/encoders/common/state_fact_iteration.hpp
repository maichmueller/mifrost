#pragma once

#include <cstdint>
#include <mimir/formalism/problem.hpp>
#include <mimir/search/state.hpp>
#include <type_traits>

namespace mifrost {

template < typename Tag >
constexpr uint32_t state_fact_tag_id()
{
   if constexpr(std::is_same_v< Tag, mimir::formalism::StaticTag >) {
      return 1U;
   }
   if constexpr(std::is_same_v< Tag, mimir::formalism::FluentTag >) {
      return 2U;
   }
   return 3U;
}

inline uint64_t pack_state_fact_key(uint32_t atom_index, uint32_t tag_id)
{
   return (static_cast< uint64_t >(atom_index) << 32) | static_cast< uint64_t >(tag_id);
}

template < typename Tag >
uint64_t state_fact_key_for_atom(const mimir::formalism::GroundAtom< Tag >& atom)
{
   return pack_state_fact_key(
      static_cast< uint32_t >(atom->get_index()), state_fact_tag_id< Tag >()
   );
}

template < typename Fn >
void for_each_state_fact_atom(const mimir::search::State& state, bool include_static, Fn&& fn)
{
   const auto& problem = state.get_problem();
   const auto& repos = problem.get_repositories();

   if(include_static) {
      for(const auto& literal : problem.get_initial_literals< mimir::formalism::StaticTag >()) {
         if(literal->get_polarity()) {
            fn(literal->get_atom());
         }
      }
   }

   const auto fluent_atoms = repos.get_ground_atoms_from_indices< mimir::formalism::FluentTag >(
      state.get_atoms< mimir::formalism::FluentTag >()
   );
   for(const auto& atom : fluent_atoms) {
      fn(atom);
   }

   const auto derived_atoms = repos.get_ground_atoms_from_indices< mimir::formalism::DerivedTag >(
      state.get_atoms< mimir::formalism::DerivedTag >()
   );
   for(const auto& atom : derived_atoms) {
      fn(atom);
   }
}

template < typename SetLike >
void collect_state_fact_keys(const mimir::search::State& state, bool include_static, SetLike& out)
{
   for_each_state_fact_atom(state, include_static, [&](const auto& atom) {
      out.insert(state_fact_key_for_atom(atom));
   });
}

}  // namespace mifrost
