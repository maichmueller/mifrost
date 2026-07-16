/**
 * @file flat_entity_context.hpp
 * @brief Pymimir per-graph node-table helpers for flat encoders.
 *
 * Flat relation and horizon encoders share one node table. This file handles
 * object rows, predicate virtual nodes, and the role ids that tell them apart.
 */
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "mifrost/backends/pymimir/encoders/common/relation_formatter.hpp"
#include "mifrost/backends/pymimir/encoders/common/state_fact_iteration.hpp"
#include "mifrost/core/common_types.hpp"
#include "mifrost/core/encoders/flat/flat_tuple_layout.hpp"

namespace mifrost {

/**
 * @brief Structural key for per-graph predicate virtual nodes.
 *
 * This avoids formatting and string lookup for every atom. A display name is
 * only built when the graph sees the predicate symbol for the first time.
 */
struct PredicateSymbolKey {
   uint32_t predicate_index = 0;
   uint32_t tag_id = 0;

   auto operator==(const PredicateSymbolKey& other) const -> bool = default;
};

struct PredicateSymbolKeyHash {
   using is_avalanching = void;

   [[nodiscard]] auto operator()(const PredicateSymbolKey& key) const noexcept -> uint64_t;
};

/// Template implementations

template < typename Tag >
PredicateSymbolKey predicate_symbol_key_for_atom(const mimir::formalism::GroundAtom< Tag >& atom)
{
   return PredicateSymbolKey{
      .predicate_index = static_cast< uint32_t >(atom->get_predicate()->get_index()),
      .tag_id = state_fact_tag_id< Tag >(),
   };
}

template < typename Context >
void reserve_common_entity_context(
   Context& context,
   size_t object_count,
   size_t extra_entity_capacity,
   size_t predicate_capacity
)
{
   // Flat encoders append extra rows after object rows. Reserving the
   // shared vectors together keeps `entity_names` and `entity_role_ids` in lockstep.
   context.entity_names.reserve(object_count + extra_entity_capacity + predicate_capacity);
   context.entity_role_ids.reserve(context.entity_names.capacity());
   context.object_names.reserve(object_count);
   context.object_indices.reserve(object_count);
   context.entity_index_by_object_id.reserve(object_count);
   context.predicate_entity_index_by_key.reserve(predicate_capacity);
}

template < typename Context, typename ObjectRange >
void append_object_entities(Context& context, const ObjectRange& ordered_objects)
{
   size_t offset = 0;
   for(const auto& obj : ordered_objects) {
      const int64_t local_index = static_cast< int64_t >(offset++);
      context.entity_index_by_object_id.emplace(
         static_cast< int64_t >(obj->get_index()), local_index
      );
      const std::string object_name = RelationFormatter::format_object(*obj);
      context.entity_names.push_back(object_name);
      context.entity_role_ids.push_back(static_cast< int64_t >(FlatEntityRole::object));
      context.object_names.push_back(object_name);
      context.object_indices.push_back(local_index);
   }
}

template < typename Context, typename AtomTag >
int64_t ensure_predicate_virtual_entity_for_atom(
   Context& context,
   const mimir::formalism::GroundAtom< AtomTag >& atom
)
{
   // Predicate virtual nodes are per-graph, per-predicate-symbol rows. They
   // use a structural key so tuple emission does not do repeated string lookup.
   const auto key = predicate_symbol_key_for_atom(atom);
   if(const auto it = context.predicate_entity_index_by_key.find(key);
      it != context.predicate_entity_index_by_key.end()) {
      return it->second;
   }
   const int64_t local_index = static_cast< int64_t >(context.entity_names.size());
   context.predicate_entity_index_by_key.emplace(key, local_index);
   context.entity_names.push_back(
      "predicate:" + RelationFormatter::format_predicate(atom->get_predicate())
   );
   context.entity_role_ids.push_back(static_cast< int64_t >(FlatEntityRole::predicate_virtual));
   return local_index;
}

}  // namespace mifrost
