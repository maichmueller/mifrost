#pragma once

#include <map>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "common_types.hpp"
#include "flat_tuple_layout.hpp"
#include "relation_dict.hpp"

namespace mifrost {

/**
 * @brief Schema entry for one flat relation lane.
 *
 * `layout.logical_arity` tracks only the logical predicate/action arity, while
 * `layout.encoded_arity()` additionally counts auxiliary slots and an optional
 * predicate virtual node slot.
 */
struct FlatRelationSchemaEntry {
   FlatTupleLayout layout;
   std::string source;
};

/**
 * @brief Relation-name keyed registry used while materializing flat schemas.
 */
class FlatRelationSchemaRegistry {
  public:
   void add(std::string name, FlatTupleLayout layout, std::string source);
   void add_or_validate(std::string name, FlatTupleLayout layout, std::string source);

   [[nodiscard]] bool contains(const std::string& name) const;
   [[nodiscard]] size_t size() const;
   [[nodiscard]] const std::map< std::string, FlatRelationSchemaEntry >& entries() const;

  private:
   std::map< std::string, FlatRelationSchemaEntry > entries_;
};

/**
 * @brief Fully materialized relation metadata exported on flat graph outputs.
 */
struct FlatRelationSchemaMetadata {
   RelationDict relation_dict;
   std::vector< std::string > relation_names;
   std::vector< int64_t > relation_arities;
   std::vector< std::string > relation_sources;
   std::vector< int64_t > relation_logical_arities;
   std::vector< int64_t > relation_encoded_arities;
   std::vector< int64_t > relation_slot_roles;
   std::vector< int64_t > relation_slot_role_offsets;
   std::vector< std::string > slot_role_names;
   hash_map< std::string, int > relation_name_to_id;
};

FlatRelationSchemaMetadata build_flat_relation_schema_metadata(
   const FlatRelationSchemaRegistry& registry,
   int max_goal_level,
   bool support_literals,
   const std::set< GoalDerivation >& goal_derivations,
   std::string_view empty_error_message
);

}  // namespace mifrost
