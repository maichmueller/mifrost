/**
 * @file flat_relation_schema.hpp
 * @brief Relation-schema registry and exported metadata for flat encoders.
 *
 * Flat encoders register relation layouts first and build graph attrs only
 * after the schema order is fixed. This keeps relation ids, arities, slot
 * roles, and exported names aligned across batches.
 */
#pragma once

#include <map>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "flat_tuple_layout.hpp"
#include "mifrost/core/common_types.hpp"
#include "mifrost/core/encoders/common/relation_dict_types.hpp"

namespace mifrost {

/**
 * @brief Schema entry for one flat relation family.
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
   /// Insert a new relation entry. Caller must ensure uniqueness.
   void add(std::string name, FlatTupleLayout layout, std::string source);
   /// Insert or validate an existing relation entry against the same layout/source.
   void add_or_validate(std::string name, FlatTupleLayout layout, std::string source);

   [[nodiscard]] bool contains(const std::string& name) const;
   [[nodiscard]] size_t size() const;
   [[nodiscard]] const std::map< std::string, FlatRelationSchemaEntry >& entries() const;

  private:
   std::map< std::string, FlatRelationSchemaEntry > entries_;
};

/**
 * @brief Full relation metadata exported on flat graph outputs.
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

/**
 * @brief Build exported relation metadata from a schema registry.
 *
 * Invariant:
 *  - all metadata vectors are emitted in the same stable relation order so
 *    relation ids, names, logical arities, encoded arities, and slot-role
 *    offsets remain index-aligned.
 */
FlatRelationSchemaMetadata build_flat_relation_schema_metadata(
   const FlatRelationSchemaRegistry& registry,
   int max_goal_level,
   bool support_literals,
   const std::set< GoalDerivation >& goal_derivations,
   std::string_view empty_error_message
);

}  // namespace mifrost
