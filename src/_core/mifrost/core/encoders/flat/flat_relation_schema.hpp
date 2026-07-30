/**
 * @file flat_relation_schema.hpp
 * @brief Persistent relation schema and exported metadata for flat encoders.
 *
 * `FlatRelationSchemaBuilder` registers relations by structured `RelationKey` and finalizes into
 * an immutable `FlatRelationSchema`, which flat encoders own as their single source of truth for
 * relation ids, arities, slot roles, and exported names.
 */
#pragma once

#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "flat_tuple_layout.hpp"
#include "mifrost/core/common_types.hpp"
#include "mifrost/core/encoders/common/relation_dict_types.hpp"
#include "mifrost/core/encoders/common/relation_key.hpp"

namespace mifrost {

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

class FlatRelationSchema;

/**
 * @brief Mutable builder for an immutable `FlatRelationSchema`.
 *
 * Registration is keyed by structured `RelationKey` rather than a formatted name:
 *  - same key, compatible layout: accepted (first-registered usage wins);
 *  - same key, incompatible layout: throws immediately;
 *  - distinct keys that format to the same exported name: throws during `finalize()`.
 *
 * There is deliberately no `add()`/`add_or_validate()` split.
 */
class FlatRelationSchemaBuilder {
  public:
   void register_relation(RelationKey key, FlatTupleLayout layout, RelationUsage usage);

   [[nodiscard]] bool contains(const RelationKey& key) const { return entries_.contains(key); }
   [[nodiscard]] size_t size() const { return entries_.size(); }

   /**
    * @brief Finalize into an immutable schema.
    *
    * Assigns stable relation ids in ascending `format_relation_name(key)` order (matching the
    * legacy registry's `std::map`-sorted iteration order). Throws if two distinct keys format to
    * the same exported name, or if no relations were registered.
    */
   [[nodiscard]] FlatRelationSchema finalize(
      int max_goal_level,
      bool support_literals,
      const std::set< GoalDerivation >& goal_derivations,
      std::string_view empty_error_message
   ) &&;

  private:
   struct PendingEntry {
      FlatTupleLayout layout;
      RelationUsage usage;
   };

   hash_map< RelationKey, PendingEntry, RelationKeyHash > entries_;
};

/**
 * @brief Immutable, persistent flat relation schema.
 *
 * Owns the stable relation entries (exported as the same struct-of-arrays shape as
 * `FlatRelationSchemaMetadata`, computed once at `finalize()` time) plus a structured
 * key-to-id index for O(1) emission-time lookups without reformatting names.
 */
class FlatRelationSchema {
  public:
   FlatRelationSchema() = default;

   [[nodiscard]] size_t size() const { return metadata_.relation_names.size(); }
   [[nodiscard]] const std::vector< std::string >& names() const
   {
      return metadata_.relation_names;
   }
   [[nodiscard]] const std::vector< int64_t >& arities() const
   {
      return metadata_.relation_arities;
   }
   [[nodiscard]] const std::vector< std::string >& sources() const
   {
      return metadata_.relation_sources;
   }
   [[nodiscard]] const std::vector< int64_t >& logical_arities() const
   {
      return metadata_.relation_logical_arities;
   }
   [[nodiscard]] const std::vector< int64_t >& encoded_arities() const
   {
      return metadata_.relation_encoded_arities;
   }
   [[nodiscard]] const std::vector< int64_t >& slot_roles() const
   {
      return metadata_.relation_slot_roles;
   }
   [[nodiscard]] const std::vector< int64_t >& slot_role_offsets() const
   {
      return metadata_.relation_slot_role_offsets;
   }
   [[nodiscard]] const std::vector< std::string >& slot_role_names() const
   {
      return metadata_.slot_role_names;
   }
   [[nodiscard]] const RelationDict& relation_dict() const { return metadata_.relation_dict; }
   [[nodiscard]] const FlatRelationSchemaMetadata& as_metadata() const { return metadata_; }

   [[nodiscard]] int id_for(const RelationKey& key) const;
   [[nodiscard]] std::optional< int > try_id_for(const RelationKey& key) const;
   [[nodiscard]] int id_for(const std::string& name) const;
   [[nodiscard]] std::optional< int > try_id_for(const std::string& name) const;

  private:
   friend class FlatRelationSchemaBuilder;

   FlatRelationSchemaMetadata metadata_;
   hash_map< RelationKey, int, RelationKeyHash > key_to_id_;
};

}  // namespace mifrost
