/**
 * @file hetero_relation_schema.hpp
 * @brief Persistent relation schema for heterogeneous-graph encoders.
 *
 * Hetero relation "shape" is fully captured by argument-position count (used directly as the
 * edge-type position suffix); there is no flat-style slot-role or predicate-virtual-node
 * machinery, so the layout type is minimal.
 *
 * Unlike `FlatRelationSchema`, hetero node/edge types are lazily-created dict keys rather than
 * indices into a closed, pre-sized array: some relation types (e.g. literal-mode reuse when
 * `support_literals=false`, successor-suffixed types) are legitimately never pre-declared.
 * `name_for()` therefore never throws — it resolves a declared key's precomputed name, or
 * formats and memoizes an undeclared key's name on first use.
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mifrost/core/common_types.hpp"
#include "mifrost/core/encoders/common/relation_dict_types.hpp"
#include "mifrost/core/encoders/common/relation_key.hpp"

namespace mifrost {

/// Minimal hetero-backend relation layout: argument-position count only.
struct HeteroRelationLayout {
   int64_t arity = 0;

   [[nodiscard]] auto operator==(const HeteroRelationLayout& other) const -> bool = default;
};

class HeteroRelationSchema;

/**
 * @brief Mutable builder for an immutable `HeteroRelationSchema`.
 *
 * Registration is keyed by structured `RelationKey`:
 *  - same key, same arity: no-op (idempotent);
 *  - same key, different arity: throws immediately.
 *
 * This is new validation relative to the legacy hetero registration path, which overwrote
 * `relation_dict_.arity[name]` with no compatibility check at all.
 */
class HeteroRelationSchemaBuilder {
  public:
   void register_relation(RelationKey key, HeteroRelationLayout layout, RelationUsage usage);

   [[nodiscard]] bool contains(const RelationKey& key) const { return entries_.contains(key); }
   [[nodiscard]] size_t size() const { return entries_.size(); }

   /// Finalize into an immutable schema. Throws if no relations were registered.
   [[nodiscard]] HeteroRelationSchema finalize(
      int max_goal_level,
      bool support_literals,
      const std::set< GoalDerivation >& goal_derivations,
      std::string_view empty_error_message
   ) &&;

  private:
   struct PendingEntry {
      HeteroRelationLayout layout;
      RelationUsage usage;
   };

   hash_map< RelationKey, PendingEntry, RelationKeyHash > entries_;
};

/**
 * @brief Immutable, persistent hetero relation schema.
 *
 * Owns the declared relation catalog (names/arities/sources plus the `RelationDict` compat/export
 * view) and resolves node/edge-type strings for both declared and undeclared keys via
 * `name_for()`.
 */
class HeteroRelationSchema {
  public:
   HeteroRelationSchema() = default;

   [[nodiscard]] size_t size() const { return names_.size(); }
   [[nodiscard]] const std::vector< std::string >& names() const { return names_; }
   [[nodiscard]] const std::vector< int64_t >& arities() const { return arities_; }
   [[nodiscard]] const std::vector< std::string >& sources() const { return sources_; }
   [[nodiscard]] const RelationDict& relation_dict() const { return relation_dict_; }

   [[nodiscard]] bool contains(const RelationKey& key) const
   {
      return declared_id_by_key_.contains(key);
   }

   /// Resolve the exported node/edge-type string for `key`. Never throws: declared keys return
   /// their precomputed name; undeclared keys are formatted and memoized on first use.
   [[nodiscard]] const std::string& name_for(const RelationKey& key) const;

  private:
   friend class HeteroRelationSchemaBuilder;

   std::vector< std::string > names_;
   std::vector< int64_t > arities_;
   std::vector< std::string > sources_;
   RelationDict relation_dict_;
   hash_map< RelationKey, int64_t, RelationKeyHash > declared_id_by_key_;
   mutable hash_map< RelationKey, std::string, RelationKeyHash > undeclared_name_cache_;
};

}  // namespace mifrost
